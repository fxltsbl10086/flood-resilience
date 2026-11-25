"""Native DEM grids, land-cover fractions and rainfall zones, with validated fraction reuse."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds, Window
from scipy.ndimage import binary_erosion, distance_transform_edt
from shapely.geometry import box


# Update this version only when aggregation, source NoData handling or normalization changes.
FRACTIONS_CACHE_METHOD = "ecovision_average_valid_area_normalized_v1"


def _file_sha256(path):
    """Stream file contents without loading high-resolution rasters into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fractions_cache_signature(source, grid):
    """Identify the fraction cache by source content and grid, independently of paths."""
    source = Path(source)
    before = source.stat()
    source_hash = _file_sha256(source)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError()
    mask = np.ascontiguousarray(grid["mask"])
    transform = grid["transform"]
    return {
        "method_version": FRACTIONS_CACHE_METHOD,
        "source_sha256": source_hash,
        "landuse_codes": list(range(1, 9)),
        "grid_shape": list(mask.shape),
        "grid_mask_dtype": str(mask.dtype),
        "grid_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        "grid_transform": [transform.a, transform.b, transform.c,
                           transform.d, transform.e, transform.f],
        "grid_crs_wkt": grid["crs"].to_wkt(),
    }


def _check_fraction_values(fractions, grid):
    """Require eight float32 fractions, summing to one in active cells and zero elsewhere."""
    if fractions.shape != (8, *grid["mask"].shape) or fractions.dtype != np.dtype("float32"):
        raise ValueError()
    active = grid["mask"] > 0
    if not active.any() or not np.isfinite(fractions).all():
        raise ValueError()
    if np.any(fractions < -1e-6) or np.any(fractions > 1 + 1e-6):
        raise ValueError()
    if np.any(fractions[:, ~active] != 0):
        raise ValueError()
    if np.max(np.abs(fractions.sum(axis=0, dtype="float64")[active] - 1)) > 1e-4:
        raise ValueError()


def read_landuse_fractions_cache(path, source, grid, signature=None):
    """Validate source, grid, cache contents and fractions; return None if the cache is incomplete."""
    path = Path(path)
    record_path = path.with_name(path.stem + "_cache.json")
    if not path.is_file() or not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = signature if signature is not None else _fractions_cache_signature(source, grid)
    if not isinstance(record, dict) or record.get("signature") != expected:
        raise ValueError()
    # Hashes detect an interrupted replacement that leaves mismatched cache files.
    if record.get("cache_sha256") != _file_sha256(path):
        raise ValueError()
    with rasterio.open(path) as dataset:
        if (dataset.count != 8 or dataset.dtypes != ("float32",) * 8
                or dataset.shape != grid["mask"].shape or dataset.crs != grid["crs"]
                or dataset.transform != grid["transform"] or dataset.nodata is not None):
            raise ValueError()
        fractions = dataset.read()
    _check_fraction_values(fractions, grid)
    return fractions


def write_landuse_fractions_cache(path, source, grid, fractions, signature=None):
    """Save fractions losslessly and atomically replace the two specified files."""
    path = Path(path)
    _check_fraction_values(fractions, grid)
    current = _fractions_cache_signature(source, grid)
    if signature is not None and current != signature:
        raise RuntimeError()
    path.parent.mkdir(parents=True, exist_ok=True)
    record_path = path.with_name(path.stem + "_cache.json")
    token = uuid4().hex
    temporary = path.with_name(path.stem + "." + token + ".tmp.tif")
    temporary_record = record_path.with_name(record_path.name + "." + token + ".tmp")
    rows, columns = grid["mask"].shape
    # Zero is a valid absent-class fraction; the model mask identifies inactive cells.
    with rasterio.open(temporary, "w", driver="GTiff", height=rows, width=columns,
                       count=8, dtype="float32", crs=grid["crs"], transform=grid["transform"],
                       tiled=True, compress="deflate", predictor=3) as destination:
        destination.write(fractions)
        for code in range(1, 9):
            destination.set_band_description(code, f"Area fraction for land-cover code {code}")
        destination.update_tags(method_version=FRACTIONS_CACHE_METHOD,
                                source_sha256=current["source_sha256"],
                                inactive_cells="0; active cells are defined by sfincs.msk")
    record = {"signature": current, "source_path": str(Path(source).resolve()),
              "cache_sha256": _file_sha256(temporary)}
    temporary_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    os.replace(temporary, path)
    os.replace(temporary_record, record_path)


def get_landuse_fractions(source, grid, model_root):
    """Reuse a validated cache or apply the existing area-aggregation algorithm."""
    path = Path(model_root) / "landuse_fractions.tif"
    signature = _fractions_cache_signature(source, grid)
    try:
        fractions = read_landuse_fractions_cache(path, source, grid, signature)
    except (OSError, ValueError, rasterio.errors.RasterioError) as error:
        print(f"Land cover: cache unavailable; aggregating source data again", flush=True)
        fractions = None
    if fractions is not None:
        print(f"Land cover: reusing validated area-fraction cache {path}", flush=True)
        return fractions
    fractions = aggregate_fractions(source, grid)
    write_landuse_fractions_cache(path, source, grid, fractions, signature)
    print(f"Land cover: area-fraction cache written to {path}", flush=True)
    return fractions


def load_grid(config):
    """Clip along native FABDEM pixel edges without terrain interpolation or filling."""
    study = gpd.read_file(config["sources"]["study_area"])
    if study.crs is None or study.crs.to_epsg() != config["epsg"]:
        raise ValueError()
    if study.empty or not study.geometry.is_valid.all():
        raise ValueError()
    core = study.geometry.union_all()
    domain = core.buffer(config["buffer_m"])
    with rasterio.open(config["sources"]["dem"]) as src:
        dx, dy = src.res
        if src.crs != study.crs or not np.isclose(dx, config["resolution_m"]) or not np.isclose(dx, dy):
            raise ValueError()
        if src.transform.b != 0 or src.transform.d != 0 or src.transform.e >= 0:
            raise ValueError()
        raw = from_bounds(*domain.bounds, src.transform)
        c0, r0 = math.floor(raw.col_off + 1e-8), math.floor(raw.row_off + 1e-8)
        c1, r1 = math.ceil(raw.col_off + raw.width - 1e-8), math.ceil(raw.row_off + raw.height - 1e-8)
        if c0 < 0 or r0 < 0 or c1 > src.width or r1 > src.height:
            extent = box(*src.bounds)
            overflow = max(src.bounds.left-domain.bounds[0], domain.bounds[2]-src.bounds.right,
                           src.bounds.bottom-domain.bounds[1], domain.bounds[3]-src.bounds.top, 0)
            # A sub-half-pixel buffer overflow contains no active cell centers.
            # Trim aligned empty outer rows/columns; require full study-area DEM coverage.
            if not core.covered_by(extent) or overflow >= min(dx, dy) / 2:
                raise ValueError()
            print(f"Buffer exceeds DEM by {overflow:.4f} m; trimming empty aligned outer rows/columns without filling elevation", flush=True)
            c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, src.width), min(r1, src.height)
        window = Window(c0, r0, c1 - c0, r1 - r0)
        terrain = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        transform, crs = src.window_transform(window), src.crs
    active = geometry_mask([domain], terrain.shape, transform, invert=True)
    if not active.any() or not np.isfinite(terrain[active]).all():
        raise ValueError()
    # Use natural interior routing and msk=3 zero-depth outflow at the domain edge.
    edge = active & ~binary_erosion(active, structure=np.ones((3, 3), dtype=bool))
    mask = np.where(active, np.where(edge, 3, 1), 0).astype("uint8")
    return dict(terrain=terrain, mask=mask, transform=transform, crs=crs,
                study=core, resolution=dx)


def aggregate_fractions(source, grid):
    """Aggregate valid class areas; declared source NoData is not an unknown class."""
    with rasterio.open(source) as src:
        if src.count != 1 or src.dtypes[0] != "uint8" or src.crs is None:
            raise ValueError()
        lines = [f'<VRTDataset rasterXSize="{src.width}" rasterYSize="{src.height}">',
                 f"<SRS>{escape(src.crs.to_wkt())}</SRS>",
                 "<GeoTransform>" + ",".join(map(str, src.transform.to_gdal())) + "</GeoTransform>"]
        for band in range(1, 10):
            # The ninth band detects unknown codes; normalize valid area after excluding NoData.
            lookup = ",".join(f"{code}:{int(code == band) if band <= 8 else int(code not in range(1, 9) and code != src.nodata)}"
                              for code in range(256))
            lines.extend([f'<VRTRasterBand dataType="Float32" band="{band}"><ComplexSource>',
                          f'<SourceFilename relativeToVRT="0">{escape(str(source))}</SourceFilename>',
                          '<SourceBand>1</SourceBand>',
                          f'<SrcRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>',
                          f'<DstRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}"/>',
                          f'<LUT>{lookup}</LUT></ComplexSource></VRTRasterBand>'])
        lines.append("</VRTDataset>")
    fractions = np.full((9, *grid["mask"].shape), np.nan, dtype="float32")
    with rasterio.Env(GDAL_CACHEMAX=256 * 1024 * 1024):
        with MemoryFile("\n".join(lines).encode("utf-8"), ext=".vrt") as memory:
            with memory.open() as vrt:
                reproject(source=rasterio.band(vrt, list(range(1, 10))), destination=fractions,
                          src_transform=vrt.transform, src_crs=vrt.crs,
                          dst_transform=grid["transform"], dst_crs=grid["crs"],
                          dst_nodata=np.nan, resampling=Resampling.average,
                          num_threads=2, warp_mem_limit=128)
    active = grid["mask"] > 0
    values = fractions[:, active]
    coverage = fractions[:8].sum(axis=0, dtype="float64")
    uncovered = active & (~np.isfinite(fractions).all(axis=0) | (coverage <= 1e-6))
    unmapped = active & (fractions[8] > 1e-6)
    if (not np.isfinite(values).all() or np.any(values[8] > 1e-6)
            or uncovered.any() or np.any(coverage[active] > 1 + 1e-4)
            or np.any(values < -1e-6) or np.any(values > 1 + 1e-6)):
        study_cells = geometry_mask([grid["study"]], grid["mask"].shape, grid["transform"], invert=True)
        raise ValueError()
    partial = active & (coverage < 1 - 1e-6)
    if partial.any():
        print(f"Land cover: {int(partial.sum())} cells contain source NoData; normalizing by valid classified area; "
              f"minimum valid coverage {float(coverage[active].min()):.4f}", flush=True)
    fractions[:8, active] /= coverage[active]
    if np.max(np.abs(fractions[:8, active].sum(axis=0) - 1)) > 1e-4:
        raise ValueError()
    fractions[:, ~active] = 0
    return fractions[:8]


def rainfall_zones(config, grid, zone_names):
    """Use the configured rainfall zones, or one rainfall zone citywide."""
    shape = grid["mask"].shape
    if "rainfall_zones" not in config["sources"]:
        if len(zone_names) != 1:
            raise ValueError()
        return np.zeros(shape, dtype="int16")
    frame = gpd.read_file(config["sources"]["rainfall_zones"]).to_crs(grid["crs"])
    field = config.get("rainfall_zone_field", "formula_zone_id")
    if field not in frame or set(frame[field]) - set(zone_names):
        raise ValueError()
    geometries = [(geometry, zone_names.index(zone))
                  for geometry, zone in zip(frame.geometry, frame[field])]
    zones = rasterize(geometries, out_shape=shape, transform=grid["transform"],
                      fill=-1, dtype="int16")
    missing = (grid["mask"] > 0) & (zones < 0)
    study_cells = geometry_mask([grid["study"]], shape, grid["transform"], invert=True)
    if np.any(missing & study_cells):
        raise ValueError()
    if missing.any():
        # Extend the nearest known zone only into the outer buffer; retain study-area zone definitions.
        if not np.any(zones >= 0):
            raise ValueError()
        nearest = distance_transform_edt(zones < 0, return_distances=False, return_indices=True)
        zones[missing] = zones[nearest[0, missing], nearest[1, missing]]
    return zones
