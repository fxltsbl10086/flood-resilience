"""Read and write SFINCS binary files, corner-coordinate NetCDF forcing and configuration."""
from datetime import datetime, timedelta
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
from rasterio import Affine
from rasterio.crs import CRS


def write_static(root, grid, drainage, manning):
    """Use the installed HydroMT-SFINCS binary writer."""
    from hydromt_sfincs import utils
    mask = np.flipud(grid["mask"])
    utils.write_binary_map_index(root / "sfincs.ind", mask)
    roughness = np.full(grid["mask"].shape, manning, dtype="float32")
    for name, values, dtype in (("sfincs.dep", grid["terrain"], "f4"),
                                ("sfincs.msk", grid["mask"], "u1"),
                                ("sfincs.man", roughness, "f4"),
                                ("sfincs.qinf", drainage, "f4")):
        utils.write_binary_map(root / name, np.flipud(values), mask, dtype=dtype)


def write_forcing(path, grid, config, frames):
    """Write solver forcing at lower-left cell corners, with y increasing northward."""
    rows, columns = grid["mask"].shape
    transform, resolution = grid["transform"], grid["resolution"]
    count = config["simulation_duration_min"] * 60 // config["forcing_step_seconds"] + 1
    start = datetime.fromisoformat(config["reference_time"])
    with Dataset(path, "w", format="NETCDF4") as nc:
        for name, size in (("time", count), ("y", rows), ("x", columns)):
            nc.createDimension(name, size)
        t = nc.createVariable("time", "f8", ("time",))
        # The SFINCS 2.4 FEWS reader expects minute-based time units.
        # Convert time values and units together; runoff intensity remains mm/hr.
        t.units = f"minutes since {start:%Y-%m-%d %H:%M:%S}"
        t.calendar = "proleptic_gregorian"
        t[:] = np.arange(count) * (config["forcing_step_seconds"] / 60.0)
        x = nc.createVariable("x", "f8", ("x",))
        y = nc.createVariable("y", "f8", ("y",))
        x[:] = transform.c + np.arange(columns) * resolution
        y[:] = transform.f - rows * resolution + np.arange(rows) * resolution
        x.units = y.units = "m"
        x.standard_name, y.standard_name = "projection_x_coordinate", "projection_y_coordinate"
        ref = nc.createVariable("spatial_ref", "i4")
        ref.assignValue(0)
        ref.spatial_ref = ref.crs_wkt = grid["crs"].to_wkt()
        precip = nc.createVariable("Precipitation", "f4", ("time", "y", "x"),
                                   zlib=True, complevel=4, fill_value=np.nan,
                                   chunksizes=(1, min(rows, 256), min(columns, 256)))
        precip.units = "mm/hr"
        precip.grid_mapping = "spatial_ref"
        precip.long_name = "CN runoff intensity"
        nc.units = "mm/hr"
        nc.solver_coordinate_convention = "cell_lower_left_corner"
        nc.precipitation_interpretation = "Class-specific CN net runoff from six hours of uniform meteorological rainfall"
        written = 0
        for written, frame in enumerate(frames, start=1):
            precip[written - 1] = np.flipud(frame)
        if written != count:
            raise ValueError()


def write_input(path, grid, config):
    """Retain the executed Fuzhou solver settings and update city, grid, time and file references."""
    source = Path(__file__).with_name("sfincs_template.inp")
    settings = read_input(source)
    start = datetime.fromisoformat(config["reference_time"])
    stop = start + timedelta(minutes=config["simulation_duration_min"])
    rows, columns = grid["mask"].shape
    transform, resolution = grid["transform"], grid["resolution"]
    settings.update(mmax=columns, nmax=rows, dx=resolution, dy=resolution,
                    x0=transform.c, y0=transform.f - rows * resolution, epsg=config["epsg"],
                    tref=start.strftime("%Y%m%d %H%M%S"), tstart=start.strftime("%Y%m%d %H%M%S"),
                    tstop=stop.strftime("%Y%m%d %H%M%S"),
                    zsini=float(grid["terrain"][grid["mask"] > 0].min()) - 1,
                    manning=config["manning_n"], manning_land=config["manning_n"],
                    depfile="../sfincs.dep", mskfile="../sfincs.msk", indexfile="../sfincs.ind",
                    manningfile="../sfincs.man", qinffile="../sfincs.qinf",
                    netamprfile="sfincs_cn_runoff.nc")
    path.write_text("".join(f"{key:<21}= {value}\n" for key, value in settings.items()), encoding="utf-8")


def read_input(path):
    """Read key=value configuration for input construction and depth export."""
    return {line.split("=", 1)[0].strip(): line.split("=", 1)[1].strip()
            for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
            if "=" in line and not line.lstrip().startswith("#")}


def read_static(input_path):
    """Reconstruct the grid from solver inputs without intermediate GIS or metadata."""
    path = Path(input_path)
    config = read_input(path)
    rows, columns = int(config["nmax"]), int(config["mmax"])
    dx, dy = float(config["dx"]), float(config["dy"])
    transform = Affine(dx, 0, float(config["x0"]), 0, -dy, float(config["y0"]) + rows * dy)
    indices = np.fromfile(path.parent / config["indexfile"], dtype="<u4")
    if not indices.size or indices[0] != len(indices) - 1:
        raise ValueError()
    positions = indices[1:].astype("int64") - 1
    if np.any(positions < 0) or np.any(positions >= rows * columns) or len(np.unique(positions)) != len(positions):
        raise ValueError()
    result = dict(transform=transform, crs=CRS.from_epsg(int(config["epsg"])), config=config)
    for key, name, dtype, fill in (("mskfile", "mask", "uint8", 0),
                                   ("depfile", "terrain", "<f4", np.nan)):
        values = np.fromfile(path.parent / config[key], dtype=dtype)
        if len(values) != len(positions):
            raise ValueError()
        flat = np.full(rows * columns, fill, dtype=dtype)
        flat[positions] = values
        result[name] = np.flipud(flat.reshape((rows, columns), order="F"))
    result["resolution"] = dx
    return result
