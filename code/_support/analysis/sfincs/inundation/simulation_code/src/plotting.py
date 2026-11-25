"""Shared depth-map plotting called by simulation_code/main.py."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import warnings

import geopandas as gpd
import matplotlib

# Render without interactive windows and save PNGs in the city result directory.
matplotlib.use("Agg")
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import box

from _release import ANALYSIS, package_path

CITY_NAMES = {
    "fuzhou": "福州", "wuhan": "武汉", "hangzhou": "杭州",
    "chengdu": "成都", "zhengzhou": "郑州",
}

# Use the same depth classes for all cities, scenarios and products.
LEVELS = [0.01, 0.10, 0.30, 0.50, 1.00, 2.00, np.inf]
COLORS = ["#e2f1f4", "#aed9e5", "#6faed2", "#327db6", "#7956a3", "#d73027"]
LABELS = ["0.01—0.10", "0.10—0.30", "0.30—0.50", "0.50—1.00", "1.00—2.00", "≥2.00"]
CMAP = ListedColormap(COLORS, name="fixed_depth_classes")
CMAP.set_bad((0, 0, 0, 0))
NORM = BoundaryNorm(LEVELS, CMAP.N, clip=True)


@dataclass
class DepthMap:
    period: int
    path: Path
    depth: np.ma.MaskedArray
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    extent: list[float]
    tags: dict[str, str]


@dataclass
class MapContext:
    city_name: str
    terrain_label: str | None
    rain_hours: float | None
    simulation_hours: float
    drainage_mm_h: float | None
    rainfall_label: str | None
    drainage_label: str | None
    resolution_m: float
    crs_label: str
    units_to_m: float

    def product_name(self, final: bool) -> str:
        hours = f"{self.simulation_hours:g}"
        return f"final_land_depth_{hours}h" if final else f"maximum_land_depth_0_to_{hours}h"

    def product_title(self, final: bool) -> str:
        hours = f"{self.simulation_hours:g}"
        return f"第{hours}小时结束水深" if final else f"0—{hours} h 最大陆地水深"

    def forcing_note(self) -> str:
        parts = []
        if self.rain_hours is not None:
            parts.append(f"{self.rain_hours:g} h {self.rainfall_label}" if self.rainfall_label
                         else f"降雨历时 {self.rain_hours:g} h")
        elif self.rainfall_label:
            parts.append(self.rainfall_label)
        if self.drainage_mm_h is not None:
            parts.append(f"{self.drainage_label or '简化排水参数'} {self.drainage_mm_h:g} mm/h")
        return "；".join(parts)


def _configure_plotting(dpi: int) -> None:
    """Use an installed font supporting historical labels without installing fonts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "savefig.dpi": dpi,
        "savefig.facecolor": "white",
    })


def _scenario_directory(result_root: Path, period: int) -> Path:
    """Find a unique scenario by return period without fixing its drainage label."""
    prefix = f"T{period:03d}"
    matcher = re.compile(rf"^{prefix}(?:_|$)", re.IGNORECASE)
    matches = sorted(path for path in result_root.iterdir()
                     if path.is_dir() and matcher.match(path.name))
    if len(matches) != 1:
        raise ValueError()
    return matches[0]


def _load_depth_maps(result_root: Path, periods: Sequence[int], filename: str) -> list[DepthMap]:
    """Read exported depth rasters without resampling or hydrological post-processing."""
    maps = []
    for period in periods:
        file = _scenario_directory(result_root, period) / filename
        if not file.exists():
            raise FileNotFoundError()
        with rasterio.open(file) as source:
            if source.count != 1 or source.crs is None:
                raise ValueError()
            raw = source.read(1, masked=True).astype(np.float64)
            # Display only: depths <=0.01 m are uncolored and NoData is transparent.
            visible = np.ma.masked_where(np.ma.getmaskarray(raw) | ~np.isfinite(raw.data)
                                         | (raw.data <= 0.01), raw)
            maps.append(DepthMap(period, file, visible, source.transform, source.crs,
                                 [source.bounds.left, source.bounds.right,
                                  source.bounds.bottom, source.bounds.top],
                                 {key.lower(): value for key, value in source.tags().items()}))
    first = maps[0]
    for item in maps[1:]:
        if (item.depth.shape != first.depth.shape or item.transform != first.transform
                or item.crs != first.crs):
            raise ValueError()
    return maps


def _load_boundary(model_root: Path, explicit: str | Path | None,
                   config: Mapping[str, Any], crs: rasterio.crs.CRS) -> gpd.GeoDataFrame:
    """Prefer the saved model boundary, falling back to the city input configuration."""
    if explicit is not None:
        file = package_path(explicit)
    else:
        file = model_root / "gis" / "study_area.gpkg"
        if not file.exists():
            value = config.get("sources", {}).get("study_area")
            if value is None:
                raise ValueError()
            file = package_path(value)
    if not file.exists():
        raise FileNotFoundError()
    boundary = gpd.read_file(file)
    if boundary.empty or boundary.crs is None:
        raise ValueError()
    return boundary.to_crs(crs)


def _load_roads(city: str, path: str | Path | None, crs: rasterio.crs.CRS,
                without_roads: bool, notes: list[str]) -> gpd.GeoDataFrame | None:
    if without_roads:
        return None
    file = package_path(path) if path is not None else ANALYSIS / "prepared" / city / "roads.shp"
    if not file.exists():
        note = f"Road file unavailable; plotting depth and study boundary only: {file}"
        notes.append(note)
        warnings.warn(note, stacklevel=2)
        return None
    roads = gpd.read_file(file)
    if roads.empty or roads.crs is None:
        note = f"Road data are empty or lack a CRS; omitting overlay: {file}"
        notes.append(note)
        warnings.warn(note, stacklevel=2)
        return None
    return roads.to_crs(crs)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError()
    return value


def _read_runtime_input(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    settings = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip().lower()] = value.strip()
    return settings


def _datetime(value: Any) -> datetime:
    text = str(value).strip()
    for pattern in ("%Y%m%d %H%M%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _duration_from_times(start: Any, stop: Any, source: str) -> float | None:
    if start is None and stop is None:
        return None
    if start is None or stop is None:
        raise ValueError()
    value = (_datetime(stop) - _datetime(start)).total_seconds() / 3600
    if not np.isfinite(value) or value <= 0:
        raise ValueError()
    return value


def _numeric_consensus(candidates: list[tuple[str, Any]], parameter: str,
                        *, required: bool = False) -> float | None:
    available = [(source, float(value)) for source, value in candidates if value is not None]
    if not available:
        if required:
            raise ValueError()
        return None
    if any(not np.isfinite(value) for _, value in available):
        raise ValueError()
    first = available[0][1]
    if any(not np.isclose(value, first, rtol=0, atol=1e-7) for _, value in available[1:]):
        raise ValueError()
    return first


def _first_recorded(records: list[Mapping[str, Any]], keys: tuple[str, ...]) -> Any:
    # Read historical scenario records here, not the current YAML configuration.
    for record in records:
        for key in keys:
            value = record.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def _context(city: str, config: Mapping[str, Any], item: DepthMap,
              notes: list[str]) -> MapContext:
    """Use recorded scenario parameters in titles; current configuration only supplies checks and boundaries."""
    if not item.crs.is_projected:
        raise ValueError()
    if not (np.isclose(item.transform.b, 0) and np.isclose(item.transform.d, 0)
            and item.transform.a > 0 and item.transform.e < 0):
        raise ValueError()
    units_to_m = float(item.crs.linear_units_factor[1])
    resolution_m = abs(item.transform.a) * units_to_m
    if not np.isclose(abs(item.transform.e) * units_to_m, resolution_m):
        raise ValueError()
    directory = item.path.parent
    summary = _read_json(directory / "result_summary.json")
    run = _read_json(directory / "run_status.json")
    inp = _read_runtime_input(directory / "sfincs.inp")
    records: list[Mapping[str, Any]] = [summary, item.tags, run]
    # Only parameter snapshots stored with results establish historical scenario settings.
    for parent in (summary, run):
        for key in ("model_configuration", "input_configuration", "parameters"):
            if isinstance(parent.get(key), dict):
                records.append(parent[key])
    actual_city = _first_recorded(records, ("city",))
    if actual_city is not None and str(actual_city).lower() != city.lower():
        raise ValueError()
    time_candidates = [
        ("运行 sfincs.inp", _duration_from_times(inp.get("tstart"), inp.get("tstop"), "运行 sfincs.inp")),
        ("run_status", _duration_from_times(run.get("simulation_start"), run.get("simulation_stop"), "run_status")),
        ("TIF tags", _duration_from_times(item.tags.get("simulation_start"), item.tags.get("simulation_stop"), "TIF tags")),
    ]
    for index, record in enumerate(records):
        for key in ("simulation_duration_hours", "simulation_duration_h", "simulation_hours"):
            time_candidates.append((f"结果记录{index}.{key}", record.get(key)))
        if record.get("simulation_duration_min") is not None:
            time_candidates.append((f"结果记录{index}.simulation_duration_min", float(record["simulation_duration_min"]) / 60))
    simulation_hours = _numeric_consensus(time_candidates, "模拟时长", required=True)
    rain_candidates = []
    drain_candidates = []
    for index, record in enumerate(records):
        for key in ("rain_duration_hours", "rain_duration_h", "rain_hours"):
            rain_candidates.append((f"结果记录{index}.{key}", record.get(key)))
        if record.get("rain_duration_min") is not None:
            rain_candidates.append((f"结果记录{index}.rain_duration_min", float(record["rain_duration_min"]) / 60))
        for key in ("drainage_mm_h", "drainage_rate_mm_h", "simplified_drainage_mm_h"):
            drain_candidates.append((f"结果记录{index}.{key}", record.get(key)))
    rain_hours = _numeric_consensus(rain_candidates, "降雨时长")
    drainage = _numeric_consensus(drain_candidates, "简化排水率")
    if rain_hours is not None and (rain_hours <= 0 or rain_hours > simulation_hours):
        raise ValueError()
    if drainage is not None and drainage < 0:
        raise ValueError()
    terrain_value = _first_recorded(records, ("terrain_label", "terrain_source", "dem_label", "source_dem", "dem_source", "dem"))
    if terrain_value is None:
        # Prefer sources.dem in a saved parameter snapshot over current configuration.
        for record in records:
            sources = record.get("sources")
            if isinstance(sources, dict) and sources.get("dem"):
                terrain_value = sources["dem"]
                break
    terrain = Path(str(terrain_value)).stem if terrain_value is not None else None
    rainfall = _first_recorded(records, ("rainfall_label", "rainfall_distribution", "rainfall_type", "rainfall_pattern"))
    if rainfall is not None and str(rainfall).lower() in ("uniform", "constant", "均匀"):
        rainfall = "均匀降雨"
    drainage_label = _first_recorded(records, ("drainage_label", "drainage_scope"))
    codes = _first_recorded(records, ("drainage_codes",))
    if drainage_label is None and isinstance(codes, (list, tuple)):
        drainage_label = "建设用地简化排水" if sorted(codes) == [1, 5, 7] else "指定地类简化排水"
    # Report configuration differences without changing historical figure titles.
    comparisons = [("resolution_m", resolution_m), ("simulation_duration_min", simulation_hours * 60),
                   ("rain_duration_min", None if rain_hours is None else rain_hours * 60),
                   ("drainage_mm_h", drainage)]
    for key, actual in comparisons:
        if actual is not None and key in config and not np.isclose(float(config[key]), actual):
            notes.append(f"T{item.period}: current YAML {key}={config[key]} differs from recorded value {actual:g}; titles use the recorded result.")
    configured_dem = config.get("sources", {}).get("dem")
    if terrain is not None and configured_dem is not None and Path(configured_dem).stem != terrain:
        notes.append(f"T{item.period}: current YAML DEM differs from the result record; titles use {terrain}.")
    if terrain is None or rain_hours is None or rainfall is None or drainage is None:
        notes.append(f"T{item.period}: some terrain, rainfall or drainage settings are undocumented; titles do not substitute current YAML values.")
    epsg = item.crs.to_epsg()
    crs_label = f"EPSG:{epsg}（{item.crs.linear_units}）" if epsg else item.crs.to_string()
    return MapContext(str(config.get("city_name") or CITY_NAMES.get(city, city)), terrain,
                      rain_hours, simulation_hours, drainage,
                      None if rainfall is None else str(rainfall),
                      None if drainage_label is None else str(drainage_label),
                      resolution_m, crs_label, units_to_m)


def _common_context(contexts: list[MapContext]) -> MapContext:
    """Show only parameters that are documented and consistent across selected scenarios."""
    first = contexts[0]
    if any(not np.isclose(item.simulation_hours, first.simulation_hours) for item in contexts[1:]):
        raise ValueError()
    optional = ("terrain_label", "rain_hours", "drainage_mm_h", "rainfall_label", "drainage_label")
    shared = {key: getattr(first, key) if all(getattr(item, key) == getattr(first, key)
              for item in contexts[1:]) else None for key in optional}
    return replace(first, **shared)


def _legend_handles(with_roads: bool) -> list:
    handles = [Patch(facecolor=color, edgecolor="#999999", linewidth=0.4,
                     label=label) for color, label in zip(COLORS, LABELS)]
    if with_roads:
        handles.append(Line2D([0], [0], color="#777777", lw=0.6, label="Road centerlines"))
    handles.append(Line2D([0], [0], color="#333333", lw=1.0, label="Study-area boundary"))
    return handles


def _map_bounds(boundary: gpd.GeoDataFrame) -> list[float]:
    x0, y0, x1, y1 = boundary.total_bounds
    padding = max(x1 - x0, y1 - y0) * 0.015
    return [x0 - padding, y0 - padding, x1 + padding, y1 + padding]


def _draw_map(ax, item: DepthMap, boundary, roads, bounds: list[float],
               context: MapContext, *, panel: bool = False) -> None:
    # Nearest-neighbor display interpolation does not change grid or depth values.
    ax.imshow(item.depth, extent=item.extent, origin="upper", cmap=CMAP, norm=NORM,
              interpolation="nearest", zorder=1)
    if roads is not None:
        selected = roads.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        if not selected.empty:
            selected.plot(ax=ax, color="#777777", linewidth=0.12 if panel else 0.20,
                          alpha=0.40, zorder=2)
    boundary.boundary.plot(ax=ax, color="#333333", linewidth=0.70 if panel else 0.90, zorder=3)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4 if panel else 5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5 if panel else 6))
    ax.tick_params(labelsize=7 if panel else 9)
    unit = "m" if context.units_to_m == 1 else item.crs.linear_units
    ax.set_xlabel(f"Projected easting / {unit}", fontsize=8 if panel else 10)
    ax.set_ylabel(f"Projected northing / {unit}", fontsize=8 if panel else 10)
    ax.grid(alpha=0.20, linewidth=0.4, color="#bbbbbb", zorder=0)
    # Choose a 1/2/5-series scale bar based on study-area width.
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    desired_m = width * context.units_to_m * 0.15
    power = 10 ** math.floor(math.log10(desired_m))
    length_m = max(candidate for candidate in (power, 2 * power, 5 * power)
                   if candidate <= desired_m)
    length = length_m / context.units_to_m
    start_x, start_y = bounds[0] + 0.045 * width, bounds[1] + 0.055 * height
    ax.plot([start_x, start_x + length], [start_y, start_y], color="#222222", lw=2, zorder=5)
    scale_label = f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"
    ax.text(start_x + length / 2, start_y + 0.02 * height, scale_label, ha="center",
            fontsize=8 if panel else 10, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1})


def _single_figure(item, boundary, roads, bounds, output_root: Path,
                   context: MapContext, *, city: str, final: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(10.4, 11.8))
    _draw_map(ax, item, boundary, roads, bounds, context)
    fig.suptitle(f"{context.city_name} T{item.period} ({item.period}-year rainfall) | Grid {context.resolution_m:g} m",
                 fontsize=16, y=0.985)
    subtitle = "；".join(value for value in (context.forcing_note(), context.product_title(final)) if value)
    fig.text(0.5, 0.948, subtitle,
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.20)
    fig.legend(handles=_legend_handles(roads is not None), loc="lower center", bbox_to_anchor=(0.52, 0.090),
               ncol=4, title="Final depth classes / m" if final else "Maximum depth classes / m",
               frameon=False, fontsize=9, title_fontsize=10, handlelength=1.7, columnspacing=1.3)
    provenance = (f"Terrain: {context.terrain_label}; " if context.terrain_label else "") + context.crs_label
    fig.text(0.5, 0.065, provenance,
             ha="center", fontsize=8, color="#555555")
    fig.text(0.5, 0.043, "Depth <=0.01 m is uncolored; water/outside NoData is transparent; no smoothing or isolated-patch filtering.",
             ha="center", fontsize=9, color="#555555")
    note = ("Final depth is not the event maximum; results have not been validated against observations." if final else
            f"Not validated against observations; the 0-{context.simulation_hours:g} h window does not establish the full-event peak.")
    fig.text(0.5, 0.023, note, ha="center", fontsize=9, color="#884422")
    # Order filenames by depth type, then zero-padded return period.
    output = output_root / f"{city}_{context.product_name(final)}_T{item.period:03d}.png"
    try:
        fig.savefig(output)
    finally:
        plt.close(fig)
    return output


def _count_name(count: int) -> str:
    return {1: "单", 2: "二", 3: "三", 4: "四", 5: "五"}.get(count, str(count)) + "重现期"


def _combined_name(maps: Sequence[DepthMap]) -> str:
    periods = sorted(item.period for item in maps)
    if periods == [10, 20, 50, 100, 200]:
        return "five_periods"
    return "_".join(f"T{period:03d}" for period in periods)


def _combined_figure(maps, boundary, roads, bounds, output_root: Path,
                     context: MapContext, *, city: str, final: bool = False) -> Path:
    cols = 2 if len(maps) <= 2 else 3
    rows = math.ceil((len(maps) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.86, rows * 6.1 + 1.0), squeeze=False)
    flat_axes = axes.ravel()
    for index, (item, ax) in enumerate(zip(maps, flat_axes)):
        _draw_map(ax, item, boundary, roads, bounds, context, panel=True)
        label = chr(97 + index) if index < 26 else str(index + 1)
        ax.set_title(f"{label}  T{item.period} ({item.period}-year rainfall)", loc="left", fontsize=12)
    legend_ax = flat_axes[len(maps)]
    legend_ax.set_axis_off()
    legend_ax.legend(handles=_legend_handles(roads is not None), loc="upper left", bbox_to_anchor=(0.10, 0.99),
                     title="Final depth classes / m" if final else "Maximum depth classes / m",
                     frameon=False, fontsize=10, title_fontsize=11, labelspacing=0.8)
    note = "Final depth is not the event maximum." if final else "The statistical window does not establish the full-event peak."
    legend_ax.text(0.10, 0.28,
                   f"{context.product_title(final)}\n{note}\n"
                   "Scenarios share the grid, map extent and fixed depth classes.\n"
                   "Depth <=0.01 m is uncolored; NoData is transparent.\n"
                   "No spatial smoothing or isolated-patch filtering.\n"
                   f"{context.crs_label}\nResults have not been validated against observations.",
                   transform=legend_ax.transAxes, va="top", fontsize=9, color="#555555", linespacing=1.6)
    for ax in flat_axes[len(maps) + 1:]:
        ax.set_axis_off()
    fig.suptitle(f"{context.city_name}｜{_count_name(len(maps))}{context.product_title(final)}",
                 fontsize=17, y=0.985)
    subtitle = "；".join(value for value in (context.forcing_note(), f"Grid {context.resolution_m:g} m") if value)
    fig.text(0.5, 0.952, subtitle,
             ha="center", fontsize=11)
    if context.terrain_label:
        fig.text(0.5, 0.925, f"Terrain: {context.terrain_label}", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.89, bottom=0.055, hspace=0.25, wspace=0.22)
    output = output_root / f"{city}_{context.product_name(final)}_{_combined_name(maps)}.png"
    try:
        fig.savefig(output)
    finally:
        plt.close(fig)
    return output


def plot_city(city: str, model_root: str | Path, result_root: str | Path,
              periods: Sequence[int], config: Mapping[str, Any] | None = None,
              dpi: int = 220, roads: str | Path | None = None,
              boundary: str | Path | None = None, without_roads: bool = False) -> dict[str, Any]:
    """Plot maximum and final depths for selected periods and return all PNG paths.

Use config for boundary lookup and scenario records for historical time/parameter labels.
Save figures under the configured result directory without changing model inputs or rasters."""
    config = config or {}
    selected = [int(period) for period in periods]
    if not selected or any(period <= 0 for period in selected) or len(set(selected)) != len(selected):
        raise ValueError()
    if dpi < 100:
        raise ValueError()
    model_root, result_root = Path(model_root), Path(result_root)
    if not result_root.exists():
        raise FileNotFoundError()
    # Require both products before plotting to avoid incomplete map sets.
    maxima = _load_depth_maps(result_root, selected, "max_depth_land_study_area_m.tif")
    finals = _load_depth_maps(result_root, selected, "final_depth_land_study_area_m.tif")
    for maximum, final in zip(maxima, finals):
        if (maximum.depth.shape != final.depth.shape or maximum.transform != final.transform
                or maximum.crs != final.crs):
            raise ValueError()
    notes: list[str] = []
    contexts = [_context(city, config, item, notes) for item in maxima]
    # Require final-depth tags to match the maximum-depth event window.
    for maximum_context, final in zip(contexts, finals):
        final_context = _context(city, config, final, [])
        if not np.isclose(maximum_context.simulation_hours, final_context.simulation_hours):
            raise ValueError()
    context = _common_context(contexts)
    study = _load_boundary(model_root, boundary, config, maxima[0].crs)
    raster_bounds = box(maxima[0].extent[0], maxima[0].extent[2],
                        maxima[0].extent[1], maxima[0].extent[3])
    if not study.intersects(raster_bounds).any():
        raise ValueError()
    road_lines = _load_roads(city, roads, maxima[0].crs, without_roads, notes)
    bounds = _map_bounds(study)
    output_root = result_root / "plot"
    output_root.mkdir(parents=True, exist_ok=True)
    _configure_plotting(dpi)
    max_outputs = [_single_figure(item, study, road_lines, bounds, output_root, item_context, city=city)
                   for item, item_context in zip(maxima, contexts)]
    max_outputs.append(_combined_figure(maxima, study, road_lines, bounds, output_root, context, city=city))
    final_outputs = [_single_figure(item, study, road_lines, bounds, output_root, item_context, city=city, final=True)
                     for item, item_context in zip(finals, contexts)]
    final_outputs.append(_combined_figure(finals, study, road_lines, bounds, output_root, context, city=city, final=True))
    return {
        "city": city, "city_name": context.city_name, "return_periods_years": selected,
        "depth_classes_m": LABELS, "terrain": context.terrain_label,
        "resolution_m": context.resolution_m, "rain_duration_hours": context.rain_hours,
        "simulation_duration_hours": context.simulation_hours,
        "spatial_filtering": False, "observation_validated": False,
        "roads_overlaid": road_lines is not None, "notes": notes,
        "title_parameters_source": "Scenario sfincs.inp, run_status, result_summary and depth-raster tags; undocumented parameters remain unspecified",
        "maximum_output_png": [str(path) for path in max_outputs],
        "final_output_png": [str(path) for path in final_outputs],
    }
