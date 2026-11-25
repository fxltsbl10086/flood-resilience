"Read native maximum-depth products and recorded scenario context for S2."
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib

# Use a noninteractive backend for PNG export.
matplotlib.use("Agg")
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from shapely.geometry import box

CITY_NAMES = {
    "fuzhou": "Fuzhou", "wuhan": "Wuhan", "hangzhou": "Hangzhou",
    "chengdu": "Chengdu", "zhengzhou": "Zhengzhou",
}

# Use fixed depth classes across cities, scenarios and products.
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

    def product_title(self, final: bool) -> str:
        hours = f"{self.simulation_hours:g}"
        return f"Final depth at {hours} h" if final else f"Maximum land depth over 0–{hours} h"

    def forcing_note(self) -> str:
        parts = []
        if self.rain_hours is not None:
            parts.append(f"{self.rain_hours:g} h {self.rainfall_label}" if self.rainfall_label
                         else f"Rainfall duration {self.rain_hours:g} h")
        elif self.rainfall_label:
            parts.append(self.rainfall_label)
        if self.drainage_mm_h is not None:
            parts.append(f"{self.drainage_label or 'Simplified drainage'} {self.drainage_mm_h:g} mm/h")
        return "；".join(parts)


def _configure_plotting(dpi: int) -> None:
    "Configure available fonts without installing new font files."
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "savefig.dpi": dpi,
        "savefig.facecolor": "white",
    })


def _scenario_directory(result_root: Path, period: int) -> Path:
    "Find one scenario directory by return period."
    prefix = f"T{period:03d}"
    matcher = re.compile(rf"^{prefix}(?:_|$)", re.IGNORECASE)
    matches = sorted(path for path in result_root.iterdir()
                     if path.is_dir() and matcher.match(path.name))
    if len(matches) != 1:
        raise ValueError()
    return matches[0]


def _load_depth_maps(result_root: Path, periods: Sequence[int], filename: str) -> list[DepthMap]:
    "Read exported depth rasters without resampling or spatial editing."
    maps = []
    for period in periods:
        file = _scenario_directory(result_root, period) / filename
        if not file.exists():
            raise FileNotFoundError()
        with rasterio.open(file) as source:
            if source.count != 1 or source.crs is None:
                raise ValueError()
            raw = source.read(1, masked=True).astype(np.float64)
            # Display depths at or below 0.01 m without fill; keep NoData transparent.
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
    # Read actual parameters from scenario records, excluding current YAML values.
    for record in records:
        for key in keys:
            value = record.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def _context(city: str, config: Mapping[str, Any], item: DepthMap,
              notes: list[str]) -> MapContext:
    "Label figures from recorded result parameters; use YAML only for boundary lookup and comparisons."
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
    # Accept parameter snapshots stored with the scenario result.
    for parent in (summary, run):
        for key in ("model_configuration", "input_configuration", "parameters"):
            if isinstance(parent.get(key), dict):
                records.append(parent[key])
    actual_city = _first_recorded(records, ("city",))
    if actual_city is not None and str(actual_city).lower() != city.lower():
        raise ValueError()
    time_candidates = [
        ("runtime sfincs.inp", _duration_from_times(inp.get("tstart"), inp.get("tstop"), "runtime sfincs.inp")),
        ("run_status", _duration_from_times(run.get("simulation_start"), run.get("simulation_stop"), "run_status")),
        ("TIF tags", _duration_from_times(item.tags.get("simulation_start"), item.tags.get("simulation_stop"), "TIF tags")),
    ]
    for index, record in enumerate(records):
        for key in ("simulation_duration_hours", "simulation_duration_h", "simulation_hours"):
            time_candidates.append((f"result record {index}.{key}", record.get(key)))
        if record.get("simulation_duration_min") is not None:
            time_candidates.append((f"result record {index}.simulation_duration_min", float(record["simulation_duration_min"]) / 60))
    simulation_hours = _numeric_consensus(time_candidates, "simulation duration", required=True)
    rain_candidates = []
    drain_candidates = []
    for index, record in enumerate(records):
        for key in ("rain_duration_hours", "rain_duration_h", "rain_hours"):
            rain_candidates.append((f"result record {index}.{key}", record.get(key)))
        if record.get("rain_duration_min") is not None:
            rain_candidates.append((f"result record {index}.rain_duration_min", float(record["rain_duration_min"]) / 60))
        for key in ("drainage_mm_h", "drainage_rate_mm_h", "simplified_drainage_mm_h"):
            drain_candidates.append((f"result record {index}.{key}", record.get(key)))
    rain_hours = _numeric_consensus(rain_candidates, "rainfall duration")
    drainage = _numeric_consensus(drain_candidates, "simplified drainage rate")
    if rain_hours is not None and (rain_hours <= 0 or rain_hours > simulation_hours):
        raise ValueError()
    if drainage is not None and drainage < 0:
        raise ValueError()
    terrain_value = _first_recorded(records, ("terrain_label", "terrain_source", "dem_label", "source_dem", "dem_source", "dem"))
    if terrain_value is None:
        # Prefer the DEM recorded in the actual parameter snapshot.
        for record in records:
            sources = record.get("sources")
            if isinstance(sources, dict) and sources.get("dem"):
                terrain_value = sources["dem"]
                break
    terrain = Path(str(terrain_value)).stem if terrain_value is not None else None
    rainfall = _first_recorded(records, ("rainfall_label", "rainfall_distribution", "rainfall_type", "rainfall_pattern"))
    if rainfall is not None and str(rainfall).lower() in ("uniform", "constant", "均匀"):
        rainfall = "Uniform rainfall"
    drainage_label = _first_recorded(records, ("drainage_label", "drainage_scope"))
    codes = _first_recorded(records, ("drainage_codes",))
    if drainage_label is None and isinstance(codes, (list, tuple)):
        drainage_label = "Simplified drainage on developed land" if sorted(codes) == [1, 5, 7] else "Simplified drainage on selected land-use classes"
    # Keep historical labels based on recorded parameters.
    comparisons = [("resolution_m", resolution_m), ("simulation_duration_min", simulation_hours * 60),
                   ("rain_duration_min", None if rain_hours is None else rain_hours * 60),
                   ("drainage_mm_h", drainage)]
    for key, actual in comparisons:
        if actual is not None and key in config and not np.isclose(float(config[key]), actual):
            notes.append(f"T{item.period}: configured {key}={config[key]} differs from recorded {actual:g}; using the recorded value.")
    configured_dem = config.get("sources", {}).get("dem")
    if terrain is not None and configured_dem is not None and Path(configured_dem).stem != terrain:
        notes.append(f"T{item.period}: configured DEM differs from the recorded source; using {terrain}.")
    if terrain is None or rain_hours is None or rainfall is None or drainage is None:
        notes.append(f"T{item.period}: some terrain, rainfall or drainage parameters are unrecorded and are omitted from labels.")
    epsg = item.crs.to_epsg()
    crs_label = f"EPSG:{epsg}（{item.crs.linear_units}）" if epsg else item.crs.to_string()
    return MapContext(str(config.get("city_name") or CITY_NAMES.get(city, city)), terrain,
                      rain_hours, simulation_hours, drainage,
                      None if rainfall is None else str(rainfall),
                      None if drainage_label is None else str(drainage_label),
                      resolution_m, crs_label, units_to_m)


def _common_context(contexts: list[MapContext]) -> MapContext:
    "Use only recorded parameters shared by all selected scenarios."
    first = contexts[0]
    if any(not np.isclose(item.simulation_hours, first.simulation_hours) for item in contexts[1:]):
        raise ValueError()
    optional = ("terrain_label", "rain_hours", "drainage_mm_h", "rainfall_label", "drainage_label")
    shared = {key: getattr(first, key) if all(getattr(item, key) == getattr(first, key)
              for item in contexts[1:]) else None for key in optional}
    return replace(first, **shared)


