"Shared TOML parameters, city inputs and command-line overrides."

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import os
import yaml
from _release import ROOT, CODE

from .missing_policy import validate_policy


PROJECT_ROOT = ROOT
DEFAULT_CONFIG = Path(os.environ.get("EXPOSURE_CONFIG", CODE / "exposure_and_roads.yaml"))


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class CityConfig:
    city: str
    config_path: Path
    road_path: Path
    edge_path: Path | None
    boundary_path: Path
    depth_paths: tuple[Path, ...]
    scenario_ids: tuple[str, ...]
    periods: tuple[int, ...]
    output_root: Path
    matching_root: Path
    facility_layer: str
    edge_layer: str
    threshold_m: float
    tolerance_m: float
    cell_batch_size: int
    depth_file: str
    missing_policy: dict[str, int | str]


def load_city(city: str, config_path: Path = DEFAULT_CONFIG, **overrides) -> CityConfig:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", city):
        raise ValueError()
    config_path = Path(config_path).resolve()
    with config_path.open(encoding="utf-8") as file:
        raw = yaml.safe_load(file)["road_flood"]
    paths, roads = raw["paths"], raw["roads"]
    scenarios, matching = raw["scenarios"], raw["matching"]
    city_data = raw.get("cities", {}).get(city, {})
    road_root = project_path(overrides.get("road_root") or paths["road_root"])
    depth_root = project_path(overrides.get("depth_root") or paths["depth_root"])
    boundary_override = overrides.get("boundary")
    if boundary_override:
        boundary_path = project_path(boundary_override)
    elif "boundary_file" in city_data:
        boundary_path = project_path(paths["boundary_root"]) / city_data["boundary_file"]
    else:
        raise ValueError()
    periods = tuple(sorted(scenarios["return_periods"]))
    if not periods or len(set(periods)) != len(periods) or any(type(p) is not int or p <= 0 for p in periods):
        raise ValueError()
    template = city_data.get("scenario_template", scenarios["scenario_template"])
    scenario_ids = tuple(template.format(period=p, city=city) for p in periods)
    if len(set(scenario_ids)) != len(periods) or any(Path(s).name != s or "/" in s or "\\" in s for s in scenario_ids):
        raise ValueError()
    threshold = float(overrides.get("threshold") if overrides.get("threshold") is not None else matching["threshold_m"])
    tolerance = float(matching["positive_length_tolerance_m"])
    batch_size = int(matching["cell_batch_size"])
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError()
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.001 or batch_size <= 0:
        raise ValueError()
    if matching["candidate_rule"] != "all_study_cells":
        raise ValueError()
    write_edges = roads.get("write_edges", True) and not overrides.get("no_edges", False)
    road_path = project_path(overrides["roads"]) if overrides.get("roads") else road_root / city / roads["facility_file"]
    config = CityConfig(
        city=city, config_path=config_path, road_path=road_path,
        edge_path=road_root / city / roads["edge_file"] if write_edges else None,
        boundary_path=boundary_path,
        depth_paths=tuple(depth_root / city / s / scenarios["depth_file"] for s in scenario_ids),
        scenario_ids=scenario_ids, periods=periods,
        output_root=project_path(overrides.get("output_root") or paths["output_root"]),
        matching_root=project_path(overrides.get("matching_root") or paths["matching_root"]),
        facility_layer=roads["facility_layer"], edge_layer=roads["edge_layer"],
        threshold_m=threshold, tolerance_m=tolerance,
        cell_batch_size=batch_size, depth_file=scenarios["depth_file"],
        missing_policy=validate_policy(raw.get("missing_policy", {})),
    )
    required = [config.road_path, config.boundary_path, *config.depth_paths]
    if config.edge_path is not None:
        required.append(config.edge_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError()
    return config
