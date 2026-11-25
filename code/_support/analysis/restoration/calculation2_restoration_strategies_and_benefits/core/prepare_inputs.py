"""Prepare formal EFAR inputs and original study-area cells without ranking roads or summarizing restoration benefits."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import config

_DLL_HANDLES = []
_bin = Path(sys.prefix) / "Library/bin"
if os.name == "nt" and _bin.is_dir():
    os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")
    _DLL_HANDLES.append(os.add_dll_directory(str(_bin)))
_gdal = Path(sys.prefix) / "Library/share/gdal"
if _gdal.is_dir():
    os.environ.setdefault("GDAL_DATA", str(_gdal))

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


DIMENSIONS = (("rescue", "应急救援"), ("medical", "医疗资源"), ("shelter", "避险转移"))
DATA_SIGNATURE_KEYS = ("GRID_PATH", "ROAD_PATH", "EDGE_PATH", "BOUNDARY_PATH", "RAW_ROAD_PATH")


def _require(condition, message):
    if not condition:
        raise ValueError("")


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _definitions():
    from _release import ROOT as root, SUPPORT, ANALYSIS
    from _support.analysis_paths import access_inputs
    engine = SUPPORT/'sar'
    source_config = _module('governance_source_geometry_config', engine/'config.py')
    cfg=source_config.config_for_city(config.CITY).copy()
    evaluation_scope=getattr(config,'EVALUATION_SCOPE','district')
    _require(evaluation_scope in ('district','study_area'), "")
    cfg.update(ORIGIN_SCOPE='district' if evaluation_scope=='district' else 'city',DISTRICT_NAME=config.DISTRICT_NAME)
    result2_path=SUPPORT/'efar/settings.py'
    inputs=access_inputs(config.CITY)
    metadata_path=inputs['risk_metadata']
    risk_metadata=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    cache_dir=ANALYSIS/'facility_counts'
    counts_path=cache_dir/config.CITY/f'{config.CITY}_grid_facility_counts.csv'
    return dict(root=root, engine=engine, cfg=cfg, result2_path=result2_path,
                inputs=inputs, risk_metadata_path=metadata_path, risk_metadata=risk_metadata,
                counts_path=counts_path,
                evaluation_scope=evaluation_scope)


def _shape_files(path):
    path = Path(path).resolve()
    paths = [path]
    if path.suffix.lower() == ".shp":
        for suffix in (".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx"):
            child = path.with_suffix(suffix)
            if child.exists():
                paths.append(child)
            elif suffix in (".dbf", ".shx", ".prj"):
                raise FileNotFoundError("")
    return paths


def _paths(definitions):
    d = definitions
    paths = []
    geometry_keys = list(DATA_SIGNATURE_KEYS)
    if d["evaluation_scope"] == "district":
        geometry_keys.append("DISTRICT_PATH")
    for key in geometry_keys:
        paths.extend(_shape_files(d["cfg"][key]))
    paths.extend([Path(config.EFARB_CSV).resolve(), Path(d["inputs"]["population_csv"]).resolve(),
                  Path(d["inputs"]["risk_csv"]).resolve(), d["risk_metadata_path"],
                  d["counts_path"], d["result2_path"],
                  *[d["engine"] / name for name in ("config.py", "functions.py", "reachability.py")]])
    return list(dict.fromkeys(path.resolve() for path in paths))


def input_paths():
    """List all source data and code first; read configuration/JSON only, without GIS or network loading."""
    return _paths(_definitions())


def _strict_bool(series, label):
    if pd.api.types.is_bool_dtype(series):
        _require(not series.isna().any(), "")
        return series.to_numpy(bool)
    normalized = series.astype(str).str.lower()
    _require(normalized.isin(["true", "false"]).all(), "")
    return normalized.eq("true").to_numpy()


def load_inputs():
    """Load the complete original network, evaluation cells, fixed facility counts and road-location inventory."""
    d = _definitions()
    evaluation_scope = d["evaluation_scope"]
    engine_functions = _module("governance_source_input_functions", d["engine"] / "functions.py")
    grids, roads, edges, boundary, clipped_district = engine_functions.read_inputs(d["cfg"])
    district = None
    if evaluation_scope == "district":
        county = gpd.read_file(d["cfg"]["DISTRICT_PATH"])
        name_field = "name" if "name" in county else "NAME"
        district = county.loc[county[name_field].eq(config.DISTRICT_NAME)].geometry.union_all()
        _require(not district.is_empty and district.is_valid, "")
        _require(county.crs == grids.crs, "")
    else:
        _require(clipped_district is None, "")
    # In district mode, road intervention locations remain relative to the full district, matching earlier trials.
    # study_area mode uses the original study boundary; neither mode clips roads or facilities.
    evaluation_area = district if evaluation_scope == "district" else boundary
    _require(not evaluation_area.is_empty and evaluation_area.is_valid,
             "")
    selected = engine_functions.select_origins(grids, boundary, clipped_district)
    # Preserve full-city index ordering; calculate priorities separately in the entry point.
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    if evaluation_scope == "study_area":
        _require(np.array_equal(selected, np.arange(len(grids))),
                 "")
    evaluation = pd.DataFrame(grids.iloc[selected].drop(columns="geometry")).copy().reset_index(drop=True)

    burden = pd.read_csv(config.EFARB_CSV, encoding="utf-8-sig", dtype={"grid_id": str},
                         float_precision="round_trip")
    burden = burden.loc[burden.city.eq(config.CITY)].copy()
    _require(burden.grid_id.is_unique and set(burden.grid_id) == set(grids.grid_id.astype(str)),
             "")
    burden = burden.set_index("grid_id").loc[evaluation.grid_id].reset_index()
    for field in ("centroid_x", "centroid_y"):
        _require(np.allclose(evaluation[field], burden[field], rtol=0, atol=1e-6),
                 "")
    for field in burden:
        if field not in evaluation:
            evaluation[field] = burden[field].to_numpy()
    evaluation["efarb_valid"] = _strict_bool(evaluation.efarb_valid, "efarb_valid")
    population = pd.read_csv(d["inputs"]["population_csv"], encoding="utf-8-sig", dtype={"grid_id": str},
                             float_precision="round_trip")
    _require(population.grid_id.is_unique and set(population.grid_id) == set(grids.grid_id.astype(str)),
             "")
    population = population.set_index("grid_id").loc[evaluation.grid_id]
    _require(np.allclose(evaluation.population, population.population, rtol=0, atol=1e-9, equal_nan=True),
             "")
    exposed_field = f"population_exposed_T{config.PERIOD}"
    evaluation[exposed_field] = population[exposed_field].to_numpy(float)
    evaluation["population_exposed"] = evaluation[exposed_field].to_numpy(float)
    evaluation["initial_EFARB"] = evaluation[f"EFARB_T{config.PERIOD}"].to_numpy(float)
    evaluation["initial_EFAR_mean"] = evaluation[f"EFAR_mean_T{config.PERIOD}"].to_numpy(float)
    risk = pd.read_csv(d["inputs"]["risk_csv"], encoding="utf-8-sig", dtype={"grid_id": str},
                       float_precision="round_trip")
    _require(risk.city.eq(config.CITY).all(), "")
    _require(not risk.duplicated(["grid_id", "评估维度"]).any(), "")
    for code, dimension in DIMENSIONS:
        current = risk.loc[risk["评估维度"].eq(dimension)].set_index("grid_id").loc[evaluation.grid_id]
        value = evaluation[f"EFAR_{code}_T{config.PERIOD}"].to_numpy(float)
        _require(np.allclose(value, current[f"风险_T{config.PERIOD:03d}"], rtol=0, atol=1e-12, equal_nan=True),
                 "")
        evaluation[f"initial_EFAR_{code}"] = value
        evaluation[f"initial_reachable_count_{code}"] = current[f"可达设施数_T{config.PERIOD:03d}"].to_numpy(np.int64)
    valid = evaluation.efarb_valid.to_numpy(bool)
    p = evaluation.effective_population.to_numpy(float)
    _require(np.isfinite(p[valid]).all() and (p[valid] >= 0).all() and np.isnan(p[~valid]).all(),
             "")
    _require(np.array_equal(p[valid], evaluation.population.to_numpy(float)[valid]),
             "")
    _require(np.isfinite(evaluation.initial_EFARB.to_numpy()[valid]).all()
             and np.isnan(evaluation.initial_EFARB.to_numpy()[~valid]).all(), "")
    expected = evaluation[[f"initial_EFAR_{code}" for code, _ in DIMENSIONS]].to_numpy(float).mean(axis=1)
    _require(np.allclose(evaluation.initial_EFARB.to_numpy()[valid], p[valid] * expected[valid],
                        rtol=1e-12, atol=1e-10), "")
    evaluation["evaluation_valid"] = valid
    resident_indices = evaluation.loc[evaluation.evaluation_valid, "grid_index"].to_numpy(np.int64)

    counts_table = pd.read_csv(d["counts_path"], encoding="utf-8-sig", dtype={"grid_id": str})
    _require(counts_table.grid_id.is_unique and set(counts_table.grid_id) == set(grids.grid_id.astype(str)),
             "")
    counts = counts_table.set_index("grid_id").loc[grids.grid_id.astype(str),
             [dimension + "数量" for _, dimension in DIMENSIONS]].to_numpy(np.int64)
    _require((counts >= 0).all(), "")
    metadata = d["risk_metadata"]
    _require(metadata.get("metric") == "inverse" and metadata.get("local") == "include"
             and metadata.get("same_grid_distance_m") == 250, "")
    _require(config.PERIOD in metadata["periods"], "")
    candidates = engine_functions.build_candidates(grids, roads, boundary, d["cfg"])
    reference = None
    reuse = {"reused": False, "cache_path": None}

    inside = shapely.length(shapely.intersection(roads.geometry.array, evaluation_area))
    outside = shapely.length(shapely.difference(roads.geometry.array, evaluation_area))
    lengths = roads.length_m.to_numpy(float)
    _require(np.allclose(inside + outside, lengths, rtol=1e-10, atol=1e-6),
             "")
    tol = 1e-6
    inventory = pd.DataFrame({"road_id": roads.road_id, "road_position": np.arange(len(roads)),
        "length_m": lengths,
        "inside_evaluation_area_length_m": inside, "outside_evaluation_area_length_m": outside,
        "has_outside_evaluation_area_part": outside > tol,
        "wholly_outside_evaluation_area": (outside > tol) & (inside <= tol),
        "crosses_evaluation_area": (outside > tol) & (inside > tol),
        "inside_district_length_m": inside if district is not None else np.nan,
        "outside_district_length_m": outside if district is not None else np.nan,
        "within_study_length_m": shapely.length(shapely.intersection(roads.geometry.array, boundary)),
        "has_outside_part": outside > tol if district is not None else pd.NA,
        "wholly_outside": (outside > tol) & (inside <= tol) if district is not None else pd.NA,
        "crosses_district": (outside > tol) & (inside > tol) if district is not None else pd.NA,
        f"flood_T{config.PERIOD:03d}": roads[f"flood_T{config.PERIOD:03d}"].to_numpy(np.int8)})
    inventory["flood_scenario"] = inventory[f"flood_T{config.PERIOD:03d}"].to_numpy(np.int8)
    speeds = pd.to_numeric(roads.get("maxspeed_raw", pd.Series(np.nan, index=roads.index)), errors="coerce")
    speed_info = {"field": "maxspeed_raw", "road_count": len(roads),
                  "positive_speed_road_count": int((speeds > 0).sum()),
                  "missing_or_nonpositive_speed_road_count": int((~(speeds > 0)).sum()),
                  "travel_time_available": False,
                  "reason": 'The original network uses distance costs; substantial missing speed coverage precludes assumed speeds or time thresholds.'}
    return {"grids": grids, "roads": roads, "edges": edges, "candidates": candidates,
            "boundary": boundary, "district_geometry": district, "evaluation_district_geometry": clipped_district,
            "evaluation_area_geometry": evaluation_area,
            "counts": counts, "evaluation_all": evaluation, "resident_indices": resident_indices,
            "road_inventory": inventory, "reference_network": reference,
            "source_paths": _paths(d), "metadata": {
                "city": config.CITY,
                "city_name": config.CITY_NAMES[config.CITY],
                "district": config.DISTRICT_NAME if district is not None else None,
                "period": config.PERIOD, "evaluation_scope": evaluation_scope,
                "evaluation_area_name": (config.DISTRICT_NAME if district is not None else
                                         f"{config.CITY_NAMES[config.CITY]}既定研究区"),
                "district_selection": ('Original cell centers must lie in both the full district and study area; retain complete cells without clipping population.'
                                       if district is not None else None),
                "evaluation_grid_selection": ('All original cells assigned to the district by their centers'
                                              if district is not None else
                                              'Retain all original study-area cells without additional center/district filtering or population clipping.'),
                "evaluation_valid_definition": 'All efarb_valid cells in the evaluation scope, including known zero population and residents completely disconnected after flooding; ranking separately requires population and initial burden >0.',
                "road_location_reference": "完整试算行政区边界" if district is not None else 'Original study-area boundary BOUNDARY_PATH',
                "outside_road_definition": 'Unique road_id values with more than 1e-6 m outside the actual evaluation scope; count only the outside portion of crossing roads, without clipping roads or facilities.',
                "legacy_district_road_fields": ("保留原行政区语义" if district is not None else
                                               "不适用，inside/outside_district_length_m及has_outside_part/wholly_outside/crosses_district留空；使用通用evaluation_area字段"),
                "physical_length_definition": 'Restore all original directional access for each road_id; intervention size uses the full physical-segment length_m.',
                "resource_failure": metadata.get("resource_failure"), "speed_availability": speed_info,
                "source_network_cache": reuse, "facility_counts_by_dimension": dict(zip(
                    [code for code, _ in DIMENSIONS], counts.sum(axis=0).tolist())),
                "grid_count": len(grids), "district_grid_count": len(evaluation) if district is not None else None,
                "evaluation_area_grid_count": len(evaluation),
                "evaluation_grid_count": len(resident_indices), "road_count": len(roads),
                "directed_edge_count": len(edges), "candidate_count": len(candidates),
                "source_crs": str(grids.crs), "same_grid_distance_m": 250.0,
            }}
