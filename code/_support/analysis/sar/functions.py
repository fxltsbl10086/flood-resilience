"Read existing 500 m grids and roads and construct multiple grid-boundary access candidates."

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import CRS


def _input_path(cfg, name):
    path = Path(cfg[name])
    return path if path.is_absolute() else Path(cfg["ROOT_MAIN"]) / path


def _require_columns(table, columns, label):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError()


def _metric_crs(table, label):
    if table.crs is None:
        raise ValueError()
    crs = CRS(table.crs)
    if not crs.is_projected or any(
        not np.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info[:2]
    ):
        raise ValueError()
    return crs


def read_inputs(cfg):
    "Retain original grid IDs, centroids, road directions and scenario labels without reading population or depth."
    grids = gpd.read_file(_input_path(cfg, "GRID_PATH")).reset_index(drop=True)
    roads = gpd.read_file(
        _input_path(cfg, "ROAD_PATH"), layer=cfg["ROAD_LAYER"]
    ).reset_index(drop=True)
    edges = gpd.read_file(
        _input_path(cfg, "EDGE_PATH"), layer=cfg["EDGE_LAYER"]
    ).reset_index(drop=True)
    boundary = gpd.read_file(_input_path(cfg, "BOUNDARY_PATH"))
    scope = cfg.get("ORIGIN_SCOPE", "district")
    if scope not in ["city", "district"]:
        raise ValueError()
    district = gpd.read_file(_input_path(cfg, "DISTRICT_PATH")) if scope == "district" else None

    _require_columns(grids, ["grid_id", "centroid_x", "centroid_y"], "既有格网")
    _require_columns(
        roads,
        ["road_id", "u", "v", "length_m", "oneway_std", "fclass",
         "is_elevated", "is_underpass", "mixed_vertical_signature", "source_osm_ids"],
        "道路设施段",
    )
    _require_columns(
        edges,
        ["edge_id", "road_id", "from_id", "to_id", "direction", "base_cost_m"],
        "有向路由边",
    )
    crs = _metric_crs(roads, "道路设施段")
    tables = [(grids, "格网"), (edges, "有向边"), (boundary, "城市边界")]
    if district is not None:
        tables.append((district, "区边界"))
    for table, label in tables:
        if _metric_crs(table, label) != crs:
            raise ValueError()
        if table.empty or table.geometry.isna().any() or table.geometry.is_empty.any():
            raise ValueError()

    for table, column, label in [(grids, "grid_id", "格网"),
                                  (roads, "road_id", "道路"),
                                  (edges, "edge_id", "有向边")]:
        if table.empty or table[column].isna().any() or not table[column].is_unique:
            raise ValueError()
    if not grids.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError()
    if not roads.geometry.geom_type.eq("LineString").all():
        raise ValueError()
    if not np.isfinite(grids[["centroid_x", "centroid_y"]].to_numpy(float)).all():
        raise ValueError()
    if not roads.oneway_std.isin(["B", "F", "T"]).all():
        raise ValueError()
    for column in ["is_elevated", "is_underpass", "mixed_vertical_signature"]:
        if roads[column].isna().any() or not roads[column].isin([False, True]).all():
            raise ValueError()
    road_ids = set(roads.road_id)
    if set(edges.road_id) != road_ids:
        raise ValueError()
    road_lookup = roads.set_index("road_id")
    reference = road_lookup.loc[edges.road_id]
    forward = edges.direction.eq("forward").to_numpy()
    reverse = edges.direction.eq("reverse").to_numpy()
    if not (forward | reverse).all():
        raise ValueError()
    expected_from = np.where(forward, reference.u, reference.v)
    expected_to = np.where(forward, reference.v, reference.u)
    if not (np.array_equal(edges.from_id, expected_from)
            and np.array_equal(edges.to_id, expected_to)):
        raise ValueError()
    allowed = np.where(forward, reference.oneway_std.isin(["B", "F"]),
                       reference.oneway_std.isin(["B", "T"]))
    expected_count = np.where(roads.oneway_std.eq("B"), 2, 1)
    observed_count = edges.groupby("road_id").size().reindex(roads.road_id).to_numpy()
    if (not allowed.all() or not np.array_equal(observed_count, expected_count)
            or edges.duplicated(["road_id", "direction"]).any()):
        raise ValueError()
    costs = edges.base_cost_m.to_numpy(float)
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError()
    if not np.allclose(costs, reference.length_m.to_numpy(float), rtol=1e-10, atol=1e-6):
        raise ValueError()
    if not np.allclose(roads.length_m, roads.geometry.length, rtol=1e-10, atol=1e-6):
        raise ValueError()
    for period in cfg["PERIODS"]:
        column = f"flood_T{int(period):03d}"
        for table, label in [(roads, "设施段"), (edges, "有向边")]:
            _require_columns(table, [column], label)
            if table[column].isna().any() or not table[column].isin([0, 1]).all():
                raise ValueError()
        if not np.array_equal(edges[column], road_lookup.loc[edges.road_id, column]):
            raise ValueError()

    # Retain stored centroid fields without recomputing centroids or creating new grids.
    grids["grid_index"] = np.arange(len(grids))
    roads["road_index"] = np.arange(len(roads))
    boundary_geom = boundary.geometry.union_all()
    if district is None:
        return grids, roads, edges, boundary_geom, None
    name_field = "name" if "name" in district.columns else "NAME"
    _require_columns(district, [name_field], "区边界")
    chosen = district.loc[district[name_field].eq(cfg["DISTRICT_NAME"])]
    if chosen.empty:
        raise ValueError()
    district_geom = chosen.geometry.union_all().intersection(boundary_geom)
    return grids, roads, edges, boundary_geom, district_geom


def select_origins(grids, boundary, district, limit=None):
    "Select district grids by centroid, or retain the complete study area, and sort by original ID."
    if district is None:
        indices = grids.sort_values("grid_id").grid_index.to_numpy(int)
        return indices if limit is None else indices[:int(limit)]
    centres = shapely.points(grids.centroid_x, grids.centroid_y)
    selected = shapely.covers(boundary, centres) & shapely.covers(district, centres)
    indices = grids.loc[selected].sort_values("grid_id").grid_index.to_numpy(int)
    return indices if limit is None else indices[:int(limit)]


def _text(value):
    # Retain the road-preprocessing attribute normalization rules.
    return "" if value is None or pd.isna(value) else str(value).strip()


def _normalize_layer(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _normalize_bool(value, tunnel=False):
    values = {"t", "true", "yes", "y", "1"}
    values.add("building_passage" if tunnel else "viaduct")
    return _text(value).lower() in values


def _intersection_points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type in ["MultiPoint", "GeometryCollection"]:
        return [point for part in geometry.geoms for point in _intersection_points(part)]
    raise ValueError()


def build_candidates(grids, roads, boundary, cfg):
    "Use the same baseline candidate set for every scenario; routing removes candidates on closed roads."
    # Reuse the spatial-join and vectorized-intersection procedure.
    # Keep real intersections inside the study area rather than projecting external roads to substitute access points.
    work = roads[["road_index", "road_id", "geometry"]].copy()
    work.geometry = shapely.intersection(work.geometry.array, boundary)
    work = work.loc[~work.geometry.is_empty]
    borders = grids[["grid_index", "geometry"]].copy()
    borders.geometry = borders.geometry.boundary
    joined = gpd.sjoin(work, borders, how="inner", predicate="intersects")
    intersections = shapely.intersection(
        joined.geometry.array, borders.geometry.array[joined.index_right.to_numpy()]
    )
    records = []
    for grid_index, road_index, road_id, geometry in zip(
        joined.grid_index, joined.road_index, joined.road_id, intersections
    ):
        for point in _intersection_points(geometry):
            records.append((int(grid_index), int(road_index), road_id, point.x, point.y))
    columns = ["grid_index", "road_index", "road_id", "x", "y"]
    candidates = pd.DataFrame(records, columns=columns).drop_duplicates().reset_index(drop=True)
    candidates = candidates.astype({"grid_index": int, "road_index": int, "x": float, "y": float})
    points = shapely.points(candidates.x, candidates.y)
    road_indices = candidates.road_index.to_numpy(int)
    grid_indices = candidates.grid_index.to_numpy(int)
    candidates["position_m"] = shapely.line_locate_point(
        roads.geometry.array[road_indices], points
    )
    candidates["access_distance_m"] = np.hypot(
        candidates.x - grids.centroid_x.to_numpy()[grid_indices],
        candidates.y - grids.centroid_y.to_numpy()[grid_indices],
    )
    excluded_classes = set(cfg["ACCESS_EXCLUDED_CLASSES"])
    pure_eligible = (
        ~roads.is_elevated.to_numpy(bool) & ~roads.is_underpass.to_numpy(bool)
        & ~roads.fclass.str.lower().isin(excluded_classes).to_numpy()
    )
    mixed = roads.mixed_vertical_signature.to_numpy(bool)
    eligible = pure_eligible[road_indices].copy()
    unknown = []
    mixed_candidate_count = int(mixed[road_indices].sum())

    if mixed_candidate_count:
        raw = gpd.read_file(_input_path(cfg, "RAW_ROAD_PATH"))
        _require_columns(raw, ["osm_id", "fclass", "layer", "bridge", "tunnel"], "来源道路")
        if _metric_crs(raw, "来源道路") != CRS(roads.crs):
            raise ValueError()
        raw["_osm_key"] = raw.osm_id.map(_text)
        layer = raw.layer.map(_normalize_layer)
        raw_eligible = (
            ~raw.bridge.map(_normalize_bool)
            & ~raw.tunnel.map(lambda value: _normalize_bool(value, tunnel=True))
            & layer.eq(0)
            & ~raw.fclass.astype(str).str.strip().str.lower().isin(excluded_classes)
        ).to_numpy()
        source_lookup = raw.groupby("_osm_key").indices
        tolerance = float(cfg["SOURCE_GEOMETRY_TOLERANCE_M"])
        # Match both geometry location and OSM ID because one ID may contain several geometries.
        for index in np.flatnonzero(mixed[road_indices]):
            road = roads.iloc[road_indices[index]]
            source_ids = _text(road.source_osm_ids).split("|")
            source_rows = np.concatenate([
                source_lookup.get(key, np.empty(0, dtype=int)) for key in source_ids
            ])
            matched = source_rows[
                shapely.distance(raw.geometry.array[source_rows], points[index]) <= tolerance
            ]
            if len(matched):
                # Permit genuine ground-access bridge-transition endpoints but not internal bridge points.
                eligible[index] = raw_eligible[matched].any()
            else:
                eligible[index] = False
                unknown.append({"grid_index": int(grid_indices[index]),
                                "road_id": road.road_id, "x": float(candidates.x.iloc[index]),
                                "y": float(candidates.y.iloc[index])})

    within_distance = candidates.access_distance_m <= float(cfg["MAX_ACCESS_DISTANCE_M"])
    result = candidates.loc[eligible & within_distance, [
        "grid_index", "road_index", "road_id", "position_m", "x", "y", "access_distance_m"
    ]].sort_values(["grid_index", "road_index", "position_m"]).reset_index(drop=True)
    result.attrs["qa"] = {
        "raw_boundary_candidate_count": len(candidates),
        "mixed_boundary_candidate_count": mixed_candidate_count,
        "source_unmatched_candidate_count": len(unknown),
        "source_unmatched_examples": unknown[:10],
        "ineligible_candidate_count": int((~eligible).sum()),
        "beyond_max_access_distance_count": int((~within_distance).sum()),
        "candidate_count": len(result),
        "grid_count_with_candidates": int(result.grid_index.nunique()),
    }
    return result


def compute_accessibility(baseline_distances, scenario_distances, origin_indices):
    "Calculate one scenario and origin batch with equal-weight destinations and a fixed baseline denominator."
    baseline = np.asarray(baseline_distances, dtype=float)
    scenario = np.asarray(scenario_distances, dtype=float)
    origins = np.asarray(origin_indices, dtype=int)
    if baseline.ndim != 2 or scenario.shape != baseline.shape:
        raise ValueError()
    if origins.shape != (baseline.shape[0],) or (origins < 0).any() or (origins >= baseline.shape[1]).any():
        raise ValueError()
    if np.isnan(baseline).any() or np.isnan(scenario).any() or (baseline < 0).any() or (scenario < 0).any():
        raise ValueError()

    baseline_mask = np.isfinite(baseline) & (baseline > 0)
    baseline_mask[np.arange(len(origins)), origins] = False
    target_count = baseline_mask.sum(axis=1)
    baseline_inverse = np.divide(
        1.0, baseline, out=np.zeros_like(baseline), where=baseline_mask
    )
    baseline_accessibility = np.divide(
        baseline_inverse.sum(axis=1), target_count,
        out=np.full(len(origins), np.nan), where=target_count > 0,
    )
    baseline_valid = np.isfinite(baseline_accessibility) & (baseline_accessibility > 0)
    if ((scenario == 0) & baseline_mask).any():
        raise ValueError()
    reachable = baseline_mask & np.isfinite(scenario) & (scenario > 0)
    scenario_inverse = np.divide(
        1.0, scenario, out=np.zeros_like(scenario), where=reachable
    )
    # Inaccessible destinations contribute zero; retain the baseline set, giving A=0 and SAR=1 if all are lost.
    scenario_accessibility = np.divide(
        scenario_inverse.sum(axis=1), target_count,
        out=np.full(len(origins), np.nan), where=baseline_valid,
    )
    sar = np.full(len(origins), np.nan)
    sar[baseline_valid] = (
        1.0 - scenario_accessibility[baseline_valid] / baseline_accessibility[baseline_valid]
    )
    return {
        "baseline_accessibility": baseline_accessibility,
        "scenario_accessibility": scenario_accessibility,
        "SAR": sar,
        "baseline_valid": baseline_valid,
        "baseline_target_count": target_count,
        "scenario_reachable_count": reachable.sum(axis=1),
        "baseline_mask": baseline_mask,
    }
