"Coordinate single-city road filtering, intersection splitting, merging and exports."

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import normalize

from config import EXPECTED_KEEP_TYPES, RoadConfig
from .road_topology import (
    build_noded_segments,
    build_nodes_facilities_edges,
    merge_degree_two_chains,
)


ROAD_GROUPS = {
    "motorway": "expressway",
    "motorway_link": "expressway",
    "trunk": "expressway",
    "trunk_link": "expressway",
    "primary": "arterial",
    "primary_link": "arterial",
    "secondary": "arterial",
    "secondary_link": "arterial",
    "tertiary": "local",
    "tertiary_link": "local",
    "residential": "local",
    "unclassified": "local",
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_oneway(value: Any, fallback: str = "B") -> tuple[str, bool]:
    text = _text(value).upper()
    if text in {"F", "YES", "Y", "TRUE", "1"}:
        return "F", False
    if text in {"T", "-1", "REVERSE"}:
        return "T", False
    if text in {"", "B", "NO", "N", "FALSE", "0", "NONE", "NAN", "<NA>"}:
        return "B", False
    return fallback.upper(), True


def normalize_layer(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def normalize_bool(value: Any, *, tunnel: bool = False) -> bool:
    values = {"t", "true", "yes", "y", "1"}
    values.add("building_passage" if tunnel else "viaduct")
    return _text(value).lower() in values


def standardize_fields(roads: gpd.GeoDataFrame, params: dict[str, Any]) -> gpd.GeoDataFrame:
    required = {"fclass", "geometry"}
    missing = required.difference(roads.columns)
    if missing:
        raise ValueError()
    if roads.crs is None:
        raise ValueError()

    work = roads.copy()
    work["_source_row"] = range(len(work))
    for column in (
        "osm_id",
        "name",
        "ref",
        "oneway",
        "maxspeed",
        "layer",
        "bridge",
        "tunnel",
        "access",
        "vehicle",
        "motor_veh",
        "motorcar",
    ):
        if column not in work.columns:
            work[column] = None
    work["oneway_raw"] = work["oneway"]
    work["layer_raw"] = work["layer"]
    work["bridge_raw"] = work["bridge"]
    work["tunnel_raw"] = work["tunnel"]

    normalized = work["oneway"].apply(
        lambda value: normalize_oneway(value, str(params["unknown_oneway_as"]))
    )
    work["oneway_std"] = normalized.str[0]
    work["oneway_uncertain"] = normalized.str[1].astype(bool)
    work["layer_std"] = work["layer"].apply(normalize_layer).astype("int16")
    work["bridge_std"] = work["bridge"].apply(normalize_bool).astype(bool)
    work["tunnel_std"] = work["tunnel"].apply(
        lambda value: normalize_bool(value, tunnel=True)
    ).astype(bool)
    work["is_elevated"] = work["bridge_std"] | (work["layer_std"] > 0)
    work["is_underpass"] = work["tunnel_std"] | (work["layer_std"] < 0)
    work["fclass"] = work["fclass"].astype(str).str.strip().str.lower()
    work["road_group"] = work["fclass"].map(ROAD_GROUPS)

    starts = work.geometry.apply(lambda line: list(line.coords)[0])
    ends = work.geometry.apply(lambda line: list(line.coords)[-1])
    work["_source_start_x"] = starts.str[0]
    work["_source_start_y"] = starts.str[1]
    work["_source_end_x"] = ends.str[0]
    work["_source_end_y"] = ends.str[1]
    return work


def prepare_roads(
    roads: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    config: RoadConfig,
) -> tuple[gpd.GeoDataFrame, Any, dict[str, object]]:
    params = config.params
    roads_utm = roads.to_crs(config.target_crs)
    boundary_utm = boundary.to_crs(config.target_crs)
    boundary_geom = boundary_utm.geometry.union_all()
    if boundary_geom.is_empty:
        raise ValueError()

    standardized = standardize_fields(roads_utm, params)
    clip_geom = boundary_geom.buffer(float(params["clip_buffer_m"]))
    clipped = gpd.clip(standardized, clip_geom)
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    clipped = clipped.loc[
        clipped.geometry.notna()
        & ~clipped.geometry.is_empty
        & clipped.geometry.geom_type.eq("LineString")
    ].copy()
    clipped["fclass"] = clipped["fclass"].astype(str).str.lower()
    kept = clipped.loc[clipped["fclass"].isin(EXPECTED_KEEP_TYPES)].copy()
    kept["length_m"] = kept.geometry.length
    kept = kept.loc[kept["length_m"] >= float(params["min_length_m"])].copy()

    kept["_geom_key"] = kept.geometry.apply(
        lambda geometry: normalize(geometry).wkb_hex
    )
    kept["_osm_sort"] = kept["osm_id"].fillna("").astype(str)
    kept = kept.sort_values(
        ["_osm_sort", "_source_row", "_geom_key"], kind="mergesort"
    )
    duplicate_subset = [
        "_geom_key",
        "oneway_std",
        "layer_std",
        "bridge_std",
        "tunnel_std",
    ]
    duplicate_mask = kept.duplicated(duplicate_subset, keep="first")
    duplicate_length = float(kept.loc[duplicate_mask, "length_m"].sum())
    kept = kept.loc[~duplicate_mask].reset_index(drop=True)
    kept["_row_id"] = range(len(kept))
    kept = kept.drop(columns=["_geom_key", "_osm_sort"])

    retained_types = sorted(kept["fclass"].unique().tolist())
    summary = {
        "input_road_count": int(len(roads)),
        "clipped_road_count_before_type_filter": int(len(clipped)),
        "filtered_road_count": int(len(kept)),
        "filtered_road_length_m": float(kept["length_m"].sum()),
        "base_duplicate_removed_length_m": duplicate_length,
        "retained_types": "|".join(retained_types),
        "service_count_after_filter": int((kept["fclass"] == "service").sum()),
        "living_street_count_after_filter": int(
            (kept["fclass"] == "living_street").sum()
        ),
        "unknown_oneway_count": int(kept["oneway_uncertain"].sum()),
        "target_crs": config.target_crs,
        "clip_buffer_m": float(params["clip_buffer_m"]),
    }
    return kept, boundary_geom, summary


def build_qa(
    facilities: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    summaries: list[dict[str, object]],
) -> pd.DataFrame:
    values = {key: value for summary in summaries for key, value in summary.items()}
    observed_types = set(facilities["fclass"].astype(str))
    node_ids = set(nodes["node_id"].astype(int))
    rows = [
        ("道路设施段非空", len(facilities), len(facilities) > 0, True),
        (
            "道路类型仅属于固定12类",
            "|".join(sorted(observed_types)),
            observed_types.issubset(set(EXPECTED_KEEP_TYPES)),
            True,
        ),
        ("service数量", int((facilities["fclass"] == "service").sum()), int((facilities["fclass"] == "service").sum()) == 0, True),
        ("living_street数量", int((facilities["fclass"] == "living_street").sum()), int((facilities["fclass"] == "living_street").sum()) == 0, True),
        ("打断长度相对误差", values["noding_relative_error"], float(values["noding_relative_error"]) <= 1e-6, True),
        ("合并长度相对误差", values["merge_relative_error"], float(values["merge_relative_error"]) <= 1e-6, True),
        ("道路编号唯一", facilities["road_id"].nunique(), facilities["road_id"].is_unique, True),
        ("节点编号唯一", nodes["node_id"].nunique(), nodes["node_id"].is_unique, True),
        ("路由边编号唯一", edges["edge_id"].nunique(), edges["edge_id"].is_unique, True),
        ("设施段自环数量", int((facilities["u"] == facilities["v"]).sum()), bool((facilities["u"] != facilities["v"]).all()), True),
        ("路由边节点引用完整", int(edges["from_id"].isin(node_ids).all() and edges["to_id"].isin(node_ids).all()), bool(edges["from_id"].isin(node_ids).all() and edges["to_id"].isin(node_ids).all()), True),
        ("路由成本全部为正", float(edges["base_cost_m"].min()), bool((edges["base_cost_m"] > 0).all()), True),
        ("来源端点优先连接次数", values["source_endpoint_connections"], True, False),
        ("内部同层交叉连接次数", values["compatible_internal_crossings"], True, False),
        ("内部立体交叉排除次数", values["grade_separated_internal_crossings"], True, False),
        ("重叠线候选数量", values["overlap_candidates"], True, False),
        ("合并连续链数量", values["merged_chain_count"], True, False),
        ("参与合并原子段数量", values["atoms_participating_in_merge"], True, False),
        ("桥梁普通道路跨垂向合并节点数", values["bridge_ground_merge_node_count"], True, False),
        ("含桥梁与普通道路的合并设施段数", values["mixed_bridge_facility_count"], True, False),
        ("垂向变化停止合并数量", values["degree_two_vertical_stop_count"], True, False),
        ("单向冲突分链数量", values["degree_two_oneway_split_count"], True, False),
        ("闭环保留原子段数量", values["cycle_atoms_kept_unmerged"], True, False),
        ("连通分量数量", values["connected_component_count"], int(values["connected_component_count"]) > 0, True),
        ("最大连通分量长度占比", values["largest_component_length_share"], 0 < float(values["largest_component_length_share"]) <= 1, True),
        ("未知单向字段数量", values["unknown_oneway_count"], True, False),
        ("裁剪缓冲_m", values["clip_buffer_m"], float(values["clip_buffer_m"]) == 500.0, True),
    ]
    return pd.DataFrame(
        rows, columns=["检查项", "检查值", "是否通过", "是否硬检查"]
    )


def expected_output_paths(
    output_root: str | Path, city_code: str
) -> dict[str, Path]:
    output_root = Path(output_root).resolve()
    city_dir = output_root / city_code
    return {
        "facilities": city_dir / "01-01__split_and_merged_road_segments.gpkg",
        "nodes": city_dir / "01-02__road_network_nodes.gpkg",
        "edges": city_dir / "01-03__directed_routing_edges.gpkg",
    }


def run_city(config: RoadConfig, output_root: str | Path) -> dict[str, object]:
    output_root = Path(output_root).resolve()
    outputs = expected_output_paths(output_root, config.city_code)
    existing = [path for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError()

    roads = gpd.read_file(config.input_paths["road"])
    boundary = gpd.read_file(config.input_paths["boundary"])
    filtered, boundary_geom, base_summary = prepare_roads(roads, boundary, config)
    noded, noding_summary, _crossing_audit = build_noded_segments(
        filtered, config.params
    )
    merge_params = {**config.params, "city_code": config.city_code}
    merged, merge_summary = merge_degree_two_chains(noded, merge_params)
    nodes, facilities, edges, graph_summary = build_nodes_facilities_edges(
        merged,
        boundary_geom,
        int(config.params["node_precision"]),
    )
    qa = build_qa(
        facilities,
        nodes,
        edges,
        [base_summary, noding_summary, merge_summary, graph_summary],
    )
    failed = qa.loc[qa["是否硬检查"] & ~qa["是否通过"].astype(bool)]
    if not failed.empty:
        details = "；".join(
            f"{row['检查项']}={row['检查值']}" for _, row in failed.iterrows()
        )
        raise RuntimeError()

    output_dir = outputs["facilities"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    facilities.to_file(
        outputs["facilities"], layer="road_facilities", driver="GPKG"
    )
    nodes.to_file(outputs["nodes"], layer="road_nodes", driver="GPKG")
    edges.to_file(outputs["edges"], layer="routing_edges", driver="GPKG")
    return {
        "city_code": config.city_code,
        "facility_count": int(len(facilities)),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "output_dir": str(output_dir),
    }
