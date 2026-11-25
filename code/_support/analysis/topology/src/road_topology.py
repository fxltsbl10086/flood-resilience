"Split road intersections, merge degree-two chains and construct nodes and directed edges."

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely import normalize
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
)
from shapely.ops import snap, split, substring


def vertical_signature(row: pd.Series) -> tuple[int, bool, bool]:
    return (
        int(row["layer_std"]),
        bool(row["bridge_std"]),
        bool(row["tunnel_std"]),
    )


def extract_points(geometry: Any) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, GeometryCollection):
        output: list[Point] = []
        for part in geometry.geoms:
            output.extend(extract_points(part))
        return output
    return []


def is_internal_point(line: LineString, point: Point, tolerance_m: float) -> bool:
    if line.distance(point) > 1e-7:
        return False
    distance = float(line.project(point))
    return tolerance_m < distance < float(line.length) - tolerance_m


def is_source_endpoint(row: pd.Series, point: Point, tolerance_m: float) -> bool:
    required = {
        "_source_start_x",
        "_source_start_y",
        "_source_end_x",
        "_source_end_y",
    }
    if not required.issubset(row.index):
        return False
    start = Point(float(row["_source_start_x"]), float(row["_source_start_y"]))
    end = Point(float(row["_source_end_x"]), float(row["_source_end_y"]))
    return start.distance(point) <= tolerance_m or end.distance(point) <= tolerance_m


def unique_points(points: Iterable[Point], precision: int = 6) -> list[Point]:
    seen: set[tuple[float, float]] = set()
    output: list[Point] = []
    for point in points:
        key = (round(point.x, precision), round(point.y, precision))
        if key not in seen:
            seen.add(key)
            output.append(point)
    return output


def collect_split_points(
    roads: gpd.GeoDataFrame,
    endpoint_tol_m: float,
    source_endpoint_tol_m: float,
) -> tuple[dict[int, list[Point]], pd.DataFrame, dict[str, int]]:
    "Collect split points. Preserve a connection when either source line has an original endpoint at the intersection. Require matching vertical signatures only for intersections internal to both lines, keeping bridge/tunnel transitions connected without falsely joining grade-separated internal crossings."
    split_points = {int(row._row_id): [] for _, row in roads.iterrows()}
    audit_rows: list[dict[str, object]] = []
    counters = {
        "source_endpoint_connections": 0,
        "compatible_internal_crossings": 0,
        "grade_separated_internal_crossings": 0,
        "overlap_candidates": 0,
    }
    spatial_index = roads.sindex

    for i, row_i in roads.iterrows():
        geom_i = row_i.geometry
        for j in spatial_index.query(geom_i, predicate="intersects"):
            j = int(j)
            if j <= i:
                continue
            row_j = roads.iloc[j]
            intersection = geom_i.intersection(row_j.geometry)
            if intersection.is_empty:
                continue
            if isinstance(intersection, (LineString, MultiLineString)):
                counters["overlap_candidates"] += 1
                audit_rows.append(
                    {
                        "road_i": int(row_i["_row_id"]),
                        "road_j": int(row_j["_row_id"]),
                        "relation": "overlap_not_split",
                        "geometry_type": intersection.geom_type,
                    }
                )
                continue

            for point in extract_points(intersection):
                endpoint_i = is_source_endpoint(
                    row_i, point, source_endpoint_tol_m
                )
                endpoint_j = is_source_endpoint(
                    row_j, point, source_endpoint_tol_m
                )
                endpoint_priority = endpoint_i or endpoint_j
                same_vertical = vertical_signature(row_i) == vertical_signature(row_j)
                if endpoint_priority:
                    connect = True
                    relation = "source_endpoint_connection"
                    counters["source_endpoint_connections"] += 1
                elif same_vertical:
                    connect = True
                    relation = "compatible_internal_crossing"
                    counters["compatible_internal_crossings"] += 1
                else:
                    connect = False
                    relation = "grade_separated_internal_crossing"
                    counters["grade_separated_internal_crossings"] += 1

                audit_rows.append(
                    {
                        "road_i": int(row_i["_row_id"]),
                        "road_j": int(row_j["_row_id"]),
                        "relation": relation,
                        "geometry_type": "Point",
                    }
                )
                if not connect:
                    continue
                if is_internal_point(geom_i, point, endpoint_tol_m):
                    split_points[int(row_i["_row_id"])].append(point)
                if is_internal_point(row_j.geometry, point, endpoint_tol_m):
                    split_points[int(row_j["_row_id"])].append(point)

    return split_points, pd.DataFrame(audit_rows), counters


def split_line(
    line: LineString,
    points: list[Point],
    split_snap_tol_m: float,
    endpoint_tol_m: float,
) -> list[LineString]:
    valid = [
        point
        for point in unique_points(points)
        if is_internal_point(line, point, endpoint_tol_m)
    ]
    if not valid:
        return [line]
    multipoint = MultiPoint(valid)
    snapped = snap(line, multipoint, split_snap_tol_m)
    pieces = split(snapped, multipoint)
    return [
        part
        for part in pieces.geoms
        if isinstance(part, LineString) and part.length > 0
    ]


def open_closed_line(line: LineString) -> list[LineString]:
    if not line.is_closed or line.length <= 0:
        return [line]
    midpoint = float(line.length) / 2.0
    return [
        part
        for part in (
            substring(line, 0.0, midpoint),
            substring(line, midpoint, float(line.length)),
        )
        if isinstance(part, LineString) and part.length > 0
    ]


def build_noded_segments(
    roads: gpd.GeoDataFrame,
    params: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, object], pd.DataFrame]:
    split_points, crossing_audit, counters = collect_split_points(
        roads,
        float(params["endpoint_tol_m"]),
        max(float(params["split_snap_tol_m"]), 1e-6),
    )
    records: list[dict[str, object]] = []
    closed_count = 0
    for _, row in roads.iterrows():
        split_parts = split_line(
            row.geometry,
            split_points[int(row["_row_id"])],
            float(params["split_snap_tol_m"]),
            float(params["endpoint_tol_m"]),
        )
        parts: list[LineString] = []
        for part in split_parts:
            opened = open_closed_line(part)
            if len(opened) > 1:
                closed_count += 1
            parts.extend(opened)
        parts = sorted(parts, key=lambda geometry: normalize(geometry).wkb_hex)
        attrs = row.drop(labels="geometry").to_dict()
        for sequence, part in enumerate(parts):
            records.append({**attrs, "split_seq": sequence, "geometry": part})

    noded = gpd.GeoDataFrame(records, geometry="geometry", crs=roads.crs)
    noded["length_m"] = noded.geometry.length
    raw_split_length = float(noded["length_m"].sum())
    short_mask = noded["length_m"] < float(params["min_length_m"])
    short_length = float(noded.loc[short_mask, "length_m"].sum())
    noded = noded.loc[~short_mask].copy()
    noded["_geom_key"] = noded.geometry.apply(
        lambda geometry: normalize(geometry).wkb_hex
    )
    duplicate_subset = [
        "_geom_key",
        "oneway_std",
        "layer_std",
        "bridge_std",
        "tunnel_std",
    ]
    duplicate_mask = noded.duplicated(duplicate_subset, keep="first")
    duplicate_length = float(noded.loc[duplicate_mask, "length_m"].sum())
    noded = noded.loc[~duplicate_mask].copy()
    noded = noded.sort_values(
        ["_row_id", "split_seq", "_geom_key"], kind="mergesort"
    ).reset_index(drop=True)
    noded["atom_id"] = range(len(noded))
    noded = noded.drop(columns="_geom_key")

    before_length = float(roads["length_m"].sum())
    summary: dict[str, object] = {
        "noding_length_before_m": before_length,
        "noding_raw_split_length_m": raw_split_length,
        "noding_relative_error": (
            abs(raw_split_length - before_length) / before_length
            if before_length
            else 0.0
        ),
        "noding_short_removed_length_m": short_length,
        "noding_duplicate_removed_length_m": duplicate_length,
        "noded_segment_count": int(len(noded)),
        "closed_line_open_count": closed_count,
        **counters,
    }
    return noded, summary, crossing_audit


def coordinate_key(point: Point, precision: int) -> str:
    "Key nodes by coordinate, preserving shared source endpoints across bridge/tunnel attribute transitions."
    return (
        f"{round(point.x, precision):.{precision}f}_"
        f"{round(point.y, precision):.{precision}f}"
    )


def reverse_line(line: LineString) -> LineString:
    return LineString(list(line.coords)[::-1])


def allowed_directions(row: pd.Series, traversed_forward: bool) -> set[str]:
    "Return permitted forward/reverse directions relative to the merged chain."
    value = str(row["oneway_std"])
    if value == "B":
        return {"forward", "reverse"}
    if value == "F":
        return {"forward"} if traversed_forward else {"reverse"}
    if value == "T":
        return {"reverse"} if traversed_forward else {"forward"}
    raise ValueError()


def _order_component(
    component: set[int],
    atoms: gpd.GeoDataFrame,
    atom_endpoints: dict[int, tuple[str, str]],
) -> tuple[list[tuple[int, bool]], bool]:
    "Order a nonbranching component physically and identify cycles with is_cycle=True."
    node_atoms: dict[str, list[int]] = defaultdict(list)
    for atom_id in component:
        u_key, v_key = atom_endpoints[atom_id]
        node_atoms[u_key].append(atom_id)
        node_atoms[v_key].append(atom_id)
    ends = sorted(key for key, values in node_atoms.items() if len(values) == 1)
    if len(ends) == 0:
        return [], True
    if len(ends) != 2:
        raise RuntimeError()

    ordered: list[tuple[int, bool]] = []
    current_node = ends[0]
    previous_atom: int | None = None
    while len(ordered) < len(component):
        candidates = sorted(
            atom_id
            for atom_id in node_atoms[current_node]
            if atom_id != previous_atom
        )
        if len(candidates) != 1:
            raise RuntimeError()
        atom_id = candidates[0]
        u_key, v_key = atom_endpoints[atom_id]
        forward = current_node == u_key
        ordered.append((atom_id, forward))
        current_node = v_key if forward else u_key
        previous_atom = atom_id
    return ordered, False


def _compatible_subchains(
    ordered: list[tuple[int, bool]],
    atoms: gpd.GeoDataFrame,
) -> list[list[tuple[int, bool]]]:
    output: list[list[tuple[int, bool]]] = []
    current: list[tuple[int, bool]] = []
    current_allowed = {"forward", "reverse"}
    for atom_id, traversed_forward in ordered:
        row = atoms.loc[atom_id]
        allowed = allowed_directions(row, traversed_forward)
        next_allowed = current_allowed.intersection(allowed)
        if current and not next_allowed:
            output.append(current)
            current = [(atom_id, traversed_forward)]
            current_allowed = set(allowed)
        else:
            current.append((atom_id, traversed_forward))
            current_allowed = next_allowed
    if current:
        output.append(current)
    return output


def _merge_geometry(
    ordered: list[tuple[int, bool]], atoms: gpd.GeoDataFrame
) -> LineString:
    coordinates: list[tuple[float, ...]] = []
    for atom_id, forward in ordered:
        line = atoms.loc[atom_id].geometry
        coords = list(line.coords if forward else reversed(line.coords))
        if coordinates and coordinates[-1][:2] == coords[0][:2]:
            coordinates.extend(coords[1:])
        else:
            coordinates.extend(coords)
    return LineString(coordinates)


def _source_text(value: Any, fallback: str) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return fallback
    return str(value).strip()


def _dominant_row(
    ordered: list[tuple[int, bool]], atoms: gpd.GeoDataFrame
) -> tuple[pd.Series, list[str]]:
    lengths: dict[str, float] = defaultdict(float)
    ground_lengths: dict[str, float] = defaultdict(float)
    atom_rows: list[pd.Series] = []
    for atom_id, _ in ordered:
        row = atoms.loc[atom_id]
        atom_rows.append(row)
        key = _source_text(row.get("osm_id"), f"row:{int(row['_row_id'])}")
        lengths[key] += float(row.geometry.length)
        if not bool(row["bridge_std"]):
            ground_lengths[key] += float(row.geometry.length)

    # Prefer ordinary-road attributes when the merged chain includes ordinary roads.
    # Use all source roads only when the complete chain is a bridge.
    ranking_lengths = ground_lengths if ground_lengths else lengths
    dominant_key = sorted(
        ranking_lengths, key=lambda key: (-ranking_lengths[key], key)
    )[0]
    candidates = [
        row
        for row in atom_rows
        if _source_text(row.get("osm_id"), f"row:{int(row['_row_id'])}")
        == dominant_key
        and (not ground_lengths or not bool(row["bridge_std"]))
    ]
    dominant = sorted(
        candidates,
        key=lambda row: (-float(row.geometry.length), int(row["_row_id"])),
    )[0]
    source_ids = sorted(lengths)
    return dominant, source_ids


def merge_degree_two_chains(
    noded: gpd.GeoDataFrame,
    params: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, object]]:
    precision = int(params["node_precision"])
    atoms = noded.copy().set_index("atom_id", drop=False)
    atom_endpoints: dict[int, tuple[str, str]] = {}
    node_atoms: dict[str, list[int]] = defaultdict(list)
    for atom_id, row in atoms.iterrows():
        coords = list(row.geometry.coords)
        u_key = coordinate_key(Point(coords[0]), precision)
        v_key = coordinate_key(Point(coords[-1]), precision)
        atom_endpoints[int(atom_id)] = (u_key, v_key)
        node_atoms[u_key].append(int(atom_id))
        node_atoms[v_key].append(int(atom_id))

    continuation = nx.Graph()
    continuation.add_nodes_from(int(value) for value in atoms.index)
    vertical_stop_count = 0
    bridge_ground_merge_node_count = 0
    source_endpoint_tol_m = max(float(params["split_snap_tol_m"]), 1e-6)
    for node_key, incident in node_atoms.items():
        if len(incident) != 2 or incident[0] == incident[1]:
            continue
        first, second = incident
        first_row = atoms.loc[first]
        second_row = atoms.loc[second]
        same_vertical = vertical_signature(first_row) == vertical_signature(second_row)
        shared_point = next(
            Point(coordinate)
            for coordinate in (
                list(first_row.geometry.coords)[0],
                list(first_row.geometry.coords)[-1],
            )
            if coordinate_key(Point(coordinate), precision) == node_key
        )
        endpoint_priority = is_source_endpoint(
            first_row, shared_point, source_endpoint_tol_m
        ) or is_source_endpoint(second_row, shared_point, source_endpoint_tol_m)
        bridge_ground_transition = (
            bool(first_row["bridge_std"]) != bool(second_row["bridge_std"])
            and not bool(first_row["tunnel_std"])
            and not bool(second_row["tunnel_std"])
        )
        if not same_vertical and not (
            bool(params["allow_bridge_ground_endpoint_merge"])
            and endpoint_priority
            and bridge_ground_transition
        ):
            vertical_stop_count += 1
            continue
        if not same_vertical:
            bridge_ground_merge_node_count += 1
        continuation.add_edge(first, second)

    records: list[dict[str, object]] = []
    cycle_atom_count = 0
    oneway_split_count = 0
    merged_atom_count = 0
    merged_chain_count = 0
    for component_values in nx.connected_components(continuation):
        component = {int(value) for value in component_values}
        ordered, is_cycle = _order_component(component, atoms, atom_endpoints)
        if is_cycle:
            cycle_atom_count += len(component)
            subchains = [[(atom_id, True)] for atom_id in sorted(component)]
        else:
            subchains = _compatible_subchains(ordered, atoms)
            if len(subchains) > 1:
                oneway_split_count += len(subchains) - 1

        for subchain in subchains:
            geometry = _merge_geometry(subchain, atoms)
            dominant, source_ids = _dominant_row(subchain, atoms)
            attrs = dominant.drop(labels="geometry").to_dict()
            bridge_length_m = float(
                sum(
                    atoms.loc[atom_id].geometry.length
                    for atom_id, _ in subchain
                    if bool(atoms.loc[atom_id, "bridge_std"])
                )
            )
            ground_length_m = float(
                sum(
                    atoms.loc[atom_id].geometry.length
                    for atom_id, _ in subchain
                    if not bool(atoms.loc[atom_id, "bridge_std"])
                )
            )
            vertical_signatures = {
                vertical_signature(atoms.loc[atom_id])
                for atom_id, _ in subchain
            }
            allowed = {"forward", "reverse"}
            for atom_id, forward in subchain:
                allowed &= allowed_directions(atoms.loc[atom_id], forward)
            if not allowed:
                raise RuntimeError()
            oneway_std = (
                "B"
                if allowed == {"forward", "reverse"}
                else "F"
                if allowed == {"forward"}
                else "T"
            )
            if len(subchain) > 1:
                merged_chain_count += 1
                merged_atom_count += len(subchain)
            records.append(
                {
                    **attrs,
                    "dominant_osm_id": _source_text(
                        dominant.get("osm_id"), f"row:{int(dominant['_row_id'])}"
                    ),
                    "source_osm_ids": "|".join(source_ids),
                    "source_osm_count": len(source_ids),
                    "merged_part_count": len(subchain),
                    "contains_bridge": bridge_length_m > 0,
                    "bridge_length_m": bridge_length_m,
                    "ground_length_m": ground_length_m,
                    "mixed_vertical_signature": len(vertical_signatures) > 1,
                    "oneway_std": oneway_std,
                    "geometry": geometry,
                }
            )

    merged = gpd.GeoDataFrame(records, geometry="geometry", crs=noded.crs)
    merged["length_m"] = merged.geometry.length
    merged["_geom_key"] = merged.geometry.apply(
        lambda geometry: normalize(geometry).wkb_hex
    )
    merged = merged.sort_values(
        ["dominant_osm_id", "source_osm_ids", "_geom_key"], kind="mergesort"
    ).reset_index(drop=True)
    city_code = str(params["city_code"])
    merged["road_id"] = [
        f"{city_code}_R{index:09d}" for index in range(len(merged))
    ]
    merged = merged.drop(columns="_geom_key")

    before_length = float(noded.geometry.length.sum())
    after_length = float(merged.geometry.length.sum())
    summary = {
        "merge_length_before_m": before_length,
        "merge_length_after_m": after_length,
        "merge_relative_error": (
            abs(after_length - before_length) / before_length
            if before_length
            else 0.0
        ),
        "merged_facility_count": int(len(merged)),
        "merged_chain_count": merged_chain_count,
        "atoms_participating_in_merge": merged_atom_count,
        "bridge_ground_merge_node_count": bridge_ground_merge_node_count,
        "mixed_bridge_facility_count": int(
            (
                merged["contains_bridge"]
                & (merged["ground_length_m"] > 0)
            ).sum()
        ),
        "degree_two_vertical_stop_count": vertical_stop_count,
        "degree_two_oneway_split_count": oneway_split_count,
        "cycle_atoms_kept_unmerged": cycle_atom_count,
    }
    return merged, summary


def build_nodes_facilities_edges(
    facilities: gpd.GeoDataFrame,
    boundary_geom: Any,
    node_precision: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    key_to_point: dict[str, Point] = {}
    endpoint_keys: list[tuple[str, str]] = []
    for _, row in facilities.iterrows():
        coords = list(row.geometry.coords)
        start = Point(coords[0])
        end = Point(coords[-1])
        start_key = coordinate_key(start, node_precision)
        end_key = coordinate_key(end, node_precision)
        key_to_point.setdefault(start_key, start)
        key_to_point.setdefault(end_key, end)
        endpoint_keys.append((start_key, end_key))

    sorted_keys = sorted(key_to_point)
    lookup = {key: index for index, key in enumerate(sorted_keys)}
    nodes = gpd.GeoDataFrame(
        {
            "node_id": range(len(sorted_keys)),
            "node_key": sorted_keys,
            "geometry": [key_to_point[key] for key in sorted_keys],
        },
        geometry="geometry",
        crs=facilities.crs,
    )
    facilities = facilities.copy()
    facilities["u"] = [lookup[keys[0]] for keys in endpoint_keys]
    facilities["v"] = [lookup[keys[1]] for keys in endpoint_keys]
    self_loops = int((facilities["u"] == facilities["v"]).sum())
    if self_loops:
        raise RuntimeError()
    facilities["target_length_m"] = facilities.geometry.apply(
        lambda geometry: float(geometry.intersection(boundary_geom).length)
    )
    facilities["in_area"] = facilities["target_length_m"] > 0

    edge_records: list[dict[str, object]] = []
    for _, row in facilities.iterrows():
        base = {
            "road_id": row["road_id"],
            "fclass": row["fclass"],
            "road_group": row["road_group"],
            "length_m": float(row["length_m"]),
            "dominant_osm_id": row["dominant_osm_id"],
        }
        direction = str(row["oneway_std"])
        if direction in {"B", "F"}:
            edge_records.append(
                {
                    **base,
                    "from_id": int(row["u"]),
                    "to_id": int(row["v"]),
                    "direction": "forward",
                    "base_cost_m": float(row["length_m"]),
                    "geometry": row.geometry,
                }
            )
        if direction in {"B", "T"}:
            edge_records.append(
                {
                    **base,
                    "from_id": int(row["v"]),
                    "to_id": int(row["u"]),
                    "direction": "reverse",
                    "base_cost_m": float(row["length_m"]),
                    "geometry": reverse_line(row.geometry),
                }
            )
    edges = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=facilities.crs)
    edges = edges.sort_values(["road_id", "direction"], kind="mergesort").reset_index(drop=True)
    edges["edge_id"] = range(len(edges))

    graph = nx.Graph()
    graph.add_nodes_from(nodes["node_id"].astype(int))
    for _, row in facilities.iterrows():
        graph.add_edge(int(row["u"]), int(row["v"]))
    components = sorted(
        nx.connected_components(graph),
        key=lambda values: (-len(values), min(values)),
    )
    node_component = {
        node_id: component_id
        for component_id, values in enumerate(components)
        for node_id in values
    }
    nodes["comp_id"] = nodes["node_id"].map(node_component).astype("int64")
    facilities["comp_id"] = facilities["u"].map(node_component).astype("int64")
    edges["comp_id"] = edges["from_id"].map(node_component).astype("int64")
    degree = dict(graph.degree())
    nodes["degree"] = nodes["node_id"].map(degree).fillna(0).astype("int64")

    total_length = float(facilities["length_m"].sum())
    largest_length = float(
        facilities.loc[facilities["comp_id"] == 0, "length_m"].sum()
    ) if components else 0.0
    summary = {
        "road_node_count": int(len(nodes)),
        "routing_edge_count": int(len(edges)),
        "self_loop_count": self_loops,
        "connected_component_count": int(len(components)),
        "largest_component_length_share": (
            largest_length / total_length if total_length else 0.0
        ),
    }

    # Retain consistent, traceable facility fields required by downstream analysis.
    # Temporary IDs and source endpoint coordinates have served network construction and QA.
    # Exclude those temporary attributes from formal exports.
    facilities = facilities.rename(
        columns={
            "maxspeed": "maxspeed_raw",
            "access": "access_raw",
            "vehicle": "vehicle_raw",
            "motor_veh": "motor_vehicle_raw",
            "motorcar": "motorcar_raw",
        }
    )
    facility_columns = [
        "road_id",
        "dominant_osm_id",
        "source_osm_ids",
        "source_osm_count",
        "merged_part_count",
        "contains_bridge",
        "bridge_length_m",
        "ground_length_m",
        "mixed_vertical_signature",
        "osm_id",
        "name",
        "ref",
        "fclass",
        "road_group",
        "oneway_raw",
        "oneway_std",
        "oneway_uncertain",
        "maxspeed_raw",
        "layer_raw",
        "layer_std",
        "bridge_raw",
        "bridge_std",
        "tunnel_raw",
        "tunnel_std",
        "is_elevated",
        "is_underpass",
        "access_raw",
        "vehicle_raw",
        "motor_vehicle_raw",
        "motorcar_raw",
        "u",
        "v",
        "length_m",
        "target_length_m",
        "in_area",
        "comp_id",
        "geometry",
    ]
    for column in facility_columns:
        if column not in facilities.columns and column != "geometry":
            facilities[column] = None
    facilities = facilities[facility_columns].copy()
    return nodes, facilities, edges, summary
