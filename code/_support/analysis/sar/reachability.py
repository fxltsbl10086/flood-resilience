"Directed network distances for 500 m grids with multiple access points. Use straight-line centroid-to-boundary access and assume within-grid reachability. Retain original road fragments inside the study area; close all directions and split fragments of a road when its flood flag is one. Preserve topology defined by original endpoint IDs."

import numpy as np
import pandas as pd
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _integers(values, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError()
    if len(values) and (np.any(pd.isna(values)) or np.any(values != values.astype(np.int64))):
        raise ValueError()
    return values.astype(np.int64)


def build_min_csr(n_nodes, rows, cols, weights):
    "Retain the minimum cost for parallel directed edges instead of CSR summation."
    if int(n_nodes) != n_nodes or n_nodes < 0 or n_nodes >= np.iinfo(np.int32).max:
        raise ValueError()
    rows, cols = _integers(rows, "rows"), _integers(cols, "cols")
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(rows) != len(cols) or len(rows) != len(weights):
        raise ValueError()
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= n_nodes) or np.any(cols >= n_nodes):
        raise ValueError()
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError()
    if not len(weights):
        return csr_matrix((n_nodes, n_nodes), dtype=float)
    order = np.lexsort((cols, rows))
    rows, cols, weights = rows[order], cols[order], weights[order]
    starts = np.flatnonzero(np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])])
    return csr_matrix(
        (np.minimum.reduceat(weights, starts),
         (rows[starts].astype(np.int32), cols[starts].astype(np.int32))),
        shape=(n_nodes, n_nodes),
    )


def _nodes_at_positions(positions, nodes, queries):
    "Find a node among this road's ordered cut points without merging different roads by XY."
    queries = np.asarray(queries, dtype=float)
    if not len(queries):
        return np.empty(0, dtype=np.int64)
    right = np.minimum(np.searchsorted(positions, queries), len(positions) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(positions[left] - queries) <= np.abs(positions[right] - queries), left, right)
    if np.any(np.abs(positions[selected] - queries) > 2e-7):
        raise AssertionError()
    return nodes[selected]


def _line_parts(geometry):
    "Keep positive-length line parts; isolated study-boundary contact points do not form travel edges."
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _line_parts(child)]
    return []


def _inside_intervals(line, boundary):
    "Map in-area fragments back to native road chainage without joining external gaps."
    length = line.length
    if boundary.covers(line):
        return [(0.0, length)]
    # Resolve repeated XY positions on closed or self-intersecting roads using original segments.
    if line.is_closed or not line.is_simple:
        coordinates = np.asarray(line.coords)[:, :2]
        intervals, offset = [], 0.0
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            segment = shapely.LineString([start, end])
            segment_length = segment.length
            if segment_length > 0:
                for piece in _line_parts(segment.intersection(boundary)):
                    a = segment.project(shapely.Point(piece.coords[0])) + offset
                    b = segment.project(shapely.Point(piece.coords[-1])) + offset
                    if a != b:
                        intervals.append((min(a, b), max(a, b)))
            offset += segment_length
    else:
        intervals = []
        for piece in _line_parts(line.intersection(boundary)):
            a = line.project(shapely.Point(piece.coords[0]))
            b = line.project(shapely.Point(piece.coords[-1]))
            if a != b:
                intervals.append((min(a, b), max(a, b)))
    # Merge only actually touching or overlapping in-area chainage intervals.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _scenario_key(period):
    value = str(period).upper().removeprefix("T")
    value = int(value)
    if value <= 0:
        raise ValueError()
    return f"T{value}", f"flood_T{value:03d}"


def build_network(roads, edges, candidates, boundary_geom, n_grids, periods):
    "Build fixed-node baseline and scenario graphs and return multi-host access tables with network audits. Candidate fields are grid_index, road_index, position_m and access_distance_m. Retain forward/reverse edge directions and split costs by original road length. Input geometry must be the full LineString so native chainage is preserved."
    if int(n_grids) != n_grids or n_grids < 0:
        raise ValueError()
    n_grids = int(n_grids)
    if boundary_geom is None or boundary_geom.is_empty or not boundary_geom.is_valid:
        raise ValueError()
    if not {"geometry", "road_id"}.issubset(roads.columns):
        raise ValueError()
    if not roads.index.is_unique or not roads.road_id.is_unique or roads.road_id.isna().any():
        raise ValueError()
    road_indices = _integers(roads.index.to_numpy(), "roads.index")
    if "road_index" in roads and not np.array_equal(road_indices, _integers(roads.road_index, "road_index")):
        raise ValueError()
    if not {"road_id", "from_id", "to_id", "direction", "base_cost_m"}.issubset(edges.columns):
        raise ValueError()
    needed = {"grid_index", "road_index", "position_m", "access_distance_m"}
    if not needed.issubset(candidates.columns):
        raise ValueError()
    candidates = candidates.copy().reset_index(drop=True)
    candidates["grid_index"] = _integers(candidates.grid_index, "grid_index")
    candidates["road_index"] = _integers(candidates.road_index, "候选road_index")
    if np.any(candidates.grid_index < 0) or np.any(candidates.grid_index >= n_grids):
        raise ValueError()
    positions = candidates.position_m.to_numpy(dtype=float)
    access = candidates.access_distance_m.to_numpy(dtype=float)
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(access)) or np.any(access < 0):
        raise ValueError()
    position_by_index = {int(index): i for i, index in enumerate(road_indices)}
    if not candidates.road_index.isin(road_indices).all():
        raise ValueError()
    candidate_road = candidates.road_index.map(position_by_index).to_numpy(dtype=np.int64)
    road_ids = roads.road_id.to_numpy()
    if "road_id" in candidates and not np.array_equal(candidates.road_id.to_numpy(), road_ids[candidate_road]):
        raise ValueError()
    candidates["road_id"] = road_ids[candidate_road]
    candidates["node_id"] = -1
    selected_edges = edges.loc[edges.road_id.isin(road_ids)].copy()
    selected_edges["from_id"] = _integers(selected_edges.from_id, "from_id")
    selected_edges["to_id"] = _integers(selected_edges.to_id, "to_id")
    if selected_edges.base_cost_m.isna().any() or np.any(~np.isfinite(selected_edges.base_cost_m)) or np.any(selected_edges.base_cost_m <= 0):
        raise ValueError()
    directions = selected_edges.direction.astype(str).str.lower().replace({"f": "forward", "t": "reverse"})
    if not directions.isin(["forward", "reverse"]).all():
        raise ValueError()
    selected_edges["direction"] = directions
    # Avoid copying unrelated wide-table attributes into every split edge.
    selected_edges = selected_edges[["road_id", "from_id", "to_id", "direction", "base_cost_m"]]
    edge_groups = {road_id: group for road_id, group in selected_edges.groupby("road_id", sort=False)}
    if not roads.road_id.isin(edge_groups).all():
        raise ValueError()

    original_nodes, next_node = {}, 0
    arc_from, arc_to, arc_cost, arc_road = [], [], [], []
    keep_candidates = np.zeros(len(candidates), dtype=bool)
    inside_length, segment_count, clipped_roads = 0.0, 0, 0
    expected_directed_cost, maximum_cost_error = 0.0, 0.0
    endpoint_count, inside_road_count = 0, 0
    road_candidate_groups = candidates.groupby("road_index", sort=False).indices
    lengths = np.empty(len(roads), dtype=float)
    for road_position, (road_index, road) in enumerate(roads.iterrows()):
        line = road.geometry
        if line is None or line.is_empty or line.geom_type != "LineString" or not line.is_valid or line.length <= 0:
            raise ValueError()
        length = float(line.length)
        lengths[road_position] = length
        group = edge_groups[road.road_id]
        road_edges = list(group.itertuples(index=False))
        first = group.iloc[0]
        if first.direction == "forward":
            u, v = int(first.from_id), int(first.to_id)
        else:
            u, v = int(first.to_id), int(first.from_id)
        if "u" in roads and "v" in roads and (int(road.u) != u or int(road.v) != v):
            raise ValueError()
        for edge in road_edges:
            expected = (u, v) if edge.direction == "forward" else (v, u)
            if (edge.from_id, edge.to_id) != expected:
                raise ValueError()
        intervals = _inside_intervals(line, boundary_geom)
        current = np.asarray(road_candidate_groups.get(road_index, []), dtype=np.int64)
        if len(current) and (np.any(positions[current] < -2e-7) or np.any(positions[current] > length + 2e-7)):
            raise ValueError()
        if not intervals:
            continue
        inside_road_count += 1
        if intervals != [(0.0, length)]:
            clipped_roads += 1
        road_inside_length = sum(end - start for start, end in intervals)
        inside_length += road_inside_length
        expected_road_cost = float(group.base_cost_m.sum()) * road_inside_length / length
        expected_directed_cost += expected_road_cost
        first_road_arc = len(arc_cost)
        query = np.clip(positions[current], 0.0, length)
        in_area = np.zeros(len(current), dtype=bool)
        for start, end in intervals:
            in_area |= (query >= start) & (query <= end)
        current, query = current[in_area], query[in_area]
        keep_candidates[current] = True
        # Split using in-area boundary chainage and candidates; do not connect internal nodes across roads by XY.
        interval_ends = np.asarray(intervals).ravel()
        cuts = np.unique(np.r_[interval_ends, query])
        nodes = np.empty(len(cuts), dtype=np.int64)
        for k, position in enumerate(cuts):
            original = u if position == 0.0 else v if position == length else None
            if original is None:
                nodes[k] = next_node
                next_node += 1
            else:
                if original not in original_nodes:
                    original_nodes[original] = next_node
                    next_node += 1
                    endpoint_count += 1
                nodes[k] = original_nodes[original]
        candidates.loc[current, "node_id"] = _nodes_at_positions(cuts, nodes, query)
        for k, (start, end) in enumerate(zip(cuts[:-1], cuts[1:])):
            midpoint = (start + end) / 2.0
            if not any(a <= midpoint <= b for a, b in intervals):
                continue
            segment_count += 1
            for edge in road_edges:
                source, target = (nodes[k], nodes[k + 1]) if edge.direction == "forward" else (nodes[k + 1], nodes[k])
                arc_from.append(source)
                arc_to.append(target)
                arc_cost.append(float(edge.base_cost_m) * (end - start) / length)
                arc_road.append(road_position)
        cost_error = abs(sum(arc_cost[first_road_arc:]) - expected_road_cost)
        maximum_cost_error = max(maximum_cost_error, cost_error)
        if cost_error > 1e-7 + 1e-10 * expected_road_cost:
            raise AssertionError()

    candidates = candidates.loc[keep_candidates].copy().reset_index(drop=True)
    candidates["node_id"] = candidates.node_id.astype(np.int64)
    candidate_road = candidates.road_index.map(position_by_index).to_numpy(dtype=np.int64)
    n_road_nodes = next_node
    source_nodes = np.arange(n_road_nodes, n_road_nodes + n_grids, dtype=np.int64)
    n_nodes = n_road_nodes + n_grids
    arc_from, arc_to = np.asarray(arc_from, dtype=np.int64), np.asarray(arc_to, dtype=np.int64)
    arc_cost, arc_road = np.asarray(arc_cost, dtype=float), np.asarray(arc_road, dtype=np.int64)
    closures = {"baseline": np.zeros(len(roads), dtype=bool), "no_closure": np.zeros(len(roads), dtype=bool)}
    for period in periods:
        scenario, field = _scenario_key(period)
        if field not in roads or roads[field].isna().any() or not roads[field].isin([0, 1]).all():
            raise ValueError()
        if scenario in closures:
            raise ValueError()
        closures[scenario] = roads[field].to_numpy(dtype=bool)
    graphs, available, scene_qa = {}, {}, {}
    for scenario, closed in closures.items():
        active = ~closed[arc_road]
        current = candidates.loc[~closed[candidate_road]].copy()
        # Give grid-source nodes outgoing edges only; minimize destination access at the terminal step.
        rows = np.r_[arc_from[active], source_nodes[current.grid_index.to_numpy(dtype=np.int64)]]
        cols = np.r_[arc_to[active], current.node_id.to_numpy(dtype=np.int64)]
        weights = np.r_[arc_cost[active], current.access_distance_m.to_numpy(dtype=float)]
        graph = build_min_csr(n_nodes, rows, cols, weights)
        if np.any(graph.indices >= n_road_nodes):
            raise AssertionError()
        graphs[scenario], available[scenario] = graph, current
        scene_qa[scenario] = {
            "closed_road_count": int(closed.sum()),
            "open_directed_road_arc_count": int(active.sum()),
            "open_directed_road_cost_m": float(arc_cost[active].sum()),
            "available_candidate_count": len(current),
            "available_grid_count": int(current.grid_index.nunique()),
            "csr_arc_count": int(graph.nnz),
            "virtual_sources_have_no_incoming_arcs": True,
        }
    qa = {
        "endpoint_connection_policy": "inherited",
        "only_in_study_area": True,
        "qualified_road_count": len(roads),
        "road_count_with_inside_length": inside_road_count,
        "clipped_road_count": clipped_roads,
        "original_geometry_length_m": float(lengths.sum()),
        "inside_geometry_length_m": float(inside_length),
        "split_inside_segment_count": segment_count,
        "expected_inside_directed_cost_m": float(expected_directed_cost),
        "split_inside_directed_cost_m": float(arc_cost.sum()),
        "maximum_split_road_cost_error_m": float(maximum_cost_error),
        "retained_original_endpoint_count": endpoint_count,
        "n_road_nodes": n_road_nodes,
        "n_nodes": n_nodes,
        "input_candidate_count": len(keep_candidates),
        "inside_candidate_count": len(candidates),
        "outside_candidate_count": int((~keep_candidates).sum()),
        "excluded_edge_count": int(len(edges) - len(selected_edges)),
        "scenarios": scene_qa,
    }
    return {
        "n_road_nodes": n_road_nodes, "n_nodes": n_nodes,
        "source_nodes": source_nodes, "candidates": candidates,
        "graphs": graphs, "available_candidates": available, "qa": qa,
    }


def compute_distances(network, origin_indices, scenario, batch_size=8):
    "Yield batched grid-to-grid total distances, minimizing over open destination access points and using infinity for unreachable grids."
    origins = _integers(origin_indices, "origin_indices")
    n_grids = len(network["source_nodes"])
    if np.any(origins < 0) or np.any(origins >= n_grids):
        raise ValueError()
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError()
    if scenario not in network["graphs"]:
        raise ValueError()
    graph = network["graphs"][scenario]
    candidates = network["available_candidates"][scenario]
    target_grid = candidates.grid_index.to_numpy(dtype=np.int64)
    target_node = candidates.node_id.to_numpy(dtype=np.int64)
    target_access = candidates.access_distance_m.to_numpy(dtype=float)
    has_access = np.zeros(n_grids, dtype=bool)
    has_access[target_grid] = True
    for start in range(0, len(origins), int(batch_size)):
        current = origins[start:start + int(batch_size)]
        result = np.full((len(current), n_grids), np.inf)
        active_rows = np.flatnonzero(has_access[current])
        if len(active_rows) and len(candidates):
            distances = dijkstra(
                graph, directed=True, indices=network["source_nodes"][current[active_rows]],
                return_predecessors=False, min_only=False,
            )
            distances = np.atleast_2d(distances)
            for row, node_distances in zip(active_rows, distances):
                np.minimum.at(result[row], target_grid, node_distances[target_node] + target_access)
        yield current, result
