"""Restoration network engine preserving the original multi-access directed EFAR semantics.

Center-to-boundary access uses straight-line distance and assumes within-cell accessibility.
Retain only original road portions inside the study area; a flood flag of 1 closes every direction and split segment of the road_id.
The static front half follows result-one reachability.py and retains source-road subarcs.
The remaining code evaluates grid projects and restored-network accessibility without final benefit summaries."""

import numpy as np
import pandas as pd
import shapely
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def _integers(values, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("")
    if len(values) and (np.any(pd.isna(values)) or np.any(values != values.astype(np.int64))):
        raise ValueError("")
    return values.astype(np.int64)


def build_min_csr(n_nodes, rows, cols, weights):
    """Take minimum cost for parallel edges with identical directed endpoints; do not let CSR construction sum them."""
    if int(n_nodes) != n_nodes or n_nodes < 0 or n_nodes >= np.iinfo(np.int32).max:
        raise ValueError("")
    rows, cols = _integers(rows, "rows"), _integers(cols, "cols")
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(rows) != len(cols) or len(rows) != len(weights):
        raise ValueError("")
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= n_nodes) or np.any(cols >= n_nodes):
        raise ValueError("")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("")
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
    """Find a node among the road's ordered split positions without merging roads by XY coordinates."""
    queries = np.asarray(queries, dtype=float)
    if not len(queries):
        return np.empty(0, dtype=np.int64)
    right = np.minimum(np.searchsorted(positions, queries), len(positions) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(positions[left] - queries) <= np.abs(positions[right] - queries), left, right)
    if np.any(np.abs(positions[selected] - queries) > 2e-7):
        raise AssertionError("")
    return nodes[selected]


def _line_parts(geometry):
    """Extract only positive-length lines; isolated study-boundary contacts do not create traversable edges."""
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _line_parts(child)]
    return []


def _inside_intervals(line, boundary):
    """Map within-study road portions to original chainage without bridging gaps outside the study area."""
    length = line.length
    if boundary.covers(line):
        return [(0.0, length)]
    # The same XY can have multiple chainages on closed/self-crossing roads; resolve ambiguity per original segment.
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
    # Merge only touching/overlapping within-study chainages; do not bridge external gaps using tolerances.
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
        raise ValueError("")
    return f"T{value}", f"flood_T{value:03d}"


def build_network(roads, edges, candidates, boundary_geom, n_grids, periods):
    """Build baseline and flood graphs on fixed nodes, returning multi-host access and graph checks.

Candidate columns are grid_index, road_index, position_m and access_distance_m.
Edges are forward/reverse, with costs proportional to original road length.
Original geometry must be the complete LineString; never clip first and reset its chainage."""
    if int(n_grids) != n_grids or n_grids < 0:
        raise ValueError("")
    n_grids = int(n_grids)
    if boundary_geom is None or boundary_geom.is_empty or not boundary_geom.is_valid:
        raise ValueError("")
    if not {"geometry", "road_id"}.issubset(roads.columns):
        raise ValueError("")
    if not roads.index.is_unique or not roads.road_id.is_unique or roads.road_id.isna().any():
        raise ValueError("")
    road_indices = _integers(roads.index.to_numpy(), "roads.index")
    if "road_index" in roads and not np.array_equal(road_indices, _integers(roads.road_index, "road_index")):
        raise ValueError("")
    if not {"road_id", "from_id", "to_id", "direction", "base_cost_m"}.issubset(edges.columns):
        raise ValueError("")
    needed = {"grid_index", "road_index", "position_m", "access_distance_m"}
    if not needed.issubset(candidates.columns):
        raise ValueError("")
    candidates = candidates.copy().reset_index(drop=True)
    candidates["grid_index"] = _integers(candidates.grid_index, "grid_index")
    candidates["road_index"] = _integers(candidates.road_index, "候选road_index")
    if np.any(candidates.grid_index < 0) or np.any(candidates.grid_index >= n_grids):
        raise ValueError("")
    positions = candidates.position_m.to_numpy(dtype=float)
    access = candidates.access_distance_m.to_numpy(dtype=float)
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(access)) or np.any(access < 0):
        raise ValueError("")
    position_by_index = {int(index): i for i, index in enumerate(road_indices)}
    if not candidates.road_index.isin(road_indices).all():
        raise ValueError("")
    candidate_road = candidates.road_index.map(position_by_index).to_numpy(dtype=np.int64)
    road_ids = roads.road_id.to_numpy()
    if "road_id" in candidates and not np.array_equal(candidates.road_id.to_numpy(), road_ids[candidate_road]):
        raise ValueError("")
    candidates["road_id"] = road_ids[candidate_road]
    candidates["node_id"] = -1
    selected_edges = edges.loc[edges.road_id.isin(road_ids)].copy()
    selected_edges["from_id"] = _integers(selected_edges.from_id, "from_id")
    selected_edges["to_id"] = _integers(selected_edges.to_id, "to_id")
    if selected_edges.base_cost_m.isna().any() or np.any(~np.isfinite(selected_edges.base_cost_m)) or np.any(selected_edges.base_cost_m <= 0):
        raise ValueError("")
    directions = selected_edges.direction.astype(str).str.lower().replace({"f": "forward", "t": "reverse"})
    if not directions.isin(["forward", "reverse"]).all():
        raise ValueError("")
    selected_edges["direction"] = directions
    # Other wide-table fields do not affect routing; avoid rebuilding large tuples per subsegment.
    selected_edges = selected_edges[["road_id", "from_id", "to_id", "direction", "base_cost_m"]]
    edge_groups = {road_id: group for road_id, group in selected_edges.groupby("road_id", sort=False)}
    if not roads.road_id.isin(edge_groups).all():
        raise ValueError("")

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
            raise ValueError("")
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
            raise ValueError("")
        for edge in road_edges:
            expected = (u, v) if edge.direction == "forward" else (v, u)
            if (edge.from_id, edge.to_id) != expected:
                raise ValueError("")
        intervals = _inside_intervals(line, boundary_geom)
        current = np.asarray(road_candidate_groups.get(road_index, []), dtype=np.int64)
        if len(current) and (np.any(positions[current] < -2e-7) or np.any(positions[current] > length + 2e-7)):
            raise ValueError("")
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
        # Split consistently at study-boundary chainages and candidates; do not join internal nodes across roads by XY.
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
            raise AssertionError("")

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
            raise ValueError("")
        if scenario in closures:
            raise ValueError("")
        closures[scenario] = roads[field].to_numpy(dtype=bool)
    graphs, available, scene_qa = {}, {}, {}
    for scenario, closed in closures.items():
        active = ~closed[arc_road]
        current = candidates.loc[~closed[candidate_road]].copy()
        # Cell source nodes have outgoing edges only; apply target access distance at terminal minimization, without permanent links.
        rows = np.r_[arc_from[active], source_nodes[current.grid_index.to_numpy(dtype=np.int64)]]
        cols = np.r_[arc_to[active], current.node_id.to_numpy(dtype=np.int64)]
        weights = np.r_[arc_cost[active], current.access_distance_m.to_numpy(dtype=float)]
        graph = build_min_csr(n_nodes, rows, cols, weights)
        if np.any(graph.indices >= n_road_nodes):
            raise AssertionError("")
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
        "arc_from": arc_from, "arc_to": arc_to, "arc_cost": arc_cost, "arc_road": arc_road,
        "road_ids": np.asarray(road_ids, dtype=str), "closures": closures,
        "candidate_road": candidate_road,
    }


def compute_distances(network, origin_indices, scenario, batch_size=8):
    """Return grid-to-grid total distances in batches, minimizing across all open target intersections; unreachable distances are inf."""
    origins = _integers(origin_indices, "origin_indices")
    n_grids = len(network["source_nodes"])
    if np.any(origins < 0) or np.any(origins >= n_grids):
        raise ValueError("")
    if int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("")
    if scenario not in network["graphs"]:
        raise ValueError("")
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


def build_scene(topology, recovered_road_positions=(), scenario="T10", reverse=False,
                baseline=False):
    """Restore all directions, subarcs and access points of physical roads; reverse only the road subgraph when requested."""
    closed = topology["closures"][scenario].copy()
    restored = np.asarray(sorted(set(map(int, recovered_road_positions))), dtype=np.int64)
    if len(restored) and (restored.min() < 0 or restored.max() >= len(closed)):
        raise ValueError("")
    if len(restored) and np.any(~closed[restored]):
        raise ValueError("")
    closed[restored] = False
    if baseline:
        closed[:] = False
    active = ~closed[topology["arc_road"]]
    current = topology["candidates"].loc[~closed[topology["candidate_road"]]].copy()
    a, b = topology["arc_from"][active], topology["arc_to"][active]
    if reverse:
        a, b = b, a
    rows = np.r_[a, topology["source_nodes"][current.grid_index.to_numpy(np.int64)]]
    cols = np.r_[b, current.node_id.to_numpy(np.int64)]
    weights = np.r_[topology["arc_cost"][active], current.access_distance_m.to_numpy(float)]
    graph = build_min_csr(topology["n_nodes"], rows, cols, weights)
    if np.any(graph.indices >= topology["n_road_nodes"]):
        raise AssertionError("")
    return {"graph": graph, "candidates": current, "closed": closed,
            "reverse": bool(reverse), "recovered_positions": restored}


def assert_same_graph(observed, reference, label):
    """Compare node ordering and every CSR arc cost to prevent incorrect topology-cache reuse."""
    if observed.shape != reference.shape or not np.array_equal(observed.indptr, reference.indptr) \
            or not np.array_equal(observed.indices, reference.indices) \
            or not np.allclose(observed.data, reference.data, rtol=1e-12, atol=1e-9):
        raise AssertionError("")


def validate_topology(topology, reference, scenario="T10"):
    if not np.array_equal(topology["source_nodes"], reference["source_nodes"]):
        raise AssertionError("")
    for name in ("baseline", scenario):
        assert_same_graph(topology["graphs"][name], reference["graphs"][name], name)
        left, right = topology["available_candidates"][name], reference["available_candidates"][name]
        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    assert_same_graph(build_scene(topology, (), scenario)["graph"],
                      reference["graphs"][scenario], "0%恢复")
    closed = np.flatnonzero(topology["closures"][scenario])
    assert_same_graph(build_scene(topology, closed, scenario)["graph"],
                      reference["graphs"]["baseline"], "全部关闭道路恢复")
    return {"baseline_graph_matches": True, "ungoverned_graph_matches": True,
            "full_recovery_matches_baseline": True, "candidate_tables_match": True}


def _candidate_arrays(scene, n_grids):
    frame = scene["candidates"]
    # Retain candidate-table ordering for reproducible terminal selection among tied shortest paths.
    grid = frame.grid_index.to_numpy(np.int64)
    node = frame.node_id.to_numpy(np.int64)
    access = frame.access_distance_m.to_numpy(float)
    index = frame.index.to_numpy(np.int64)
    groups = [np.flatnonzero(grid == i) for i in range(n_grids)]
    return frame, grid, node, access, index, groups


def _od_from_nodes(node_distances, facility_indices, candidate_arrays, return_choice=False):
    if not return_choice:
        # Minimize over each target cell's open access points; preserve Dijkstra and subsequent risk sums.
        # Retain the original tracing loop and first-occurrence ordering for tied terminals.
        _, grid, nodes, access, _, groups = candidate_arrays
        grid_distances = np.full(len(groups), np.inf, dtype=np.float64)
        np.minimum.at(grid_distances, grid, node_distances[nodes] + access)
        return grid_distances[facility_indices]
    _, _, nodes, access, indices, groups = candidate_arrays
    distances = np.full(len(facility_indices), np.inf, dtype=np.float64)
    chosen = np.full(len(facility_indices), -1, dtype=np.int64)
    for column, grid in enumerate(facility_indices):
        candidates = groups[int(grid)]
        if not len(candidates):
            continue
        values = node_distances[nodes[candidates]] + access[candidates]
        best = int(np.argmin(values))
        distances[column] = values[best]
        if np.isfinite(values[best]):
            chosen[column] = indices[candidates[best]]
    return (distances, chosen) if return_choice else distances


def _arc_provenance(topology, reverse=False):
    """Match CSR minimum-cost parallel-arc semantics; break equal-cost ties stably by road_id."""
    a, b = topology["arc_from"], topology["arc_to"]
    if reverse:
        a, b = b, a
    road = topology["arc_road"]
    ids = topology["road_ids"]
    order = np.lexsort((ids[road], topology["arc_cost"], b, a))
    mapping = {}
    for j in order:
        mapping.setdefault((int(a[j]), int(b[j])), int(road[j]))
    # Virtual access arcs can share endpoints; minimize access distance, then break ties by road_id.
    candidates = topology["candidates"]
    road_positions = topology["candidate_road"]
    access_order = np.lexsort((ids[road_positions], candidates.access_distance_m,
                              candidates.node_id, candidates.grid_index))
    for j in access_order:
        row = candidates.iloc[int(j)]
        mapping.setdefault((int(topology["source_nodes"][int(row.grid_index)]), int(row.node_id)),
                           int(road_positions[j]))
    return mapping


def _trace_roads(topology, predecessor, source, target_candidate_index, arc_lookup):
    candidate = topology["candidates"].loc[int(target_candidate_index)]
    node = int(candidate.node_id)
    # Terminal access is outside the graph; explicitly include the road carrying the target-cell terminal.
    roads = {int(topology["candidate_road"][int(target_candidate_index)])}
    visited = set()
    while node != source:
        if node in visited:
            raise AssertionError("")
        visited.add(node)
        previous = int(predecessor[node])
        if previous < 0:
            raise AssertionError("")
        key = (previous, node)
        if key not in arc_lookup:
            raise AssertionError("")
        roads.add(arc_lookup[key])
        node = previous
    return roads


def prepare_baseline(topology, resident_indices, counts, scenario="T10", local_distance=250.0,
                     progress=None, trace_all_projects=True):
    """Save costly baseline OD and grid-road projects; resident-origin routing is equivalent to the original facility-origin convention.

The fixed set contains all baseline-reachable facility cells, weighted by facility counts per category.
Rescue routes use the reversed road network from residents; medical and shelter routes use the forward network.
Same-cell resources always use 250 m and do not generate road-restoration projects.
The maximum-loss-contact strategy disables trace_all_projects, saving baseline OD and tracing selected contacts separately."""
    residents = _integers(resident_indices, "resident_indices")
    counts = np.asarray(counts, dtype=np.int64)
    n_grids = len(topology["source_nodes"])
    if counts.shape != (n_grids, 3) or np.any(counts < 0):
        raise ValueError("")
    if len(set(residents)) != len(residents) or np.any(residents < 0) or np.any(residents >= n_grids):
        raise ValueError("")
    if not np.isfinite(local_distance) or local_distance <= 0:
        raise ValueError("")
    projects = [set() for _ in residents]
    base_inverse = np.zeros((len(residents), 3), dtype=float)
    base_count = np.zeros((len(residents), 3), dtype=np.int64)
    modes = []
    for dims, reverse in [([0], True), ([1, 2], False)]:
        facilities = np.flatnonzero(counts[:, dims].sum(axis=1) > 0)
        scene = build_scene(topology, scenario=scenario, reverse=reverse, baseline=True)
        arrays = _candidate_arrays(scene, n_grids)
        lookup = _arc_provenance(topology, reverse) if trace_all_projects else None
        distances = np.full((len(residents), len(facilities)), np.inf, dtype=float)
        for row, resident in enumerate(residents):
            source = int(topology["source_nodes"][resident])
            if trace_all_projects:
                node_dist, predecessors = dijkstra(scene["graph"], directed=True, indices=source,
                                                   return_predecessors=True)
                od, choice = _od_from_nodes(node_dist, facilities, arrays, return_choice=True)
            else:
                node_dist = dijkstra(scene["graph"], directed=True, indices=source,
                                     return_predecessors=False)
                od = _od_from_nodes(node_dist, facilities, arrays)
            local = facilities == resident
            od[local] = local_distance
            fixed = np.isfinite(od) & (od > 0)
            distances[row] = od
            inv = np.divide(1.0, od, out=np.zeros_like(od), where=fixed)
            for dim in dims:
                weights = counts[facilities, dim]
                base_count[row, dim] = int(np.dot(fixed, weights))
                base_inverse[row, dim] = float(np.dot(inv, weights))
            if trace_all_projects:
                for column in np.flatnonzero(fixed & ~local):
                    route_roads = _trace_roads(topology, predecessors, source, choice[column], lookup)
                    projects[row].update(r for r in route_roads if topology["closures"][scenario][r])
            if progress and (row % 20 == 0 or row + 1 == len(residents)):
                    progress(f"Baseline OD {dims}: {row + 1}/{len(residents)}")
        modes.append({"dims": dims, "reverse": reverse, "facility_indices": facilities,
                      "distances": distances,
                      "fixed_mask": np.isfinite(distances) & (distances > 0)})
    projects = [np.asarray(sorted(v), dtype=np.int64) for v in projects]
    return {"resident_indices": residents, "counts": counts.copy(), "scenario": scenario,
            "local_distance": float(local_distance), "modes": modes,
            "base_inverse": base_inverse, "base_count": base_count,
            "dimensions_valid": (base_count > 0) & (base_inverse > 0),
            "project_road_positions": projects, "projects": projects,
            "all_contact_projects_traced": bool(trace_all_projects),
            "path_policy": ("one deterministic baseline shortest path per reachable facility grid; includes both access hosts"
                            if trace_all_projects else
                            "baseline OD only; selected maximum-loss contacts are traced separately")}


def evaluate(topology, recovered_road_positions, baseline, progress=None):
    """Reevaluate all specified resident cells for the union of complete restored roads without aggregating totals or improvements."""
    residents = baseline["resident_indices"]
    counts = baseline["counts"]
    n_grids = len(topology["source_nodes"])
    scene_inverse = np.zeros((len(residents), 3), dtype=float)
    reachable_count = np.zeros((len(residents), 3), dtype=np.int64)
    nearest_distance = np.full((len(residents), 3), np.inf, dtype=float)
    timings = {"network_update_seconds": 0.0, "routing_seconds": 0.0, "efar_seconds": 0.0}
    for mode in baseline["modes"]:
        stamp = time.perf_counter()
        scene = build_scene(topology, recovered_road_positions, baseline["scenario"], mode["reverse"])
        facilities, dims = mode["facility_indices"], mode["dims"]
        arrays = _candidate_arrays(scene, n_grids)
        timings["network_update_seconds"] += time.perf_counter() - stamp
        for row, resident in enumerate(residents):
            stamp = time.perf_counter()
            source = int(topology["source_nodes"][resident])
            node_dist = dijkstra(scene["graph"], directed=True, indices=source,
                                 return_predecessors=False)
            od = _od_from_nodes(node_dist, facilities, arrays)
            od[facilities == resident] = baseline["local_distance"]
            timings["routing_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            valid = mode["fixed_mask"][row] & np.isfinite(od) & (od > 0)
            if np.any(od[valid] + 1e-6 < mode["distances"][row, valid]):
                raise AssertionError("")
            inv = np.divide(1.0, od, out=np.zeros_like(od), where=valid)
            for dim in dims:
                weights = counts[facilities, dim]
                scene_inverse[row, dim] = float(np.dot(inv, weights))
                reachable_count[row, dim] = int(np.dot(valid, weights))
                eligible = valid & (weights > 0)
                if np.any(eligible):
                    nearest_distance[row, dim] = float(np.min(od[eligible]))
            timings["efar_seconds"] += time.perf_counter() - stamp
            if progress and (row % 100 == 0 or row + 1 == len(residents)):
                progress(f"Restoration network evaluation {dims}: {row + 1}/{len(residents)}")
    stamp = time.perf_counter()
    denominator = baseline["base_inverse"]
    valid = (baseline["base_count"] > 0) & (denominator > 0)
    risks = 1 - np.divide(scene_inverse, denominator, out=np.full_like(denominator, np.nan), where=valid)
    if np.any(risks[np.isfinite(risks)] < -1e-9) or np.any(risks[np.isfinite(risks)] > 1 + 1e-9):
        raise AssertionError("")
    timings["efar_seconds"] += time.perf_counter() - stamp
    return {"resident_indices": residents.copy(), "base_inverse": denominator.copy(),
            "base_count": baseline["base_count"].copy(), "scene_inverse": scene_inverse,
            "reachable_count": reachable_count, "nearest_distance_m": nearest_distance,
            "nearest_distance": nearest_distance, "timings": timings,
            "risk": np.clip(risks, 0, 1)}

