"""Select facility-grid contacts with the largest EFAR-loss contribution per cell and category; do not summarize final benefits."""
from __future__ import annotations

import time
import config  # Use the formal entry's Windows DLL/GDAL initialization.
import numpy as np
from scipy.sparse.csgraph import dijkstra
import network_engine as engine

DIMENSIONS = ("rescue", "medical", "shelter")
DIRECTIONS = ("设施→居民", "居民→设施", "居民→设施")
SELECTION_POLICY = "max_weighted_inverse_loss_per_dimension_exact_ties_facility_grid_id_v1"


def _require(condition, message):
    if not condition:
        raise ValueError("")


def _inverse(values, mask):
    values = np.asarray(values, dtype=float)
    allowed = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values > 0)
    return np.divide(1.0, values, out=np.zeros_like(values), where=allowed)


def _losses(base_distances, scene_distances, fixed_mask, weights):
    """Compute unrounded contact losses; remove only negative floating-point noise and preserve positive values."""
    weights = np.asarray(weights, dtype=np.int64)
    candidates = np.asarray(fixed_mask, bool) & (weights > 0)
    base = weights * _inverse(base_distances, candidates)
    scene = weights * _inverse(scene_distances, candidates)
    loss = base - scene
    tolerance = 64 * np.finfo(float).eps * np.maximum(np.maximum(np.abs(base), np.abs(scene)), np.finfo(float).tiny)
    if np.any(loss < -tolerance):
        raise AssertionError("")
    negative = loss < 0
    loss[negative] = 0.0
    loss[~candidates] = 0.0
    return loss, candidates, int(negative.sum()), float(np.max(tolerance)) if len(tolerance) else 0.0


def _choose_contact(losses, candidates, facility_grid_ids):
    """Use exact maxima without rounding or approximate ties; break exact ties by ascending grid_id string."""
    positive = np.flatnonzero(np.asarray(candidates, bool) & (np.asarray(losses) > 0))
    if not len(positive):
        return None, 0, 0
    maximum = float(np.max(np.asarray(losses)[positive]))
    tied = positive[np.asarray(losses)[positive] == maximum]
    selected = min(map(int, tied), key=lambda column: str(facility_grid_ids[column]))
    return selected, int(len(positive)), int(len(tied))


def prepare_ungoverned_od(topology, baseline, progress=None):
    """Calculate unrestored contact OD for the current return period once; the caller manages caching."""
    total_started = time.perf_counter()
    residents = np.asarray(baseline["resident_indices"], dtype=np.int64)
    counts = np.asarray(baseline["counts"], dtype=np.int64)
    n_grids = len(topology["source_nodes"])
    _require(counts.shape == (n_grids, 3), "")
    scene_inverse = np.zeros((len(residents), 3), dtype=float)
    inverse_losses = np.zeros((len(residents), 3), dtype=float)
    reachable_count = np.zeros((len(residents), 3), dtype=np.int64)
    nearest = np.full((len(residents), 3), np.inf, dtype=float)
    modes = []
    timings = {"network_update_seconds": 0.0, "routing_seconds": 0.0, "efar_seconds": 0.0}
    negative_count, max_decomposition_error = 0, 0.0
    for mode in baseline["modes"]:
        stamp = time.perf_counter()
        facilities = np.asarray(mode["facility_indices"], dtype=np.int64)
        scene = engine.build_scene(topology, (), baseline["scenario"], mode["reverse"])
        arrays = engine._candidate_arrays(scene, n_grids)
        timings["network_update_seconds"] += time.perf_counter() - stamp
        distances = np.full((len(residents), len(facilities)), np.inf, dtype=float)
        for row, resident in enumerate(residents):
            stamp = time.perf_counter()
            node_dist = dijkstra(scene["graph"], directed=True,
                                 indices=int(topology["source_nodes"][resident]), return_predecessors=False)
            od = engine._od_from_nodes(node_dist, facilities, arrays)
            od[facilities == resident] = baseline["local_distance"]
            distances[row] = od
            timings["routing_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            fixed = mode["fixed_mask"][row]
            reachable = fixed & np.isfinite(od) & (od > 0)
            if np.any(od[reachable] + 1e-6 < mode["distances"][row, reachable]):
                raise AssertionError("")
            inverse = _inverse(od, fixed)
            for dim in mode["dims"]:
                weights = counts[facilities, dim]
                scene_inverse[row, dim] = float(np.dot(inverse, weights))
                reachable_count[row, dim] = int(np.dot(reachable, weights))
                eligible = reachable & (weights > 0)
                if np.any(eligible):
                    nearest[row, dim] = float(np.min(od[eligible]))
                losses, _, negative, _ = _losses(mode["distances"][row], od, fixed, weights)
                inverse_losses[row, dim] = float(np.sum(losses))
                negative_count += negative
                expected_loss = float(baseline["base_inverse"][row, dim] - scene_inverse[row, dim])
                error = abs(float(np.sum(losses)) - expected_loss)
                max_decomposition_error = max(max_decomposition_error, error)
                if not np.isclose(np.sum(losses), expected_loss, rtol=1e-11, atol=1e-13):
                    raise AssertionError("")
            timings["efar_seconds"] += time.perf_counter() - stamp
            if progress and (row % 50 == 0 or row + 1 == len(residents)):
                progress(f"{baseline['scenario']} contact OD {mode['dims']}: {row + 1}/{len(residents)}")
        modes.append({"dims": list(mode["dims"]), "reverse": bool(mode["reverse"]),
                      "facility_indices": facilities.copy(), "distances": distances})
    denominator = np.asarray(baseline["base_inverse"], float)
    valid = (np.asarray(baseline["base_count"]) > 0) & (denominator > 0)
    risk = 1 - np.divide(scene_inverse, denominator, out=np.full_like(denominator, np.nan), where=valid)
    if np.any(risk[np.isfinite(risk)] < -1e-10) or np.any(risk[np.isfinite(risk)] > 1 + 1e-10):
        raise AssertionError("")
    decomposed_risk = np.divide(inverse_losses, denominator, out=np.full_like(denominator, np.nan), where=valid)
    _require(np.allclose(risk, decomposed_risk, rtol=1e-10, atol=1e-10, equal_nan=True),
             "")
    risk_error = float(np.max(np.abs(risk[valid] - decomposed_risk[valid]))) if np.any(valid) else 0.0
    timings["total_seconds"] = time.perf_counter() - total_started
    return {"resident_indices": residents.copy(), "scenario": baseline["scenario"], "modes": modes,
            "scene_inverse": scene_inverse, "reachable_count": reachable_count,
            "total_inverse_loss": inverse_losses,
            "risk": np.clip(risk, 0, 1), "nearest_distance": nearest,
            "nearest_distance_m": nearest, "timings": timings,
            "qa": {"loss_decomposition_checked": True,
                   "maximum_inverse_loss_decomposition_error": max_decomposition_error,
                   "maximum_EFAR_decomposition_error": risk_error,
                   "floating_negative_contacts_clipped": negative_count,
                   "baseline_facility_set_fixed": True, "same_grid_distance_m": baseline["local_distance"]}}


def select_loss_projects(topology, baseline, ungoverned_od, grid_ids, progress=None):
    """Select at most one maximum-loss contact per resident cell and category, then union obstructed physical roads across categories."""
    started = time.perf_counter()
    grid_ids = np.asarray(grid_ids, dtype=str)
    residents = np.asarray(baseline["resident_indices"], dtype=np.int64)
    counts = np.asarray(baseline["counts"], dtype=np.int64)
    _require(len(grid_ids) == len(topology["source_nodes"]) and len(set(grid_ids)) == len(grid_ids),
             "")
    _require(np.array_equal(residents, ungoverned_od["resident_indices"]), "")
    _require(ungoverned_od["scenario"] == baseline["scenario"], "")
    _require(len(ungoverned_od["modes"]) == len(baseline["modes"]), "")
    projects = [set() for _ in residents]
    selections, contact_roads = [], []
    timings = {"contact_selection_seconds": 0.0, "network_update_seconds": 0.0,
               "baseline_predecessor_seconds": 0.0, "path_trace_seconds": 0.0}
    max_decomposition_error, negative_count = 0.0, 0
    selected_by_mode = []
    for mode, scene_mode in zip(baseline["modes"], ungoverned_od["modes"]):
        stamp = time.perf_counter()
        facilities = np.asarray(mode["facility_indices"], dtype=np.int64)
        _require(mode["dims"] == scene_mode["dims"] and mode["reverse"] == scene_mode["reverse"]
                 and np.array_equal(facilities, scene_mode["facility_indices"]), "")
        _require(scene_mode["distances"].shape == mode["distances"].shape, "")
        chosen_rows = []
        for row, resident in enumerate(residents):
            for dim in mode["dims"]:
                base_sum = float(baseline["base_inverse"][row, dim])
                scene_sum = float(ungoverned_od["scene_inverse"][row, dim])
                valid = bool(baseline["base_count"][row, dim] > 0 and base_sum > 0)
                losses, candidates, negative, _ = _losses(mode["distances"][row], scene_mode["distances"][row],
                                                        mode["fixed_mask"][row], counts[facilities, dim])
                negative_count += negative
                total_loss = float(np.sum(losses))
                expected_loss = base_sum - scene_sum
                error = abs(total_loss - expected_loss)
                max_decomposition_error = max(max_decomposition_error, error)
                _require(np.isclose(total_loss, expected_loss, rtol=1e-11, atol=1e-13),
                         "")
                selected, positive_count, tied_count = _choose_contact(losses, candidates, grid_ids[facilities])
                status = "invalid_baseline" if not valid else ("selected" if selected is not None else "zero_loss")
                if not valid:
                    _require(selected is None, "")
                initial_risk = float(np.clip(1 - scene_sum / base_sum, 0, 1)) if valid else np.nan
                if valid:
                    _require(np.isclose(total_loss / base_sum, initial_risk, rtol=1e-10, atol=1e-10),
                             "")
                record = {"grid_id": str(grid_ids[resident]), "grid_index": int(resident),
                          "dimension": DIMENSIONS[dim], "direction": DIRECTIONS[dim],
                          "selected_facility_grid_id": None, "selected_facility_grid_index": None,
                          "facility_count": None, "baseline_distance_m": np.nan,
                          "initial_scenario_distance_m": np.nan, "scenario": baseline["scenario"],
                          "base_inverse": base_sum,
                          "initial_scene_inverse": scene_sum, "initial_EFAR": initial_risk,
                          "total_inverse_loss": total_loss, "selected_inverse_loss": 0.0,
                          "selected_EFAR_contribution": 0.0, "selected_loss_share": np.nan,
                          "candidate_count": int(candidates.sum()), "positive_loss_contact_count": positive_count,
                          "max_tie_count": tied_count, "selection_status": status, "project_road_count": 0}
                if selected is not None:
                    facility = int(facilities[selected])
                    value = float(losses[selected])
                    record.update(selected_facility_grid_id=str(grid_ids[facility]),
                                  selected_facility_grid_index=facility, facility_count=int(counts[facility, dim]),
                                  baseline_distance_m=float(mode["distances"][row, selected]),
                                  initial_scenario_distance_m=float(scene_mode["distances"][row, selected]),
                                  selected_inverse_loss=value, selected_EFAR_contribution=value / base_sum,
                                  selected_loss_share=value / total_loss)
                    chosen_rows.append((row, dim, int(selected), record))
                selections.append(record)
        selected_by_mode.append(chosen_rows)
        timings["contact_selection_seconds"] += time.perf_counter() - stamp
    for mode, chosen_rows in zip(baseline["modes"], selected_by_mode):
        if not chosen_rows:
            continue
        stamp = time.perf_counter()
        scene = engine.build_scene(topology, scenario=baseline["scenario"], reverse=mode["reverse"], baseline=True)
        arrays = engine._candidate_arrays(scene, len(grid_ids))
        lookup = engine._arc_provenance(topology, mode["reverse"])
        facilities = np.asarray(mode["facility_indices"], dtype=np.int64)
        timings["network_update_seconds"] += time.perf_counter() - stamp
        by_resident = {}
        for item in chosen_rows:
            by_resident.setdefault(item[0], []).append(item)
        for row, selected_items in by_resident.items():
            resident = int(residents[row])
            source = int(topology["source_nodes"][resident])
            stamp = time.perf_counter()
            distances, predecessors = dijkstra(scene["graph"], directed=True, indices=source,
                                               return_predecessors=True)
            # Find terminals and trace paths only for selected contacts, rather than all facilities.
            selected_facilities = np.asarray([facilities[item[2]] for item in selected_items], dtype=np.int64)
            od, choices = engine._od_from_nodes(distances, selected_facilities, arrays, return_choice=True)
            timings["baseline_predecessor_seconds"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            for current, (row, dim, column, record) in enumerate(selected_items):
                _require(selected_facilities[current] != resident, "")
                _require(np.isclose(od[current], mode["distances"][row, column], rtol=1e-12, atol=1e-9),
                         "")
                road_set = engine._trace_roads(topology, predecessors, source, choices[current], lookup)
                closed = sorted(road for road in road_set if topology["closures"][baseline["scenario"]][road])
                _require(bool(closed), "")
                record["project_road_count"] = len(closed)
                projects[row].update(closed)
                for road in closed:
                    contact_roads.append({"grid_id": str(grid_ids[resident]), "grid_index": resident,
                                          "dimension": DIMENSIONS[dim],
                                          "selected_facility_grid_id": record["selected_facility_grid_id"],
                                          "road_position": int(road)})
            timings["path_trace_seconds"] += time.perf_counter() - stamp
            if progress and (row % 50 == 0 or row + 1 == len(residents)):
                progress(f"{baseline['scenario']} selected-contact baseline paths {mode['dims']}: {row + 1}/{len(residents)}")
    selections.sort(key=lambda record: (record["grid_index"], DIMENSIONS.index(record["dimension"])))
    contact_roads.sort(key=lambda record: (record["grid_index"], DIMENSIONS.index(record["dimension"]), record["road_position"]))
    _require(len(selections) == 3 * len(residents), "")
    timings["total_seconds"] = time.perf_counter() - started
    return {"resident_indices": residents.copy(), "scenario": baseline["scenario"],
            "projects": [np.asarray(sorted(roads), dtype=np.int64) for roads in projects],
            "selection_records": selections, "contact_road_records": contact_roads,
            "selection_policy": SELECTION_POLICY, "timings": timings,
            "qa": {"quantity_weighted_facility_grid_contacts": True,
                   "maximum_inverse_loss_decomposition_error": max_decomposition_error,
                   "floating_negative_contacts_clipped": negative_count,
                   "selected_contacts_have_blocked_physical_roads": True,
                   "maximum_contacts_per_resident_dimension": 1,
                   "exact_tie_policy": "raw loss maximum; facility grid_id ascending",
                   "zero_loss_share": "NaN; no selected contact and no project",
                   "risk_weighting_and_facility_set_unchanged": True}}
