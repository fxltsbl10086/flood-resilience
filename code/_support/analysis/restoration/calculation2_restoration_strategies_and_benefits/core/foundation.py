"""Validate accessibility baselines and selected facility contacts."""
from __future__ import annotations


import config
import numpy as np
import pandas as pd
import contact_projects


PERIODS = (10, 20, 50, 100, 200)


def _require(condition, message):
    if not condition:
        raise ValueError("")


def bind_baseline(shared, scenario):
    """Bind a scenario to shared baseline accessibility data."""
    _require(scenario in {f"T{period}" for period in PERIODS}, "")
    _require(shared["all_contact_projects_traced"] is False,
             "")
    _require(all(len(project) == 0 for project in shared["projects"])
             and all(len(project) == 0 for project in shared["project_road_positions"]),
             "")
    result = dict(shared)
    result["scenario"] = scenario
    return result


def _validate_baseline(pack, baseline):
    residents = np.asarray(pack["resident_indices"], dtype=np.int64)
    np.testing.assert_array_equal(baseline["resident_indices"], residents)
    np.testing.assert_array_equal(baseline["counts"], pack["counts"])
    _require(baseline["local_distance"] == config.LOCAL_DISTANCE_M, "")
    _require(baseline["dimensions_valid"].shape == (len(residents), 3)
             and baseline["dimensions_valid"].all(), "")
    _require(len(baseline["modes"]) == 2, "")
    counts = np.asarray(pack["counts"])
    for mode, dims, reverse in zip(baseline["modes"], ([0], [1, 2]), (True, False)):
        _require(mode["dims"] == dims and mode["reverse"] is reverse, "")
        facilities = np.flatnonzero(counts[:, dims].sum(axis=1) > 0)
        np.testing.assert_array_equal(mode["facility_indices"], facilities)
        distances = np.asarray(mode["distances"])
        _require(distances.shape == (len(residents), len(facilities)), "")
        fixed = np.isfinite(distances) & (distances > 0)
        np.testing.assert_array_equal(mode["fixed_mask"], fixed)
        inverse = np.divide(1.0, distances, out=np.zeros_like(distances), where=fixed)
        for dim in dims:
            np.testing.assert_array_equal(baseline["base_count"][:, dim], fixed @ counts[facilities, dim])
            _require(np.allclose(baseline["base_inverse"][:, dim], inverse @ counts[facilities, dim],
                                rtol=1e-12, atol=1e-13), "")
    bind_baseline(baseline, f"T{config.PERIOD}")


def _validate_ungoverned(pack, baseline, od):
    np.testing.assert_array_equal(od["resident_indices"], baseline["resident_indices"])
    _require(od["scenario"] == baseline["scenario"], "")
    _require(len(od["modes"]) == 2, "")
    expected = pack["evaluation_all"].set_index("grid_index").loc[baseline["resident_indices"]]
    errors = {}
    for column, dim in enumerate(config.DIMENSION_KEYS):
        difference = np.abs(od["risk"][:, column] - expected[f"initial_EFAR_{dim}"].to_numpy(float))
        errors[dim] = float(difference.max()) if len(difference) else 0.0
        _require(errors[dim] <= config.RISK_ATOL,
                 "")
        np.testing.assert_array_equal(od["reachable_count"][:, column],
                                      expected[f"initial_reachable_count_{dim}"].to_numpy(np.int64))
    for mode, scene in zip(baseline["modes"], od["modes"]):
        _require(mode["dims"] == scene["dims"] and mode["reverse"] == scene["reverse"], "")
        np.testing.assert_array_equal(mode["facility_indices"], scene["facility_indices"])
        _require(mode["distances"].shape == scene["distances"].shape, "")
    return errors


def _validate_selection(pack, baseline, selection, od):
    np.testing.assert_array_equal(selection["resident_indices"], baseline["resident_indices"])
    _require(selection["scenario"] == baseline["scenario"], "")
    _require(selection["selection_policy"] == contact_projects.SELECTION_POLICY, "")
    records = pd.DataFrame(selection["selection_records"])
    links = pd.DataFrame(selection["contact_road_records"], columns=[
        "grid_id", "grid_index", "dimension", "selected_facility_grid_id", "road_position"])
    residents = baseline["resident_indices"]
    pairs = {(int(i), dim) for i in residents for dim in config.DIMENSION_KEYS}
    _require(len(records) == len(pairs) and not records.duplicated(["grid_index", "dimension"]).any()
             and set(zip(records.grid_index, records.dimension)) == pairs, "")
    _require(records.scenario.eq(baseline["scenario"]).all(), "")
    _require(not links.duplicated(["grid_index", "dimension", "road_position"]).any(), "")
    grid_ids = pack["grids"].grid_id.astype(str).to_numpy()
    grid_rows = {int(value): index for index, value in enumerate(residents)}
    grouped_links = {key: part for key, part in links.groupby(["grid_index", "dimension"], sort=False)}
    chosen_check_count = 0
    for record in records.to_dict("records"):
        grid_index, dim = int(record["grid_index"]), record["dimension"]
        row, column = grid_rows[grid_index], config.DIMENSION_KEYS.index(dim)
        _require(record["grid_id"] == grid_ids[grid_index], "")
        _require(abs(record["base_inverse"] - baseline["base_inverse"][row, column]) <= 1e-12
                 and abs(record["initial_EFAR"] - od["risk"][row, column]) <= config.RISK_ATOL,
                 "")
        mode_index = 0 if column == 0 else 1
        mode, scene_mode = baseline["modes"][mode_index], od["modes"][mode_index]
        facilities = mode["facility_indices"]
        loss, mask, _, _ = contact_projects._losses(mode["distances"][row], scene_mode["distances"][row],
                                                  mode["fixed_mask"][row], baseline["counts"][facilities, column])
        chosen, _, _ = contact_projects._choose_contact(loss, mask, grid_ids[facilities])
        current = grouped_links.get((grid_index, dim))
        associated_count = 0 if current is None else len(current)
        _require(associated_count == record["project_road_count"], "")
        if chosen is None:
            _require(record["selection_status"] == "zero_loss" and associated_count == 0
                     and record["selected_EFAR_contribution"] == 0, "")
        else:
            target = int(facilities[chosen])
            _require(record["selection_status"] == "selected"
                     and int(record["selected_facility_grid_index"]) == target
                     and record["selected_facility_grid_id"] == grid_ids[target],
                     "")
            _require(associated_count > 0 and current.selected_facility_grid_id.eq(grid_ids[target]).all(),
                     "")
            _require(np.isclose(record["initial_scenario_distance_m"], scene_mode["distances"][row, chosen],
                                rtol=1e-12, atol=1e-9), "")
            _require(np.isclose(record["selected_EFAR_contribution"],
                                loss[chosen] / baseline["base_inverse"][row, column], rtol=1e-12, atol=1e-13)
                     and np.isclose(record["selected_loss_share"], loss[chosen] / loss.sum(),
                                    rtol=1e-12, atol=1e-13), "")
            chosen_check_count += 1
    closed = pack["topology"]["closures"][baseline["scenario"]]
    _require(len(selection["projects"]) == len(residents), "")
    for index, project in zip(residents, selection["projects"]):
        positions = set(map(int, project))
        expected = set(links.loc[links.grid_index.eq(index), "road_position"].astype(int))
        _require(positions == expected and all(0 <= p < len(closed) and closed[p] for p in positions),
                 "")
    return {"max_loss_contacts_checked_against_all_cached_candidates": chosen_check_count,
            "scenario_and_road_closures_match": True, "contact_projects_match": True}
