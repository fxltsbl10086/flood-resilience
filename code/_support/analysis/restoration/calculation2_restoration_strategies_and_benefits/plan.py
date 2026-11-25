"""Fix road ordering independently per period; restored inundated length is divided by all inundated length in the same city/scenario."""
from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left

import numpy as np
import pandas as pd


STRATEGY = "scenario_priority_inundated_length_targets_v1"
TARGET_PCTS = (0, 1, 2, *range(5, 101, 5))
LENGTH_THRESHOLD_TOLERANCE_M = 1e-6
SAMPLING = "targets 0,1,2,5,10,...,100 percent of all same-city same-period inundated length; select shortest fixed road prefix reaching each target; retain actual length and overshoot"


def road_hash(road_ids):
    return hashlib.sha256(json.dumps(sorted(map(str, road_ids)), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def validate_baseline_inventory(inventory):
    required = {"road_id", "road_position", "length_m",
                "inside_evaluation_area_length_m", "outside_evaluation_area_length_m"}
    if not required <= set(inventory):
        raise ValueError("")
    if inventory.road_id.isna().any() or inventory.road_id.astype(str).duplicated().any():
        raise ValueError("")
    if inventory.road_position.duplicated().any():
        raise ValueError("")
    lengths = inventory.length_m.to_numpy(float)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("")
    if not np.allclose(inventory.inside_evaluation_area_length_m + inventory.outside_evaluation_area_length_m,
                       lengths, rtol=1e-10, atol=1e-6):
        raise ValueError("")
    return math.fsum(map(float, lengths))


def validate_inundated_inventory(inventory):
    """Use the full-city analysis network for the denominator without filtering by candidate roads, district boundaries or closure flags."""
    validate_baseline_inventory(inventory)
    if "inundated_length_m" not in inventory:
        raise ValueError("")
    wet = inventory.inundated_length_m.to_numpy(float)
    full = inventory.length_m.to_numpy(float)
    if not np.isfinite(wet).all() or (wet < 0).any() or (wet > full + 1e-6).any():
        raise ValueError("")
    total = math.fsum(map(float, wet))
    if total <= 0:
        raise ValueError("")
    return total


def extend_inundated_priority(demand_priority, inventory):
    """Queue all inundated roads; roads without selected contacts have zero scores and follow in fixed ID order."""
    validate_inundated_inventory(inventory)
    roads = inventory.copy()
    roads["road_id"] = roads.road_id.astype(str)
    all_wet = roads.loc[roads.inundated_length_m.gt(0)].copy()
    if not all_wet.flood_scenario.eq(1).all():
        raise ValueError("")
    positive = demand_priority.copy()
    positive["road_id"] = positive.road_id.astype(str)
    if positive.road_id.duplicated().any() or not set(positive.road_id) <= set(all_wet.road_id):
        raise ValueError("")
    if not np.isfinite(positive.associated_risk_burden).all() or not positive.associated_risk_burden.gt(0).all():
        raise ValueError("")
    expected = positive.sort_values(["associated_risk_burden", "best_grid_rank", "road_id"],
                                   ascending=[False, True, True], kind="stable")
    if list(expected.road_id) != list(positive.road_id):
        raise ValueError("")
    positive["demand_linked_road"] = True
    remaining = all_wet.loc[~all_wet.road_id.isin(positive.road_id)].copy()
    for column in ("associated_risk_burden", "rescue_associated_risk_burden",
                   "medical_associated_risk_burden", "shelter_associated_risk_burden"):
        remaining[column] = 0.0
    for column in ("associated_grid_count", "associated_contact_count"):
        remaining[column] = 0
    remaining["best_grid_rank"] = np.nan
    remaining["demand_linked_road"] = False
    remaining = remaining.sort_values("road_id", kind="stable")
    priority = pd.concat([positive.drop(columns="road_rank"), remaining], ignore_index=True)
    priority.insert(0, "road_rank", np.arange(1, len(priority) + 1, dtype=int))
    if set(priority.road_id) != set(all_wet.road_id):
        raise ValueError("")
    return priority


def build_length_plan(priority, inventory, period):
    """Select the first complete-road prefix reaching the target of total inundated length, rather than sampling a percentage of road counts."""
    total = validate_baseline_inventory(inventory)
    total_wet = validate_inundated_inventory(inventory)
    count = len(priority)
    if not np.array_equal(priority.road_rank.to_numpy(), np.arange(1, count + 1)):
        raise ValueError("")
    if priority.road_id.astype(str).duplicated().any():
        raise ValueError("")
    reference = inventory.set_index(inventory.road_id.astype(str))
    if not set(priority.road_id.astype(str)) <= set(reference.index):
        raise ValueError("")
    if not np.array_equal(priority.length_m.to_numpy(float),
                          reference.loc[priority.road_id.astype(str), "length_m"].to_numpy(float)):
        raise ValueError("")
    if not np.array_equal(priority.inundated_length_m.to_numpy(float),
                          reference.loc[priority.road_id.astype(str), "inundated_length_m"].to_numpy(float)):
        raise ValueError("")
    if not set(priority.road_id.astype(str)) == set(inventory.loc[inventory.inundated_length_m.gt(0), "road_id"].astype(str)):
        raise ValueError("")
    wet_values = priority.inundated_length_m.to_numpy(float)
    cumulative = [0.0] + [math.fsum(map(float, wet_values[:m])) for m in range(1, count + 1)]
    records = []
    for index, target_pct in enumerate(TARGET_PCTS):
        target_length = total_wet * target_pct / 100.0
        m = 0 if target_pct == 0 else bisect_left(cumulative, target_length - LENGTH_THRESHOLD_TOLERANCE_M)
        if m > count:
            raise ValueError("")
        prefix = priority.iloc[:m]
        length = math.fsum(map(float, prefix.length_m))
        wet_length = math.fsum(map(float, prefix.inundated_length_m))
        previous_length = cumulative[m - 1] if m else 0.0
        if target_pct == 100 and m != count:
            raise ValueError("")
        if not 0 <= length <= total + 1e-6:
            raise ValueError("")
        if not 0 <= wet_length <= total_wet + 1e-6:
            raise ValueError("")
        records.append({
            "period": int(period), "plan_index": index, "planned_road_count": m,
            "candidate_road_count": count, "planned_road_length_m": length,
            "baseline_network_total_length_m": total,
            "treated_inundated_length_m": wet_length,
            "scenario_total_inundated_length_m": total_wet,
            "target_treated_length_pct": target_pct,
            "target_inundated_length_m": target_length,
            "previous_prefix_inundated_length_m": previous_length,
            "length_overshoot_m": wet_length - target_length,
            "length_overshoot_pct": 100 * (wet_length - target_length) / total_wet,
            "length_threshold_tolerance_m": LENGTH_THRESHOLD_TOLERANCE_M,
            "target_reached": wet_length + LENGTH_THRESHOLD_TOLERANCE_M >= target_length,
            "treated_length_fraction": wet_length / total_wet,
            "treated_length_pct": 100 * wet_length / total_wet,
            "planned_road_set_hash": road_hash(prefix.road_id),
            "planned_roads_file": f'plans/T{period}_step{index:02d}_target{target_pct:03d}pct_selected_roads.csv',
        })
    return pd.DataFrame(records)
