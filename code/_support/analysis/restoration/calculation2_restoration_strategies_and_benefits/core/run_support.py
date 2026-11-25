"""Shared road-restoration outputs, unrounded demand ranking and zero-restoration checks."""
import hashlib
import json
import config
import numpy as np

def road_hash(road_ids):
    return hashlib.sha256(json.dumps(sorted(map(str, road_ids)), ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def make_priority(evaluation):
    evaluation = evaluation.copy()
    valid = evaluation.efarb_valid.to_numpy(bool)
    eligible = valid & evaluation.effective_population.gt(0) & evaluation.initial_EFARB.gt(0)
    if not np.isfinite(evaluation.loc[eligible, "population_exposed"]).all():
        raise ValueError("")
    ranked = evaluation.loc[eligible].sort_values(
        ["initial_EFARB", "effective_population", "population_exposed", "grid_id"],
        ascending=[False, False, False, True], kind="stable").copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    mapping = ranked.set_index("grid_id")["rank"]
    evaluation["ranking_eligible"] = eligible
    evaluation["rank"] = evaluation.grid_id.map(mapping).fillna(0).astype(int)
    return evaluation, ranked


def make_result(evaluation, baseline, result, selected):
    rows = evaluation.loc[evaluation.efarb_valid].copy().reset_index(drop=True)
    if not np.array_equal(rows.grid_index, baseline["resident_indices"]):
        raise AssertionError("")
    rows = rows[["grid_id", "grid_index", "effective_population", "population_status", "rank"]]
    rows["selected"] = rows.grid_id.isin(selected)
    for k, name in enumerate(config.DIMENSION_KEYS):
        rows[f"EFAR_{name}"] = result["risk"][:, k]
        rows[f"reachable_count_{name}"] = result["reachable_count"][:, k]
        rows[f"nearest_distance_m_{name}"] = result["nearest_distance"][:, k]
        rows[f"base_count_{name}"] = baseline["base_count"][:, k]
        rows[f"base_inverse_{name}"] = baseline["base_inverse"][:, k]
        rows[f"scene_inverse_{name}"] = result["scene_inverse"][:, k]
    rows["EFAR_mean"] = result["risk"].mean(axis=1)
    rows["EFARB"] = rows.effective_population * rows.EFAR_mean
    if not np.isfinite(rows[["EFARB", "EFAR_mean"]].to_numpy()).all():
        raise AssertionError("")
    return rows


def check_no_governance(evaluation, result):
    expected = evaluation.loc[evaluation.efarb_valid].set_index("grid_id").loc[result.grid_id]
    errors = {}
    for name in config.DIMENSION_KEYS:
        errors[name] = float(np.max(np.abs(result[f"EFAR_{name}"].to_numpy() -
                                           expected[f"initial_EFAR_{name}"].to_numpy())))
        if errors[name] > config.RISK_ATOL:
            raise AssertionError("")
        if not np.array_equal(result[f"reachable_count_{name}"].to_numpy(),
                              expected[f"initial_reachable_count_{name}"].to_numpy()):
            raise AssertionError("")
    errors["EFARB"] = float(np.max(np.abs(result.EFARB.to_numpy() - expected.initial_EFARB.to_numpy())))
    if errors["EFARB"] > config.BURDEN_ATOL:
        raise AssertionError("")
    return errors
