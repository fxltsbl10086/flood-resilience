"""Independently verify and summarize restoration benefits after all network calculations finish.

Usage: python -X utf8 -B summarize.py --run-dir <batch-directory>
The main horizontal axis is cumulative confirmed-inundated restored-road length divided by all confirmed-inundated road length in the same city/scenario.
Each return period has its own fixed ranking; this script does not build graphs, route or reselect roads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys as _path_sys
_RESULT4_ROOT = Path(__file__).resolve().parents[1]
if str(_RESULT4_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_RESULT4_ROOT))
import time

import numpy as np
import pandas as pd
from _support.analysis.restoration.paths import read_json, resolve_path


DIMENSIONS = ("rescue", "medical", "shelter")
DIMENSION_NAMES = {"rescue": "应急救援", "medical": "医疗资源", "shelter": "避险转移"}
STRATEGY = "scenario_priority_inundated_length_targets_v1"
TARGET_PCTS = [0, 1, 2, *range(5, 101, 5)]
RISK_TOL = 1e-10
VALUE_TOL = 1e-6
LENGTH_TOL = 1e-6


def require(condition, message):
    if not bool(condition):
        raise ValueError("")


def close(actual, expected, label, atol=RISK_TOL, rtol=0.0):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape, "")
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), "")


def numeric(values, label, *, finite=True):
    result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    require(not np.isnan(result).any(), "")
    require(not finite or np.isfinite(result).all(), "")
    return result


def integers(values, label):
    result = numeric(values, label)
    require((result >= 0).all() and (result == np.floor(result)).all(), "")
    return result.astype(np.int64)


def boolean(values, label):
    text = values.astype(str).str.strip().str.lower()
    require(text.isin(["true", "false", "0", "1"]).all(), "")
    return text.isin(["true", "1"]).to_numpy(bool)


def columns(frame, required, label):
    missing = set(required) - set(frame.columns)
    require(not missing, "")


def table(path, id_column=None):
    frame = pd.read_csv(resolve_path(path), encoding="utf-8-sig", float_precision="round_trip",
                        dtype={"road_id": str, "grid_id": str})
    if id_column:
        columns(frame, [id_column], str(path))
        require(frame[id_column].notna().all() and frame[id_column].str.strip().ne("").all()
                and frame[id_column].is_unique, "")
    return frame


def signature(path):
    path = resolve_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def road_hash(ids):
    payload = json.dumps(sorted(map(str, ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def within_run(run, relative):
    relative = Path(str(relative))
    require(not relative.is_absolute(), "")
    path = (run / relative).resolve()
    require(path.is_relative_to(run) and path.is_file(), "")
    return path


def total(values):
    return float(math.fsum(np.asarray(values, dtype=float).ravel()))


def fraction(numerator, denominator, *, tol=RISK_TOL):
    if denominator == 0:
        require(abs(numerator) <= VALUE_TOL, "")
        return np.nan
    require(denominator > 0, "")
    result = float(numerator / denominator)
    require(-tol <= result <= 1 + tol, "")
    # Remove only floating-point noise; do not hide out-of-tolerance negative benefits or benefits over 100%.
    return min(1.0, max(0.0, result))


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Audit:
    def __init__(self):
        self.files = {}
        # Path resolver used by current statistics; do not rewrite historical source records.
        self.track(_RESULT4_ROOT / "paths.py")

    def track(self, path, expected=None):
        path = resolve_path(path)
        current = signature(path)
        if expected:
            require(current["sha256"] == expected["sha256"], "")
            if "bytes" in expected:
                require(current["bytes"] == expected["bytes"], "")
        previous = self.files.setdefault(str(path), current)
        require(previous == current, "")
        return current

    def recheck(self):
        for saved in self.files.values():
            require(signature(saved["path"]) == saved, "")


def validate_lengths(frame, label):
    fields = ["length_m", "inside_evaluation_area_length_m", "outside_evaluation_area_length_m"]
    columns(frame, fields, label)
    for field in fields:
        require((numeric(frame[field], f"{label}/{field}") >= 0).all(), "")
    close(frame.length_m, frame.inside_evaluation_area_length_m + frame.outside_evaluation_area_length_m,
          "", atol=LENGTH_TOL)


def load_roads(run, period, baseline, audit):
    path = run / "inputs" / f"T{period}_road_inventory.csv"
    audit.track(path)
    roads = table(path, "road_id").set_index("road_id")
    require(set(roads.index) == set(baseline.index), "")
    roads = roads.loc[baseline.index]
    columns(roads, ["flood_scenario", "inundated_length_m", "within_study_length_m", "missing_coverage"], str(path))
    validate_lengths(roads, f"T{period} 道路清单")
    for field in ["length_m", "inside_evaluation_area_length_m", "outside_evaluation_area_length_m"]:
        close(roads[field], baseline[field], "", atol=LENGTH_TOL)
    if "road_position" in roads:
        close(roads.road_position, baseline.road_position, "", atol=0)
    wet = numeric(roads.inundated_length_m, f"T{period} 淹没长度")
    require((wet >= 0).all() and (wet <= roads.length_m.to_numpy() + LENGTH_TOL).all(),
            "")
    within = numeric(roads.within_study_length_m, "研究区道路长度")
    require((within >= 0).all() and (within <= roads.length_m.to_numpy() + LENGTH_TOL).all(),
            "")
    require((wet <= within + LENGTH_TOL).all(), "")
    boolean(roads.missing_coverage, f"T{period} missing_coverage")
    if "flood_vertical_uncertain" in roads:
        boolean(roads.flood_vertical_uncertain, f"T{period} flood_vertical_uncertain")
    closed = boolean(roads.flood_scenario, f"T{period} flood_scenario")
    return roads, set(roads.index[closed])


def load_evaluation(run, period, audit):
    path = run / "inputs" / f"T{period}_evaluation_grids.csv"
    audit.track(path)
    all_grids = table(path, "grid_id")
    fields = ["efarb_valid", "effective_population", "population_exposed", "initial_EFARB",
              *[f"initial_EFAR_{d}" for d in DIMENSIONS]]
    columns(all_grids, fields, str(path))
    valid = boolean(all_grids.efarb_valid, f"T{period} efarb_valid")
    evaluation = all_grids.loc[valid].set_index("grid_id").sort_index()
    require(len(evaluation) > 0, "")
    population = numeric(evaluation.effective_population, "有效人口")
    exposure = numeric(evaluation.population_exposed, "淹没人口")
    require((population >= 0).all() and total(population) > 0, "")
    require((exposure >= -VALUE_TOL).all() and (exposure <= population + VALUE_TOL).all(),
            "")
    exposure_clean = np.maximum(0.0, np.minimum(population, exposure))
    exposure_qa = {"roundoff_adjusted_grid_count": int(np.count_nonzero(exposure != exposure_clean)),
                   "maximum_roundoff_adjustment_persons": float(np.max(np.abs(exposure_clean - exposure))),
                   "tolerance_persons": VALUE_TOL}
    risk = evaluation[[f"initial_EFAR_{d}" for d in DIMENSIONS]].to_numpy(float)
    require(np.isfinite(risk).all() and ((risk >= -RISK_TOL) & (risk <= 1 + RISK_TOL)).all(),
            "")
    close(evaluation.initial_EFARB, population * risk.mean(axis=1),
          "", atol=VALUE_TOL)
    return evaluation, population, exposure_clean, exposure_qa, len(all_grids)


def rebuild_demand_scores(run, period, evaluation, audit):
    """Independently reconstruct demand-associated road scores from original population, selected category contacts and their path-road mappings."""
    paths = [run / "plans" / f"T{period}_{name}.csv" for name in
             ["maximum_loss_facility_links", "facility_link_road_lookup", "road_link_burden_details"]]
    for path in paths:
        audit.track(path)
    contacts, links, reported = [table(path) for path in paths]
    columns(contacts, ["grid_id", "dimension", "selected_EFAR_contribution", "selected_loss_share",
                       "selected_inverse_loss", "total_inverse_loss", "base_inverse", "initial_EFAR"], "所选设施联系")
    require(not contacts.duplicated(["grid_id", "dimension"]).any(), "")
    columns(links, ["grid_id", "dimension", "road_id", "road_position"], "facility_link_road_lookup")
    require(links.dimension.isin(DIMENSIONS).all(), "")
    require(not links.duplicated(["grid_id", "dimension", "road_id"]).any(), "")
    eligible = evaluation.loc[(evaluation.effective_population > 0) & (evaluation.initial_EFARB > 0)].copy()
    require("rank" in eligible, "")
    ranked = eligible.reset_index().sort_values(["initial_EFARB", "effective_population", "population_exposed", "grid_id"],
                                              ascending=[False, False, False, True], kind="stable")
    close(ranked["rank"], np.arange(1, len(ranked) + 1), "", atol=0)
    links = links.loc[links.grid_id.isin(eligible.index)].copy()
    lookup = contacts.set_index(["grid_id", "dimension"])
    pairs = pd.MultiIndex.from_frame(links[["grid_id", "dimension"]])
    require(set(pairs) <= set(lookup.index), "")
    source = lookup.loc[pairs]
    contribution = numeric(source.selected_EFAR_contribution, "所选联系EFAR贡献")
    selected_loss = numeric(source.selected_inverse_loss, "所选联系逆距离损失")
    total_loss = numeric(source.total_inverse_loss, "类别逆距离总损失")
    base_inverse = numeric(source.base_inverse, "类别基准逆距离")
    loss_share = numeric(source.selected_loss_share, "所选联系损失比例")
    require((base_inverse > 0).all() and (total_loss > 0).all() and (selected_loss > 0).all(),
            "")
    close(contribution, selected_loss / base_inverse, "", atol=1e-12, rtol=1e-12)
    close(loss_share, selected_loss / total_loss, "", atol=1e-12, rtol=1e-12)
    close(source.initial_EFAR, total_loss / base_inverse, "", atol=1e-12, rtol=1e-12)
    rebuilt = links[["grid_id", "dimension", "road_id", "road_position"]].copy()
    rebuilt["effective_population"] = eligible.loc[links.grid_id, "effective_population"].to_numpy()
    rebuilt["initial_EFARB"] = eligible.loc[links.grid_id, "initial_EFARB"].to_numpy()
    rebuilt["grid_rank"] = eligible.loc[links.grid_id, "rank"].to_numpy()
    rebuilt["selected_EFAR_contribution"] = contribution
    rebuilt["selected_loss_share"] = loss_share
    rebuilt["contact_risk_burden"] = rebuilt.effective_population / 3 * contribution
    for k, dimension in enumerate(DIMENSIONS):
        mask = links.dimension.eq(dimension).to_numpy()
        close(source.loc[mask, "initial_EFAR"], eligible.loc[links.loc[mask, "grid_id"], f"initial_EFAR_{dimension}"],
              "", atol=VALUE_TOL)
    columns(reported, rebuilt.columns, "road_link_burden_details")
    require(not reported.duplicated(["grid_id", "dimension", "road_id"]).any(), "")
    sort_keys = ["grid_id", "dimension", "road_id"]
    rebuilt, reported = [frame.sort_values(sort_keys, kind="stable").reset_index(drop=True)
                         for frame in [rebuilt, reported]]
    require(len(rebuilt) == len(reported), "")
    for field in sort_keys:
        require(list(rebuilt[field]) == list(reported[field]), "")
    for field in set(rebuilt.columns) - set(sort_keys):
        close(rebuilt[field], reported[field], "", atol=1e-10, rtol=1e-12)
    scores = []
    for road_id, group in rebuilt.groupby("road_id", sort=True):
        record = {"road_id": road_id, "associated_risk_burden": total(group.contact_risk_burden),
                  "best_grid_rank": int(group.grid_rank.min()), "associated_grid_count": group.grid_id.nunique(),
                  "associated_contact_count": len(group)}
        for dimension in DIMENSIONS:
            record[f"{dimension}_associated_risk_burden"] = total(group.loc[group.dimension.eq(dimension), "contact_risk_burden"])
        scores.append(record)
    fields = ["road_id", "associated_risk_burden", "best_grid_rank", "associated_grid_count", "associated_contact_count",
              *[f"{dimension}_associated_risk_burden" for dimension in DIMENSIONS]]
    return pd.DataFrame(scores, columns=fields).set_index("road_id")


def load_priority(run, period, baseline, roads, closed, evaluation, audit):
    path = run / "plans" / f"T{period}_road_priority_order.csv"
    audit.track(path)
    priority = table(path, "road_id")
    rank_field = "road_rank" if "road_rank" in priority else "rank"
    columns(priority, [rank_field, "length_m", "inundated_length_m", "demand_linked_road", "associated_risk_burden",
                       "best_grid_rank", "associated_grid_count", "associated_contact_count",
                       *[f"{d}_associated_risk_burden" for d in DIMENSIONS]], str(path))
    rank = integers(priority[rank_field], f"T{period} 道路序号")
    require(np.array_equal(np.sort(rank), np.arange(1, len(priority) + 1)), "")
    priority = priority.sort_values(rank_field, kind="stable").reset_index(drop=True)
    require(set(priority.road_id) <= closed, "")
    all_wet_ids = set(roads.index[roads.inundated_length_m > 0])
    require(set(priority.road_id) == all_wet_ids, "")
    require(set(priority.road_id) <= set(baseline.index), "")
    close(priority.length_m, baseline.loc[priority.road_id, "length_m"], "", atol=LENGTH_TOL)
    close(priority.inundated_length_m, roads.loc[priority.road_id, "inundated_length_m"], "", atol=LENGTH_TOL)
    require((numeric(priority.inundated_length_m, "治理道路正受淹长度") > 0).all(), "")
    linked = boolean(priority.demand_linked_road, "demand_linked_road")
    expected_scores = rebuild_demand_scores(run, period, evaluation, audit)
    require(set(priority.loc[linked, "road_id"]) == set(expected_scores.index), "")
    require((numeric(priority.loc[linked, "associated_risk_burden"], "需求关联道路负担") > 0).all(),
            "")
    for field in expected_scores.columns:
        close(priority.loc[linked, field], expected_scores.loc[priority.loc[linked, "road_id"], field],
              "", atol=1e-10, rtol=1e-12)
    zero_fields = ["associated_risk_burden", "associated_grid_count", "associated_contact_count",
                   *[f"{d}_associated_risk_burden" for d in DIMENSIONS]]
    require((priority.loc[~linked, zero_fields].to_numpy(float) == 0).all(), "")
    require(priority.loc[~linked, "best_grid_rank"].isna().all(), "")
    expected = priority.sort_values(["associated_risk_burden", "best_grid_rank", "road_id"],
                                    ascending=[False, True, True], kind="stable", na_position="last")
    require(list(expected.road_id) == list(priority.road_id), "")
    return priority


def validate_road_list(run, relative, expected_ids, inventory, audit, expected_sha=None):
    path = within_run(run, relative)
    sig = audit.track(path)
    if expected_sha is not None:
        require(sig["sha256"] == str(expected_sha), "")
    actual = table(path, "road_id").set_index("road_id")
    require(set(actual.index) == expected_ids, "")
    columns(actual, ["length_m", "inundated_length_m"], str(path))
    for field in ["length_m", "inundated_length_m", "inside_evaluation_area_length_m",
                  "outside_evaluation_area_length_m", "road_position"]:
        if field in actual and field in inventory:
            close(actual[field], inventory.loc[actual.index, field], "",
                  atol=0 if field == "road_position" else LENGTH_TOL)
    for field in ["missing_coverage", "flood_vertical_uncertain"]:
        if field in inventory:
            columns(actual, [field], str(path))
            require(np.array_equal(boolean(actual[field], f"道路文件 {field}"),
                                   boolean(inventory.loc[actual.index, field], f"道路主表 {field}")),
                    "")
    return sig


def result_arrays(frame, ids, population, label):
    fields = ["population_status", "effective_population", "EFARB"]
    fields += [f"{prefix}_{d}" for d in DIMENSIONS for prefix in
               ["EFAR", "reachable_count", "nearest_distance_m", "base_count", "base_inverse"]]
    columns(frame, fields, label)
    require(set(frame.index) == set(ids), "")
    frame = frame.loc[ids]
    require(frame.population_status.notna().all(), "")
    close(frame.effective_population, population, "", atol=0)
    get = lambda prefix: frame[[f"{prefix}_{d}" for d in DIMENSIONS]].to_numpy(float)
    risk, count, nearest = get("EFAR"), get("reachable_count"), get("nearest_distance_m")
    base_count, base_inverse = get("base_count"), get("base_inverse")
    burden = numeric(frame.EFARB, f"{label}/EFARB")
    require(np.isfinite(risk).all() and ((risk >= -RISK_TOL) & (risk <= 1 + RISK_TOL)).all(),
            "")
    require(np.isfinite(count).all() and (count >= 0).all() and (count == np.floor(count)).all(),
            "")
    require(np.isfinite(base_count).all() and (base_count > 0).all()
            and (base_count == np.floor(base_count)).all() and (count <= base_count).all(),
            "")
    require(np.isfinite(base_inverse).all() and (base_inverse > 0).all(), "")
    require(not np.isnan(nearest).any() and (nearest > 0).all()
            and np.array_equal(np.isposinf(nearest), count == 0), "")
    inverse_fields = [f"scene_inverse_{d}" for d in DIMENSIONS]
    checked_efar_formula = all(field in frame for field in inverse_fields)
    if checked_efar_formula:
        scene = frame[inverse_fields].to_numpy(float)
        require(np.isfinite(scene).all() and (scene >= 0).all() and (scene <= base_inverse + 1e-12).all(),
                "")
        close(risk, 1 - scene / base_inverse, "")
    close(burden, population * risk.mean(axis=1), "", atol=VALUE_TOL)
    return {"risk": risk, "count": count, "nearest": nearest, "base_count": base_count,
            "base_inverse": base_inverse, "burden": burden,
            "population_status": frame.population_status.astype(str).to_numpy(),
            "efar_formula_checked": checked_efar_formula,
            "formula_max_absolute_error": float(np.max(np.abs(burden - population * risk.mean(axis=1))))}


def exposure_fields(population, exposure, risk_before, risk_after):
    fields = {}
    means_before, means_after = risk_before.mean(axis=1), risk_after.mean(axis=1)
    for label, weights in [("exposed", exposure), ("unexposed", population - exposure)]:
        before, after = total(weights * means_before), total(weights * means_after)
        fields.update({f"{label}_population": total(weights), f"{label}_initial_EFARB": before,
                       f"{label}_EFARB": after, f"{label}_EFARB_reduction": before - after,
                       f"{label}_EFARB_reduction_rate": fraction(before - after, before)})
    return fields


def validate_target_plans(plans, priority, denominator, period):
    """Resum original segment lengths and independently identify the first complete-road prefix reaching each nominal target."""
    targets = numeric(plans.target_treated_length_pct, "标称治理受淹长度百分数")
    require(list(targets) == TARGET_PCTS, "")
    tolerances = numeric(plans.length_threshold_tolerance_m, "首达长度容差")
    require((tolerances == LENGTH_TOL).all(), "")
    require(boolean(plans.target_reached, "target_reached").all(), "")
    lengths = priority.inundated_length_m.to_numpy(float)
    cumulative = np.array([total(lengths[:m]) for m in range(len(lengths) + 1)])
    close(cumulative[-1], denominator, "", atol=LENGTH_TOL, rtol=1e-12)
    expected_counts = []
    for _, row in plans.iterrows():
        percentage = float(row.target_treated_length_pct)
        threshold = denominator * percentage / 100
        close(row.target_inundated_length_m, threshold, "", atol=LENGTH_TOL, rtol=1e-12)
        if percentage == 0:
            m = 0
        else:
            passing = np.flatnonzero(cumulative + LENGTH_TOL >= threshold)
            require(len(passing) > 0, "")
            m = int(passing[0])
        if percentage == 100:
            require(m == len(priority), "")
        expected_counts.append(m)
        actual = cumulative[m]
        previous = cumulative[m - 1] if m else 0.0
        close(row.previous_prefix_inundated_length_m, previous, "", atol=LENGTH_TOL)
        close(row.treated_inundated_length_m, actual, "", atol=LENGTH_TOL)
        close(row.length_overshoot_m, actual - threshold, "", atol=LENGTH_TOL)
        close(row.length_overshoot_pct, 100 * (actual - threshold) / denominator,
              "", atol=1e-10)
        require(actual + LENGTH_TOL >= threshold, "")
        if percentage > 0 and m > 0:
            require(previous + LENGTH_TOL < threshold, "")
    counts = integers(plans.planned_road_count, "计划道路数")
    require(list(counts) == expected_counts, "")
    return counts


def target_rows(summary):
    rows = []
    for period, group in summary.groupby("period", sort=True):
        group = group.sort_values("plan_index", kind="stable")
        for dimension in ["composite", *DIMENSIONS]:
            rate_field = ("EFARB_reduction_rate" if dimension == "composite"
                          else f"{dimension}_risk_burden_reduction_rate")
            for target in [0.5, 0.8]:
                reached = group.loc[group[rate_field] >= target]
                row = {"period": int(period), "dimension": dimension,
                       "维度": "综合" if dimension == "composite" else DIMENSION_NAMES[dimension],
                       "target_reduction_rate": target, "target_reached": bool(len(reached)),
                       "selection_method": 'First sampled stage reaching the threshold; no interpolation or continuous minimum-cost estimate',
                       "sampled_maximum_reduction_rate": group[rate_field].max(),
                       "baseline_network_total_length_m": float(group.iloc[0].baseline_network_total_length_m),
                       "scenario_total_inundated_length_m": float(group.iloc[0].scenario_total_inundated_length_m)}
                keys = ["plan_index", "target_treated_length_pct", "target_inundated_length_m", "length_overshoot_m",
                        "length_overshoot_pct", "planned_road_count", "planned_road_length_m", "treated_inundated_length_m", "treated_length_fraction",
                        "treated_length_pct", "restored_road_count", "restored_length_m"]
                row.update({k: reached.iloc[0][k] if len(reached) else np.nan for k in keys})
                row["observed_reduction_rate"] = reached.iloc[0][rate_field] if len(reached) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def load_checkpoint_validation(run, config, completed, signed_outputs, audit):
    """Read signed continuation evidence after table-level checks of the rebuilt model; do not count continuation as a cache hit."""
    count = int(completed.get("checkpoint_reused_count", 0))
    require(count >= 0, "")
    record = config.get("resume_checkpoint_manifest_signature")
    if not record:
        require(count == 0, "")
        return {}, None
    path = Path(record["path"]).resolve()
    require(path.is_relative_to(run) and str(path) in signed_outputs,
            "")
    audit.track(path, record)
    proof = read_json(path)
    require(proof.get("status") == "PASS", "")
    for source_signature in proof.get("source_signatures", []):
        audit.track(source_signature["path"], source_signature)
    source_run = Path(proof["source_run"]).resolve()
    require(source_run != run, "")
    source_config = proof["source_config_signature"]
    require(Path(source_config["path"]).resolve() == source_run / "run_config.json", "")
    audit.track(source_config["path"], source_config)
    source_manifests = {}
    for item in proof["source_scene_manifest_signatures"]:
        origin = Path(item["path"]).resolve()
        require(origin.is_relative_to(source_run) and origin.name == "scenario_inventory.csv",
                "")
        require(str(origin) not in source_manifests, "")
        audit.track(origin, item)
        source_manifests[str(origin)] = (item, table(origin))
    rows = {}
    for item in proof["rows"]:
        key = (int(item["period"]), int(item["plan_index"]))
        require(key not in rows and item.get("model_inputs_equal") is True, "")
        require(key[0] in list(map(int, proof["periods_verified"])), "")
        origin_sig = item["source_scene_manifest_signature"]
        origin = str(Path(origin_sig["path"]).resolve())
        require(origin in source_manifests and origin_sig["sha256"] == source_manifests[origin][0]["sha256"],
                "")
        source = source_manifests[origin][1]
        selected = source.loc[source.period.eq(key[0]) & source.plan_index.eq(key[1])]
        require(len(selected) == 1, "")
        source_row = selected.iloc[0]
        require(str(source_row.model_key) == str(item["source_model_key"]), "")
        require(not boolean(pd.Series([source_row.cache_used]), "source cache_used")[0]
                and pd.isna(source_row.reused_from_plan_index), "")
        for file_field, hash_field in [("result_file", "result_sha256"),
                                       ("planned_roads_file", "planned_roads_sha256"),
                                       ("restored_roads_file", "restored_roads_sha256")]:
            require(str(source_row[hash_field]) == item[hash_field], "")
            source_path = within_run(source_run, source_row[file_field])
            require(audit.track(source_path)["sha256"] == item[hash_field], "")
        rows[key] = item
    require(len(rows) == count, "")
    return rows, {"validation_signature": signature(path), "source_run": str(source_run), "row_count": len(rows)}


def summarize(run_dir, output_dir=None):
    started = time.perf_counter()
    run = resolve_path(run_dir)
    audit = Audit()
    output = Path(output_dir).resolve() if output_dir is not None else run / "statistics"
    require(output != run and output.is_relative_to(run), "")
    output_names = ["restoration_effect_summary.csv", "three_facility_improvements.csv", "restoration_extent_for_target_reduction.csv", "statistics_validation.json",
                    "metric_definitions.json"]
    require(not any((output / name).exists() for name in output_names), "")
    completed_path, config_path = run / "completion.json", run / "run_config.json"
    require(completed_path.is_file(), "")
    audit.track(completed_path)
    audit.track(config_path)
    completed, config = read_json(completed_path), read_json(config_path)
    require(completed.get("status") == "PASS", "")
    require(config.get("schema_version") == 3 and config.get("strategy") == STRATEGY,
            "")
    require(completed.get("strategy", STRATEGY) == STRATEGY, "")
    periods = list(map(int, config["evaluation_periods"]))
    require(len(periods) == len(set(periods)) and periods, "")
    require(config.get("target_treated_length_pcts") == TARGET_PCTS, "")
    require(config.get("length_threshold_tolerance_m") == LENGTH_TOL, "")
    require(sorted(map(int, completed["periods"])) == sorted(periods), "")
    require(config.get("evaluation_scope") in {"district", "study_area"}, "")
    scope = config["evaluation_scope"]
    city_name = str(config["city_name"])
    scope_description = (f"{city_name}原研究区全部有效居民格网" if scope == "study_area" else
                         f"{city_name}{config.get('district', '')}内原研究区有效居民格网")
    require(isinstance(completed.get("output_signatures"), list), "")
    signed_outputs = set()
    for record in completed["output_signatures"]:
        path = Path(record["path"]).resolve()
        require(path.is_relative_to(run), "")
        require(str(path) not in signed_outputs, "")
        signed_outputs.add(str(path))
        audit.track(path, record)
    require(str(config_path) in signed_outputs, "")
    for record in config["source_signatures"]:
        audit.track(record["path"], record)
    for record in config["scripts"].values():
        audit.track(record["path"], record)
    checkpoint_rows, checkpoint_proof = load_checkpoint_validation(run, config, completed, signed_outputs, audit)
    checkpoint_seen = set()
    own_signature = audit.track(Path(__file__))

    baseline_path = run / "inputs" / "fixed_baseline_road_inventory.csv"
    plan_path = run / "plans" / "restoration_targets.csv"
    audit.track(baseline_path)
    audit.track(plan_path)
    baseline = table(baseline_path, "road_id").set_index("road_id")
    columns(baseline, ["road_position"], "fixed_baseline_road_inventory")
    positions = integers(baseline.road_position, "固定基准 road_position")
    require(len(set(positions)) == len(positions), "")
    validate_lengths(baseline, "fixed_baseline_road_inventory")
    denominator = total(baseline.length_m)
    require(denominator > 0, "")
    close(denominator, config["baseline_network_total_length_m"], "",
          atol=LENGTH_TOL, rtol=1e-12)
    require(len(baseline) == int(config["baseline_road_count"]), "")
    all_plans = table(plan_path)
    plan_columns = ["period", "plan_index", "planned_road_count", "candidate_road_count", "planned_road_length_m",
                    "baseline_network_total_length_m", "treated_inundated_length_m", "scenario_total_inundated_length_m",
                    "treated_length_fraction", "treated_length_pct",
                    "target_treated_length_pct", "target_inundated_length_m", "previous_prefix_inundated_length_m",
                    "length_overshoot_m", "length_overshoot_pct", "length_threshold_tolerance_m", "target_reached",
                    "planned_road_set_hash", "planned_roads_file"]
    columns(all_plans, plan_columns, "治理档位")
    require(not all_plans.duplicated(["period", "plan_index"]).any(), "")
    require(set(integers(all_plans.period, "档位 period")) == set(periods), "")

    summaries, dimensions, checks, period_checks = [], [], [], []
    common_population = common_ids = common_base_count = common_base_inverse = common_population_status = None
    for period in periods:
        roads, closed = load_roads(run, period, baseline, audit)
        inundated_denominator = total(roads.inundated_length_m)
        require(inundated_denominator > 0,
                "")
        close(inundated_denominator, config["scenario_total_inundated_length_m_by_period"][str(period)],
              "", atol=LENGTH_TOL, rtol=1e-12)
        missing_depth = boolean(roads.missing_coverage, f"T{period} missing_coverage")
        vertical_uncertain = (boolean(roads.flood_vertical_uncertain, f"T{period} flood_vertical_uncertain")
                              if "flood_vertical_uncertain" in roads else None)
        evaluation, population, exposure, exposure_qa, all_grid_count = load_evaluation(run, period, audit)
        priority = load_priority(run, period, baseline, roads, closed, evaluation, audit)
        ids = evaluation.index
        if common_ids is None:
            common_ids, common_population = ids.copy(), population.copy()
        else:
            require(list(ids) == list(common_ids), "")
            close(population, common_population, "", atol=0)
        zero_qa_path = run / f"T{period}" / 'zero_restoration_validation.json'
        audit.track(zero_qa_path)
        require(read_json(zero_qa_path).get("status") == "PASS", "")
        manifest_path = run / f"T{period}" / "scenario_inventory.csv"
        audit.track(manifest_path)
        manifest = table(manifest_path)
        columns(manifest, [*plan_columns, "restored_road_count", "restored_length_m", "actual_restored_road_hash",
                           "restored_roads_file", "result_file", "cache_used", "reused_from_plan_index",
                           "result_sha256", "restored_roads_sha256"], str(manifest_path))
        plans = all_plans.loc[all_plans.period.eq(period)].sort_values("plan_index", kind="stable")
        manifest = manifest.sort_values("plan_index", kind="stable")
        require(len(manifest) == len(plans) and len(plans) > 0, "")
        expected_plan_indices = integers(plans.plan_index, "方案 plan_index")
        require(np.array_equal(expected_plan_indices, np.arange(len(plans))), "")
        for field in plan_columns:
            if field in ["planned_road_set_hash", "planned_roads_file"]:
                require(list(manifest[field]) == list(plans[field]), "")
            elif field == "target_reached":
                require(np.array_equal(boolean(manifest[field], field), boolean(plans[field], field)),
                        "")
            else:
                close(manifest[field], plans[field], "", atol=0)
        counts = validate_target_plans(plans, priority, inundated_denominator, period)
        require((integers(plans.candidate_road_count, "候选道路数") == len(priority)).all(), "")
        cache_flags = boolean(manifest.cache_used, "cache_used")
        checkpoint_flags = (boolean(manifest.checkpoint_reused, "checkpoint_reused")
                            if "checkpoint_reused" in manifest else np.zeros(len(manifest), dtype=bool))
        require(not (checkpoint_flags & cache_flags).any(), "")
        if config.get("use_cache") is False:
            require(not cache_flags.any(), "")
        blocked_table = roads.loc[sorted(closed)]
        blocked_length = total(blocked_table.length_m)
        blocked_inundated = total(blocked_table.inundated_length_m)
        baseline_arrays = previous_arrays = None
        previous_ids = set()
        restored_seen = {}
        for row_number, (_, row) in enumerate(manifest.iterrows()):
            plan_index, n = int(row.plan_index), int(row.planned_road_count)
            label = f"T{period}/plan={plan_index}"
            checkpoint_key = (period, plan_index)
            checkpoint_reused = bool(checkpoint_flags[row_number])
            if checkpoint_reused:
                require(checkpoint_key in checkpoint_rows, "")
                proof_row = checkpoint_rows[checkpoint_key]
                require(str(row.checkpoint_validation) == "PASS"
                        and Path(row.source_run).resolve() == Path(checkpoint_proof["source_run"]).resolve()
                        and str(row.source_model_key) == str(proof_row["source_model_key"])
                        and str(row.model_key) == str(proof_row["new_model_key"]), "")
                for hash_field in ["result_sha256", "planned_roads_sha256", "restored_roads_sha256"]:
                    require(str(row[hash_field]) == str(proof_row[hash_field]), "")
                checkpoint_seen.add(checkpoint_key)
            else:
                require(checkpoint_key not in checkpoint_rows, "")
            planned_ids = set(priority.iloc[:n].road_id)
            require(previous_ids <= planned_ids, "")
            length = total(baseline.loc[sorted(planned_ids), "length_m"])
            treated_inundated_length = total(roads.loc[sorted(planned_ids), "inundated_length_m"])
            close(length, row.planned_road_length_m, "", atol=LENGTH_TOL, rtol=1e-12)
            close(denominator, row.baseline_network_total_length_m, "", atol=0)
            close(treated_inundated_length, row.treated_inundated_length_m,
                  "", atol=LENGTH_TOL, rtol=1e-12)
            close(inundated_denominator, row.scenario_total_inundated_length_m,
                  "", atol=LENGTH_TOL, rtol=1e-12)
            close(treated_inundated_length / inundated_denominator, row.treated_length_fraction,
                  "")
            close(100 * treated_inundated_length / inundated_denominator, row.treated_length_pct,
                  "")
            fraction(treated_inundated_length, inundated_denominator)
            fraction(length, denominator)
            require(road_hash(planned_ids) == row.planned_road_set_hash, "")
            validate_road_list(run, row.planned_roads_file, planned_ids, roads, audit,
                               row.planned_roads_sha256 if "planned_roads_sha256" in row else None)
            restored_ids = planned_ids & closed
            require(restored_ids == planned_ids, "")
            require(len(restored_ids) == int(row.restored_road_count), "")
            restored = roads.loc[sorted(restored_ids)]
            restored_length = total(restored.length_m)
            restored_inundated_length = total(restored.inundated_length_m)
            close(restored_length, row.restored_length_m, "", atol=LENGTH_TOL)
            close(restored_inundated_length, treated_inundated_length,
                  "", atol=LENGTH_TOL)
            restored_hash = road_hash(restored_ids)
            require(restored_hash == row.actual_restored_road_hash, "")
            road_sig = validate_road_list(run, row.restored_roads_file, restored_ids, roads, audit,
                                         row.restored_roads_sha256)
            result_path = within_run(run, row.result_file)
            result_sig = audit.track(result_path)
            require(result_sig["sha256"] == str(row.result_sha256), "")
            arrays = result_arrays(table(result_path, "grid_id").set_index("grid_id"), ids, population, label)
            risk, count, nearest = (arrays[key] for key in ["risk", "count", "nearest"])
            if row_number == 0:
                require(n == 0 and length == 0 and treated_inundated_length == 0,
                        "")
                close(risk, evaluation[[f"initial_EFAR_{d}" for d in DIMENSIONS]],
                      "", atol=VALUE_TOL)
                close(arrays["burden"], evaluation.initial_EFARB, "",
                      atol=VALUE_TOL)
                for k, d in enumerate(DIMENSIONS):
                    if f"initial_reachable_count_{d}" in evaluation:
                        close(count[:, k], evaluation[f"initial_reachable_count_{d}"],
                              "", atol=0)
                baseline_arrays = arrays
                if common_base_count is None:
                    common_base_count, common_base_inverse = arrays["base_count"].copy(), arrays["base_inverse"].copy()
                    common_population_status = arrays["population_status"].copy()
                else:
                    close(arrays["base_count"], common_base_count, "", atol=0)
                    close(arrays["base_inverse"], common_base_inverse, "", atol=0)
                    require(np.array_equal(arrays["population_status"], common_population_status), "")
            else:
                close(arrays["base_count"], baseline_arrays["base_count"], "", atol=0)
                close(arrays["base_inverse"], baseline_arrays["base_inverse"], "", atol=0)
                require(np.array_equal(arrays["population_status"], baseline_arrays["population_status"]), "")
                require((risk <= previous_arrays["risk"] + RISK_TOL).all(), "")
                require((count >= previous_arrays["count"]).all(), "")
                require((nearest <= previous_arrays["nearest"] + LENGTH_TOL).all(), "")
            reused = row.reused_from_plan_index
            if pd.notna(reused):
                reused_index = int(float(reused))
                require(float(reused) == reused_index and 0 <= reused_index < plan_index,
                        "")
                require(restored_hash in restored_seen and restored_seen[restored_hash][0] == reused_index,
                        "")
                # One prefix can serve multiple nominal targets; require equal content hashes only for explicitly marked reuse.
                require(restored_seen[restored_hash][1] == result_sig["sha256"], "")
            if config.get("use_cache") is False:
                require(pd.isna(reused), "")
            restored_seen.setdefault(restored_hash, (plan_index, result_sig["sha256"]))
            delta = baseline_arrays["risk"] - risk
            improved = (delta > RISK_TOL).any(axis=1)
            reconnected = (baseline_arrays["count"] == 0) & (count > 0)
            before, after = total(baseline_arrays["burden"]), total(arrays["burden"])
            reduction = before - after
            dim_reductions = [total(population * delta[:, k]) for k in range(3)]
            close(reduction, total(dim_reductions) / 3, "", atol=VALUE_TOL)
            before_risk = baseline_arrays["risk"]
            split = exposure_fields(population, exposure, before_risk, risk)
            close(split["exposed_initial_EFARB"] + split["unexposed_initial_EFARB"], before,
                  "", atol=VALUE_TOL)
            close(split["exposed_EFARB_reduction"] + split["unexposed_EFARB_reduction"], reduction,
                  "", atol=VALUE_TOL)
            outside = restored.outside_evaluation_area_length_m.to_numpy() > LENGTH_TOL
            inside = restored.inside_evaluation_area_length_m.to_numpy() > LENGTH_TOL
            common = {"city": config["city"], "city_name": city_name, "district": config.get("district"),
                      "evaluation_scope": config["evaluation_scope"], "period": period, "plan_index": plan_index,
                      "planned_road_count": n, "candidate_road_count": len(priority),
                      "demand_linked_road_count": int(boolean(priority.demand_linked_road, "demand_linked_road").sum()),
                      "planned_road_length_m": length, "baseline_network_total_length_m": denominator,
                      "target_treated_length_pct": float(row.target_treated_length_pct),
                      "target_inundated_length_m": float(row.target_inundated_length_m),
                      "previous_prefix_inundated_length_m": float(row.previous_prefix_inundated_length_m),
                      "length_overshoot_m": float(row.length_overshoot_m),
                      "length_overshoot_pct": float(row.length_overshoot_pct),
                      "length_threshold_tolerance_m": float(row.length_threshold_tolerance_m),
                      "target_reached": True,
                      "treated_inundated_length_m": treated_inundated_length,
                      "scenario_total_inundated_length_m": inundated_denominator,
                      "treated_length_fraction": treated_inundated_length / inundated_denominator,
                      "treated_length_pct": 100 * treated_inundated_length / inundated_denominator,
                      "evaluation_grid_count": len(ids), "evaluation_population": total(population)}
            summary = {**common, "initial_EFARB_total": before, "EFARB_total": after,
                       "EFARB_reduction": reduction, "EFARB_reduction_rate": fraction(reduction, before),
                       "incremental_EFARB_reduction": total(previous_arrays["burden"]) - after if previous_arrays else 0.0,
                       "improved_grid_count": int(improved.sum()), "improved_population": total(population[improved]),
                       "improved_population_share": fraction(total(population[improved]), total(population)),
                       "reconnected_any_service_grid_count": int(reconnected.any(axis=1).sum()),
                       "reconnected_any_service_population": total(population[reconnected.any(axis=1)]),
                       "newly_reachable_facilities_population": total(population[(count > baseline_arrays["count"]).any(axis=1)]),
                       "restored_road_count": len(restored_ids), "restored_length_m": restored_length,
                       "restored_inundated_length_m": restored_inundated_length,
                       "restored_inside_evaluation_area_length_m": total(restored.inside_evaluation_area_length_m),
                       "restored_outside_evaluation_area_length_m": total(restored.outside_evaluation_area_length_m),
                       "restored_with_outside_part_count": int(outside.sum()),
                       "restored_wholly_outside_count": int((outside & ~inside).sum()),
                       "restored_crossing_count": int((outside & inside).sum()),
                       "blocked_road_count": len(closed), "blocked_road_total_length_m": blocked_length,
                       "blocked_road_inundated_length_m": blocked_inundated,
                       "scenario_missing_depth_coverage_road_count": int(missing_depth.sum()),
                       "scenario_known_inundated_length_on_missing_coverage_roads_m": total(roads.loc[missing_depth, "inundated_length_m"]),
                       "scenario_vertical_uncertain_road_count": int(vertical_uncertain.sum()) if vertical_uncertain is not None else np.nan,
                       "restored_blocked_road_fraction": fraction(len(restored_ids), len(closed)),
                       "restored_blocked_length_fraction": fraction(restored_length, blocked_length),
                       "restored_blocked_inundated_length_fraction": fraction(total(restored.inundated_length_m), blocked_inundated),
                       "planned_road_set_hash": row.planned_road_set_hash,
                       "actual_restored_road_hash": restored_hash,
                       "cache_used": bool(cache_flags[row_number]), "reused_from_plan_index": reused,
                       "checkpoint_reused": checkpoint_reused,
                       "routing_recomputed_this_run": not checkpoint_reused and not bool(cache_flags[row_number])
                                                       and pd.isna(reused), **split}
            if "missing_coverage" in restored:
                summary["restored_missing_depth_coverage_count"] = int(boolean(restored.missing_coverage, "missing_coverage").sum())
            for k, d in enumerate(DIMENSIONS):
                dim_before = total(population * before_risk[:, k])
                dim_after = total(population * risk[:, k])
                dim_reduction = dim_before - dim_after
                dim_rate = fraction(dim_reduction, dim_before)
                summary[f"{d}_risk_burden_reduction_rate"] = dim_rate
                dim_row = {**common, "dimension": d, "维度": DIMENSION_NAMES[d],
                           "initial_population_weighted_risk_burden": dim_before,
                           "population_weighted_risk_burden": dim_after,
                           "risk_burden_reduction": dim_reduction, "risk_burden_reduction_rate": dim_rate,
                           "initial_population_weighted_EFAR": dim_before / total(population),
                           "population_weighted_EFAR": dim_after / total(population),
                           "improved_grid_count": int((delta[:, k] > RISK_TOL).sum()),
                           "improved_population": total(population[delta[:, k] > RISK_TOL]),
                           "reconnected_grid_count": int(reconnected[:, k].sum()),
                           "reconnected_population": total(population[reconnected[:, k]])}
                for group_name, weights in [("exposed", exposure), ("unexposed", population - exposure)]:
                    b, a = total(weights * before_risk[:, k]), total(weights * risk[:, k])
                    dim_row.update({f"{group_name}_initial_risk_burden": b, f"{group_name}_risk_burden": a,
                                    f"{group_name}_risk_burden_reduction": b - a,
                                    f"{group_name}_risk_burden_reduction_rate": fraction(b - a, b)})
                dimensions.append(dim_row)
            summaries.append(summary)
            checks.append({"period": period, "plan_index": plan_index, "status": "PASS",
                           "checkpoint_reused": checkpoint_reused,
                           "EFAR_formula_checked": arrays["efar_formula_checked"],
                           "EFARB_formula_max_absolute_error": arrays["formula_max_absolute_error"],
                           "result_signature": result_sig, "restored_roads_signature": road_sig})
            previous_ids, previous_arrays = planned_ids, arrays
        period_checks.append({"period": period, "status": "PASS", "all_input_grid_count": all_grid_count,
                              "evaluation_grid_count": len(ids), "evaluation_population": total(population),
                              "scenario_count": len(manifest), "candidate_road_count": len(priority),
                              "demand_linked_road_count": int(boolean(priority.demand_linked_road, "demand_linked_road").sum()),
                              "target_schedule": TARGET_PCTS,
                              "all_targets_use_first_reaching_complete_prefix": True,
                              "scenario_total_inundated_length_m": inundated_denominator,
                              "inundation_denominator_source": 'Sum inundated_length_m over all unique road_id values in the period road table, without candidate filtering or clipping to evaluation boundaries.',
                              "inundation_spatial_coverage": f'原{city_name}Confirmed-inundated portions of the analysis network within valid study-area flood-raster coverage, not a claim of full administrative-city coverage.',
                              "missing_depth_coverage_road_count": int(missing_depth.sum()),
                              "known_inundated_length_on_missing_coverage_roads_m": total(roads.loc[missing_depth, "inundated_length_m"]),
                              "vertical_uncertain_road_count": int(vertical_uncertain.sum()) if vertical_uncertain is not None else None,
                              "exposure_decomposition_validation": exposure_qa})
    require(len(summaries) == int(completed["scenario_count"]), "")
    require(checkpoint_seen == set(checkpoint_rows), "")
    checkpoint_count = sum(row["checkpoint_reused"] for row in summaries)
    recomputed_count = sum(row["routing_recomputed_this_run"] for row in summaries)
    if config.get("use_cache") is False:
        require(completed.get("routing_cache_hit_count") == 0, "")
        require(int(completed.get("checkpoint_reused_count", 0)) == checkpoint_count
                and int(completed["routing_recomputed_count"]) == recomputed_count
                and checkpoint_count + recomputed_count == len(summaries), "")
    # Completion markers sign this batch's computed outputs only. External raw CSV data are checked using
    # source_signatures before reading and again in audit.recheck(); do not
    # incorrectly require those raw inputs to appear in this batch's output_signatures.
    observed_csvs = {path for path in audit.files
                     if Path(path).suffix.lower() == ".csv" and Path(path).is_relative_to(run)}
    require(observed_csvs <= signed_outputs, "")
    audit.recheck()
    summary_frame = pd.DataFrame(summaries)
    dimension_frame = pd.DataFrame(dimensions)
    targets = target_rows(summary_frame)
    explanations = {
        "strategy": STRATEGY,
        "标称长度档位": 'target_treated_length_pct takes 0, 1, 2, 5, 10, ..., 100 across 23 stages, using all confirmed-inundated length in this city/period as denominator. Each stage selects the shortest complete-road prefix in fixed order first reaching the target, with 1e-6 m tolerance.',
        "标称与实际": "target_inundated_length_m为标称目标长度；treated_inundated_length_m及treated_length_pct为完整物理路段治理后的实际长度和比例，不用标称值替换。length_overshoot_m/pct记录实际超过标称的长度/百分点，允许容差内的浮点微差。同一前缀可对应多个目标，但各标称档位分别保留。",
        "道路全集与排序": 'The intervention universe includes every road with confirmed positive inundated length in this city/period. Demand-associated roads retain positive-burden scores and tie rules; remaining roads receive zero scores and contact counts, empty original grid ranks, and follow in lexicographic road_id order. Added roads do not fabricate demand contacts.',
        "横轴": "treated_length_pct = treated_inundated_length_m / scenario_total_inundated_length_m × 100。分子为累计入选物理道路本期已确认受淹部分长度；分母为同城市同情景道路清单全部唯一road_id已确认受淹部分长度之和，不限评价行政区。0档为0；100%档来自全部正受淹道路实际纳入，不通过绘图拉伸实现。",
        "受淹长度空间口径": f"Current original inundated_length_m covers only the original {city_name} analysis network's confirmed-inundated length within valid study-area flood coverage. The denominator represents all confirmed-inundated analysis-network length in this city/scenario, not a full administrative-city survey.",
        "受淹长度阈值": "Use a 0.30 m depth threshold and road-centerline overlap. Independently verify and aggregate this batch's inundated_length_m inputs without rebuilding GIS or shortest paths during final statistics.",
        "本批次数据生成方式": (
            f'Run configuration use_cache=False with zero network-result cache hits; current city run recalculated {recomputed_count} stages and continued {checkpoint_count} stages after signatures and rebuilt model inputs were verified. Continued stages use checkpoint_reused and count as neither current recalculations nor cache hits. Recalculate every remaining nominal stage independently even when adjacent targets share a prefix. Retain verified population/facility inputs; final statistics do not build graphs or route.'
            if config.get("use_cache") is False else
            'The calculation may reuse verified intermediate data or scenario results as configured. Actual scenario cache use is recorded in cache_used and logs; statistics do not assume reuse at every stage.'),
        "缺测与不确定": 'missing_coverage indicates unconfirmed depth coverage; zero inundated length does not imply the full road is dry. Roads with missing coverage and confirmed inundated portions contribute those known lengths to numerator and denominator. Retain flood_vertical_uncertain without changing restoration rules or inventing additional inundated lengths.',
        "固定与变化": "Generate candidate roads and fixed priorities once per period, using that period's total road-inundated-length denominator, which may differ between periods. Keep valid evaluation cells, population and no-flood accessibility denominators fixed. Full baseline-network length is contextual only.",
        "评价范围": scope_description + ', not just cells directly associated with selected roads. Preserve the original network/facility extent without clipping paths to restoration-demand cells.',
        "负担": "Composite EFARB = effective population times mean EFAR across three categories; single-category burden = effective population times that category's EFAR, without further division by 3.",
        "削减率": "全部削减率以本期零治理负担为分母；零分母时留空，不记为0或100%。rate/fraction为0到1，treated_length_pct为百分数0到100。",
        "受益人口": "Relative to this period's zero-restoration stage, union effective population in cells with an EFAR decrease >1e-10 in at least one category. Do not sum improved population across categories.",
        "恢复可达人口": "Relative to this period's zero stage, a category's reachable-facility count changes from zero to positive; overall population is the union of cells restored in at least one category.",
        "人口分解假设": "同一格网内暴露人口和未暴露人口承担相同的该格 EFAR；以淹没人口/剩余有效人口分别乘风险分配负担。不是个体级可达损失观测，不表示已分别建模两组路径。",
        "人口分解边界": 'Strictly verify 0<=inundated population<=effective population. Allow and record only floating-point adjustments within 1e-6 persons; stop statistics beyond tolerance.',
        "道路长度": "planned_road_length_m/ restored_length_m仍为完整物理路段长度；treated_inundated_length_m/restored_inundated_length_m为入选/实际恢复道路本期已确认受淹部分长度。横轴仅使用受淹部分长度，不以完整路段长度代替。",
        "受阻道路分母": "blocked_road_total_length_m is the full length of obstructed roads in this period's complete baseline network; restored_blocked_length_fraction is supplementary only, not the main-figure horizontal axis.",
        "评价范围外道路": ('Outside length is the restored physical-road portion outside the original study area' if scope == "study_area" else
                         'Outside length is the restored-road portion outside the evaluated district') +
                        '; with_outside includes crossing roads, wholly_outside includes only entirely external roads, and crossing includes both internal and external portions.',
        "目标治理范围": 'For 50% and 80% burden reduction, find the first sampled stage reaching each threshold without interpolation. If not reached, target_reached=False and extent is missing; this is not a continuous minimum-length solution.',
        "收益": 'Benefits are recalculated on the actual network after combined road restoration; associated road scores cannot be summed as realized benefits.',
        "缓存": 'cache_used records actual network-result cache hits and must be False throughout when use_cache=False. checkpoint_reused records complete stages continued after signature and rebuilt-input checks. routing_recomputed_this_run counts actual recalculation within this city run; a pooled batch reusing a complete existing city run must exclude those historical recalculations from its current-run count. Statistics do not trigger network recalculation.',
    }
    # Write final statistics together only after all calculations and checks finish, preventing conclusions from incomplete batches.
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [(output_names[0], summary_frame), (output_names[1], dimension_frame),
                            (output_names[2], targets)]:
        frame.to_csv(output / filename, index=False, encoding="utf-8-sig", mode="x", na_rep="")
    with (output / "metric_definitions.json").open("x", encoding="utf-8") as stream:
        json.dump(explanations, stream, ensure_ascii=False, indent=2)
    qa = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
          "run_dir": str(run), "strategy": STRATEGY, "periods": periods, "scenario_count": len(summaries),
          "statistics_output_dir": str(output),
          "city": config["city"], "city_name": city_name, "evaluation_scope": scope,
          "evaluation_area_description": scope_description,
          "evaluation_grid_count": len(common_ids), "evaluation_population": total(common_population),
          "baseline_road_count": len(baseline), "baseline_network_total_length_m": denominator,
          "cache_policy": {"use_cache": config.get("use_cache"),
                           "manifest_cache_hit_count": int(summary_frame.cache_used.sum()),
                           "checkpoint_reused_count": checkpoint_count,
                           "routing_recomputed_count": recomputed_count,
                           "checkpoint_evidence": checkpoint_proof,
                           "no_cache_requested_and_verified": config.get("use_cache") is False},
          "checks": {"all_network_jobs_completed_before_statistics": True,
                     "road_prefixes_nested_deduplicated_and_hashed": True,
                     "period_specific_priority_and_restoration_intersection_checked": True,
                     "all_positive_inundated_roads_included_without_fabricated_demand": True,
                     "road_scores_rebuilt_from_population_contacts_and_path_mapping": True,
                     "length_target_schedule_and_first_reaching_prefix_checked": True,
                     "nominal_actual_and_overshoot_lengths_checked": True,
                     "scenario_inundation_denominator_rebuilt_from_all_inventory_roads": True,
                     "treated_inundation_numerator_rebuilt_from_selected_roads": True,
                     "same_population_and_no_flood_baseline_across_periods": True,
                     "monotone_risk_count_and_nearest_distance": True,
                     "zero_governance_and_EFARB_formula_checked": True,
                     "all_EFAR_formulas_checked": all(x["EFAR_formula_checked"] for x in checks),
                     "exposed_unexposed_burden_decomposition_closed": True,
                     "input_output_code_sha_verified_before_and_after": True},
          "tolerances": {"risk": RISK_TOL, "zero_and_burden": VALUE_TOL, "length_m": LENGTH_TOL},
          "statistic_script_signature": own_signature, "source_signatures": list(audit.files.values()),
          "period_checks": period_checks, "scenario_checks": checks,
          "elapsed_seconds": time.perf_counter() - started,
          "outputs": [signature(output / name) for name in output_names if name != "statistics_validation.json"]}
    with (output / "statistics_validation.json").open("x", encoding="utf-8") as stream:
        json.dump(safe_json(qa), stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": "PASS", "scenario_count": len(summaries), "periods": periods,
                      "statistics_dir": str(output), "elapsed_seconds": qa["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help='New statistics-check directory inside the city batch; existing results are never overwritten')
    args = parser.parse_args()
    summarize(args.run_dir, output_dir=args.output_dir)
