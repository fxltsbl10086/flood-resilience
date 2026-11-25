"""Independently summarize exposure-priority and random-restoration experiments without network calculation or manuscript plotting.

First aggregate the agreed repeats within each case, then weight all 25 cases equally. Random quantile ranges describe ordering variation only.
Reuse the existing Fig. 4c population criteria and Fig. 4d2 complete-cell inundation classification; population exposure cannot replace spatial classification."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from _release import ANALYSIS, ROOT as PACKAGE_ROOT
from _support.analysis.restoration.paths import DEFAULT_BATCH
HERE = Path(__file__).resolve().parent
ROOT = HERE
RESULT4 = ROOT.parent
CITIES = ("zhengzhou", "hangzhou", "chengdu", "wuhan", "fuzhou")
CITY_NAMES = dict(zip(CITIES, ("郑州", "杭州", "成都", "武汉", "福州")))
PERIODS = (10, 20, 50, 100, 200)
TARGETS = (0, 1, 2, *range(5, 101, 5))
DIMENSIONS = ("rescue", "medical", "shelter")
EPSILON = 1e-8
RISK_TOLERANCE = 1e-10
AREA_TOLERANCE = 1e-4
DEFAULT_GEOMETRY = ANALYSIS / "restoration_dry_grid"
DEFAULT_SOURCE = DEFAULT_BATCH
STATE_KEY = ["city", "period", "replicate_id", "target_treated_length_pct"]
GOOD_STATUSES = {"complete", "completed", "pass", "ok", "cached", "reused"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def total(values):
    return float(math.fsum(np.asarray(values, dtype=float).ravel().tolist()))


def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator > 0 else np.nan


def signature(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip", **kwargs)


def _d2_module():
    name = "_comparison_existing_dry_statistics"
    if name not in sys.modules:
        if str(RESULT4) not in sys.path:
            sys.path.insert(0, str(RESULT4))
        path = RESULT4 / "fig4d2_/calculate_fig4d2data.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def population_improvement(population, risk_before, risk_after, reachable_before, reachable_after):
    """Use the Fig. 4c criterion: total-risk decrease >1e-8; deduplicate cells into three mutually exclusive resident groups."""
    population = np.asarray(population, dtype=float)
    before, after = np.asarray(risk_before, float), np.asarray(risk_after, float)
    rb, ra = np.asarray(reachable_before), np.asarray(reachable_after)
    require(before.shape == after.shape == rb.shape == ra.shape == (len(population), 3), "")
    require(np.isfinite(population).all() and (population >= 0).all(), "")
    require(np.isfinite(before).all() and np.isfinite(after).all(), "")
    require((after <= before + RISK_TOLERANCE).all(), "")
    require((ra >= rb).all(), "")
    delta = before.mean(axis=1) - after.mean(axis=1)
    improved = delta > EPSILON
    reconnected = ((rb == 0) & (ra > 0)).any(axis=1)
    flags = (improved & reconnected, improved & ~reconnected, ~improved)
    values = [total(population[flag]) for flag in flags]
    p = total(population)
    require(math.isclose(math.fsum(values), p, abs_tol=1e-6, rel_tol=1e-11), "")
    pi = values[0] + values[1]
    record = {"P_valid": p, "P_reconnected": values[0], "P_improved_other": values[1],
              "P_unimproved": values[2], "P_improved": pi,
              "S_reconnected_pct": pct(values[0], p), "S_improved_other_pct": pct(values[1], p),
              "S_unimproved_pct": pct(values[2], p), "S_improved_pct": pct(pi, p),
              "N_reconnected": int(flags[0].sum()), "N_improved_other": int(flags[1].sum()),
              "N_unimproved": int(flags[2].sum()),
              "mean_delta_risk_improved": total(population[improved] * delta[improved]) / pi if pi > 0 else np.nan,
              "P_reconnected_without_improvement": total(population[reconnected & ~improved]),
              "N_reconnected_without_improvement": int((reconnected & ~improved).sum()),
              "population_classification_epsilon": EPSILON}
    return record


def _booleans(series, label):
    text = series.astype(str).str.lower()
    require(text.isin(("true", "false", "0", "1")).all(), "")
    return text.isin(("true", "1")).to_numpy(bool)


def validate_geometry(geometry):
    g = geometry.copy()
    fields = ["grid_area_m2", "depth_coverage_area_m2", "depth_unknown_area_m2", "inundation_area_m2"]
    require(np.isfinite(g[fields].to_numpy(float)).all(), "")
    require(np.allclose(g.grid_area_m2, 250000.0, rtol=0, atol=1e-3), "")
    require((g[fields].to_numpy(float) >= 0).all(), "")
    require(np.allclose(g.depth_unknown_area_m2, np.maximum(g.grid_area_m2-g.depth_coverage_area_m2, 0),
                        rtol=0, atol=AREA_TOLERANCE), "")
    full = g.depth_unknown_area_m2.to_numpy(float) <= AREA_TOLERANCE
    dry = full & (g.inundation_area_m2.to_numpy(float) <= AREA_TOLERANCE)
    require(np.array_equal(full, _booleans(g.depth_full_coverage, "depth_full_coverage")), "")
    require(np.array_equal(dry, _booleans(g.confirmed_dry, "confirmed_dry")), "")
    classification = np.where(dry, "confirmed_dry", np.where(g.inundation_area_m2 > AREA_TOLERANCE, "flooded", "unknown"))
    require(np.array_equal(classification, g.classification.to_numpy()), "")
    g["confirmed_dry"] = dry
    return g


def dry_improvement(population, risk_before, risk_after, geometry):
    """Match geometry to positive-population resident ordering and call the existing Fig. 4d2 coverage/burden functions."""
    p = np.asarray(population, float)
    before, after = np.asarray(risk_before, float), np.asarray(risk_after, float)
    require(len(geometry) == len(p) and before.shape == after.shape == (len(p), 3), "")
    require((p > 0).all(), "")
    g = validate_geometry(geometry)
    g["effective_population"] = p
    g["risk_before"] = before.mean(axis=1)
    g["risk_after"] = after.mean(axis=1)
    g["delta_risk"] = g.risk_before-g.risk_after
    require((g.delta_risk >= -RISK_TOLERANCE).all(), "")
    selected = g.loc[g.confirmed_dry]
    d2 = _d2_module()
    coverage = d2.coverage_summary(selected, 0.05)
    burden = d2.burden_summary(selected)
    return {"N_dry": coverage["N_dry"], "N_benefit": coverage["N_benefit"],
            "G_dry_pct": coverage["G_dry"], "P_dry": coverage["P_dry"],
            "P_benefit_dry_005": coverage["P_benefit"], "H_dry_005_pct": coverage["H_dry_threshold"],
            "R_dry_pct": burden["R_dry"], "B0_dry": burden["B0_dry"],
            "B_after_dry": burden["B_after_dry"], "DeltaB_dry": burden["DeltaB_dry"],
            "N_unknown": int(g.classification.eq("unknown").sum()),
            "P_unknown": total(g.loc[g.classification.eq("unknown"), "effective_population"]),
            "dry_improvement_threshold": 0.05, "dry_threshold_operator": ">=",
            "dry_denominator": "all_confirmed_dry_valid_positive_population_full_grids"}


def load_geometry(directory, sources):
    directory = Path(directory).resolve()
    packaged = directory == DEFAULT_GEOMETRY.resolve()
    csv_path = directory / "grid_flood_classification.csv"
    metadata_path = directory / ("method.json" if packaged else "grid_classification_validation.json")
    metadata = read_json(metadata_path)
    thresholds = metadata.get("thresholds", {})
    require(metadata.get("area_tolerance_m2") == AREA_TOLERANCE
            and thresholds.get("flood_depth_m") == 0.3
            and thresholds.get("flood_operator") == ">=",
            "Unexpected inundation classification thresholds.")
    actual = signature(csv_path)
    metadata_signature = signature(metadata_path)

    if packaged:
        included = metadata.get("included_cities", [])
        require(len(included) == len(CITIES) and set(included) == set(CITIES)
                and tuple(metadata.get("periods", [])) == PERIODS
                and metadata.get("data_file") == csv_path.name
                and metadata.get("algorithm") == "native_axis_aligned_exact_pixel_area_v1"
                and thresholds.get("geometry_area_tolerance_m2") == AREA_TOLERANCE
                and all(metadata.get(field) is False for field in (
                    "population_exposure_used_for_classification", "raster_resampled", "grid_clipped",
                )), "Unexpected packaged dry-grid method or scope.")
        manifest_path = ANALYSIS / "manifest.csv"
        manifest = read_csv(manifest_path, dtype={"path": str, "bytes": str, "sha256": str})
        for path, observed in ((metadata_path, metadata_signature), (csv_path, actual)):
            relative = path.relative_to(PACKAGE_ROOT.resolve()).as_posix()
            matches = manifest.loc[manifest["path"].eq(relative)]
            require(len(matches) == 1
                    and matches.iloc[0]["bytes"] == str(observed["bytes"])
                    and matches.iloc[0]["sha256"] == observed["sha256"],
                    f"Analysis manifest does not match {relative}.")
        sources.append(signature(manifest_path))
    else:
        require(metadata.get("status") == "PASS", "The supplied dry-grid validation did not pass.")
        matches = [record for record in metadata.get("outputs", [])
                   if Path(record["path"]).name == csv_path.name]
        require(len(matches) == 1 and all(actual[key] == matches[0][key] for key in ("bytes", "sha256")),
                "The supplied dry-grid table does not match its validation record.")

    g = validate_geometry(read_csv(csv_path, dtype={"grid_id": str}))
    if packaged:
        require(metadata.get("row_count") == len(g), "Packaged dry-grid row count differs from its method.")
    require(not g.duplicated(["city", "period", "grid_id"]).any(), "")
    require(set(zip(g.city, g.period)) == {(c, p) for c in CITIES for p in PERIODS}, "")
    sources.extend([metadata_signature, actual, signature(RESULT4 / "fig4d2_/calculate_fig4d2data.py")])
    return g


def burden_metrics(population, before, after):
    b0 = total(population * before.mean(axis=1))
    b1 = total(population * after.mean(axis=1))
    delta = b0-b1
    row = {"initial_EFARB_total": b0, "EFARB_total": b1, "EFARB_reduction": delta,
           "EFARB_reduction_fraction": delta / b0 if b0 > 0 else np.nan,
           "EFARB_reduction_pct": pct(delta, b0), "EFARB_reduction_defined": b0 > 0}
    dimension_deltas = []
    for k, name in enumerate(DIMENSIONS):
        db0, db1 = total(population*before[:, k]), total(population*after[:, k])
        dd = db0-db1
        dimension_deltas.append(dd)
        row.update({f"{name}_initial_risk_burden": db0, f"{name}_risk_burden": db1,
                    f"{name}_risk_burden_reduction": dd,
                    f"{name}_risk_burden_reduction_fraction": dd/db0 if db0 > 0 else np.nan,
                    f"{name}_risk_burden_reduction_pct": pct(dd, db0),
                    f"{name}_composite_contribution_pct": pct(dd/3.0, b0)})
    require(math.isclose(delta, math.fsum(dimension_deltas)/3, abs_tol=1e-6, rel_tol=1e-11), "")
    return row


def _read_state(row, base_inverse, sources, verified):
    path = Path(row.state_path).resolve()
    metadata_path = path.with_suffix(".json")
    metadata = read_json(metadata_path)
    expected_key = hashlib.sha256(json.dumps(
        {"model_key": str(row.model_key), "road_set_hash": str(row.planned_road_set_hash), "array_schema": 1},
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(metadata.get("model_key") == row.model_key, "")
    require(metadata.get("road_set_hash") == row.planned_road_set_hash, "")
    require(metadata.get("state_key") == row.state_key == expected_key, "")
    require(metadata.get("schema_version") == 1, "")
    require(metadata.get("sha256") == row.state_sha256, "")
    require(Path(metadata.get("path", "")).resolve() == path, "")
    if str(metadata_path) not in verified:
        sources.append(signature(metadata_path))
        verified.add(str(metadata_path))
    if str(path) not in verified:
        record = signature(path)
        require(record["sha256"] == row.state_sha256, "")
        sources.append(record)
        verified.add(str(path))
    else:
        require(next(x["sha256"] for x in sources if x["path"] == str(path)) == row.state_sha256, "")
    with np.load(path, allow_pickle=False) as state:
        inverse = state["scene_inverse"]
        count = state["reachable_count"]
        risk = state["risk"].copy() if "risk" in state else 1.0 - inverse/base_inverse
        nearest = state["nearest_distance"].copy() if "nearest_distance" in state else state["nearest_distance_m"].copy()
        require(inverse.shape == count.shape == risk.shape == nearest.shape == base_inverse.shape, "")
        require(np.isfinite(inverse).all() and np.isfinite(risk).all() and np.isfinite(count).all(), "")
        require((inverse >= 0).all() and (count >= 0).all() and (count == np.floor(count)).all(), "")
        require(np.allclose(risk, 1.0-inverse/base_inverse, atol=1e-10, rtol=1e-10), "")
        require((risk >= -RISK_TOLERANCE).all() and (risk <= 1+RISK_TOLERANCE).all(), "")
        require((nearest >= 0).all() and not np.isnan(nearest).any(), "")
    return risk, count, nearest


def _bind_baseline(case_dir, sources):
    npz_path, csv_path = case_dir / "evaluation_baseline.npz", case_dir / "evaluation_baseline.csv"
    signatures = [signature(npz_path), signature(csv_path)]
    qa_path = case_dir / "baseline_validation.json"
    require(qa_path.exists(), "")
    qa = read_json(qa_path)
    require(isinstance(qa.get("model_key"), str) and len(qa["model_key"]) == 64, "")
    signed = qa.get("files", qa.get("outputs", []))
    for item in signatures:
        found = [x for x in signed if Path(x["path"]).name == Path(item["path"]).name]
        require(len(found) == 1 and all(found[0][k] == item[k] for k in ("bytes", "sha256")), "")
    sources.extend([*signatures, signature(qa_path)])
    with np.load(npz_path, allow_pickle=False) as source:
        base = {key: source[key].copy() for key in ("grid_id", "population", "base_inverse", "base_count")}
    base["model_key"] = qa["model_key"]
    n = len(base["population"])
    require(base["base_inverse"].shape == base["base_count"].shape == (n, 3), "")
    require(np.isfinite(base["base_inverse"]).all() and (base["base_inverse"] > 0).all(), "")
    require(np.isfinite(base["population"]).all() and (base["population"] >= 0).all(), "")
    require(len(set(base["grid_id"].astype(str))) == n, "")
    frame = read_csv(csv_path, dtype={"grid_id": str})
    require(np.array_equal(frame.grid_id.astype(str), base["grid_id"].astype(str)), "")
    require(np.allclose(frame.effective_population.to_numpy(float), base["population"], rtol=0, atol=1e-8), "")
    return base, frame


def _reference_at30(source_root, city, period, sources):
    path = Path(source_root) / city / "statistics/restoration_effect_summary.csv"
    qa_path = path.parent/"statistics_validation.json"
    qa = read_json(qa_path)
    actual = signature(path)
    require(qa.get("status") == "PASS", "")
    matches = [x for x in qa.get("outputs", []) if Path(x["path"]).name == path.name]
    require(len(matches) == 1 and all(matches[0][k] == actual[k] for k in ("bytes", "sha256")), "")
    sources.extend([actual, signature(qa_path)])
    frame = read_csv(path)
    rows = frame.loc[frame.period.eq(period) & frame.target_treated_length_pct.eq(30)]
    require(len(rows) == 1, "")
    return rows.iloc[0]


def aggregate(long, targets, expected_repeats):
    """Produce formal case statistics only with all required repeats, and the overall curve only with all 25 complete cases."""
    metrics = ["EFARB_reduction_pct", *(f"{d}_risk_burden_reduction_pct" for d in DIMENSIONS),
               "treated_length_pct", "treated_inundated_length_m", "planned_road_count", "length_overshoot_pct"]
    records = []
    for city in CITIES:
        for period in PERIODS:
            for target in targets:
                subset = long.loc[long.city.eq(city) & long.period.eq(period) & long.target_treated_length_pct.eq(target)]
                successful = subset.loc[subset.status.eq("PASS")]
                complete = set(successful.replicate_id.astype(int)) == set(range(expected_repeats))
                for metric in metrics:
                    values = successful[metric].to_numpy(float) if metric in successful else np.empty(0)
                    finite = np.isfinite(values)
                    valid = values[finite]
                    metric_complete = complete and len(valid) == expected_repeats
                    records.append({"city": city, "city_name": CITY_NAMES[city], "period": period,
                        "target_treated_length_pct": target, "metric": metric,
                        "expected_repeats": expected_repeats, "successful_repeats": len(successful),
                        "valid_repeats": len(valid), "undefined_repeats": int((~finite).sum()),
                        "complete": metric_complete,
                        "mean": float(np.mean(valid)) if metric_complete else np.nan,
                        "std": float(np.std(valid, ddof=1)) if metric_complete and len(valid)>1 else np.nan,
                        "quantile_025": float(np.quantile(valid, .025, method="linear")) if metric_complete else np.nan,
                        "quantile_975": float(np.quantile(valid, .975, method="linear")) if metric_complete else np.nan})
    within = pd.DataFrame(records)
    overall_records = []
    for (target, metric), subset in within.groupby(["target_treated_length_pct", "metric"], sort=True):
        valid = subset.loc[subset.complete]
        complete = len(valid) == 25
        overall_records.append({"target_treated_length_pct": target, "metric": metric,
            "expected_city_scenarios": 25, "complete_city_scenarios": len(valid),
            "repeats_per_city_scenario": expected_repeats, "complete": complete,
            "mean_25_equal_weight": float(valid["mean"].mean()) if complete else np.nan})
    return within, pd.DataFrame(overall_records)


def run(run_dir, expected_repeats=None, output_dir=None):
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    config_path = run_dir/"run_config.json"
    config = read_json(config_path)
    strategy = config["strategy"]
    require(strategy in ("population_priority", "random_priority"), "")
    score_field = config.get("population_score_field")
    require(score_field == ("population_exposed" if strategy == "population_priority" else None),
            "")
    score_definition = config.get("population_score_definition")
    if score_definition is None and strategy == "population_priority":
        score_definition = ('Deduplicate physical-road/resident-cell IDs and aggregate scenario inundation-exposed population from population_exposed_T{period}, represented as population_exposed in frozen inputs. Use this for ranking only; evaluation still uses effective_population.')
    require(set(config.get("cities", CITIES)) <= set(CITIES), "")
    require(set(map(int, config.get("periods", PERIODS))) <= set(PERIODS), "")
    planned = int(config.get("planned_repeats", 100 if strategy == "random_priority" else 1))
    expected_repeats = int(expected_repeats or planned)
    require(0 < expected_repeats <= planned and (strategy != "population_priority" or expected_repeats == 1), "")
    targets = tuple(map(int, config.get("targets", TARGETS)))
    require(set(targets) <= set(TARGETS) and 0 in targets, "")
    source_root = Path(config.get("source_root", DEFAULT_SOURCE))
    geometry_dir = Path(config.get("geometry_dir", DEFAULT_GEOMETRY))
    output_dir = Path(output_dir) if output_dir else run_dir/"statistics"/datetime.now().strftime("%Y%m%d_%H%M%S_standalone_statistics")
    require(not output_dir.exists(), "")
    sources = [signature(config_path), signature(__file__)]
    for filename in ("run_comparison.py", "comparison_common.py", "comparison_context.py"):
        code_path = HERE/filename if (HERE/filename).exists() else ROOT/filename
        sources.append(signature(code_path))
    geometry = load_geometry(geometry_dir, sources)
    records, at30, failures, case_audits = [], [], [], []
    verified_states = set()
    for city in CITIES:
        for period in PERIODS:
            case_dir = run_dir/'cases'/city/f"T{period}"
            state_table = case_dir/"state_record.csv"
            if not state_table.exists():
                for replicate_id in range(expected_repeats):
                    for target in targets:
                        records.append({"city": city, "period": period, "replicate_id": replicate_id,
                            "target_treated_length_pct": target, "status": "MISSING", "error": "案例状态表尚未生成"})
                continue
            try:
                sources.append(signature(state_table))
                states = read_csv(state_table)
                require(not states.duplicated(["replicate_id", "target_treated_length_pct"]).any(), "")
                require(states.city.eq(city).all() and states.period.eq(period).all(), "")
                base, baseline_frame = _bind_baseline(case_dir, sources)
                require("model_key" in states and states.model_key.eq(base["model_key"]).all(), "")
                population, inverse = base["population"], base["base_inverse"]
                grid_ids = base["grid_id"].astype(str)
                positive = population > 0
                g = geometry.loc[geometry.city.eq(city) & geometry.period.eq(period)].set_index("grid_id")
                require(set(grid_ids[positive]) == set(g.index), "")
                ordered_geometry = g.loc[grid_ids[positive]].reset_index()
                before_rows = states.loc[states.target_treated_length_pct.eq(0) & states.status.astype(str).str.lower().isin(GOOD_STATUSES)]
                require(len(before_rows) > 0, "")
                before, before_count, before_distance = _read_state(before_rows.iloc[0], inverse, sources, verified_states)
                for k, dimension in enumerate(DIMENSIONS):
                    require(np.allclose(before[:, k], baseline_frame[f"initial_EFAR_{dimension}"], rtol=0, atol=1e-8), "")
                    if f"initial_reachable_count_{dimension}" in baseline_frame:
                        require(np.array_equal(before_count[:, k], baseline_frame[f"initial_reachable_count_{dimension}"].to_numpy()), "")
                reference = _reference_at30(source_root, city, period, sources) if 30 in targets else None
                for replicate_id in range(expected_repeats):
                    previous = None
                    for target in targets:
                        selected = states.loc[states.replicate_id.eq(replicate_id) & states.target_treated_length_pct.eq(target)]
                        common = {"city": city, "city_name": CITY_NAMES[city], "period": period,
                                  "replicate_id": replicate_id, "target_treated_length_pct": target, "strategy": strategy}
                        if len(selected) == 0:
                            records.append({**common, "status": "MISSING", "error": "此重复/档位尚未完成"})
                            continue
                        row = selected.iloc[0]
                        if str(row.status).lower() not in GOOD_STATUSES:
                            records.append({**row.to_dict(), **common, "status": "FAILED", "error": str(row.get("error", "网络状态未完成"))})
                            failures.append({**common, "error": str(row.get("error", "网络状态未完成"))})
                            continue
                        try:
                            after, count, distance = _read_state(row, inverse, sources, verified_states)
                            require((after <= before + RISK_TOLERANCE).all(), "")
                            require((count >= before_count).all(), "")
                            require((distance <= before_distance + 1e-6).all(), "")
                            if target == 0:
                                require(np.allclose(after, before, rtol=0, atol=RISK_TOLERANCE), "")
                                require(np.array_equal(count, before_count), "")
                            if previous is not None:
                                require((after <= previous[0]+RISK_TOLERANCE).all(), "")
                                require((count >= previous[1]).all(), "")
                                require(row.treated_length_pct+1e-10 >= previous[2], "")
                                require(row.planned_road_count >= previous[3], "")
                            previous = (after, count, float(row.treated_length_pct), int(row.planned_road_count))
                            require(float(row.treated_length_pct)+1e-6 >= target, "")
                            require(float(row.treated_length_pct) <= 100+1e-8, "")
                            require(math.isclose(float(row.treated_length_pct), 100*float(row.treated_inundated_length_m)/float(row.scenario_total_inundated_length_m), rel_tol=1e-11, abs_tol=1e-8), "")
                            summary = {**row.to_dict(), **common, **burden_metrics(population, before, after),
                                       "evaluation_population": total(population), "evaluation_grid_count": len(population), "status": "PASS"}
                            if target == 30:
                                detail = {**common, **population_improvement(population, before, after, before_count, count),
                                    **dry_improvement(population[positive], before[positive], after[positive], ordered_geometry),
                                    **{k: summary[k] for k in ("treated_length_pct", "treated_inundated_length_m", "planned_road_count", "length_overshoot_pct", "EFARB_reduction_pct")}}
                                detail.update({"risk_priority_actual_pct": float(reference.treated_length_pct),
                                    "risk_priority_EFARB_reduction_pct": 100*float(reference.EFARB_reduction_rate),
                                    "actual_pct_difference_vs_risk": float(row.treated_length_pct-reference.treated_length_pct),
                                    "reduction_pp_difference_vs_risk": summary["EFARB_reduction_pct"]-100*float(reference.EFARB_reduction_rate)})
                                at30.append(detail)
                            records.append(summary)
                        except Exception as error:
                            records.append({**common, "status": "INVALID", "error": str(error)})
                            failures.append({**common, "error": str(error)})
                case_audits.append({"city": city, "period": period, "zero_reference_verified": True,
                                    "evaluation_grid_count": len(population), "positive_population_grid_count": int(positive.sum())})
            except Exception as error:
                failures.append({"city": city, "period": period, "error": str(error)})
                # A failed case must retain its expected status rather than silently disappearing from the 25-case summary.
                existing = {(int(x["replicate_id"]), int(x["target_treated_length_pct"])) for x in records if x["city"] == city and x["period"] == period}
                for replicate_id in range(expected_repeats):
                    for target in targets:
                        if (replicate_id, target) not in existing:
                            records.append({"city": city, "period": period, "replicate_id": replicate_id,
                                "target_treated_length_pct": target, "status": "INVALID", "error": str(error)})
    long = pd.DataFrame(records)
    require(not long.duplicated(STATE_KEY).any(), "")
    within, overall = aggregate(long, targets, expected_repeats)
    detail30 = pd.DataFrame(at30)
    detail_summary = []
    if len(detail30):
        metric_columns = [c for c in detail30.select_dtypes(include=np.number) if c not in ("period", "replicate_id", "target_treated_length_pct")]
        for (city, period), group in detail30.groupby(["city", "period"], sort=False):
            for metric in metric_columns:
                values = group[metric].to_numpy(float)
                valid = values[np.isfinite(values)]
                complete = set(group.replicate_id.astype(int)) == set(range(expected_repeats)) and len(valid) == expected_repeats
                detail_summary.append({"city": city, "period": period, "metric": metric, "valid_repeats": len(valid),
                    "expected_repeats": expected_repeats, "complete": complete,
                    "mean": float(np.mean(valid)) if complete else np.nan,
                    "std": float(np.std(valid, ddof=1)) if complete and len(valid)>1 else np.nan,
                    "quantile_025": float(np.quantile(valid, .025, method="linear")) if complete else np.nan,
                    "quantile_975": float(np.quantile(valid, .975, method="linear")) if complete else np.nan})
    # All inputs must remain unchanged throughout independent statistics; deduplicate references by path.
    unique_sources = {item["path"]: item for item in sources}
    for item in unique_sources.values():
        require(signature(item["path"]) == item, "")
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    budget_fields = [*STATE_KEY, "status", "treated_length_pct", "treated_inundated_length_m", "scenario_total_inundated_length_m", "planned_road_count", "length_overshoot_m", "length_overshoot_pct", "planned_road_set_hash"]
    tables = {"benefits_by_repeat_and_target.csv": long, "city_period_statistics.csv": within,
              "equal_weight_benefits_across_25_cases.csv": overall, "target30_population_and_dry_grid_benefits.csv": detail30,
              "target30_repeat_statistics_and_reference.csv": pd.DataFrame(detail_summary),
              "actual_budget_and_overshoot.csv": long.reindex(columns=budget_fields),
              "incomplete_or_failed_cases.csv": long.loc[~long.status.eq("PASS")]}
    for filename, frame in tables.items():
        path = output_dir/filename
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g", na_rep="")
        outputs.append(signature(path))
    successful = int(long.status.eq("PASS").sum())
    expected_states = 25*len(targets)*expected_repeats
    complete = successful == expected_states and not failures
    report = {"status": "PASS" if complete else "INCOMPLETE", "strategy": strategy,
        "population_score_field": score_field, "population_score_definition": score_definition,
        "evaluation_population_field": "effective_population",
        "created_at": datetime.now().astimezone().isoformat(), "run_dir": str(run_dir),
        "planned_repeats": planned, "summarized_repeat_ids": list(range(expected_repeats)),
        "expected_states_in_this_summary": expected_states, "passed_states": successful,
        "missing_failed_or_invalid_states": expected_states-successful,
        "full_planned_experiment_complete": complete and expected_repeats==planned and targets==TARGETS,
        "validation_status": "PASS" if not failures else "FAIL", "validation_error_count": len(failures),
        "city_scenario_count": 25, "included_cities": list(CITIES), "excluded_cities": [],
        "periods": list(PERIODS), "targets": list(targets), "epsilon": EPSILON,
        "dry_improvement_threshold": .05, "dry_area_tolerance_m2": AREA_TOLERANCE,
        "aggregation": "equal city-scenario means after within-case repetitions; no averaging of quantile limits",
        "random_quantile_interpretation": "random ordering variability, not city-population inference confidence interval",
        "standard_deviation_ddof": 1, "quantile_method": "linear",
        "case_checks": case_audits, "source_signatures": list(unique_sources.values()), "outputs": outputs,
        "elapsed_seconds": time.perf_counter()-started,
        "notes": ['Ratios with zero initial burden or zero dry-cell denominator are missing, not zero-filled.',
                  'Risk scores measure associations and are not directly summed as restoration benefits.',
                  'population_priority uses inundation-exposed population for road ordering only; EFARB and population improvements still use original effective-population evaluation.',
                  'Exposure priority uses a risk-screened mapping and is neither fully risk-independent nor a strict single-factor ablation.',
                  'Random quantiles describe ordering variability; this analysis uses no bootstrap.']}
    print(json.dumps({k: report[k] for k in ("status", "strategy", "planned_repeats", "passed_states", "expected_states_in_this_summary", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    print(str(output_dir), flush=True)
    return report, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, help='Required complete repeat count: 10 for the initial batch and 100 for the full run')
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report, _ = run(args.run_dir, args.expected_repeats, args.output_dir)
    # Mark trials or unrun cases INCOMPLETE; return nonzero only for actual data, conservation or task failures.
    sys.exit(0 if report["validation_status"] == "PASS" else 1)
