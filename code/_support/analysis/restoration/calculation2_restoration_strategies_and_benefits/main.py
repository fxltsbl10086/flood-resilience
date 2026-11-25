"""Unified entry: select roads independently by period using all inundated length in the same city/scenario; summarize final benefits separately."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys as _path_sys
_RESULT4_ROOT = Path(__file__).resolve().parents[1]
if str(_RESULT4_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_RESULT4_ROOT))
import platform
import shutil
import sys
import time
from zoneinfo import ZoneInfo

import runtime
from _support.analysis.restoration.paths import read_json, resolve_path
from plan import (STRATEGY, SAMPLING, build_length_plan, validate_baseline_inventory,
                  validate_inundated_inventory, extend_inundated_priority, road_hash,
                  TARGET_PCTS, LENGTH_THRESHOLD_TOLERANCE_M)


HERE = Path(__file__).resolve().parent
PERIODS = (10, 20, 50, 100, 200)
CITIES = {"chengdu": "成都", "zhengzhou": "郑州", "hangzhou": "杭州", "wuhan": "武汉", "fuzhou": "福州"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def json_key(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_signature(path):
    path = resolve_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _json(path):
    return read_json(path)


def _require(value, message):
    if not value:
        raise ValueError("")


def _checked_path(root, relative):
    path = (root / str(relative)).resolve()
    _require(path.is_relative_to(root), "")
    return path


class CheckpointResume:
    """Read-only verification of final cell-level results without loading old topology, OD or pickle caches."""

    def __init__(self, args, current_scripts):
        import pandas as pd

        self.source = resolve_path(args.resume_from)
        self.config_path = self.source / "run_config.json"
        self.config = _json(self.config_path)
        self.config_signature = file_signature(self.config_path)
        self.frozen = {str(self.config_path): self.config_signature}
        self.verified_periods, self.rows, self.rejected = [], [], []
        self.manifest_signatures, self.table_comparisons, self.periods = [], [], {}
        timing_path = self.source / 'calculation_runtime.json'
        self.source_timings = []
        if timing_path.is_file():
            self._freeze(timing_path)
            self.source_timings = _json(timing_path).get("records", [])
        _require(self.config.get("schema_version") == 3 and self.config.get("strategy") == STRATEGY,
                 "")
        _require(self.config.get("use_cache") is False, "")
        for key, expected in (("city", args.city), ("evaluation_scope", args.evaluation_scope),
                              ("district", args.district if args.evaluation_scope == "district" else None),
                              ("evaluation_periods", sorted(set(args.periods)))):
            _require(self.config.get(key) == expected, "")
        source_records = self.config.get("source_signatures", [])
        _require(source_records, "")
        for record in source_records:
            self._verify(record)
        if self.config.get("resume_checkpoint_manifest_signature"):
            self._verify(self.config["resume_checkpoint_manifest_signature"])
        for name, current in current_scripts.items():
            previous = self.config["scripts"][name]
            _require(previous == current, "")
            self._verify(previous)
        self.old_plans = self._read_table("plans/restoration_targets.csv")
        for period in sorted(set(args.periods)):
            path = self.source / f"T{period}/scenario_inventory.csv"
            if not path.exists():
                continue
            manifest_signature = self._freeze(path)
            try:
                manifest = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
                self.rejected.append({"period": period, "reason": f"Incomplete scenario inventory ({type(exc).__name__})"})
                continue
            _require({"period", "plan_index"} <= set(manifest), "")
            _require(not manifest.duplicated(["period", "plan_index"]).any(), "")
            _require(manifest.period.eq(period).all(), "")
            self.periods[period] = (manifest, manifest_signature)
            self.manifest_signatures.append(manifest_signature)

    def _freeze(self, path):
        record = file_signature(path)
        self.frozen[record["path"]] = record
        return record

    def _verify(self, expected):
        actual = self._freeze(expected["path"])
        _require(actual == expected, "")
        return actual

    def _read_table(self, relative):
        import pandas as pd
        path = _checked_path(self.source, relative)
        self._freeze(path)
        return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip",
                           dtype={"road_id": str, "grid_id": str, "selected_facility_grid_id": str})

    def compare_configuration(self, fresh):
        ignored = {"scripts", "created_at", "environment", "source_signatures", "resume_from",
                   "resume_checkpoint_manifest_signature", "resume_code_archive_config_signature",
                   "resume_compatibility_signature", "checkpoint_resume_enabled"}
        for key, value in self.config.items():
            if key not in ignored:
                _require(key in fresh and fresh[key] == value, "")
        old = {item["path"]: item for item in self.config["source_signatures"]}
        new = {item["path"]: item for item in fresh["source_signatures"]}
        _require(old == new, "")

    def verify_period(self, period, output, plans, routing_manifest):
        import pandas as pd
        if period not in self.periods:
            return
        tables = ["inputs/fixed_baseline_road_inventory.csv", f"inputs/T{period}_evaluation_grids.csv",
                  f"inputs/T{period}_road_inventory.csv", f"plans/T{period}_maximum_loss_facility_links.csv",
                  f"plans/T{period}_facility_link_road_lookup.csv", f"plans/T{period}_road_link_burden_details.csv",
                  f"plans/T{period}_road_priority_order.csv", f"plans/T{period}_grid_project_status.csv"]
        for relative in tables:
            old = self._read_table(relative)
            new = pd.read_csv(output / relative, encoding="utf-8-sig", float_precision="round_trip",
                              dtype={"road_id": str, "grid_id": str, "selected_facility_grid_id": str})
            pd.testing.assert_frame_equal(old, new, check_exact=True)
            self.table_comparisons.append({"period": period, "relative_path": relative,
                                           "exact_values_and_order_equal": True,
                                           "source_signature": self.frozen[str(_checked_path(self.source, relative))],
                                           "rebuilt_signature": file_signature(output / relative)})
        old_plan = self.old_plans.loc[self.old_plans.period.eq(period)].reset_index(drop=True)
        fresh_plan = pd.read_csv(output / "plans/restoration_targets.csv", encoding="utf-8-sig", float_precision="round_trip")
        fresh_plan = fresh_plan.loc[fresh_plan.period.eq(period)].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_plan, fresh_plan, check_exact=True)
        path = self.source / f"inputs/T{period}_network_calculation_fingerprint.json"
        self._freeze(path)
        old_model = _json(path)
        _require(old_model.get("runtime") == self.config["scripts"]["runtime.py"], "")
        _require({k: v for k, v in old_model.items() if k != "runtime"}
                 == {k: v for k, v in routing_manifest.items() if k != "runtime"},
                 "")
        self.periods[period] = (*self.periods[period], json_key(old_model), json_key(routing_manifest))
        self.verified_periods.append(period)

    def candidate(self, period, plan, roads, output, evaluation, baseline):
        """Do not skip computation for corrupt/incomplete results; reject model, population or road-plan mismatches earlier."""
        import pandas as pd
        import numpy as np
        if period not in self.verified_periods:
            return None
        manifest, manifest_signature, old_key, new_key = self.periods[period]
        matches = manifest.loc[manifest.plan_index.eq(plan["plan_index"])]
        if matches.empty:
            return None
        row = matches.iloc[0].to_dict()
        required = ["model_key", "result_file", "restored_roads_file", "planned_roads_file",
                    "result_sha256", "restored_roads_sha256", "planned_roads_sha256", "cache_used"]
        try:
            _require(all(key in row and pd.notna(row[key]) for key in required), "")
            _require(row["model_key"] == old_key, "")
            _require(str(row["cache_used"]).lower() == "false", "")
            for key, value in plan.items():
                _require(key in row and row[key] == value, "")
            source_files = {}
            for file_key, hash_key in (("result_file", "result_sha256"), ("restored_roads_file", "restored_roads_sha256"),
                                       ("planned_roads_file", "planned_roads_sha256")):
                source = _checked_path(self.source, row[file_key])
                signed = self._freeze(source)
                _require(signed["sha256"] == row[hash_key], "")
                source_files[file_key] = source
            for key in ("planned_roads_file", "restored_roads_file"):
                original_roads = pd.read_csv(source_files[key], encoding="utf-8-sig", float_precision="round_trip", dtype={"road_id": str})
                fresh_roads = pd.read_csv(output / plan["planned_roads_file"], encoding="utf-8-sig", float_precision="round_trip", dtype={"road_id": str})
                pd.testing.assert_frame_equal(original_roads, fresh_roads, check_exact=True)
            rows = pd.read_csv(source_files["result_file"], encoding="utf-8-sig", float_precision="round_trip")
            identifiers = ["grid_id", "grid_index", "effective_population", "population_status", "rank"]
            _require(set(identifiers) <= set(rows) and len(rows) > 0 and not rows.grid_id.duplicated().any(),
                     "")
            expected = evaluation.loc[evaluation.efarb_valid, identifiers].reset_index(drop=True)
            pd.testing.assert_frame_equal(rows[identifiers], expected, check_exact=True, check_dtype=False)
            for k, dimension in enumerate(("rescue", "medical", "shelter")):
                np.testing.assert_array_equal(rows[f"base_count_{dimension}"], baseline["base_count"][:, k])
                np.testing.assert_array_equal(rows[f"base_inverse_{dimension}"], baseline["base_inverse"][:, k])
            timing = next((item for item in self.source_timings
                           if item.get("period") == period and item.get("target_treated_length_pct") == plan["target_treated_length_pct"]
                           and item.get("planned_road_count") == plan["planned_road_count"]), {})
            return {"source_row": row, "rows": rows, "source_files": source_files,
                    "source_original_compute_seconds": (timing.get("cache_record", {}).get("original_compute_seconds")
                                                          if timing.get("cache_record", {}).get("original_compute_seconds") is not None
                                                          else timing.get("source_original_compute_seconds")),
                    "proof": {"period": period, "plan_index": int(plan["plan_index"]),
                    "source_run": str(self.source), "result_sha256": row["result_sha256"],
                    "planned_roads_sha256": row["planned_roads_sha256"], "restored_roads_sha256": row["restored_roads_sha256"],
                    "source_scene_manifest_signature": manifest_signature, "source_model_key": old_key,
                    "new_model_key": new_key, "model_inputs_equal": True}}
        except (ValueError, AssertionError, KeyError, FileNotFoundError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
            self.rejected.append({"period": period, "plan_index": int(plan["plan_index"]),
                                  "reason": str(exc), "action": "recompute_from_fresh_network"})
            return None

    def copy(self, candidate, output, planned_file, restored_file, result_file):
        for key, relative in (("planned_roads_file", planned_file), ("restored_roads_file", restored_file), ("result_file", result_file)):
            source, target = candidate["source_files"][key], _checked_path(output, relative)
            shutil.copyfile(source, target)
            _require(file_signature(source)["sha256"] == file_signature(target)["sha256"], "")
        self.rows.append(candidate["proof"])

    def report(self):
        return {"status": "PASS", "source_run": str(self.source), "source_config_signature": self.config_signature,
                "source_scene_manifest_signatures": self.manifest_signatures,
                "periods_verified": self.verified_periods, "rows": self.rows, "rejected_checkpoints": self.rejected,
                "table_comparisons": self.table_comparisons, "source_signatures": list(self.frozen.values()),
                "cache_used": False, "checkpoint_rows_are_final_results_not_computation_cache": True}

    def verify_frozen(self):
        for expected in self.frozen.values():
            _require(file_signature(expected["path"]) == expected, "")


def run(args):
    import numpy as np
    import pandas as pd

    started = time.perf_counter()
    periods = tuple(sorted(set(args.periods)))
    if not periods or not set(periods) <= set(PERIODS):
        raise ValueError("")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%Hh%Mmin%Ss")
    scenario_label = "_".join(f"T{period}" for period in periods)
    city_name = CITIES[args.city]
    district = args.district if args.evaluation_scope == "district" else None
    scripts = {name: file_signature(HERE / name) for name in ("main.py", "runtime.py", "plan.py")}
    resume = CheckpointResume(args, scripts) if args.resume_from else None
    output = (args.output_dir or HERE / "outputs" / f"{stamp}_{args.city}_{scenario_label}_target_flooded_length_restoration").resolve()
    if resume:
        _require(output != resume.source and not output.is_relative_to(resume.source), "")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "plans", *[f"T{p}" for p in periods]):
        (output / name).mkdir()
    print(f'Output batch: {output}', flush=True)
    records = []
    stage_start = time.perf_counter()
    ctx = runtime.load_context_fresh(city=args.city, district=district, periods=periods,
                                     evaluation_scope=args.evaluation_scope)
    api = ctx["api"]
    engine, signature = api["engine"], api["signature"]
    def save(path, payload):
        # Create a new output directory and commit atomically; interruption leaves only incomplete tmp files, preserving the last valid checkpoint.
        temporary = path.with_suffix(path.suffix + ".tmp")
        api["dump_json"](temporary, payload)
        temporary.replace(path)
    records.append({"stage": "Prepare network and flood inputs", "period": None,
                    "elapsed_seconds": time.perf_counter() - stage_start, "cache_used": False})
    source_signatures = ctx["source_signatures"]
    baseline_columns = ["road_id", "road_position", "length_m", "inside_evaluation_area_length_m",
                        "outside_evaluation_area_length_m"]
    first_pack = ctx["packs"][periods[0]]
    inventory = first_pack["road_inventory"][baseline_columns].copy()
    inventory["road_id"] = inventory.road_id.astype(str)
    total_length = validate_baseline_inventory(inventory)
    inventory.to_csv(output / "inputs/fixed_baseline_road_inventory.csv", index=False, encoding="utf-8-sig")
    baseline_source = ctx["baselines"][periods[0]]
    fixed_eval = first_pack["evaluation_all"].loc[first_pack["evaluation_all"].efarb_valid,
                    ["grid_id", "grid_index", "effective_population"]].reset_index(drop=True)
    configuration = {
        "schema_version": 3, "strategy": STRATEGY, "city": args.city, "city_name": city_name,
        "district": district, "evaluation_scope": args.evaluation_scope, "evaluation_periods": list(periods),
        "baseline_network_total_length_m": total_length, "baseline_road_count": len(inventory),
        "baseline_network_scope": f'原{city_name}Full analysis network: full lengths of all unique physical road_id values, without clipping to target district boundaries.',
        "scenario_total_inundated_length_m_by_period": {
            str(p): validate_inundated_inventory(ctx["packs"][p]["road_inventory"]) for p in periods},
        "inundated_length_definition": 'Intersection length of original study-area road centerlines with the union of valid depth cells >=0.30 m; period-specific length metadata record raster geometry.',
        "inundated_length_threshold_m": 0.30,
        "inundated_denominator_scope": 'All confirmed-inundated portions of the original analysis network in the same city/scenario, unrestricted by candidate roads or target districts.',
        "inundated_length_missing_data_policy": 'Retain missing-coverage and vertical-uncertainty flags; count confirmed portions only and do not interpret missing coverage as dry.',
        "restored_full_length_definition": 'Restored traversable-road length is the full length of unique restored physical segments; record separately, not as the main horizontal-axis numerator.',
        "evaluation_area": (f'{district} valid resident cells in the original study area; do not split original cell population at district boundaries' if district
                            else f'{city_name}All valid resident cells in the original study area; this does not represent the full administrative city.'),
        "evaluation_grid_count": len(fixed_eval),
        "evaluation_population": math.fsum(map(float, fixed_eval.effective_population)),
        "road_plan_is_shared_across_periods": False, "road_order_is_fixed_within_period": True,
        "length_used_in_ranking": False, "score_is_realized_benefit": False,
        "target_treated_length_pcts": list(TARGET_PCTS),
        "length_threshold_tolerance_m": LENGTH_THRESHOLD_TOLERANCE_M,
        "road_selection_rule": "first whole-road prefix whose cumulative inundated length plus tolerance reaches the target",
        "priority_road_scope": "all roads with positive confirmed inundated length in the same city and period",
        "zero_score_road_rule": "no selected demand contact: score and associated counts zero; best_grid_rank null; append in road_id lexicographic order",
        "nominal_target_is_not_actual_length": True,
        "road_score": "sum(effective_population/3*selected_EFAR_contribution) over associated contacts",
        "road_sort": ["associated_risk_burden descending", "best_grid_rank ascending", "road_id lexicographic ascending"],
        "sampling": SAMPLING, "x_definition": "100*unique selected road inundated length / all same-city same-period road inundated length",
        "y_definition": "whole evaluation-area EFARB reduction relative to the same period's zero-road state",
        "sources_are_read_only": True, "source_signatures": source_signatures, "scripts": scripts,
        "require_existing_cache": False,
        "use_cache": False,
        "computation_mode": "fresh_without_computation_cache",
        "created_at": now(), "environment": {"python": sys.version, "executable": sys.executable,
                                            "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    if resume:
        resume.compare_configuration(configuration)
        configuration.update({"checkpoint_resume_enabled": True, "resume_from": str(resume.source)})
    save(output / "run_config.json", configuration)
    save(output / "inputs/cache_reuse_validation.json", ctx["reuse_metadata"])
    all_plans, completed_periods = [], []
    for period in periods:
        period_start = time.perf_counter()
        pack, baseline = ctx["packs"][period], ctx["baselines"][period]
        current_inventory = pack["road_inventory"].copy()
        current_inventory["road_id"] = current_inventory.road_id.astype(str)
        pd.testing.assert_frame_equal(current_inventory[baseline_columns], inventory, check_exact=True)
        valid_rows = pack["evaluation_all"].loc[pack["evaluation_all"].efarb_valid,
                         ["grid_id", "grid_index", "effective_population"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(valid_rows, fixed_eval, check_exact=True)
        for key in ("resident_indices", "base_inverse", "base_count", "dimensions_valid", "counts"):
            np.testing.assert_array_equal(baseline[key], baseline_source[key])
        if baseline["scenario"] != f"T{period}":
            raise ValueError("")
        np.testing.assert_array_equal(pack["topology"]["closures"][f"T{period}"],
                                      current_inventory.flood_scenario.to_numpy(bool))
        evaluation, ranked = api["make_priority"](pack["evaluation_all"])
        evaluation.to_csv(output / f"inputs/T{period}_evaluation_grids.csv", index=False, encoding="utf-8-sig")
        current_inventory.to_csv(output / f"inputs/T{period}_road_inventory.csv", index=False, encoding="utf-8-sig")
        selection = ctx["selections"][period]
        selection_table = pd.DataFrame(selection["selection_records"])
        selection_table = selection_table.drop(columns=["rank", "ranking_eligible"], errors="ignore").merge(
            evaluation[["grid_id", "rank", "ranking_eligible"]], on="grid_id", validate="many_to_one")
        links = pd.DataFrame(selection["contact_road_records"], columns=[
            "grid_id", "grid_index", "dimension", "selected_facility_grid_id", "road_position"])
        links["road_id"] = links.road_position.map(current_inventory.set_index("road_position").road_id)
        demand_priority, contacts, priority_meta = api["build_road_priority"](
            evaluation, selection_table, links, current_inventory)
        priority = extend_inundated_priority(demand_priority, current_inventory)
        priority_meta.update({"candidate_road_count": len(priority),
                             "demand_linked_road_count": len(demand_priority),
                             "zero_score_road_count": len(priority) - len(demand_priority),
                             "candidate_scope": "all positive confirmed inundated roads in this city and scenario",
                             "zero_score_meaning": "no association under the selected maximum-loss contacts; does not imply zero realized network benefit"})
        selection_table.to_csv(output / f"plans/T{period}_maximum_loss_facility_links.csv", index=False, encoding="utf-8-sig")
        links.to_csv(output / f"plans/T{period}_facility_link_road_lookup.csv", index=False, encoding="utf-8-sig")
        contacts.to_csv(output / f"plans/T{period}_road_link_burden_details.csv", index=False, encoding="utf-8-sig")
        priority.to_csv(output / f"plans/T{period}_road_priority_order.csv", index=False, encoding="utf-8-sig")
        status = evaluation[["grid_id", "ranking_eligible", "rank"]].copy()
        project_counts = links.loc[links.grid_id.isin(set(ranked.grid_id))].drop_duplicates(["grid_id", "road_id"]).groupby("grid_id").size()
        status["project_road_count"] = status.grid_id.map(project_counts).fillna(0).astype(int)
        status["no_project"] = status.project_road_count.eq(0)
        status.to_csv(output / f"plans/T{period}_grid_project_status.csv", index=False, encoding="utf-8-sig")
        plans = build_length_plan(priority, current_inventory, period)
        all_plans.append(plans)
        pd.concat(all_plans, ignore_index=True).to_csv(output / "plans/restoration_targets.csv", index=False, encoding="utf-8-sig")
        save(output / f"plans/T{period}_ranking_method.json", {**priority_meta, "strategy": STRATEGY,
             "fixed_within_period": True, "shared_across_periods": False, "sampling": SAMPLING})
        records.append({"stage": 'Current-period demand-contact selection, road ordering and target-stage construction', "period": period,
                        "elapsed_seconds": time.perf_counter() - period_start, "cache_used": False})
        # Single/multiple-period calls share routing keys; other scenarios' selection or plotting settings do not invalidate this period.
        period_source = ctx["reuse_metadata"]["period_sources"][period]
        routing_manifest = {"sources": source_signatures, "runtime": scripts["runtime.py"],
                 "source_scenario_model_key": period_source["scenario_model_key"],
                 "city": args.city, "period": period, "district": district, "evaluation_scope": args.evaluation_scope,
                 "residents": list(map(int, baseline["resident_indices"])),
                 "facility_rule": "fixed_baseline_reachable_all_original_facilities", "local_distance_m": 250.0}
        model_key = json_key(routing_manifest)
        save(output / f"inputs/T{period}_network_calculation_fingerprint.json", routing_manifest)
        if resume:
            resume.verify_period(period, output, plans, routing_manifest)
        scenes, previous = [], set()
        print(f'T{period}: evaluating {len(valid_rows)} cells; demand in {len(ranked)} cells; candidates {len(priority)} roads; {len(plans)} stages; final horizontal-axis value {plans.treated_length_pct.iloc[-1]:.6f}%', flush=True)
        for plan in plans.to_dict("records"):
            scene_start = time.perf_counter()
            m = int(plan["planned_road_count"])
            roads = priority.iloc[:m].copy()
            if not roads.flood_scenario.eq(1).all():
                raise ValueError("")
            positions = set(map(int, roads.road_position))
            if len(positions) != m or not previous <= positions:
                raise ValueError("")
            road_set_hash = road_hash(roads.road_id)
            if road_set_hash != plan["planned_road_set_hash"]:
                raise ValueError("")
            planned_file = plan["planned_roads_file"]
            scene_stem = f'step{int(plan['plan_index']):02d}_target{int(plan['target_treated_length_pct']):03d}pct'
            restored_file = f'T{period}/{scene_stem}_restored_roads.csv'
            result_file = f'T{period}/{scene_stem}_grid_results.csv'
            roads.to_csv(output / planned_file, index=False, encoding="utf-8-sig")
            roads.to_csv(output / restored_file, index=False, encoding="utf-8-sig")
            checkpoint = resume.candidate(period, plan, roads, output, evaluation, baseline) if resume else None
            if checkpoint is not None:
                resume.copy(checkpoint, output, planned_file, restored_file, result_file)
                rows = checkpoint["rows"]
                cache_info = {"cache_used": False, "elapsed_seconds": 0.0,
                              "original_compute_seconds": None, "cache_file": None, "cache_disabled": True}
                timings = {}
            else:
                compute_started = time.perf_counter()
                result = engine.evaluate(pack["topology"], sorted(positions), baseline)
                compute_elapsed = time.perf_counter() - compute_started
                cache_info = {"cache_used": False, "elapsed_seconds": compute_elapsed,
                              "original_compute_seconds": compute_elapsed,
                              "cache_file": None, "cache_disabled": True}
            if checkpoint is None:
                rows = api["make_result"](evaluation, baseline, result, set()).drop(columns="selected")
                rows.to_csv(output / result_file, index=False, encoding="utf-8-sig")
            if m == 0:
                errors = api["check_no_governance"](evaluation, rows)
                save(output / f"T{period}/zero_restoration_validation.json", {"status": "PASS", "period": period,
                     "maximum_absolute_error": errors, "reachable_counts_match": True})
            if checkpoint is None:
                timings = {} if cache_info["cache_used"] else result["timings"]
            records.append({"stage": "Continue completed scenario" if checkpoint is not None else "Recalculate road restoration and grid risks", "period": period,
                            "target_treated_length_pct": plan["target_treated_length_pct"],
                            "planned_road_count": m, "elapsed_seconds": time.perf_counter() - scene_start,
                            "cache_used": cache_info["cache_used"], "cache_record": cache_info,
                            "checkpoint_reused": checkpoint is not None,
                            "source_original_compute_seconds": checkpoint["source_original_compute_seconds"] if checkpoint else None,
                            "network_update_seconds": timings.get("network_update_seconds", 0),
                            "routing_seconds": timings.get("routing_seconds", 0), "efar_seconds": timings.get("efar_seconds", 0)})
            scenes.append({**plan, "restored_road_count": len(roads),
                  "restored_length_m": math.fsum(map(float, roads.length_m)),
                  "restored_inundated_length_m": math.fsum(map(float, roads.inundated_length_m)),
                  "actual_restored_road_hash": road_set_hash, "restored_roads_file": restored_file,
                  "result_file": result_file, "cache_used": cache_info["cache_used"],
                  "checkpoint_reused": checkpoint is not None,
                  "source_run": str(resume.source) if checkpoint else None,
                  "source_model_key": checkpoint["proof"]["source_model_key"] if checkpoint else None,
                  "checkpoint_validation": "PASS" if checkpoint else None,
                  "reused_from_plan_index": None,
                  "result_sha256": signature(output / result_file)["sha256"],
                  "restored_roads_sha256": signature(output / restored_file)["sha256"],
                  "planned_roads_sha256": signature(output / planned_file)["sha256"], "model_key": model_key})
            manifest_file = output / f"T{period}/scenario_inventory.csv"
            manifest_temporary = manifest_file.with_suffix(".csv.tmp")
            pd.DataFrame(scenes).to_csv(manifest_temporary, index=False, encoding="utf-8-sig")
            manifest_temporary.replace(manifest_file)
            if resume:
                proof_path = output / "inputs/resume_validation.json"
                save(proof_path, resume.report())
                configuration["resume_checkpoint_manifest_signature"] = signature(proof_path)
                save(output / "run_config.json", configuration)
            save(output / 'calculation_runtime.json', {"records": records, "run_wall_seconds": time.perf_counter() - started,
                 "final_statistics_computed": False, "timing_note": 'Child timings are included in parent stages and must not be added twice. Current checkpoint verification/copying is separate from historical computation; checkpoint reuse is not current routing recalculation.'})
            previous = positions
            action = 'verified checkpoint transfer' if checkpoint is not None else 'network recalculation'
            print(f'  T{period} length target {plan['target_treated_length_pct']}%; actual {plan['treated_length_pct']:.6f}%; selected {m}/{len(priority)} roads: {action}', flush=True)
        completed_periods.append({"period": period, "scenario_count": len(scenes),
                                  "candidate_road_count": len(priority), "elapsed_seconds": time.perf_counter() - period_start})
    for record in [*source_signatures, *scripts.values()]:
        if signature(record["path"]) != record:
            raise RuntimeError("")
    if resume:
        resume.verify_frozen()
    output_records = [signature(path) for path in sorted(output.rglob("*.csv"))]
    output_records += [signature(output / "run_config.json")]
    output_records += [signature(path) for path in sorted((output / "inputs").glob("*.json"))]
    output_records += [signature(path) for path in sorted((output / "plans").glob("*.json"))]
    scenario_count = sum(item["scenario_count"] for item in completed_periods)
    routing_recomputed_count = sum(not row["cache_used"] and not row.get("checkpoint_reused", False)
                                   for row in records if "planned_road_count" in row)
    checkpoint_reused_count = sum(bool(row.get("checkpoint_reused", False))
                                  for row in records if "planned_road_count" in row)
    routing_cache_hit_count = sum(bool(row["cache_used"]) for row in records if "planned_road_count" in row)
    _require(scenario_count == len(periods) * len(TARGET_PCTS)
             and routing_recomputed_count + checkpoint_reused_count + routing_cache_hit_count == scenario_count,
             "")
    _require(routing_cache_hit_count == 0, "")
    save(output / "completion.json", {"status": "PASS", "strategy": STRATEGY, "periods": list(periods),
         "scenario_count": scenario_count,
         "period_details": completed_periods, "source_files_unchanged": True,
         "output_signatures": output_records, "cross_period_population_and_baseline_exactly_identical": True,
         "routing_recomputed_count": routing_recomputed_count,
         "checkpoint_reused_count": checkpoint_reused_count,
         "routing_cache_hit_count": routing_cache_hit_count,
         "run_wall_seconds": time.perf_counter() - started, "completed_at": now(), "statistics_stage": "not_started"})
    print(f'All cell-level network calculations completed: {output}\nRun summarize.py separately with --run-dir pointing to this directory, then use the appropriate plotting entry.', flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=tuple(CITIES), default="fuzhou")
    parser.add_argument("--evaluation-scope", choices=("study_area", "district"), default="study_area")
    parser.add_argument("--district", default="台江区")
    parser.add_argument("--periods", type=int, nargs="+", default=list(PERIODS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path, help='Read the old run directory, rebuild and verify the model, then transfer complete final-result checkpoints into a new directory.')
    run(parser.parse_args())
