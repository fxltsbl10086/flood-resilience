"""Road-ordering controls with fixed original project mappings. Calculate network states only; run summarize_comparison.py separately for benefits."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

from comparison_common import (ROOT, SOURCE_ROOT, CITY_CODES, CITY_NAMES, PERIODS,
    TARGET_PCTS, STRATEGIES, atomic_json, atomic_csv, atomic_npz, read_json, sha256,
    canonical_hash, road_hash, wet_pool, population_priority, random_priority,
    load_state, save_state, assert_monotone)
from plan import build_length_plan


def log(message):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def routing_progress(message):
    """Log only start and completion per direction to avoid excessive batch logs."""
    position = message.rsplit(": ", 1)[-1].split("/")
    if len(position) == 2 and (position[0] == "1" or position[0] == position[1]):
        log(message)


def signature(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def record_timing(run_dir, record):
    with (run_dir / "run_times.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": datetime.now().isoformat(), **record}, ensure_ascii=False) + "\n")
        stream.flush()


def save_baseline(case_dir, context):
    base = context["baseline"]
    evaluation = context["pack"]["evaluation_all"]
    evaluation = evaluation.loc[evaluation.efarb_valid].reset_index(drop=True).copy()
    np.testing.assert_array_equal(evaluation.grid_index.to_numpy(), base["resident_indices"])
    arrays = {"grid_id": evaluation.grid_id.astype(str).to_numpy(dtype=str),
              "population": evaluation.effective_population.to_numpy(float),
              "base_inverse": base["base_inverse"], "base_count": base["base_count"]}
    path = case_dir / "evaluation_baseline.npz"
    if (case_dir / "baseline_validation.json").exists():
        old_audit = read_json(case_dir / "baseline_validation.json")
        if old_audit["model_key"] != context["model_key"]:
            raise ValueError("")
        for item in old_audit["files"]:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("")
        with np.load(path, allow_pickle=False) as saved:
            for field, value in arrays.items():
                np.testing.assert_array_equal(saved[field], value)
    else:
        atomic_npz(path, **arrays)
        atomic_csv(case_dir / "evaluation_baseline.csv", evaluation)
        atomic_json(case_dir / "baseline_validation.json", {
            "model_key": context["model_key"],
            "files": [signature(path), signature(case_dir / "evaluation_baseline.csv")],
            "source_signatures": context["source_signatures"],
            "context_audit": context.get("audit", {}),
            "zero_population_valid_grids_retained": True})
    return evaluation


def compare_reference(context, city, period, target, arrays, source_root):
    if target not in (0, 100):
        return False
    index = list(TARGET_PCTS).index(target)
    path = source_root / city / f"T{period}" / f'step{index:02d}_target{target:03d}pct_grid_results.csv'
    source = pd.read_csv(path, dtype={"grid_id": str})
    evaluation = context["pack"]["evaluation_all"]
    ids = evaluation.loc[evaluation.efarb_valid].grid_id.astype(str)
    if source.grid_id.duplicated().any() or set(source.grid_id) != set(ids):
        raise ValueError("")
    source = source.set_index("grid_id").loc[ids]
    population = evaluation.loc[evaluation.efarb_valid].effective_population.to_numpy(float)
    np.testing.assert_allclose(population, source.effective_population, atol=1e-8, rtol=0)
    for k, name in enumerate(("rescue", "medical", "shelter")):
        np.testing.assert_allclose(context["baseline"]["base_inverse"][:, k], source[f"base_inverse_{name}"],
                                   atol=1e-13, rtol=1e-12)
        np.testing.assert_array_equal(context["baseline"]["base_count"][:, k], source[f"base_count_{name}"])
        np.testing.assert_allclose(arrays["risk"][:, k], source[f"EFAR_{name}"], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(arrays["reachable_count"][:, k], source[f"reachable_count_{name}"])
    return True


def prepare_ranking(case_dir, context, strategy, city, period, replicate, seed):
    inventory, contacts = context["inventory"], context["contacts"]
    pool = wet_pool(inventory)
    if set(pool.road_id) != set(context["original_priority"].road_id.astype(str)):
        raise ValueError("")
    pool_path = case_dir / "common_candidate_roads.csv"
    if not pool_path.exists():
        atomic_csv(pool_path, pool)
    else:
        old_pool = pd.read_csv(pool_path, dtype={"road_id": str}, float_precision="round_trip")
        pd.testing.assert_frame_equal(old_pool, pool, check_dtype=False, check_exact=True)
    if strategy == "population_priority":
        ranking = population_priority(inventory, contacts, context["original_evaluation"])
        ranking_path = case_dir / "fixed_flood_exposure_population_ranking.csv"
        if ranking_path.exists():
            old = pd.read_csv(ranking_path, dtype={"road_id": str}, float_precision="round_trip")
            pd.testing.assert_frame_equal(old, ranking, check_dtype=False, check_exact=True)
        else:
            atomic_csv(ranking_path, ranking)
    else:
        ranking = random_priority(inventory, city, period, replicate, seed)
        ranking_path = case_dir / "permutations" / f"repeat{replicate:03d}.npz"
        if ranking_path.exists():
            with np.load(ranking_path, allow_pickle=False) as old:
                np.testing.assert_array_equal(old["road_position"], ranking.road_position.to_numpy(np.int64))
        else:
            atomic_npz(ranking_path, road_position=ranking.road_position.to_numpy(np.int64))
    plan = build_length_plan(ranking, inventory, period)
    plan["strategy"] = strategy
    plan["city"] = city
    plan["replicate_id"] = replicate
    plan["model_key"] = context["model_key"]
    plan["ranking_file"] = str(ranking_path.resolve())
    plan["pool_hash"] = road_hash(pool.road_id)
    # Rebuild each prefix from fixed ordering and counts rather than saving 23 duplicate road tables.
    plan = plan.drop(columns=["planned_roads_file"])
    atomic_csv(case_dir / "plans" / f"repeat{replicate:03d}_length_targets.csv", plan)
    return ranking, plan


def upsert_record(path, records, row):
    records[(int(row["replicate_id"]), int(row["target_treated_length_pct"]))] = row
    atomic_csv(path, pd.DataFrame([records[key] for key in sorted(records)]))


def run_case(args, city, period):
    from comparison_context import load_comparison_context
    started = time.perf_counter()
    case_dir = args.run_dir / 'cases' / city / f"T{period}"
    case_dir.mkdir(parents=True, exist_ok=True)
    log(f'Preparing {city.title()} T{period}')
    context = load_comparison_context(city, period, args.source_root, args.cache_root, progress=log)
    evaluation = save_baseline(case_dir, context)
    record_timing(args.run_dir, {"stage": "context_preparation", "city": city, "period": period,
        "elapsed_seconds": time.perf_counter() - started, "cache_used": context["cache_used"]})
    records_path = case_dir / "state_record.csv"
    records = {}
    if records_path.exists():
        for row in pd.read_csv(records_path).to_dict("records"):
            records[(int(row["replicate_id"]), int(row["target_treated_length_pct"]))] = row
    repetitions = range(args.repeat_start, args.repeat_start + args.repeat_count) if args.strategy == "random_priority" else (0,)
    baseline = context["baseline"]
    api = context["api"]
    for replicate in repetitions:
        replicate_start = time.perf_counter()
        ranking, plan = prepare_ranking(case_dir, context, args.strategy, city, period, replicate, args.seed)
        previous = None
        previous_ids = set()
        for row in plan.to_dict("records"):
            target = int(row["target_treated_length_pct"])
            if target not in args.targets:
                continue
            row_start = time.perf_counter()
            prefix = ranking.iloc[:int(row["planned_road_count"]) ]
            ids = list(prefix.road_id.astype(str))
            if not previous_ids <= set(ids) or road_hash(ids) != row["planned_road_set_hash"]:
                raise ValueError("")
            existing = records.get((replicate, target))
            row.update({"status": "running", "model_key": context["model_key"], "error": ""})
            if not existing or existing.get("status") != "complete":
                upsert_record(records_path, records, row)
            try:
                cached = load_state(args.cache_root, context["model_key"], ids, len(evaluation))
                if existing and existing.get("status") == "complete":
                    if existing.get("model_key") != context["model_key"] or existing.get("planned_road_set_hash") != row["planned_road_set_hash"]:
                        raise ValueError("")
                    if cached is None or existing.get("state_sha256") != cached[1]["sha256"]:
                        raise ValueError("")
                if cached is None:
                    log(f'{city.title()} T{period} repeat {replicate:03d} target {target}%: recalculating {len(ids)} roads')
                    result = api["engine"].evaluate(context["pack"]["topology"],
                        prefix.road_position.to_numpy(np.int64), baseline, progress=routing_progress)
                    arrays, metadata = save_state(args.cache_root, context["model_key"], ids, result, len(evaluation))
                    cache_used = False
                else:
                    arrays, metadata = cached
                    cache_used = True
                if previous is not None:
                    assert_monotone(previous, arrays)
                reference_verified = compare_reference(context, city, period, target, arrays, args.source_root)
                verification = False
                # Always compare original-strategy 0/100 endpoints cell by cell; independently recompute sampled nontrivial 30% states.
                # Do not reroute a verified endpoint with the same road set merely because its ordering fields changed.
                if args.verify_cache and target == 30 and replicate == args.repeat_start:
                    log(f'Independent recalculation check: {city} T{period} {target}%')
                    fresh = api["engine"].evaluate(context["pack"]["topology"], prefix.road_position.to_numpy(np.int64), baseline,
                                                   progress=routing_progress)
                    for field in ("risk", "scene_inverse", "reachable_count", "nearest_distance"):
                        np.testing.assert_array_equal(fresh[field], arrays[field])
                    verification = True
                row.update({"status": "complete", "state_path": metadata["path"], "state_sha256": metadata["sha256"],
                    "state_key": metadata["state_key"], "cache_used": cache_used,
                    "reference_endpoints_verified": reference_verified,
                    "independent_recompute_verified": verification or bool(existing and existing.get("independent_recompute_verified") is True),
                    "elapsed_seconds": time.perf_counter() - row_start,
                    "original_network_compute_seconds": sum(metadata.get("timings", {}).values())})
                upsert_record(records_path, records, row)
                previous, previous_ids = arrays, set(ids)
                record_timing(args.run_dir, {"stage": "network_state", "city": city, "period": period,
                    "replicate_id": replicate, "target_pct": target, "cache_used": cache_used,
                    "state_key": metadata["state_key"], "elapsed_seconds": row["elapsed_seconds"]})
                log(f'Completed {city} T{period} rep={replicate} target={target}% actual={row['treated_length_pct']:.6f}% cache={cache_used}')
            except Exception as error:
                row.update({"status": "failed", "error": repr(error), "elapsed_seconds": time.perf_counter() - row_start})
                upsert_record(records_path, records, row)
                raise
        record_timing(args.run_dir, {"stage": "replicate_total", "city": city, "period": period,
            "replicate_id": replicate, "elapsed_seconds": time.perf_counter() - replicate_start})
    return {"city": city, "period": period, "status": "complete", "requested_targets": args.targets,
            "requested_replicates": list(repetitions), "elapsed_seconds": time.perf_counter() - started}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache/strategy_comparison")
    parser.add_argument("--cities", nargs="+", choices=list(CITY_CODES), default=list(CITY_CODES))
    parser.add_argument("--periods", nargs="+", type=int, choices=PERIODS, default=list(PERIODS))
    parser.add_argument("--targets", nargs="+", type=int, choices=TARGET_PCTS, default=list(TARGET_PCTS))
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--planned-repeats", type=int, default=100)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-cache", action="store_true", help='Independently recalculate the first repeat at the 30 percent target; always check 0/100 endpoints against original results')
    args = parser.parse_args(argv)
    if args.seed < 0 or args.repeat_start < 0 or args.repeat_count < 1 or args.planned_repeats < args.repeat_start + args.repeat_count:
        parser.error("")
    args.targets = sorted(set(args.targets))
    args.cities = list(dict.fromkeys(args.cities))
    args.periods = sorted(set(args.periods))
    if args.strategy == "population_priority":
        args.repeat_start, args.repeat_count, args.planned_repeats = 0, 1, 1
    if args.run_dir is None:
        args.run_dir = ROOT / "outputs" / args.strategy / f"{datetime.now():%Y%m%d_%H%M%S}"
    args.run_dir = args.run_dir.resolve()
    args.source_root = args.source_root.resolve()
    args.cache_root = args.cache_root.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    config = {"schema_version": 1, "strategy": args.strategy, "cities": list(CITY_CODES),
        "city_codes": CITY_CODES, "periods": list(PERIODS), "targets": list(TARGET_PCTS),
        "planned_repeats": args.planned_repeats, "seed": args.seed,
        "source_root": str(args.source_root), "cache_root": str(args.cache_root),
        "rng": "PCG64/SeedSequence", "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "fixed_mapping_comparison": True,
        "population_score_field": "population_exposed" if args.strategy == "population_priority" else None,
        "population_score_definition": ("sum of scenario-specific exposed population after road-grid deduplication"
                                        if args.strategy == "population_priority" else None),
        "evaluation_population_field": "effective_population",
        "length_rule": "first complete-road prefix reaching target; original 1e-6 m tolerance",
        "statistics_in_separate_script": True,
        "implementation_fingerprint": canonical_hash({name: sha256(Path(__file__).parent / name)
            for name in ("run_comparison.py", "comparison_common.py", "comparison_context.py")})}
    config_path = args.run_dir / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise ValueError("")
        if read_json(config_path) != config:
            raise ValueError("")
    else:
        atomic_json(config_path, config)
    lock_path = args.run_dir / "run_lock.lock"
    with lock_path.open("a+b") as lock:
        lock.seek(0)
        if lock.read(1) == b"":
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        invocation = {"started": datetime.now().isoformat(), "strategy": args.strategy,
            "cities": args.cities, "periods": args.periods, "targets": args.targets,
            "repeat_start": args.repeat_start, "repeat_count": args.repeat_count,
            "status": "running", "cases": [], "command": sys.argv,
            "scripts": [signature(Path(__file__)), signature(Path(__file__).parent / "comparison_common.py"),
                        signature(Path(__file__).parent / "comparison_context.py")]}
        invocation_path = args.run_dir / 'invocations' / f"{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.json"
        atomic_json(invocation_path, invocation)
        try:
            for city in args.cities:
                for period in args.periods:
                    invocation["cases"].append(run_case(args, city, period))
                    atomic_json(invocation_path, invocation)
            invocation["status"] = "complete_requested_subset"
            invocation["finished"] = datetime.now().isoformat()
            atomic_json(invocation_path, invocation)
            log(f'Requested scope completed: {args.run_dir}; run the independent statistics script for final summaries')
        except BaseException as error:
            invocation.update({"status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                               "finished": datetime.now().isoformat()})
            atomic_json(invocation_path, invocation)
            raise


if __name__ == "__main__":
    main()
