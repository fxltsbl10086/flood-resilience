"Run 500 m SAR: read inputs, construct access/network graphs, route and export."

import json
from datetime import datetime
from time import perf_counter

import numpy as np
import pandas as pd

from config import CONFIG
from functions import read_inputs, select_origins, build_candidates, compute_accessibility
from reachability import build_network, compute_distances


def candidate_counts(network, scenario, n_grids):
    "Use counts only to distinguish missing access from network disconnection."
    frame = network["available_candidates"][scenario]
    return np.bincount(frame.grid_index.to_numpy(dtype=int), minlength=n_grids)


def get_distances(network, indices, scenario, batch_size):
    "Keep only the current batch in memory, avoiding a city-wide OD matrix."
    blocks = [distances for _, distances in compute_distances(
        network, indices, scenario, batch_size=batch_size
    )]
    return np.vstack(blocks)


def export_od(stream, grids, indices, distances, scenario, target_indices,
              baseline_mask, available_counts, header):
    "Represent inaccessibility with missing distances plus status fields; propagate calculation errors."
    records = []
    for row, origin_index in enumerate(indices):
        targets = target_indices[target_indices != origin_index]
        distance = distances[row, targets]
        reachable = np.isfinite(distance)
        state = np.where(reachable, "reachable", "no_path").astype(object)
        if available_counts[origin_index] == 0:
            state[:] = "origin_no_access"
        else:
            state[available_counts[targets] == 0] = "target_no_access"
        state[reachable & (distance == 0)] = "zero_distance"
        records.append(pd.DataFrame({
            "origin_grid_id": grids.iloc[origin_index].grid_id,
            "target_grid_id": grids.iloc[targets].grid_id.to_numpy(),
            "scenario": scenario,
            "distance_m": np.where(reachable, distance, np.nan),
            "reachable": reachable,
            "reachable_state": state,
            "in_baseline_targets": baseline_mask[row, targets],
        }))
    frame = pd.concat(records, ignore_index=True)
    frame.to_csv(stream, index=False, header=header, float_format="%.12g")


def run_phase(grids, network, origins, output, cfg, check_no_closure=False,
              baseline_cache=None, save_cache=False):
    "Calculate baseline distances once per origin batch and reuse them across scenarios."
    output.mkdir(parents=True, exist_ok=False)
    n_grids = len(grids)
    baseline_counts = candidate_counts(network, "baseline", n_grids)
    target_indices = np.flatnonzero(baseline_counts > 0)
    if not len(target_indices):
        raise ValueError()
    counts = {f"T{period}": candidate_counts(network, f"T{period}", n_grids)
              for period in cfg["PERIODS"]}
    cache = baseline_cache if baseline_cache is not None else {}
    risk_blocks = []
    header = True
    with (output / "grid_od_distance.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        for start in range(0, len(origins), cfg["BATCH_SIZE"]):
            indices = np.asarray(origins[start:start + cfg["BATCH_SIZE"]], dtype=int)
            baseline = np.empty((len(indices), n_grids))
            missing = [int(index) for index in indices if int(index) not in cache]
            if missing:
                computed = get_distances(network, np.asarray(missing), "baseline", cfg["BATCH_SIZE"])
                missing_lookup = {index: row for row, index in enumerate(missing)}
            for row, index in enumerate(indices):
                baseline[row] = cache[int(index)] if int(index) in cache else computed[missing_lookup[int(index)]]
                if save_cache:
                    cache[int(index)] = baseline[row].copy()
            base_result = compute_accessibility(baseline, baseline, indices)
            base_valid = base_result["baseline_valid"]
            base_state = np.where(base_valid, "valid", "no_baseline_target")
            base_state[baseline_counts[indices] == 0] = "no_baseline_access"
            original_columns = [column for column in
                                ["grid_id", "row", "col", "cell_m", "area_m2", "int_area", "int_ratio",
                                 "centroid_x", "centroid_y"] if column in grids]
            risk = grids.iloc[indices][original_columns].reset_index(drop=True).copy()
            risk["access_candidate_count"] = baseline_counts[indices]
            risk["baseline_target_count"] = base_result["baseline_target_count"]
            risk["baseline_accessibility"] = base_result["baseline_accessibility"]
            risk["baseline_valid"] = base_valid
            risk["baseline_state"] = base_state
            export_od(stream, grids, indices, baseline, "baseline", target_indices,
                      base_result["baseline_mask"], baseline_counts, header)
            header = False

            if check_no_closure:
                no_closure = get_distances(network, indices, "no_closure", cfg["BATCH_SIZE"])
                if not np.array_equal(np.isfinite(baseline), np.isfinite(no_closure)):
                    raise AssertionError()
                finite = np.isfinite(baseline)
                if not np.allclose(baseline[finite], no_closure[finite], rtol=0,
                                   atol=cfg["DISTANCE_TOLERANCE_M"]):
                    raise AssertionError()
                unchanged = compute_accessibility(baseline, no_closure, indices)["SAR"]
                error = float(np.max(np.abs(unchanged[base_valid]))) if base_valid.any() else 0.0
                if error > cfg["RISK_TOLERANCE"]:
                    raise AssertionError()

            for period in cfg["PERIODS"]:
                scenario = f"T{period}"
                distance = get_distances(network, indices, scenario, cfg["BATCH_SIZE"])
                finite_base = np.isfinite(baseline)
                finite_scene = np.isfinite(distance)
                newly_reachable = int((~finite_base & finite_scene).sum())
                both = finite_base & finite_scene
                shortening = float(np.max(baseline[both] - distance[both])) if both.any() else 0.0
                if newly_reachable or shortening > cfg["DISTANCE_TOLERANCE_M"]:
                    raise AssertionError()
                result = compute_accessibility(baseline, distance, indices)
                sar = result["SAR"]
                if np.any(sar[base_valid] < -cfg["RISK_TOLERANCE"]) or np.any(
                        sar[base_valid] > 1 + cfg["RISK_TOLERANCE"]):
                    raise AssertionError()
                state = np.where(result["scenario_reachable_count"] > 0, "valid", "all_targets_unreachable")
                state[counts[scenario][indices] == 0] = "origin_no_access"
                state[~base_valid] = "baseline_invalid"
                risk[f"accessibility_T{period}"] = result["scenario_accessibility"]
                risk[f"SAR_T{period}"] = sar
                risk[f"reachable_count_T{period}"] = result["scenario_reachable_count"]
                risk[f"state_T{period}"] = state
                export_od(stream, grids, indices, distance, scenario, target_indices,
                          base_result["baseline_mask"], counts[scenario], header=False)
            risk_blocks.append(risk)
            if start == 0 or start + len(indices) == len(origins) or (start // cfg["BATCH_SIZE"] + 1) % 10 == 0:
                print(f"  Origins completed {min(start + len(indices), len(origins))}/{len(origins)}", flush=True)

    risk = pd.concat(risk_blocks, ignore_index=True)
    risk.to_csv(output / "grid_risk.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    valid = risk.baseline_valid.to_numpy(dtype=bool)
    # Summarize with equal grid weights; this stage does not read population.
    summary = pd.DataFrame([{
        "city": cfg["CITY"], "test_area": cfg.get("ANALYSIS_AREA", cfg["DISTRICT_NAME"]), "scenario": f"T{period}",
        "origin_grid_count": len(risk), "baseline_valid_grid_count": int(valid.sum()),
        "grid_coverage_rate": float(valid.mean()), "target_grid_count": len(target_indices),
        "grid_mean_SAR": float(risk.loc[valid, f"SAR_T{period}"].mean()),
        "all_targets_unreachable_origin_count": int((valid & (risk[f"reachable_count_T{period}"] == 0)).sum()),
    } for period in cfg["PERIODS"]])
    summary.to_csv(output / "regional_summary.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    sar_columns = [f"SAR_T{period}" for period in cfg["PERIODS"]]
    invalid_sar_blank = risk.loc[~valid, sar_columns].isna().all().all()
    valid_sar_complete = risk.loc[valid, sar_columns].notna().all().all()
    if not invalid_sar_blank or not valid_sar_complete:
        raise AssertionError()
    return summary


def run(cfg=None):
    cfg = CONFIG.copy() if cfg is None else cfg.copy()
    scope = cfg.get("ORIGIN_SCOPE", "district")
    area = "全研究区" if scope == "city" else cfg["DISTRICT_NAME"]
    cfg["ANALYSIS_AREA"] = area
    area_path = "full_study_area" if scope == "city" else "district"
    started = perf_counter()
    output = cfg["OUTPUT_DIR"] / f"{cfg['CITY']}_{area_path}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "running", "config": {key: str(value) if hasattr(value, "__fspath__") else value
                                                for key, value in cfg.items()}}
    try:
        print("Reading existing grids, road-inundation results and administrative boundary...", flush=True)
        grids, roads, edges, boundary, district = read_inputs(cfg)
        print("Constructing grid-boundary access candidates inside the study area...", flush=True)
        candidates = build_candidates(grids, roads, boundary, cfg)
        candidate_qa = candidates.attrs.get("qa", {})
        print(f"Grids {len(grids)}; facility segments {len(roads)}; candidate pairs {len(candidates)}", flush=True)
        print("Splitting roads once and building baseline and flood-scenario graphs...", flush=True)
        network = build_network(roads, edges, candidates, boundary, len(grids), cfg["PERIODS"])
        origins = select_origins(grids, boundary, district)
        if not len(origins):
            raise ValueError()
        # Prefer smoke-test origins with candidates so the no-closure check has valid comparisons.
        accessible = network["available_candidates"]["baseline"].grid_index.unique()
        smoke = origins[np.isin(origins, accessible)][:cfg["TEST_ORIGIN_LIMIT"]]
        cache = {}
        print(f"Smoke test: {len(smoke)} origins and all study-area destinations.", flush=True)
        run_phase(grids, network, smoke, output / "small_sample", cfg, check_no_closure=True,
                  baseline_cache=cache, save_cache=True)
        if cfg.get("RUN_FULL_AFTER_TEST", cfg.get("RUN_DISTRICT_AFTER_TEST", True)):
            print(f"Smoke test passed; computing all {len(origins)} origins in {area}.", flush=True)
            summary = run_phase(grids, network, origins, output / area_path, cfg,
                                baseline_cache=cache)
            print(summary.to_string(index=False), flush=True)
        record.update(status="success", grid_count=len(grids), origin_scope=scope,
                      origin_count=len(origins), analysis_area=area,
                      candidate_qa=candidate_qa, network_qa=network.get("qa", {}))
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["elapsed_seconds"] = perf_counter() - started
        (output / "run_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"Output directory: {output}", flush=True)
    return output


if __name__ == "__main__":
    run()
