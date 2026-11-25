"""Calculate population-weighted essential-facility accessibility risk burden (EFARB), retaining all original grid cells.

EFARB_i,T = population_i * mean(EFAR_rescue_i,T, EFAR_medical_i,T,
                              EFAR_shelter_i,T)

Use the formal result-two EFAR without renormalization, inundated-population filtering or missing-value imputation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSION_KEYS = {"应急救援": "rescue", "医疗资源": "medical", "避险转移": "shelter"}
CITY_NAMES_ZH = {"chengdu": "成都", "zhengzhou": "郑州", "hangzhou": "杭州",
                 "wuhan": "武汉", "fuzhou": "福州"}
POPULATION_STATUSES = {"complete", "partial", "no_data"}


def _require(condition, message):
    if not condition:
        raise ValueError("")


def file_signature(path):
    """Return a traceable input signature computed from the original file bytes."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_city(city, inputs, periods, dimensions):
    metadata = json.loads(Path(inputs["risk_metadata"]).read_text(encoding="utf-8-sig"))
    _require(metadata.get("metric") == "inverse" and metadata.get("local") == "include"
             and metadata.get("same_grid_distance_m") == 250
             and metadata.get("periods") == list(periods),
             "")
    if "cities" in metadata:
        _require(city in metadata["cities"], "")
    population = pd.read_csv(inputs["population_csv"], encoding="utf-8-sig", dtype={"grid_id": str})
    risk = pd.read_csv(inputs["risk_csv"], encoding="utf-8-sig", dtype={"grid_id": str})
    required_pop = {"grid_id", "city", "row", "col", "cell_m", "centroid_x", "centroid_y",
                    "population", "population_status", "population_data_coverage_ratio"}
    required_risk = {"city", "grid_id", "评估维度", "状态", "指标", "同网格处理",
                     "同网格距离m", "centroid_x", "centroid_y",
                     *[f"风险_T{t:03d}" for t in periods]}
    _require(required_pop.issubset(population.columns), "")
    _require(required_risk.issubset(risk.columns), "")
    ids = population.grid_id
    _require(not ids.isna().any() and not ids.str.strip().eq("").any() and ids.is_unique,
             "")
    _require(population.city.eq(city).all() and risk.city.eq(city).all(), "")
    _require(not population.duplicated(["row", "col"]).any(), "")
    _require(np.allclose(population.cell_m, 500, rtol=0, atol=1e-6),
             "")
    for coord in ("centroid_x", "centroid_y"):
        _require(np.isfinite(population[coord].to_numpy(float)).all(), "")
    _require(set(risk["评估维度"]) == set(dimensions), "")
    _require(not risk.duplicated(["grid_id", "评估维度"]).any(), "")
    _require(risk["指标"].eq("inverse").all() and risk["同网格处理"].eq("include").all()
             and risk["同网格距离m"].eq(250).all(), "")

    p = pd.to_numeric(population.population, errors="raise").to_numpy(float)
    known = ~np.isnan(p)
    _require(np.isfinite(p[known]).all() and (p[known] >= 0).all(), "")
    _require(population.population_status.isin(POPULATION_STATUSES).all(), "")
    _require(np.array_equal(known, population.population_status.isin(["complete", "partial"])),
             "")
    coverage = population.population_data_coverage_ratio.to_numpy(float)
    _require(np.isfinite(coverage).all() and (coverage >= 0).all() and (coverage <= 1).all(),
             "")

    # Retain cells, population and coverage; direct-exposure fields neither define this metric nor enter its output table.
    retained = [c for c in population.columns if not c.startswith(
        ("population_exposed_", "population_nonexposed_", "exposed_ratio_"))]
    data = population[retained].copy()
    baseline_flags, parts = {}, {}
    for dimension in dimensions:
        key = DIMENSION_KEYS[dimension]
        part = risk.loc[risk["评估维度"] == dimension].set_index("grid_id")
        _require(set(part.index) == set(ids), "")
        part = part.loc[ids]
        for coord in ("centroid_x", "centroid_y"):
            _require(np.allclose(part[coord], population[coord], rtol=0, atol=1e-5),
                     "")
        _require(part["状态"].isin(["有效", "基准无可达资源"]).all(),
                 "")
        valid = part["状态"].eq("有效").to_numpy()
        values = part[[f"风险_T{t:03d}" for t in periods]].to_numpy(float)
        _require(np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all(),
                 "")
        _require((values[valid] >= 0).all() and (values[valid] <= 1).all(),
                 "")
        if "基准加权可达水平" in part:
            base = part["基准加权可达水平"].to_numpy(float)
            _require(np.isfinite(base[valid]).all() and (base[valid] > 0).all()
                     and (np.isnan(base[~valid]) | (base[~valid] == 0)).all(),
                     "")
        data[f"baseline_valid_{key}"] = valid
        baseline_flags[key], parts[key] = valid, values
        for j, period in enumerate(periods):
            data[f"EFAR_{key}_T{period}"] = values[:, j]

    baseline_valid = np.logical_and.reduce(list(baseline_flags.values()))
    valid = known & baseline_valid
    partial = population.population_status.eq("partial").to_numpy()
    data["service_baseline_valid"] = baseline_valid
    data["population_known"] = known
    data["population_partial"] = partial
    data["efarb_valid"] = valid
    data["effective_population"] = np.where(valid, p, np.nan)
    data["efarb_status"] = np.select(
        [~known & ~baseline_valid, ~known, ~baseline_valid, valid & (p == 0), valid & partial],
        ["population_unknown_and_baseline_invalid", "population_unknown", "baseline_invalid",
         "valid_zero_known_population", "valid_partial_known_population"],
        default="valid_complete_population")
    checks = []
    for j, period in enumerate(periods):
        all_risks = np.column_stack([parts[key][:, j] for key in DIMENSION_KEYS.values()])
        mean = np.full(len(data), np.nan)
        mean[baseline_valid] = all_risks[baseline_valid].sum(axis=1) / 3.0
        burden = np.full(len(data), np.nan)
        burden[valid] = p[valid] * mean[valid]
        data[f"EFAR_mean_T{period}"] = mean
        data[f"EFARB_T{period}"] = burden
        _require(np.isfinite(mean[baseline_valid]).all() and np.isnan(mean[~baseline_valid]).all(),
                 "")
        _require(np.isfinite(burden[valid]).all() and np.isnan(burden[~valid]).all(),
                 "")
        _require((burden[valid] >= 0).all() and (burden[valid] <= p[valid]).all(),
                 "")
        _require((burden[valid & (p == 0)] == 0).all(), "")
        residual = np.abs(burden[valid] - p[valid] * all_risks[valid].mean(axis=1))
        checks.append({"city": city, "return_period": int(period), "status": "PASS",
                       "grid_count": len(data), "valid_grid_count": int(valid.sum()),
                       "missing_burden_count": int((~valid).sum()),
                       "valid_zero_population_count": int((valid & (p == 0)).sum()),
                       "formula_max_abs_error": float(residual.max()) if residual.size else 0.0,
                       "minimum": float(burden[valid].min()) if valid.any() else None,
                       "maximum": float(burden[valid].max()) if valid.any() else None})
    return data, checks


def prepare(output: Path, city_inputs: dict, city_order: list, periods: list,
            service_dimensions: list):
    """Return (frames, summary, coverage, metadata) and write CSV/JSON with the existing filenames.

frames contains all-grid wide tables indexed by city. EFAR_mean requires the three baseline-valid dimensions; EFARB additionally requires known population.
For partial population coverage, retain the known number of residents without extrapolation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _require(list(periods) == [10, 20, 50, 100, 200], "")
    _require(len(service_dimensions) == 3 and set(service_dimensions) == set(DIMENSION_KEYS),
             "")
    _require(bool(city_order) and len(set(city_order)) == len(city_order)
             and set(city_order).issubset(CITY_NAMES_ZH) and set(city_inputs) == set(city_order),
             "")
    scope = 'five_cities' if len(city_order) == 5 else 'selected_cities'
    output_names = [f"{scope}_grid_essential_facility_risk_burden_wide.csv",
                    f"{scope}_grid_essential_facility_risk_burden_long.csv",
                    f"{scope}_risk_burden_summary_by_period.csv", f"{scope}_valid_population_coverage.csv",
                    "data_sources_and_methods.json"]
    _require(not any((output / name).exists() for name in output_names),
             "")
    frames, summary_rows, coverage_rows, long_blocks, checks, sources = {}, [], [], [], [], {}
    summary_checks = []
    signature_cache = {}
    for city in city_order:
        inputs = city_inputs[city]
        # Sign inputs before numeric reads and recheck after calculation to prevent mixed input versions.
        source_signatures = {}
        for key in ("population_csv", "risk_csv", "risk_metadata"):
            path = Path(inputs[key]).resolve()
            if path not in signature_cache:
                signature_cache[path] = file_signature(path)
            source_signatures[key] = signature_cache[path]
        sources[city] = source_signatures
        frames[city], city_checks = _read_city(city, inputs, periods, service_dimensions)
        checks.extend(city_checks)
        data = frames[city]
        p = data.population.to_numpy(float)
        valid = data.efarb_valid.to_numpy(bool)
        known = data.population_known.to_numpy(bool)
        baseline = data.service_baseline_valid.to_numpy(bool)
        partial = data.population_partial.to_numpy(bool)
        known_total = float(p[known].sum())
        valid_total = float(p[valid].sum())
        _require(known_total > 0 and valid_total > 0, "")
        coverage_rows.append({
            "city": city, "城市": CITY_NAMES_ZH[city], "total_grid_count": len(data),
            "population_known_grid_count": int(known.sum()), "population_unknown_grid_count": int((~known).sum()),
            "population_partial_grid_count": int(partial.sum()),
            "three_service_baseline_valid_grid_count": int(baseline.sum()),
            "efarb_valid_grid_count": int(valid.sum()),
            "efarb_valid_positive_population_grid_count": int((valid & (p > 0)).sum()),
            "efarb_valid_zero_population_grid_count": int((valid & (p == 0)).sum()),
            "efarb_valid_partial_population_grid_count": int((valid & partial).sum()),
            "known_total_population": known_total, "analysis_population": valid_total,
            "excluded_known_population": known_total - valid_total,
            "known_population_coverage": valid_total / known_total,
            "partial_known_population": float(p[known & partial].sum()),
            "analysis_partial_known_population": float(p[valid & partial].sum()),
            **{f"baseline_valid_{key}_grid_count": int(data[f"baseline_valid_{key}"].sum())
               for key in DIMENSION_KEYS.values()}})
        id_columns = ["city", "grid_id", "centroid_x", "centroid_y", "population", "effective_population", "population_status",
                      "population_data_coverage_ratio", "baseline_valid_rescue", "baseline_valid_medical",
                      "baseline_valid_shelter", "service_baseline_valid", "population_known",
                      "population_partial", "efarb_valid", "efarb_status"]
        for period in periods:
            values = data[f"EFARB_T{period}"].to_numpy(float)[valid]
            means = data[f"EFAR_mean_T{period}"].to_numpy(float)[valid]
            maximum_index = data[f"EFARB_T{period}"].idxmax()
            total = float(values.sum())
            # Independently compute population-weighted means per risk dimension, then average to check the city identity.
            dimension_weighted_means = {
                key: float(np.sum(p[valid] * data[f"EFAR_{key}_T{period}"].to_numpy(float)[valid])
                           / valid_total)
                for key in DIMENSION_KEYS.values()}
            independent_mean = float(np.mean(list(dimension_weighted_means.values())))
            summary_mean = total / valid_total
            summary_error = abs(summary_mean - independent_mean)
            _require(np.isclose(summary_mean, independent_mean, rtol=1e-12, atol=1e-12),
                     "")
            summary_checks.append({
                "city": city, "return_period": int(period), "status": "PASS",
                "population_weighted_EFAR_by_dimension": dimension_weighted_means,
                "mean_of_three_population_weighted_EFAR": independent_mean,
                "EFARB_total_divided_by_analysis_population": summary_mean,
                "absolute_residual": summary_error,
            })
            summary_rows.append({
                "city": city, "城市": CITY_NAMES_ZH[city], "return_period": int(period), "scenario": f"T{period}",
                "total_grid_count": len(data), "included_grid_count": int(valid.sum()),
                "positive_population_grid_count": int((valid & (p > 0)).sum()),
                "analysis_population": valid_total, "known_total_population": known_total,
                "known_population_coverage": valid_total / known_total,
                "EFARB_total": total, "population_weighted_EFAR_mean": total / valid_total,
                "EFAR_mean_grid_average": float(means.mean()), "EFARB_grid_average": float(values.mean()),
                "EFARB_grid_median": float(np.median(values)), "EFARB_grid_maximum": float(values.max()),
                "maximum_grid_id": data.loc[maximum_index, "grid_id"],
                "zero_burden_grid_count": int((values == 0).sum())})
            block = data[id_columns].copy()
            block["return_period"] = int(period)
            block["scenario"] = f"T{period}"
            for key in DIMENSION_KEYS.values():
                block[f"EFAR_{key}"] = data[f"EFAR_{key}_T{period}"]
            block["EFAR_mean"] = data[f"EFAR_mean_T{period}"]
            block["EFARB"] = data[f"EFARB_T{period}"]
            long_blocks.append(block)
        print(f'{city.title()}: total {len(data):,} cells; valid EFARB in {int(valid.sum()):,} cells; evaluable known-population coverage {valid_total / known_total:.2%}', flush=True)

    summary, coverage = pd.DataFrame(summary_rows), pd.DataFrame(coverage_rows)
    wide, long = pd.concat(frames.values(), ignore_index=True), pd.concat(long_blocks, ignore_index=True)
    _require(len(long) == len(wide) * len(periods), "")
    source_stability_checks = []
    for path, before in signature_cache.items():
        after = file_signature(path)
        _require(before == after, "")
        source_stability_checks.append({"path": str(path), "status": "PASS",
                                        "sha256_before": before["sha256"], "sha256_after": after["sha256"]})
    metadata = {
        "name_zh": "关键设施可达风险负担", "name_en": "Essential Facility Accessibility Risk Burden",
        "abbreviation": "EFARB", "indicator_type": "人口加权的风险负担指标",
        "formula": "EFARB_i,T = population_i * (EFAR_rescue_i,T + EFAR_medical_i,T + EFAR_shelter_i,T) / 3",
        "service_mean_formula": "EFAR_mean_i,T = (EFAR_rescue_i,T + EFAR_medical_i,T + EFAR_shelter_i,T) / 3",
        "population_definition": 'population from the formal result-one 500 m population table: known valid total cell population, not inundated population.',
        "effective_population_definition": 'effective_population is known population in EFARB-evaluable cells and NaN elsewhere, corresponding to p_i in the formula.',
        "population_scope": 'Cells with known population and valid baselines for all three services; use a fixed set across five scenarios, including known zero-population cells.',
        "partial_population": 'For partial cells, retain the known population, coverage fraction and partial flag; do not scale by coverage or infer unknown population.',
        "missing_population": 'For no_data cells, EFARB is NaN; retain EFAR_mean when all three services are valid.',
        "missing_service_risk": 'If any service baseline has no reachable resources, the three-service mean and EFARB are NaN; do not average remaining dimensions or fill zeros.',
        "zero_population": 'EFARB=0 for known zero-population cells with three valid baselines; zero population cannot validate an undefined service risk.',
        "unit_interpretation": 'Persons times dimensionless accessibility risk: population-weighted risk burden, not observed affected persons, deaths or a count of distinct persons.',
        "grid_extent": 'Use population for original complete 500 m cells; preserve source areas, study-area intersection fractions and coordinates without rescaling boundary-cell population.',
        "direct_flood_exposure": 'Do not use population_exposed or exposed_ratio as weights, composite indicators or filters.',
        "service_source": 'Use the formal three-dimensional result-two EFAR: relative loss of count-weighted inverse-distance accessibility.',
        "source_settings": {"metric": "inverse", "local": "include", "same_grid_distance_m": 250},
        "cities": list(city_order), "periods": list(periods), "service_dimensions": list(service_dimensions),
        "dimension_keys": DIMENSION_KEYS, "row_count": len(wide), "long_row_count": len(long),
        "summary_row_count": len(summary), "sources": sources,
        "output_files": output_names,
    }
    scientific_qa = {
        "status": "PASS", "all_original_grids_preserved": True,
        "risk_dimension_ids_unique": True, "source_coordinate_match_tolerance_m": 1e-5,
        "all_source_risks_in_unit_interval": True, "baseline_nan_consistency": True,
        "population_status_consistency": True, "no_population_or_risk_imputation": True,
        "no_flood_exposure_filter": True, "fixed_population_set_across_scenarios": True,
        "known_zero_population_preserved": True, "partial_population_flag_preserved": True,
        "formula_max_abs_error": max(row["formula_max_abs_error"] for row in checks),
        "city_summary_identity_max_abs_error": max(row["absolute_residual"] for row in summary_checks),
        "input_files_unchanged_during_calculation": True,
        "distinct_input_file_count": len(source_stability_checks),
        "grid_count": len(wide), "long_row_count": len(long), "summary_row_count": len(summary),
        "city_scenario_checks": checks,
        "city_summary_identity_checks": summary_checks,
        "source_stability_checks": source_stability_checks,
    }
    metadata["scientific_qa"] = {k: v for k, v in scientific_qa.items() if k not in
                               {"city_scenario_checks", "city_summary_identity_checks", "source_stability_checks"}}
    for name, table in zip(output_names[:4], (wide, long, summary, coverage)):
        table.to_csv(output / name, index=False, encoding="utf-8-sig", na_rep="", mode="x")
    with (output / output_names[4]).open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return frames, summary, coverage, metadata
