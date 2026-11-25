"Aggregate existing grid SAR by origin population using CSV inputs only."

from pathlib import Path

import numpy as np
import pandas as pd


def weighted_mean(values, weights):
    "Use the existing weighted-mean formula and stop on missing eligible SAR."
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights) & (weights > 0)
    if not np.isfinite(values[valid]).all():
        raise ValueError()
    denominator = float(weights[valid].sum())
    if denominator == 0:
        return float("nan"), 0.0
    return float(np.sum(values[valid] * weights[valid]) / denominator), denominator


def normalize_risk_fields(table):
    "Normalize the legacy NAR column name to SAR, requiring equal values and missingness when both exist."
    for old_column in list(table.columns):
        if "NAR" not in old_column:
            continue
        new_column = old_column.replace("NAR", "SAR")
        if new_column in table:
            old_values = pd.to_numeric(table[old_column], errors="raise")
            new_values = pd.to_numeric(table[new_column], errors="raise")
            same = old_values.eq(new_values) | (old_values.isna() & new_values.isna())
            if not same.all():
                raise ValueError()
            table = table.drop(columns=old_column)
        else:
            table = table.rename(columns={old_column: new_column})
    return table


def risk_setting(cfg, name, old_name):
    "Accept legacy configuration keys without permitting conflicting aliases."
    if name in cfg and old_name in cfg and cfg[name] != cfg[old_name]:
        raise ValueError()
    if name in cfg:
        return cfg[name]
    if old_name in cfg:
        return cfg[old_name]
    raise KeyError()


def read_grid_table(path, required_columns):
    "Read existing CSVs while retaining grid IDs and original fields."
    path = Path(path)
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"grid_id": str})
    table = normalize_risk_fields(table)
    missing = set(required_columns) - set(table.columns)
    if missing:
        raise ValueError()
    ids = table["grid_id"]
    if table.empty or ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError()
    return table


def merge_grid_results(cfg):
    "Merge one-to-one by ID and use the fixed baseline-valid set with known population."
    scenarios = [f"T{period}" for period in cfg["PERIODS"]]
    population_columns = ["grid_id", "population"]
    for scenario in scenarios:
        population_columns += [f"population_exposed_{scenario}", f"population_nonexposed_{scenario}"]
    population = read_grid_table(cfg["POPULATION_CSV"], population_columns)
    sar = read_grid_table(risk_setting(cfg, "SAR_CSV", "NAR_CSV"),
                          ["grid_id", "baseline_valid"] + [f"SAR_{s}" for s in scenarios])
    if set(population["grid_id"]) != set(sar["grid_id"]):
        raise ValueError()
    if "city" in population and not population["city"].eq(cfg["CITY"]).all():
        raise ValueError()

    # Validate shared spatial fields before retaining the population-table version.
    aligned_sar = sar.set_index("grid_id").loc[population["grid_id"]]
    for column in ["row", "col", "cell_m", "centroid_x", "centroid_y"]:
        if column in population and column in sar:
            tolerance = cfg["COORDINATE_TOLERANCE_M"] if column.startswith("centroid_") else 0
            if not np.allclose(population[column], aligned_sar[column], rtol=0, atol=tolerance):
                raise ValueError()
    overlap = (set(population.columns) & set(sar.columns)) - {"grid_id"}
    # Prevent risk/population primary-field collisions after checking common geometry attributes.
    if overlap & {"baseline_valid", "population", *[f"SAR_{s}" for s in scenarios]}:
        raise ValueError()
    result = population.merge(sar.drop(columns=sorted(overlap)), on="grid_id", how="left",
                              validate="one_to_one", sort=False)

    selected_ids = cfg.get("GRID_IDS")
    if selected_ids is not None:
        if not selected_ids or len(selected_ids) != len(set(selected_ids)):
            raise ValueError()
        missing = set(selected_ids) - set(result["grid_id"])
        if missing:
            raise ValueError()
        result = result.loc[result["grid_id"].isin(selected_ids)].copy()

    flags = result["baseline_valid"].astype(str).str.strip().str.lower()
    if not flags.isin(["true", "false", "1", "0"]).all():
        raise ValueError()
    result["baseline_valid"] = flags.isin(["true", "1"])
    result["population"] = pd.to_numeric(result["population"], errors="raise")
    p = result["population"]
    known = p.notna()
    if not np.isfinite(p[known]).all() or p[known].lt(0).any():
        raise ValueError()
    baseline_valid = result["baseline_valid"]
    # Use only known population in partially missing grids, without estimating or zero-filling unknown counts.
    result["weighting_valid"] = baseline_valid & known
    result["population_used"] = p.where(result["weighting_valid"])

    for scenario in scenarios:
        exposed = f"population_exposed_{scenario}"
        nonexposed = f"population_nonexposed_{scenario}"
        sar_column = f"SAR_{scenario}"
        for column in [exposed, nonexposed, sar_column]:
            result[column] = pd.to_numeric(result[column], errors="raise")
        groups = result[[exposed, nonexposed]]
        if not np.isfinite(groups.loc[known].to_numpy()).all() or groups.loc[known].lt(0).any().any():
            raise ValueError()
        if groups.loc[~known].notna().any().any():
            raise ValueError()
        if not np.allclose(groups.loc[known].sum(axis=1), p[known], rtol=0,
                           atol=cfg["POPULATION_TOLERANCE"]):
            raise ValueError()
        risks = result[sar_column]
        if not np.isfinite(risks[baseline_valid]).all() or risks[~baseline_valid].notna().any():
            raise ValueError()
        tolerance = risk_setting(cfg, "SAR_TOLERANCE", "NAR_TOLERANCE")
        if risks[baseline_valid].lt(-tolerance).any() or risks[baseline_valid].gt(1 + tolerance).any():
            raise ValueError()
        # This is the weighted numerator contribution, not a new grid SAR or affected-population estimate.
        result[f"population_SAR_{scenario}"] = (p * risks).where(result["weighting_valid"])
    return result


def summarize_population_risk(result, cfg):
    "Use fixed eligible population for totals and scenario-specific eligible group population for groups."
    valid = result["weighting_valid"].to_numpy()
    known = result["population"].notna().to_numpy()
    rows = []
    for period in cfg["PERIODS"]:
        scenario = f"T{period}"
        risks = result[f"SAR_{scenario}"].to_numpy()
        group_columns = {
            "all": "population",
            "exposed": f"population_exposed_{scenario}",
            "nonexposed": f"population_nonexposed_{scenario}",
        }
        for group, column in group_columns.items():
            weights = result[column].to_numpy()
            mean, denominator = weighted_mean(risks[valid], weights[valid])
            total = float(weights[known].sum())
            excluded = float(weights[known & ~valid].sum())
            rows.append({
                "city": cfg["CITY"],
                "test_area": cfg["ANALYSIS_AREA"],
                "scenario": scenario,
                "population_group": group,
                "origin_grid_count": len(result),
                "baseline_valid_grid_count": int(result["baseline_valid"].sum()),
                "weighting_valid_grid_count": int(valid.sum()),
                "population_unknown_grid_count": int((~known).sum()),
                "group_total_population": total,
                "group_valid_population": denominator,
                "group_excluded_population": excluded,
                "population_coverage_rate": denominator / total if total > 0 else np.nan,
                "group_positive_valid_grid_count": int((valid & (weights > 0)).sum()),
                "population_weighted_SAR": mean,
            })
    return pd.DataFrame(rows)


def export_results(result, summary, output_dir, city):
    "Exclusively create the two batch CSVs using UTF-8 with BOM."
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    grid_path = output_dir / f"{city}_500mgrid_population_and_sar.csv"
    summary_path = output_dir / f"{city}_population_weighted_sar_summary.csv"
    for table, path in [(result, grid_path), (summary, summary_path)]:
        table.to_csv(path, mode="x", index=False, encoding="utf-8-sig", na_rep="")
    return grid_path, summary_path
