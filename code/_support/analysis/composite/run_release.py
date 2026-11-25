"""Recompute the compound index, bivariate classes and Gi* hotspots."""
import argparse
from types import SimpleNamespace
import json
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
from config import (CITIES, PERIODS, FUNCTION_COLORS, FIGURE_MM, DPI,
                    FIGURE_STYLE_METADATA, p, SUPPORTED_CITIES, CITY_SEED_OFFSETS)
from hotspot_stats import queen_weights, permutation_hotspots
from prepare_data import prepare_all
from bivariate import assign_bivariate
from _support.analysis_paths import output_root

def compute(args, output, cities=CITIES):
    datasets, sources, reference = prepare_all(cities)
    bivariate_metadata, class_counts = assign_bivariate(datasets)
    class_counts.to_csv(output / "Bivariate_class_counts.csv", index=False, encoding="utf-8-sig")
    coverage, seeds = [], {}
    for city, (data, boundary) in datasets.items():
        selected = data.analysis_included.eq(1)
        subset = data.loc[selected]
        weights = queen_weights(subset.row.to_numpy(), subset.col.to_numpy())
        values = subset[[f"R_T{t}" for t in PERIODS]].to_numpy(float)
        seed = args.seed + CITY_SEED_OFFSETS[city]
        seeds[city] = seed
        print(f"Gi* {city}: {len(subset):,} observations, {args.permutations:,} permutations", flush=True)
        stats = permutation_hotspots(values, weights, permutations=args.permutations,
                                     seed=seed, alpha=args.alpha, fdr=args.fdr,
                                     progress=lambda current, total: print(f"  {city}: {current:,}/{total:,}", flush=True))
        data["queen_neighbor_count"] = np.nan
        data.loc[selected, "queen_neighbor_count"] = stats["neighbor_count"]
        for j, t in enumerate(PERIODS):
            for key, value in [("Gi_z", stats["z"]), ("Gi_p_upper", stats["p_upper"]),
                               ("Gi_q_BH", stats["q_bh"]), ("Gi_q_BY", stats["q_by"]),
                               ("Gi_q", stats["q"]), ("hotspot", stats["hotspot"].astype(int)),
                               ("Gi_tested", stats["tested"].astype(int))]:
                column = f"{key}_T{t}"
                data[column] = np.nan
                data.loc[selected, column] = value[:, j]
            tested = stats["tested"][:, j]
            coverage.append({"city": city, "return_period": t, "total_grid_count": len(data),
                "population_known_count": int(data.population.notna().sum()),
                "positive_known_population_count": int(data.population.gt(0).sum()),
                "joint_evaluable_count": len(subset), "statistically_tested_count": int(tested.sum()),
                "neighborless_count": int((stats["neighbor_count"] == 0).sum()),
                "queen_neighbor_mean": float(stats["neighbor_count"].mean()),
                "queen_neighbor_min": int(stats["neighbor_count"].min()),
                "hotspot_count": int(stats["hotspot"][:, j].sum()),
                "BH_hotspot_count": int(((stats["z"][:, j] > 0) & (stats["q_bh"][:, j] <= args.alpha)).sum()),
                "BY_hotspot_count": int(((stats["z"][:, j] > 0) & (stats["q_by"][:, j] <= args.alpha)).sum()),
                "R_min": float(values[:, j].min()), "R_max": float(values[:, j].max()),
                "partial_population_coverage_included": int(subset.population_data_coverage_ratio.lt(1-1e-9).sum())})
        data_dir = output / "spatial_data"
        data_dir.mkdir(exist_ok=True)
        data.drop(columns="geometry").to_csv(data_dir / f"{city}_composite_risk_and_hotspots.csv",
                                              index=False, encoding="utf-8-sig")
        data.to_file(data_dir / f"{city}_composite_risk_and_hotspots.gpkg", layer="grid_composite_risk", driver="GPKG")
        gpd.GeoDataFrame({"city": [city], "geometry": [boundary]}, crs=data.crs).to_file(
            data_dir / f"{city}_composite_risk_and_hotspots.gpkg", layer="study_area", driver="GPKG")
        print(f"{city} hotspot cells (T10/T20/T50/T100/T200): {[row['hotspot_count'] for row in coverage if row['city']==city]}", flush=True)
    pd.DataFrame(coverage).to_csv(output / "Analysis_coverage_and_hotspot_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "exposure": "Existing estimated inundated population count; not a population fraction or area fraction.",
        "exposure_reference": reference,
        "normalization": f"E = exposed_people / fixed maximum across the {len(datasets)} reference cities and {len(PERIODS)} scenarios.",
        "reference_universe": f"All population-known source cells in {len(datasets)} cities and {len(PERIODS)} scenarios, irrespective of service baseline availability.",
        "reference_cities": list(datasets),
        "formula": "R = 0.5*E + (A_rescue + A_medical + A_shelter)/6",
        "weights": {"exposure": 0.5, "rescue": 1/6, "medical": 1/6, "shelter": 1/6},
        "analysis_set": "Fixed per-city cells with known positive population and three baseline-valid EFAR dimensions; same set in all five scenarios.",
        "missingness": "Missing population or any undefined service baseline is excluded, never filled with zero or one. Zero known population is excluded. Partial coverage retains only source known people.",
        "cooccurrence": "affected_dimension_count is the number of dimensions with a positive increase (EFAR >1e-10 or inundated people >0); it is distinct from the weighted burden and is not the mapped hotspot criterion.",
        "hotspot_method": "Getis–Ord Gi* standardized local sum, binary first-order queen weights including self.",
        "neighborhood": "Original 500 m cell edge or corner adjacency; up to eight other evaluable cells; no artificial neighbors across missing cells.",
        "inference": "Unconditional random-label permutations within each city, including focal values; shared location permutations across scenarios.",
        "pvalue": "Inclusive upper-tail Monte Carlo p=(1+count(permuted local deviation >= observed local deviation))/(B+1).",
        "permutations": args.permutations, "seed_by_city": seeds, "minimum_pvalue": 1/(args.permutations+1),
        "fdr_method": args.fdr.upper(), "fdr_family": "All tested cells in each city and scenario; no cross-city or cross-scenario family-wide claim.",
        "alpha": args.alpha, "hotspot_criterion": "positive Gi* z and selected FDR q<=alpha",
        "fdr_rationale": "BY is default because local neighborhood tests are dependent; BH adjusted values are also exported for sensitivity inspection.",
        "islands": "No other valid queen neighbor: burden remains mapped; Gi* and p/q remain unavailable; not a hotspot.",
        "spatial_interpretation": "Relative high-value clusters within each city; not absolute high-risk thresholds, calibrated loss probabilities, or proof of risk co-occurrence.",
        "simulation_limit": "Uses the existing 0–12 h maximum-depth-based population exposure and road-access products; source limitations remain applicable.",
        "bivariate": bivariate_metadata,
        "figure": {"layout": "2×3", "city_order": list(datasets), "encoding": "3×3 bivariate population/service classes", "index_scale_for_R": [0,1],
                   "existing_function_colors": FUNCTION_COLORS, "size_mm": FIGURE_MM, "png_dpi": DPI,
                   "geometry": "Original complete source grid cells; no clipping or simplification.", "map_scales": "independent per city",
                   **FIGURE_STYLE_METADATA},
        "sources": sources, "versions": {"python": sys.version, "geopandas": gpd.__version__, "scipy": scipy.__version__},
        "method_sources": [
            "https://doc.esri.com/en/arcgis-pro/latest/tool-reference/spatial-statistics/h-how-hot-spot-analysis-getis-ord-gi-spatial-stati.html",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html"],
        "source_invariance": "No input data or pre-existing experimental results were modified.",
    }
    (output / "Calculation_and_map_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", choices=SUPPORTED_CITIES, default=CITIES)
    selected = set(parser.parse_args().cities)
    cfg=p['composite']
    output=output_root('composite')
    output.mkdir(parents=True, exist_ok=False)
    compute(SimpleNamespace(seed=cfg['city_seed_base'], permutations=cfg['permutations'],
                            alpha=cfg['alpha'], fdr=cfg['fdr']),output,
            cities=[city for city in SUPPORTED_CITIES if city in selected])
