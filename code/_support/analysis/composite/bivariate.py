"""Shared pooled bins for exposed people and mean service-access risk."""
import numpy as np
import pandas as pd
from config import PERIODS, DIMENSION_KEYS, BIVARIATE_PALETTE


def assign_bivariate(datasets):
    exposure_pool, service_pool = [], []
    for data, _ in datasets.values():
        valid = data.analysis_included.eq(1).to_numpy()
        for t in PERIODS:
            services = data[[f"A_{key}_T{t}" for key in DIMENSION_KEYS]].to_numpy(float)
            service_mean = services.sum(axis=1) / 3  # Never average a missing dimension away.
            data[f"service_composite_T{t}"] = np.where(valid, service_mean, np.nan)
            exposure = data[f"exposed_people_T{t}"].to_numpy(float)
            if not (np.isfinite(exposure[valid]).all() and np.isfinite(service_mean[valid]).all()):
                raise ValueError()
            if not np.array_equal(valid, data[f"R_T{t}"].notna().to_numpy()):
                raise ValueError()
            exposure_pool.append(exposure[valid])
            service_pool.append(service_mean[valid])
    exposure_pool, service_pool = np.concatenate(exposure_pool), np.concatenate(service_pool)
    exposure_cuts = np.quantile(exposure_pool, [1/3, 2/3], method="linear")
    service_cuts = np.quantile(service_pool, [1/3, 2/3], method="linear")
    if not (exposure_cuts[0] < exposure_cuts[1] and service_cuts[0] < service_cuts[1]):
        raise ValueError()
    counts = []
    for city, (data, _) in datasets.items():
        valid = data.analysis_included.eq(1).to_numpy()
        for t in PERIODS:
            ex = data[f"exposed_people_T{t}"].to_numpy(float)
            service = data[f"service_composite_T{t}"].to_numpy(float)
            x = np.searchsorted(exposure_cuts, ex[valid], side="left")
            y = np.searchsorted(service_cuts, service[valid], side="left")
            code = y * 3 + x
            for name, values in [("population_exposure_level", x), ("service_risk_level", y), ("bivariate_class", code)]:
                result = np.full(len(data), np.nan)
                result[valid] = values
                data[f"{name}_T{t}"] = result
            values, frequencies = np.unique(code, return_counts=True)
            for c, n in zip(values, frequencies):
                counts.append({"city": city, "return_period": t, "bivariate_class": int(c),
                    "population_exposure_level": int(c % 3), "service_risk_level": int(c // 3),
                    "cell_count": int(n), "color": BIVARIATE_PALETTE[c // 3][c % 3]})
    metadata = {
        "encoding": "3×3 bivariate choropleth",
        "horizontal_dimension": "Existing inundated population count (people)",
        "vertical_dimension": "S=(A_rescue+A_medical+A_shelter)/3; three valid dimensions required",
        "population_exposure_cuts_people": exposure_cuts.tolist(),
        "mean_service_risk_cuts": service_cuts.tolist(),
        "classification": f"Shared pooled terciles across {len(datasets)} cities and {len(PERIODS)} scenarios, with common cutpoints for all panels.",
        "classification_cities": list(datasets),
        "classification_pool_n": len(exposure_pool),
        "cutpoint_inclusion": "lower: value<=cut1; middle: cut1<value<=cut2; upper: value>cut2",
        "interpretation": "Relative pooled ranks, not absolute high/low risk or population thresholds.",
        "palette_rows_low_to_high_service_cols_low_to_high_population": BIVARIATE_PALETTE,
        "palette_semantics": "Exposure increases toward cyan; service risk toward magenta; joint upper classes dark blue-purple.",
        "class_code": "service_level*3+population_level; levels 0,1,2; unavailable cells NaN",
        "hotspot_overlay": "Gi* and FDR hotspots of R=0.5E+0.5S.",
    }
    return metadata, pd.DataFrame(counts)
