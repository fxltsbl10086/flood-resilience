"""Reproduce frozen S6 means and 95% intervals from included city-period curves.

This stage starts from fig_s6_case_curves.csv and aggregates the saved outcomes.
"""
from itertools import product
import numpy as np
import pandas as pd
def _city_means(case):
    return case.groupby(["city", "strategy", "restoration_pct"], as_index=False, sort=False).agg(
        mean_EFARB_reduction=("mean_EFARB_reduction", "mean"),
        mean_actual_restoration_pct=("mean_actual_restoration_pct", "mean"),
        period_count=("period", "size"),
        return_periods=("period", lambda values: ",".join(map(str, sorted(values)))),
        complete_repeat_count=("complete_repeat_count", "sum"),
    )

def _period_bootstrap_intervals(case: pd.DataFrame) -> pd.DataFrame:
    "Calculate pointwise percentile intervals using matched whole-curve scenario resamples."
    intervals = []
    for city, rows in case.groupby("city", sort=False):
        by_period = rows.pivot(
            index="period", columns=["strategy", "restoration_pct"],
            values="mean_EFARB_reduction",
        )
        values = by_period.to_numpy(dtype=float)
        n_periods = len(by_period)
        if n_periods < 2 or not np.isfinite(values).all():
            raise ValueError()
        # Enumerate equally probable resamples for the at-most-five return periods.
        draws = np.array(list(product(range(n_periods), repeat=n_periods)))
        bootstrap_means = values[draws].mean(axis=1)
        lower, upper = np.percentile(
            bootstrap_means, [2.5, 97.5], axis=0, method="inverted_cdf",
        )
        intervals.append(by_period.columns.to_frame(index=False).assign(
            city=city, CI95_lower=lower, CI95_upper=upper,
        ))
    return pd.concat(intervals, ignore_index=True)

def calculate(case):
    return _city_means(case).merge(_period_bootstrap_intervals(case),
        on=['city','strategy','restoration_pct'],validate='one_to_one')
