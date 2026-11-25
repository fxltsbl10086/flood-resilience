"""Calculate road-restoration benefits for resident cells with confirmed dry conditions."""
import math

import numpy as np

EPSILON = 1e-8
FULL_GRID_AREA_DESCRIPTION = "full_500m_grid_area_not_residential_land_area"


def require(test, message):
    if not test:
        raise ValueError("")


def close(actual, expected, message, atol=1e-6, rtol=1e-11):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), "")


def total(values):
    return float(math.fsum(np.asarray(values, dtype=float).tolist()))


def ratio_pct(numerator, denominator):
    return 100 * numerator / denominator if denominator > 0 else np.nan


def burden_summary(frame):
    population = frame.effective_population.to_numpy(float)
    before = frame.risk_before.to_numpy(float)
    after = frame.risk_after.to_numpy(float)
    delta = frame.delta_risk.to_numpy(float)
    improved = delta > EPSILON
    p, b0, b1 = total(population), total(population * before), total(population * after)
    db = total(population * delta)
    close(db, b0 - b1, "")
    pi = total(population[improved])
    return {'N_dry': len(frame), 'P_dry': p, 'B0_dry': b0, 'B_after_dry': b1, 'DeltaB_dry': db,
            'R_dry': ratio_pct(db, b0), 'P_improved_dry': pi, 'N_improved_dry': int(improved.sum()),
            'H_dry': ratio_pct(pi, p),
            'mean_delta_risk_dry': db / p if p > 0 else np.nan,
            'mean_delta_risk_improved_dry': total(population[improved] * delta[improved]) / pi if pi > 0 else np.nan,
            'N_at_risk_dry': int((before > EPSILON).sum()),
            'P_at_risk_dry': total(population[before > EPSILON]),
            'S_at_risk_dry': ratio_pct(total(population[before > EPSILON]), p),
            'mean_risk_before_dry': b0 / p if p > 0 else np.nan,
            'R_dry_defined': b0 > 0, 'H_dry_defined': p > 0}


def coverage_summary(frame, threshold):
    """Use all valid dry resident cells as the denominator; apply the absolute-improvement threshold only to the numerator."""
    population = frame.effective_population.to_numpy(float)
    delta = frame.delta_risk.to_numpy(float)
    area = frame.grid_area_m2.to_numpy(float)
    require(np.isfinite(population).all() and np.isfinite(delta).all() and np.isfinite(area).all(),
            "")
    require((population > 0).all(), "")
    flag = delta >= threshold  # Do not preround, add floating-point slack or replace >= with >.
    n, nb = len(frame), int(flag.sum())
    p, pb = total(population), total(population[flag])
    a, ab = total(area), total(area[flag])
    close(a, n * 250000., "", atol=1e-2)
    close(ab, nb * 250000., "", atol=1e-2)
    return {'threshold': threshold, 'operator': '>=', 'N_dry': n, 'N_benefit': nb,
            'G_dry': ratio_pct(nb, n), 'P_dry': p, 'P_benefit': pb,
            'H_dry_threshold': ratio_pct(pb, p), 'A_dry_full_grid_m2': a,
            'A_benefit_full_grid_m2': ab, 'A_dry_full_grid_km2': a/1e6,
            'A_benefit_full_grid_km2': ab/1e6,
            'G_dry_defined': n > 0, 'H_dry_threshold_defined': p > 0,
            'grid_weighting': 'each_valid_residential_grid_has_equal_weight',
            'population_weighting': 'effective_population',
            'area_definition': FULL_GRID_AREA_DESCRIPTION}
