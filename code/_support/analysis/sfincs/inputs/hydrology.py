"""Uniform meteorological rainfall, class-specific CN runoff and area-weighted mixing."""
import numpy as np
import pandas as pd


def read_rainfall(config, periods):
    """Read the project rainfall table without changing formulae or rainfall totals."""
    table = pd.read_csv(config["rainfall_table"])
    table = table[table.city_code == config["city"]]
    result, expected_zones = {}, None
    for period in periods:
        rows = table[table.return_period_year == period].sort_values("formula_zone_id")
        if rows.empty or rows.formula_zone_id.duplicated().any():
            raise ValueError()
        if not (rows.duration_min == config["rain_duration_min"]).all():
            raise ValueError()
        totals = rows.rainfall_6h_mm.to_numpy(dtype=float)
        if not np.all(np.isfinite(totals) & (totals > 0)):
            raise ValueError()
        zones = tuple(rows.formula_zone_id)
        if expected_zones is not None and zones != expected_zones:
            raise ValueError()
        expected_zones = zones
        result[period] = dict(zip(zones, totals))
    return result


def class_runoff_rates(total_mm, cn, abstraction, duration, step):
    """Calculate nonlinear runoff by class before mixing; do not use an averaged CN."""
    storage = 25400.0 / cn - 254.0
    rainfall = np.linspace(0, total_mm, duration // step + 1)[:, None]
    excess = np.maximum(rainfall - abstraction * storage, 0)
    denominator = excess + storage
    cumulative = np.divide(excess ** 2, denominator, out=np.zeros_like(excess), where=denominator > 0)
    return np.diff(cumulative, axis=0) * (3600.0 / step)


def runoff_frames(config, fractions, cn, rainfall, zones, active):
    """Generate forcing one time frame at a time to limit memory use."""
    step = config["forcing_step_seconds"]
    duration = config["rain_duration_min"] * 60
    simulation = config["simulation_duration_min"] * 60
    if step <= 0 or duration % step or simulation % step or simulation <= duration:
        raise ValueError()
    rates = [class_runoff_rates(total, cn, config["initial_abstraction_ratio"], duration, step)
             for total in rainfall.values()]
    for i in range(simulation // step + 1):
        frame = np.zeros(active.shape, dtype="float32")
        if i < duration // step:
            for zone, class_rates in enumerate(rates):
                mixed = np.einsum("k,kyx->yx", class_rates[i], fractions, dtype="float64")
                selected = active & (zones == zone)
                frame[selected] = mixed[selected]
        # Forcing is zero from minute 360 onward; the solver applies drainage dynamically.
        yield frame
