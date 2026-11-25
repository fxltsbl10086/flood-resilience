"""Population-weighted Lorenz curves using complete, unsmoothed grid values."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import (CITIES, CITY_NAMES, CITY_NAMES_ZH, INPUTS, PERIODS,
                    SERVICE_DIMENSIONS, DIMENSIONS, DIMENSION_NAMES)
from _support.analysis_paths import access_inputs

def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

def lorenz(values, weights):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError()
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError()
    if np.any(values < 0) or np.any(weights <= 0):
        raise ValueError()
    order = np.argsort(values, kind='stable')
    ranked, pop = values[order], weights[order]
    weighted_risk = ranked * pop
    total_pop, total_burden = float(pop.sum()), float(weighted_risk.sum())
    if total_burden == 0:
        return np.array([]), np.array([]), np.nan, 0.0, 'zero_total_burden'
    x = np.r_[0.0, np.cumsum(pop) / total_pop]
    y = np.r_[0.0, np.cumsum(weighted_risk) / total_burden]
    x[-1], y[-1] = 1.0, 1.0  # normalize accumulated floating-point residual only
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < -1e-12) or np.any(y - x > 1e-10):
        raise ValueError()
    gini = float(1.0 - np.sum(np.diff(x) * (y[1:] + y[:-1])))
    if abs(gini) < 1e-13:
        gini = 0.0
    if not 0 <= gini <= 1:
        raise ValueError()
    return x, y, gini, total_burden / total_pop, 'defined'

def pairwise_gini(values, weights):
    """Independent weighted mean-absolute-difference definition for QA subsets."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    burden = float(np.dot(values, weights))
    if burden == 0:
        return np.nan
    numerator = np.sum(np.abs(values[:, None] - values[None, :]) * weights[:, None] * weights[None, :])
    return float(numerator / (2 * weights.sum() * burden))

def check_math():
    cases = [([1, 1, 1], [2, 3, 5], 0), ([0, 1], [1, 1], 0.5),
             ([0, 1], [3, 1], 0.75), ([0, 1], [1, 3], 0.25),
             ([0, 0, 0, 4], [1, 1, 1, 1], 0.75),
             ([1, 2, 3], [1, 1, 1], 2 / 9), ([0.4], [3.7], 0)]
    for values, weights, expected in cases:
        actual = lorenz(values, weights)[2]
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise AssertionError()
    values, weights = np.array([0, 0.2, 0.7, 1.0]), np.array([7.3, 2.0, 9.1, 3.3])
    expected = pairwise_gini(values, weights)
    for scale, weight_scale in [(1, 1), (5, 1), (1, 1000), (0.001, 1000)]:
        actual = lorenz(values[::-1] * scale, weights[::-1] * weight_scale)[2]
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise AssertionError()
    if lorenz([0, 0], [1, 2])[4] != 'zero_total_burden' or not np.isnan(lorenz([0, 0], [1, 2])[2]):
        raise AssertionError()
    for values, weights in [([-1, 2], [1, 1]), ([1, 2], [0, 1]), ([np.nan, 1], [1, 1])]:
        try:
            lorenz(values, weights)
        except ValueError:
            continue
        raise AssertionError()
    return {'analytic_gini_cases': 7, 'invariance_cases': 4, 'zero_burden_case': 1,
            'invalid_input_rejections': 3}

def read_city(city):
    cfg = access_inputs(city)
    metadata = json.loads(cfg['risk_metadata'].read_text(encoding='utf-8'))
    if (metadata.get('metric') != 'inverse' or metadata.get('local') != 'include'
            or metadata.get('same_grid_distance_m') != 250 or metadata.get('periods') != PERIODS):
        raise ValueError()
    pop = pd.read_csv(cfg['population_csv'], encoding='utf-8-sig', dtype={'grid_id': str})
    risk = pd.read_csv(cfg['risk_csv'], encoding='utf-8-sig', dtype={'grid_id': str})
    required_pop = {'grid_id', 'population', 'population_status', 'centroid_x', 'centroid_y', 'cell_m',
                    *[f'exposed_ratio_T{t}' for t in PERIODS], *[f'population_exposed_T{t}' for t in PERIODS]}
    required_risk = {'grid_id', 'city', '评估维度', '状态', 'centroid_x', 'centroid_y',
                     '指标', '同网格处理', '同网格距离m', *[f'风险_T{t:03d}' for t in PERIODS]}
    if not required_pop.issubset(pop) or not required_risk.issubset(risk):
        raise ValueError()
    if pop.grid_id.isna().any() or pop.grid_id.str.strip().eq('').any() or pop.grid_id.duplicated().any():
        raise ValueError()
    if set(risk['评估维度']) != set(SERVICE_DIMENSIONS) or risk.duplicated(['grid_id', '评估维度']).any():
        raise ValueError()
    if not risk.city.eq(city).all() or ('city' in pop and not pop.city.eq(city).all()):
        raise ValueError()
    if not (risk['指标'].eq('inverse').all() and risk['同网格处理'].eq('include').all()
            and risk['同网格距离m'].eq(250).all() and pop.cell_m.eq(500).all()):
        raise ValueError()
    weights = pd.to_numeric(pop.population, errors='raise').to_numpy(float)
    known = ~np.isnan(weights)
    if not np.isfinite(weights[known]).all() or np.any(weights[known] < 0):
        raise ValueError()
    positive = known & (weights > 0)
    data = {'exposure': pop[[f'exposed_ratio_T{t}' for t in PERIODS]].to_numpy(float)}
    if not np.isfinite(data['exposure'][positive]).all() or np.any(data['exposure'][positive] < 0) or np.any(data['exposure'][positive] > 1):
        raise ValueError()
    exposed_counts = pop[[f'population_exposed_T{t}' for t in PERIODS]].to_numpy(float)
    if not np.allclose(data['exposure'][positive] * weights[positive, None], exposed_counts[positive], rtol=1e-10, atol=1e-6):
        raise ValueError()
    common, baseline = positive.copy(), {}
    for dimension in SERVICE_DIMENSIONS:
        part = risk[risk['评估维度'].eq(dimension)].set_index('grid_id')
        if set(part.index) != set(pop.grid_id):
            raise ValueError()
        part = part.loc[pop.grid_id].reset_index()
        for coord in ['centroid_x', 'centroid_y']:
            if not np.allclose(part[coord], pop[coord], rtol=0, atol=1e-5):
                raise ValueError()
        if not part['状态'].isin(['有效', '基准无可达资源']).all():
            raise ValueError()
        valid = part['状态'].eq('有效').to_numpy()
        values = part[[f'风险_T{t:03d}' for t in PERIODS]].to_numpy(float)
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError()
        if np.any(values[valid] < 0) or np.any(values[valid] > 1):
            raise ValueError()
        data[dimension], baseline[dimension] = values, valid
        common &= valid
    common &= np.isfinite(data['exposure']).all(axis=1)
    if not common.any():
        raise ValueError()
    denominator, total_known = float(weights[common].sum()), float(weights[known].sum())
    coverage = {'city': city, 'city_name': CITY_NAMES[city], 'total_grid_count': len(pop),
        'known_positive_population_grid_count': int(positive.sum()), 'analytic_grid_count': int(common.sum()),
        'population_missing_grid_count': int((~known).sum()), 'known_zero_population_grid_count': int((known & (weights == 0)).sum()),
        'excluded_positive_population_grid_count': int((positive & ~common).sum()),
        'partial_population_analytic_grid_count': int((common & pop.population_status.eq('partial').to_numpy()).sum()),
        'known_total_population': total_known, 'analytic_population': denominator,
        'known_population_coverage': denominator / total_known, 'excluded_known_population': total_known - denominator}
    for dimension in SERVICE_DIMENSIONS:
        coverage[f'baseline_valid_grid_count_{dimension}'] = int(baseline[dimension].sum())
    grids = []
    for j, period in enumerate(PERIODS):
        table = pop[['grid_id', 'centroid_x', 'centroid_y', 'population', 'population_status']].copy()
        table.insert(0, 'city', city)
        table['return_period'], table['included_in_analysis'] = period, common
        for dimension in DIMENSIONS:
            table[dimension] = data[dimension][:, j]
        grids.append(table)
    return pop, weights, data, common, coverage, pd.concat(grids, ignore_index=True)

def prepare(output):
    summaries, coverages, grids, curve_tables = [], [], [], []
    curves, sources = {}, {}
    qa = check_math()
    independent_errors = []
    for city in CITIES:
        sources[city] = {key: signature(path) for key, path in INPUTS[city].items()}
        pop, weights, data, common, coverage, grid_table = read_city(city)
        coverages.append(coverage)
        grids.append(grid_table)
        population = weights[common]
        for dimension in DIMENSIONS:
            for j, period in enumerate(PERIODS):
                values = data[dimension][common, j]
                x, y, gini, mean, status = lorenz(values, population)
                curves[(city, dimension, period)] = (x, y)
                subset = np.unique(np.linspace(0, len(values) - 1, min(64, len(values))).astype(int))
                gini_subset = lorenz(values[subset], population[subset])[2]
                pairwise = pairwise_gini(values[subset], population[subset])
                if not ((np.isnan(gini_subset) and np.isnan(pairwise)) or np.isclose(gini_subset, pairwise, rtol=0, atol=1e-11)):
                    raise AssertionError()
                if np.isfinite(pairwise):
                    independent_errors.append(abs(gini_subset - pairwise))
                summaries.append({'city': city, 'city_name': CITY_NAMES[city], '城市': CITY_NAMES_ZH[city],
                    'dimension': dimension, 'dimension_name': DIMENSION_NAMES[dimension], 'return_period': period,
                    'gini': gini, 'population_weighted_mean_risk': mean, 'curve_status': status,
                    'analytic_grid_count': int(common.sum()), 'analytic_population': float(population.sum()),
                    'population_weighted_total_burden': float(np.dot(values, population)),
                    'known_population_coverage': coverage['known_population_coverage'],
                    'highest_risk_10pct_population_burden_share': float(1 - np.interp(0.9, x, y)) if x.size else np.nan})
                if x.size:
                    curve_tables.append(pd.DataFrame({'city': city, 'dimension': dimension, 'return_period': period,
                        'vertex': np.arange(x.size), 'cumulative_population_share': x, 'cumulative_risk_burden_share': y}))
        print(f'{CITY_NAMES[city]}: {common.sum():,} analytic grids; known-population coverage {coverage["known_population_coverage"]:.2%}', flush=True)
    summary, coverage = pd.DataFrame(summaries), pd.DataFrame(coverages)
    pd.concat(grids, ignore_index=True).to_csv(output / '500m_grid_four_dimensions.csv.gz', index=False,
                                               encoding='utf-8-sig', compression={'method': 'gzip', 'mtime': 0})
    pd.concat(curve_tables, ignore_index=True).to_csv(output / 'Lorenz_curve_vertices.csv.gz', index=False,
                                                     encoding='utf-8-sig', compression={'method': 'gzip', 'mtime': 0})
    summary.to_csv(output / 'Gini_and_mean_risk.csv', index=False, encoding='utf-8-sig')
    coverage.to_csv(output / 'Analysis_population_coverage.csv', index=False, encoding='utf-8-sig')
    qa.update({'real_data_independent_pairwise_checks': len(summaries),
               'maximum_pairwise_gini_difference': max(independent_errors, default=0),
               'all_curve_count': len(curves), 'defined_curve_count': sum(bool(x.size) for x, _ in curves.values())})
    return summary, coverage, curves, sources, qa
