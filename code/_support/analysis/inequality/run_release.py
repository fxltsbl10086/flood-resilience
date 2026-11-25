"""Reproduce Gini and Lorenz summary tables from signed compound-risk data."""
import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from config import CITIES, CITY_NAMES, PERIODS, DIMENSIONS as ALL_DIMENSIONS
from prepare_data import lorenz, pairwise_gini, read_city, signature, check_math
from _release import ANALYSIS, CODE
from _support.analysis_paths import CITY_CODES, city_selection, output_root, access_inputs

KEYS=['rescue','medical','shelter']
DIMENSIONS=ALL_DIMENSIONS[1:]
MAP_DIR=CODE/'_support/analysis/composite'
INEQUALITY_DIR=CODE/'_support/analysis/inequality'
def prepare_composite(source, output, cities=CITIES):
    cities = city_selection(cities)
    metadata_path = source / 'Calculation_and_map_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    ref = float(metadata['exposure_reference']['people'])
    expected_weights = {'exposure': 0.5, 'rescue': 1 / 6, 'medical': 1 / 6, 'shelter': 1 / 6}
    if not np.isfinite(ref) or ref <= 0 or set(metadata['weights']) != set(expected_weights):
        raise ValueError()
    if not all(np.isclose(metadata['weights'][key], value, rtol=0, atol=1e-14) for key, value in expected_weights.items()):
        raise ValueError()
    city_order = metadata['figure']['city_order']
    if (not city_order or len(set(city_order)) != len(city_order)
            or set(city_order) != set(cities)
            or set(metadata['sources']) != set(city_order)):
        raise ValueError()
    selected_cities = list(cities)
    signatures = {'map_metadata': signature(metadata_path), 'map_style_config': signature(MAP_DIR / 'config.py'),
                  'inequality_config': signature(INEQUALITY_DIR / 'config.py'),
                  'math_and_input_helpers': signature(INEQUALITY_DIR / 'prepare_data.py')}
    for city,city_sources in metadata['sources'].items():
        for key in ['population', 'services', 'service_metadata']:
            recorded = city_sources[key]
            local={'population':'population_csv','services':'risk_csv','service_metadata':'risk_metadata'}[key]
            if signature(access_inputs(city)[local])['sha256'] != recorded['sha256']:
                raise ValueError()
    curves, summaries, coverages, grid_tables, vertices = {}, [], [], [], []
    qa = check_math()
    max_reference = 0.0
    composite_errors, exposure_errors, membership_checks, pairwise_errors, mean_errors = [], [], [], [], []
    for city in selected_cities:
        path = source / 'spatial_data' / f'{city}_composite_risk_and_hotspots.csv'
        signatures[f'{city}_stored_composite_table'] = signature(path)
        needed = ['grid_id', 'city', 'population', 'population_status', 'centroid_x', 'centroid_y', 'analysis_included']
        needed += [f'{prefix}_T{t}' for t in PERIODS for prefix in ['E', 'R', 'A_rescue', 'A_medical', 'A_shelter']]
        frame = pd.read_csv(path, encoding='utf-8-sig', dtype={'grid_id': str}, usecols=needed)
        pop, original_weights, original_dims, original_common, coverage, _ = read_city(city)
        if not frame.grid_id.is_unique or set(frame.grid_id) != set(pop.grid_id):
            raise ValueError()
        frame = frame.set_index('grid_id').loc[pop.grid_id].reset_index()
        if not frame.city.eq(city).all() or not frame.analysis_included.isin([0, 1]).all():
            raise ValueError()
        included = frame.analysis_included.eq(1).to_numpy()
        if not np.array_equal(included, original_common):
            raise ValueError()
        membership_checks.append(city)
        if not np.allclose(frame.population, original_weights, rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError()
        for coordinate in ['centroid_x', 'centroid_y']:
            if not np.allclose(frame[coordinate], pop[coordinate], rtol=0, atol=1e-5):
                raise ValueError()
        population = original_weights[included]
        known = np.isfinite(original_weights)
        coverages.append(coverage)
        for j, period in enumerate(PERIODS):
            counts = pop[f'population_exposed_T{period}'].to_numpy(float)
            max_reference = max(max_reference, float(np.max(counts[known])))
            expected_exposure = counts[included] / ref
            exposure = frame[f'E_T{period}'].to_numpy(float)
            risk = frame[f'R_T{period}'].to_numpy(float)
            service = np.column_stack([original_dims[dimension][included, j] for dimension in DIMENSIONS])
            expected_risk = expected_exposure / 2 + service.sum(axis=1) / 6
            if not (np.isfinite(risk[included]).all() and np.isnan(risk[~included]).all()
                    and np.all(risk[included] >= 0) and np.all(risk[included] <= 1)):
                raise ValueError()
            if not np.allclose(exposure[included], expected_exposure, rtol=0, atol=1e-12):
                raise ValueError()
            if not np.allclose(risk[included], expected_risk, rtol=0, atol=1e-12):
                raise ValueError()
            for key, dimension in zip(KEYS, DIMENSIONS):
                if not np.allclose(frame.loc[included, f'A_{key}_T{period}'], original_dims[dimension][included, j], rtol=0, atol=1e-12):
                    raise ValueError()
            composite_errors.append(float(np.max(np.abs(risk[included] - expected_risk))))
            exposure_errors.append(float(np.max(np.abs(exposure[included] - expected_exposure))))
            values = risk[included]
            x, y, gini, mean, status = lorenz(values, population)
            curves[(city, period)] = (x, y)
            independent_mean = np.average(expected_exposure, weights=population) / 2 + np.sum(np.average(service, axis=0, weights=population)) / 6
            mean_errors.append(abs(mean - independent_mean))
            if mean_errors[-1] > 1e-12:
                raise AssertionError()
            subset = np.unique(np.linspace(0, len(values) - 1, min(128, len(values))).astype(int))
            subset_gini, independent_gini = lorenz(values[subset], population[subset])[2], pairwise_gini(values[subset], population[subset])
            if not ((np.isnan(subset_gini) and np.isnan(independent_gini)) or np.isclose(subset_gini, independent_gini, rtol=0, atol=1e-11)):
                raise AssertionError()
            if np.isfinite(independent_gini):
                pairwise_errors.append(abs(subset_gini - independent_gini))
            summaries.append({'city': city, 'city_name': CITY_NAMES[city], 'return_period': period,
                'gini': gini, 'population_weighted_mean_CFRI': mean, 'curve_status': status,
                'analytic_grid_count': int(included.sum()), 'analytic_population': float(population.sum()),
                'population_weighted_CFRI_burden': float(np.dot(values, population)),
                'known_population_coverage': coverage['known_population_coverage'],
                'highest_risk_10pct_population_burden_share': float(1 - np.interp(0.9, x, y)) if x.size else np.nan})
            if x.size:
                vertices.append(pd.DataFrame({'city': city, 'return_period': period, 'vertex': np.arange(x.size),
                    'cumulative_population_share': x, 'cumulative_population_weighted_risk_share': y}))
            table = frame[['grid_id', 'city', 'population', 'population_status', 'centroid_x', 'centroid_y', 'analysis_included']].copy()
            table['return_period'] = period
            table['inundated_people'] = counts
            table['E'], table['CFRI'] = exposure, risk
            for key in KEYS:
                table[f'A_{key}'] = frame[f'A_{key}_T{period}'].to_numpy()
            table['population_times_CFRI'] = original_weights * risk
            grid_tables.append(table)
        print(f'{CITY_NAMES[city]}: {included.sum():,} analytic grids; stored R verified for all five scenarios.', flush=True)
    if not np.isclose(max_reference, ref, rtol=0, atol=1e-9):
        raise ValueError()
    summary, coverage = pd.DataFrame(summaries), pd.DataFrame(coverages)
    summary.to_csv(output / 'CFRI_Gini_and_mean_risk.csv', index=False, encoding='utf-8-sig')
    coverage.to_csv(output / 'Analysis_population_coverage.csv', index=False, encoding='utf-8-sig')
    pd.concat(grid_tables, ignore_index=True).to_csv(output / '500m_grid_CFRI.csv.gz', index=False,
                                                    encoding='utf-8-sig', compression={'method': 'gzip', 'mtime': 0})
    pd.concat(vertices, ignore_index=True).to_csv(output / 'CFRI_Lorenz_vertices.csv.gz', index=False,
                                                 encoding='utf-8-sig', compression={'method': 'gzip', 'mtime': 0})
    city_period_count = len(selected_cities) * len(PERIODS)
    qa.update({'stored_R_formula_checks': city_period_count, 'max_stored_R_difference': max(composite_errors),
               'max_stored_E_difference': max(exposure_errors), 'fixed_exposure_reference_people': ref,
               'original_analytic_membership_checks': membership_checks,
               'independent_real_data_gini_checks': city_period_count, 'max_pairwise_gini_difference': max(pairwise_errors, default=0),
               'max_linear_combination_mean_difference': max(mean_errors), 'curve_count': len(curves)})
    return summary, coverage, curves, signatures, metadata, city_order, qa

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',choices=CITY_CODES,default=CITIES)
    cities=city_selection(parser.parse_args().cities)
    output=output_root('inequality')
    output.mkdir(parents=True,exist_ok=False)
    source = Path(os.environ.get('RELEASE_COMPOSITE_INPUT', ANALYSIS/'validated_bivariate'))
    prepare_composite(source,output,cities)
