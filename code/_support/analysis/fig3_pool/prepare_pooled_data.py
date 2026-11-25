"""Pool original resident-grid records across five cities in 1-km intervals.

Manuscript tables are archived; model values are generated separately. This
module does not rerun routes or calculate grid-level EFAR.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DLL_HANDLES = []
_lib = Path(sys.prefix)/'Library/bin'
if os.name == 'nt' and _lib.is_dir():
    os.environ['PATH'] = str(_lib)+os.pathsep+os.environ.get('PATH', '')
    DLL_HANDLES.append(os.add_dll_directory(str(_lib)))

import numpy as np
import pandas as pd

CITIES = ['zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan']
FUNCTIONS = ['Emergency rescue', 'Healthcare', 'Emergency shelter']
PERIODS = [10, 20, 50, 100, 200]
BIN_WIDTH_M = 1000.0
from _release import ROOT as PROJECT
RELEASE_DATA=PROJECT/'results/analysis/fig3/frozen'
RAW_DIR=RELEASE_DATA/'raw'
CITY_DIR=RELEASE_DATA/'city'
DIRECTIONS = {'Emergency rescue': 'facility_to_resident',
              'Healthcare': 'resident_to_facility', 'Emergency shelter': 'resident_to_facility'}


def require(condition, message):
    if not condition:
        raise RuntimeError()


def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def _verified_source(folder, name):
    path = folder/name
    manifest_path = folder/'Figure_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    actual = signature(path)
    matches = sum(item['sha256'] == actual['sha256'] and item['bytes'] == actual['bytes']
                  for item in manifest['outputs'])
    require(matches == 1, '')
    return path, [actual, signature(manifest_path)]


def prepare_pooled_data():
    """Return (pooled means, city population composition, audit).

    The composition table is fixed across all rainfall scenarios because the
    baseline distance and resident population used for grouping are fixed.
    """
    raw_path, raw_provenance = _verified_source(RAW_DIR, 'Grid_source_data.csv.gz')
    city_path, city_provenance = _verified_source(CITY_DIR, 'Distance_bin_1km_source_data.csv')
    points = pd.read_csv(raw_path, dtype={'grid_id': str}, float_precision='round_trip')
    city_means = pd.read_csv(city_path, float_precision='round_trip')
    require(set(points.city) == set(CITIES) and set(points.function) == set(FUNCTIONS)
            and set(points.period) == set(PERIODS), '')
    require(not points.duplicated(['city', 'function', 'period', 'grid_id']).any(), '')
    require(np.isfinite(points.population).all() and points.population.gt(0).all(), '')
    require(np.isfinite(points.nearest_baseline_distance_m).all()
            and points.nearest_baseline_distance_m.gt(0).all(), '')
    require(np.isfinite(points.EFAR).all() and points.EFAR.between(0, 1).all(), '')
    local_flag = points.same_grid_facility.astype(str).str.lower()
    require(local_flag.isin(['true', 'false', '1', '0']).all(), '')
    points['same_grid_facility'] = local_flag.isin(['true', '1'])
    require(points.loc[points.same_grid_facility, 'nearest_baseline_distance_m'].eq(250).all(), '')
    require(city_means.bin_upper_m.sub(city_means.bin_lower_m).eq(1000).all(), '')

    pooled_frames, composition_frames, checks, support = [], [], [], []
    common_ids = common_population = None
    max_mean_error = max_city_merge_error = 0.0
    for function in FUNCTIONS:
        original = points.loc[points.function.eq(function)]
        require(original.nearest_distance_direction.eq(DIRECTIONS[function]).all()
                and original.EFAR_direction.eq(DIRECTIONS[function]).all(), '')
        reference = original.loc[original.period.eq(PERIODS[0])].set_index(['city', 'grid_id']).sort_index()
        ids = reference.index
        p = reference.population.to_numpy(float)
        distance = reference.nearest_baseline_distance_m.to_numpy(float)
        cities = reference.index.get_level_values('city').to_numpy()
        if common_ids is None:
            common_ids, common_population = ids.copy(), p.copy()
        require(ids.equals(common_ids) and np.array_equal(p, common_population), '')
        n = int(np.ceil(distance.max()/BIN_WIDTH_M))
        assigned = np.ceil(distance/BIN_WIDTH_M).astype(np.int64)-1
        lower = np.arange(n)*BIN_WIDTH_M
        upper = lower+BIN_WIDTH_M
        require((distance > lower[assigned]).all() and (distance <= upper[assigned]).all(), '')
        counts = np.bincount(assigned, minlength=n)
        pop = np.bincount(assigned, weights=p, minlength=n)
        total = float(p.sum())
        require(counts.sum() == len(ids) and np.isclose(pop.sum(), total, atol=1e-6, rtol=1e-12), '')
        city_count = np.zeros(n, dtype=np.int64)
        for city in CITIES:
            mask = cities == city
            city_pop = np.bincount(assigned[mask], weights=p[mask], minlength=n)
            city_grids = np.bincount(assigned[mask], minlength=n)
            city_count += city_grids > 0
            composition_frames.append(pd.DataFrame({
                'city': city, 'function': function, 'bin_id': np.arange(n),
                'bin_lower_m': lower, 'bin_upper_m': upper,
                'grid_count': city_grids, 'population': city_pop,
                'pooled_bin_grid_count': counts, 'pooled_bin_population': pop,
                'share_of_pooled_bin_population': np.divide(city_pop, pop, out=np.full(n, np.nan), where=pop > 0),
                'share_of_city_analysis_population': city_pop/p[mask].sum(),
            }))
        support.append({'function': function, 'grid_count_per_scenario': len(ids),
                        'pooled_analysis_population': total, 'number_of_bins': n,
                        'empty_bins': int((counts == 0).sum()),
                        'max_baseline_distance_m': float(distance.max())})
        for period in PERIODS:
            frame = original.loc[original.period.eq(period)].set_index(['city', 'grid_id']).sort_index()
            require(frame.index.equals(ids) and np.array_equal(frame.population.to_numpy(), p)
                    and np.array_equal(frame.nearest_baseline_distance_m.to_numpy(), distance),
                    '')
            y = frame.EFAR.to_numpy(float)
            numerator = np.bincount(assigned, weights=p*y, minlength=n)
            mean = np.divide(numerator, pop, out=np.full(n, np.nan), where=pop > 0)
            require(np.array_equal(np.isnan(mean), counts == 0), '')
            direct_mean = float(np.dot(p, y)/total)
            reconstructed = float(np.nansum(mean*pop)/total)
            error = abs(direct_mean-reconstructed)
            max_mean_error = max(max_mean_error, error)
            require(error < 1e-12, '')
            comparison = city_means.loc[city_means.function.eq(function) & city_means.period.eq(period)]
            comp_ids = comparison.bin_id.to_numpy(dtype=np.int64)
            comp_pop = np.bincount(comp_ids, weights=comparison.population, minlength=n)
            comp_counts = np.bincount(comp_ids, weights=comparison.grid_count, minlength=n)
            comp_numerator = np.bincount(comp_ids,
                weights=np.nan_to_num(comparison.mean_EFAR_population_weighted.to_numpy())*comparison.population.to_numpy(), minlength=n)
            merged = np.divide(comp_numerator, comp_pop, out=np.full(n, np.nan), where=comp_pop > 0)
            require(np.array_equal(comp_counts, counts) and np.allclose(comp_pop, pop, rtol=1e-12, atol=1e-6),
                    '')
            merge_error = float(np.max(np.abs(merged[counts > 0]-mean[counts > 0])))
            max_city_merge_error = max(max_city_merge_error, merge_error)
            require(np.array_equal(np.isnan(merged), np.isnan(mean)) and merge_error < 1e-12,
                    '')
            checks.append({'function': function, 'period': period,
                           'original_pooled_population_weighted_EFAR': direct_mean,
                           'reconstructed_from_bins_EFAR': reconstructed,
                           'absolute_difference': error, 'max_city_bin_merge_difference': merge_error})
            pooled_frames.append(pd.DataFrame({
                'function': function, 'period': period, 'bin_id': np.arange(n),
                'bin_lower_m': lower, 'bin_upper_m': upper, 'distance_midpoint_km': (lower+upper)/2000,
                'grid_count': counts, 'population': pop, 'population_share': pop/total,
                'city_count': city_count, 'mean_EFAR_population_weighted': mean,
            }))
    pooled = pd.concat(pooled_frames, ignore_index=True)
    composition = pd.concat(composition_frames, ignore_index=True)
    require(len(checks) == 15 and not pooled.duplicated(['function','period','bin_id']).any(), '')
    audit = {
        'definitions': {
            'cities': CITIES, 'functions': FUNCTIONS, 'periods': PERIODS, 'bin_width_m': BIN_WIDTH_M,
            'bin_convention': '(lower, upper], starting with (0, 1000] m',
            'x': 'Midpoint of original baseline nearest-facility network-distance interval, in km',
            'y': 'sum(population_i * original_EFAR_i) / sum(population_i) across all city grids inside the interval',
            'aggregation': 'Resident grids are pooled before calculating the population-weighted mean; cities are not equally weighted',
            'population_domain': 'Original common valid resident population for three functions within each city',
            'same_grid': 'Existing 250-m assumption retained in the first 1-km bin',
            'directions': DIRECTIONS, 'empty_bins': 'Explicit NaN means; no interpolation',
            'source_values': 'Model values are generated separately from archived manuscript tables',
            'city_composition': 'Population contribution by city in each function/distance bin, fixed across all scenarios',
        },
        'validation': {
            'source_hashes': 'passed', 'original_scope_and_directions': 'passed',
            'common_population_and_scenario_membership': 'passed', 'right_closed_bin_membership': 'passed',
            'all_grids_and_population_retained': 'passed', 'pooled_mean_reconstruction': 'passed',
            'max_pooled_mean_reconstruction_error': max_mean_error,
            'city_bin_population_weighted_merge': 'passed', 'max_city_bin_merge_error': max_city_merge_error,
            'function_scenario_combinations': len(checks), 'raw_record_rows': len(points),
            'pooled_grid_count_per_scenario_and_function': len(common_ids),
            'pooled_analysis_population': float(common_population.sum()),
            'bin_rows': len(pooled), 'empty_bin_rows': int(pooled.grid_count.eq(0).sum()),
            'route_calls': 0, 'grid_EFAR_recalculated': False, 'original_data_changed': False,
        },
        'sources': raw_provenance+city_provenance, 'distance_support': support, 'mean_checks': checks,
    }
    return pooled, composition, audit


if __name__ == '__main__':
    pooled, composition, audit = prepare_pooled_data()
    print(json.dumps(audit['validation'], ensure_ascii=False, indent=2))
    print(pooled.groupby('function').mean_EFAR_population_weighted.agg(['min','max']).to_string())
