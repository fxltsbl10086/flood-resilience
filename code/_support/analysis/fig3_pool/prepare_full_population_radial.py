"""Rebuild radial EFAR components for the complete fixed analysis population.

Reads the original route-decomposition archives and per-grid EFAR export. It
does not calculate routes or modify inputs. Running this file writes only the
radial source CSV in the output directory.
"""
from pathlib import Path
from itertools import product
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DLL_HANDLES = []
_library = Path(sys.prefix) / 'Library/bin'
if os.name == 'nt' and _library.is_dir():
    os.environ['PATH'] = str(_library) + os.pathsep + os.environ.get('PATH', '')
    DLL_HANDLES.append(os.add_dll_directory(str(_library)))

import numpy as np
import pandas as pd

CITIES = ['zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan']
CITY_LABELS = dict(zip(CITIES, ['ZZ', 'HZ', 'FZ', 'CD', 'WH']))
FUNCTIONS = ['Emergency rescue', 'Healthcare', 'Emergency shelter']
PERIODS = [10, 20, 50, 100, 200]
KEYS = ['city', 'function', 'period']
from _release import ROOT as PROJECT
import os
RELEASE_DATA=PROJECT/'results/analysis/fig3/frozen'
CACHE=RELEASE_DATA/'decomposition'
TOTALS=RELEASE_DATA/'city_function_loss_mechanisms.csv'
RAW_DIR=RELEASE_DATA/'raw'
RAW=RAW_DIR/'Grid_source_data.csv.gz'
OLD_DIR=RELEASE_DATA/'previous_radial'
OLD_SOURCE=OLD_DIR/'Radial_figure_source_data.csv'
OUT_DIR=Path(os.environ.get('RELEASE_ANALYSIS_OUTPUT',str(PROJECT/'results/analysis/recomputed/fig3_radial')))
OUT_CSV=OUT_DIR/'Radial_figure_source_data.csv'


def require(condition, message):
    if not condition:
        raise RuntimeError()


def signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': digest.hexdigest()}


def verified_manifest_source(path, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    actual = signature(path)
    matches = sum(item['sha256'] == actual['sha256'] for item in manifest['outputs'])
    require(matches == 1, '')
    return [actual, signature(manifest_path)]


def prepare_data():
    """Return 75 full-population summaries and their reproducibility audit."""
    sources = verified_manifest_source(RAW, RAW_DIR / 'Figure_manifest.json')
    sources += verified_manifest_source(OLD_SOURCE, OLD_DIR / 'Figure_manifest.json')
    sources.append(signature(TOTALS))
    raw = pd.read_csv(RAW, dtype={'grid_id': str}, float_precision='round_trip')
    totals = pd.read_csv(TOTALS, float_precision='round_trip')
    old = pd.read_csv(OLD_SOURCE, float_precision='round_trip')
    expected = set(product(CITIES, FUNCTIONS, PERIODS))
    for label, data in [('grid export', raw), ('city summary', totals), ('previous radial source', old)]:
        require(set(data[KEYS].itertuples(index=False, name=None)) == expected,
                '')
        unique_key = KEYS + ['grid_id'] if label == 'grid export' else KEYS
        require(not data.duplicated(unique_key).any(), '')
    totals = totals.set_index(KEYS)
    old = old.set_index(KEYS)
    errors = {name: 0.0 for name in [
        'max_grid_EFAR_reconstruction_error', 'max_grid_component_additivity_error',
        'max_mean_component_additivity_error', 'max_fully_disconnected_EFAR_error',
        'max_fully_disconnected_D_error', 'max_fully_disconnected_U_error',
        'max_existing_city_summary_error', 'max_previous_all_grid_summary_error',
        'max_previous_reachable_summary_error', 'max_population_partition_reconstruction_error',
        'max_disconnection_share_reconstruction_error']}
    rows, support = [], []

    def check_error(name, value, message, tolerance=1e-9):
        value = float(value)
        errors[name] = max(errors[name], value)
        require(value < tolerance, '')

    for city in CITIES:
        path = CACHE / city / 'resource_distance_decomposition.npz'
        record_path = path.with_suffix('.json')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        actual = signature(path)
        require(actual['sha256'] == record['output_signature']['sha256'],
                '')
        sources.extend([actual, signature(record_path)])
        with np.load(path, allow_pickle=False) as archive:
            ix = archive['analysis_indices']
            ids = archive['analysis_grid_ids'].astype(str)
            population = archive['population']
            require(np.array_equal(archive['grid_ids'][ix].astype(str), ids),
                    '')
            base = archive['baseline_access'][ix]
            scene = archive['scenario_access'][:, ix]
            lost = archive['disconnected_access_loss'][:, ix]
            detour = archive['detour_access_loss'][:, ix]
            remaining = archive['scenario_resource_count'][:, ix].sum(axis=3)
            baseline_count = archive['baseline_resource_count'][ix].sum(axis=2)
            local = archive['counts'][ix] > 0
        n = len(ids)
        require(len(set(ids)) == n and population.shape == (n,), '')
        require(np.isfinite(population).all() and (population > 0).all(),
                '')
        require(base.shape == (n, 3, 7) and scene.shape == (5, n, 3, 7)
                and lost.shape == scene.shape and detour.shape == scene.shape,
                '')
        for label, values in [('baseline', base), ('scenario', scene), ('disconnection', lost),
                              ('detour', detour), ('remaining', remaining)]:
            require(np.isfinite(values).all() and (values >= 0).all(), '')
        base_total = base.sum(axis=2)
        scene_total = scene.sum(axis=3)
        require((base_total > 0).all() and (baseline_count > 0).all(),
                '')
        risk = 1 - scene_total / base_total[None, :, :]
        d = lost.sum(axis=3) / base_total[None, :, :]
        u = detour.sum(axis=3) / base_total[None, :, :]
        check_error('max_grid_component_additivity_error', np.max(np.abs(d + u - risk)),
                    f'{city}: per-grid components are not additive')
        disconnected = remaining == 0
        require(np.array_equal(disconnected, scene_total == 0),
                '')
        if disconnected.any():
            for name, values, target in [('EFAR', risk, 1), ('D', d, 1), ('U', u, 0)]:
                check_error(f'max_fully_disconnected_{name}_error',
                            np.max(np.abs(values[disconnected] - target)),
                            f'{city}: fully disconnected {name} differs from {target}')
        require(not np.any(disconnected & local[None, :, :]), '')
        require(np.allclose(base[:, :, 0], scene[:, :, :, 0], rtol=0, atol=1e-12)
                and (lost[:, :, :, 0] == 0).all() and (detour[:, :, :, 0] == 0).all(),
                '')
        pop_sum = float(population.sum())
        weight = population / pop_sum
        city_raw = raw.loc[raw.city.eq(city)]
        for fi, function in enumerate(FUNCTIONS):
            for si, period in enumerate(PERIODS):
                key = (city, function, period)
                label = f'{city}/{function}/T{period}'
                original = city_raw.loc[city_raw.function.eq(function) & city_raw.period.eq(period)]
                require(len(original) == n and set(original.grid_id) == set(ids),
                        '')
                original = original.set_index('grid_id').loc[ids]
                require(np.allclose(original.population.to_numpy(), population, rtol=0, atol=1e-6),
                        '')
                original_risk = original.EFAR.to_numpy()
                require(np.isfinite(original_risk).all() and ((original_risk >= 0) & (original_risk <= 1)).all(),
                        '')
                check_error('max_grid_EFAR_reconstruction_error', np.max(np.abs(original_risk - risk[si, :, fi])),
                            f'{label}: per-grid original EFAR differs')
                means = np.array([np.sum(weight * original_risk), np.sum(weight * d[si, :, fi]),
                                  np.sum(weight * u[si, :, fi])], dtype=float)
                mean, mean_d, mean_u = means.tolist()
                check_error('max_mean_component_additivity_error', abs(mean_d + mean_u - mean),
                            f'{label}: full-population mean components differ')
                reference, previous = totals.loc[key], old.loc[key]
                require(int(reference.analysis_grid_count) == n and int(previous.analysis_grid_count) == n
                        and abs(float(reference.analysis_population) - pop_sum) < 1e-5
                        and abs(float(previous.analysis_population) - pop_sum) < 1e-5,
                        '')
                reference_means = np.array([reference.weighted_mean_EFAR, reference.disconnection_EFAR_contribution,
                                            reference.detour_EFAR_contribution], dtype=float)
                previous_means = np.array([previous.all_grid_mean_EFAR, previous.all_grid_disconnection_loss,
                                           previous.all_grid_detour_loss], dtype=float)
                check_error('max_existing_city_summary_error', np.max(np.abs(means - reference_means)),
                            f'{label}: city summary does not reconstruct')
                check_error('max_previous_all_grid_summary_error', np.max(np.abs(means - previous_means)),
                            f'{label}: previous all-grid summary does not reconstruct')
                reachable = remaining[si, :, fi] > 0
                n_reachable = int(reachable.sum())
                reachable_pop = float(population[reachable].sum())
                require(n_reachable > 0 and reachable_pop > 0, '')
                reachable_weight = population[reachable] / reachable_pop
                reachable_means = np.array([np.sum(reachable_weight * original_risk[reachable]),
                                            np.sum(reachable_weight * d[si, reachable, fi]),
                                            np.sum(reachable_weight * u[si, reachable, fi])], dtype=float)
                previous_reachable = np.array([previous.mean_EFAR, previous.disconnection_loss,
                                               previous.detour_loss], dtype=float)
                check_error('max_previous_reachable_summary_error', np.max(np.abs(reachable_means - previous_reachable)),
                            f'{label}: previous conditional summary does not reconstruct')
                disconnected_pop = float(population[~reachable].sum())
                reachable_share = reachable_pop / pop_sum
                disconnected_share = disconnected_pop / pop_sum
                partition_means = reachable_share * reachable_means + disconnected_share * np.array([1., 1., 0.])
                check_error('max_population_partition_reconstruction_error', np.max(np.abs(means - partition_means)),
                            f'{label}: reachable and disconnected populations do not reconstruct full means')
                require(mean_d + mean_u > 0, '')
                d_share = mean_d / (mean_d + mean_u)
                check_error('max_disconnection_share_reconstruction_error', abs(d_share - mean_d / mean),
                            f'{label}: disconnection contribution ratio differs')
                rows.append({
                    'city': city, 'city_label': CITY_LABELS[city], 'function': function, 'period': period,
                    'population': pop_sum, 'grid_count': n,
                    'mean_EFAR': mean, 'disconnection_loss': mean_d, 'detour_loss': mean_u,
                    'analysis_population': pop_sum, 'analysis_grid_count': n,
                    'all_grid_mean_EFAR': mean, 'all_grid_disconnection_loss': mean_d, 'all_grid_detour_loss': mean_u,
                    'reachable_grid_count': n_reachable, 'reachable_population': reachable_pop,
                    'reachable_population_share': reachable_share, 'disconnected_grid_count': n - n_reachable,
                    'disconnected_population': disconnected_pop, 'disconnected_population_share': disconnected_share,
                    'disconnected_grid_share': (n - n_reachable) / n, 'reachable_grid_share': n_reachable / n,
                    'same_grid_grid_count': int(local[:, fi].sum()),
                    'reachable_mean_EFAR': float(reachable_means[0]),
                    'reachable_mean_disconnection_loss': float(reachable_means[1]),
                    'reachable_mean_detour_loss': float(reachable_means[2]),
                    'disconnection_share': d_share, 'disconnection_share_labelled': period in [10, 200],
                })
        support.append({'city': city, 'analysis_grid_count': n, 'analysis_population': pop_sum})

    frame = pd.DataFrame(rows)
    require(len(frame) == 75 and not frame.duplicated(KEYS).any(), '')
    require((frame.groupby('city')[['population', 'grid_count']].nunique() == 1).all().all(),
            '')
    require((frame.population == frame.analysis_population).all()
            and (frame.grid_count == frame.analysis_grid_count).all(), '')
    require(int(frame.disconnection_share_labelled.sum()) == 30, '')
    range_columns = ['mean_EFAR', 'disconnection_loss', 'detour_loss', 'disconnection_share',
                     'reachable_mean_EFAR', 'reachable_population_share', 'disconnected_population_share']
    for column in range_columns:
        require(np.isfinite(frame[column]).all() and frame[column].between(0, 1).all(), '')
    audit = {
        'schema_version': 1,
        'definitions': {
            'scope': 'Five cities; all common-valid, positive-population resident grids in each original analysis_indices set, including fully disconnected grids',
            'denominator': 'population and grid_count equal analysis_population and analysis_grid_count and are fixed within each city across all three functions and five scenarios',
            'mean_EFAR': 'sum_i(population_i * EFAR_i) / sum_i(population_i) over the full fixed analysis set',
            'components': 'D and U use the same full-population weights; D + U = mean_EFAR',
            'disconnection_share': 'D / (D + U); ratio of population-weighted components, not disconnected-link or disconnected-population fraction',
            'fully_disconnected': 'remaining facility count = 0 and scenario accessibility = 0; EFAR = D = 1 and U = 0',
            'reachable_auxiliary': 'reachable_mean_* recompute and verify the previous reachable-only conditional summaries for audit; they are not plotted values',
            'normalization': 'Original inverse-distance and destination-POI-count-weighted accessibility, with each grid normalized by its own full baseline accessibility',
            'same_grid': '250 m; same-grid facilities remain reachable with zero loss in all scenarios under the original model',
            'directions': {'Emergency rescue': 'facility_to_resident', 'Healthcare': 'resident_to_facility',
                           'Emergency shelter': 'resident_to_facility'},
            'top_row': 'Manuscript tables are archived; model values are generated separately',
        },
        'validation': {'combinations': len(frame), 'source_manifest_signatures': 'passed',
                       'fixed_common_analysis_population': 'passed', 'original_grid_scope_and_weights': 'passed',
                       'zero_count_equals_zero_accessibility': 'passed', 'fully_disconnected_components': 'passed',
                       'same_grid_accessibility': 'passed', 'share_labels_T10_T200': 30, **errors},
        'support': support,
        'analysis_totals': {'grid_count': sum(item['analysis_grid_count'] for item in support),
                            'population': sum(item['analysis_population'] for item in support)},
        'ranges': {column: [float(frame[column].min()), float(frame[column].max())] for column in range_columns},
        'sources': sources,
        'code': signature(Path(__file__)),
    }
    return frame, audit


def main():
    frame, _ = prepare_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(OUT_CSV)


if __name__ == '__main__':
    main()
