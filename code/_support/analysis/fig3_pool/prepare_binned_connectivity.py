"""Raw POI-link connectivity grouped by each resident's nearest facility.

The horizontal grouping matches the original pooled EFAR figure. It does not
group individual resident--facility links by their own route length. Functions
return source tables without writing or changing any input file.
"""
import json

from prepare_pooled_data import (PROJECT, RAW_DIR, CITIES, FUNCTIONS, PERIODS,
                                 BIN_WIDTH_M, DIRECTIONS, _verified_source,
                                 require, signature)
import numpy as np
import pandas as pd

CACHE = PROJECT / 'results/analysis/fig3/frozen/decomposition'
KEYS = ['function', 'period', 'bin_id']


def prepare_connectivity(pooled):
    """Return all five scenarios' connectivity and a JSON-serializable audit.

    A link is one baseline-reachable resident-grid--facility-POI pair. A bin's
    disconnected_share is disconnected link count / baseline link count,
    pooling all five cities without population or inverse-distance weights.
    Reachable links include unchanged and detoured routes. Population-weighted
    EFAR components are retained separately only for source validation.
    """
    require(not pooled.duplicated(KEYS).any(), '')
    require(set(pooled.function) == set(FUNCTIONS) and set(pooled.period) == set(PERIODS),
            '')
    raw_path, sources = _verified_source(RAW_DIR, 'Grid_source_data.csv.gz')
    raw = pd.read_csv(raw_path, dtype={'grid_id': str}, float_precision='round_trip')
    require(not raw.duplicated(['city', 'function', 'period', 'grid_id']).any(),
            '')
    frames, support = [], []
    max_grid_error = max_additivity_error = 0.0
    for city in CITIES:
        path = CACHE / city / 'resource_distance_decomposition.npz'
        meta_path = path.with_suffix('.json')
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        archive_sig = signature(path)
        require(archive_sig['sha256'] == metadata['output_signature']['sha256'],
                '')
        sources.extend([archive_sig, signature(meta_path)])
        with np.load(path, allow_pickle=False) as archive:
            ix = archive['analysis_indices']
            ids = archive['analysis_grid_ids'].astype(str)
            population = archive['population']
            require(np.array_equal(archive['grid_ids'][ix].astype(str), ids),
                    '')
            baseline = archive['baseline_access'][ix]
            scenario = archive['scenario_access'][:, ix]
            lost = archive['disconnected_access_loss'][:, ix]
            detour = archive['detour_access_loss'][:, ix]
            baseline_count = archive['baseline_resource_count'][ix].sum(axis=2)
            remaining_count = archive['scenario_resource_count'][:, ix].sum(axis=3)
            local_count = archive['counts'][ix]
            require(not any('detour' in key.lower() and 'count' in key.lower()
                            for key in archive.files),
                    '')
        n = len(ids)
        require(population.shape == (n,) and np.isfinite(population).all()
                and (population > 0).all(), '')
        require(baseline.shape == (n, 3, 7) and scenario.shape == (5, n, 3, 7),
                '')
        require(baseline_count.shape == (n, 3) and remaining_count.shape == (5, n, 3),
                '')
        for label, values in [('baseline_count', baseline_count), ('remaining_count', remaining_count)]:
            require(np.isfinite(values).all() and (values >= 0).all()
                    and np.array_equal(values, np.rint(values)),
                    '')
        baseline_count = baseline_count.astype(np.int64)
        remaining_count = remaining_count.astype(np.int64)
        require((baseline_count > 0).all()
                and (remaining_count <= baseline_count[None, :, :]).all(),
                '')
        require((remaining_count >= local_count[None, :, :]).all(),
                '')
        for label, values in [('baseline', baseline), ('scenario', scenario),
                              ('lost', lost), ('detour', detour)]:
            require(np.isfinite(values).all() and (values >= 0).all(),
                    '')
        baseline_total = baseline.sum(axis=2)
        require((baseline_total > 0).all(), '')
        d = lost.sum(axis=3) / baseline_total[None, :, :]
        u = detour.sum(axis=3) / baseline_total[None, :, :]
        efar = 1 - scenario.sum(axis=3) / baseline_total[None, :, :]
        add_error = float(np.max(np.abs(d + u - efar)))
        max_additivity_error = max(max_additivity_error, add_error)
        require(add_error < 1e-9, '')
        require(np.allclose(baseline[:, :, 0], scenario[:, :, :, 0], rtol=0, atol=1e-12)
                and (lost[:, :, :, 0] == 0).all() and (detour[:, :, :, 0] == 0).all(),
                '')

        city_raw = raw.loc[raw.city.eq(city)]
        for fi, function in enumerate(FUNCTIONS):
            reference_distance = None
            for ti, period in enumerate(PERIODS):
                original = city_raw.loc[city_raw.function.eq(function) & city_raw.period.eq(period)]
                require(len(original) == n and set(original.grid_id) == set(ids),
                        '')
                original = original.set_index('grid_id').loc[ids]
                require(np.allclose(original.population.to_numpy(), population, rtol=0, atol=1e-6),
                        '')
                require(original.nearest_distance_direction.eq(DIRECTIONS[function]).all()
                        and original.EFAR_direction.eq(DIRECTIONS[function]).all(),
                        '')
                distances = original.nearest_baseline_distance_m.to_numpy(float)
                require(np.isfinite(distances).all() and (distances > 0).all(),
                        '')
                if reference_distance is None:
                    reference_distance = distances.copy()
                require(np.array_equal(reference_distance, distances),
                        '')
                local = original.same_grid_facility.astype(str).str.lower().isin(['true', '1']).to_numpy()
                require((distances[local] == 250).all(), '')
                original_risk = original.EFAR.to_numpy(float)
                error = float(np.max(np.abs(original_risk - efar[ti, :, fi])))
                max_grid_error = max(max_grid_error, error)
                require(error < 1e-8, '')
                bin_ids = np.ceil(distances / BIN_WIDTH_M).astype(np.int64) - 1
                require(((distances > bin_ids * BIN_WIDTH_M)
                         & (distances <= (bin_ids + 1) * BIN_WIDTH_M)).all(),
                        '')
                frames.append(pd.DataFrame({
                    'city': city, 'function': function, 'period': period,
                    'bin_id': bin_ids, 'population': population,
                    'weighted_D': population * d[ti, :, fi],
                    'weighted_U': population * u[ti, :, fi],
                    'baseline_links': baseline_count[:, fi],
                    'reachable_links': remaining_count[ti, :, fi],
                }))
        support.append({'city': city, 'grid_count': n, 'population': float(population.sum())})

    records = pd.concat(frames, ignore_index=True)
    groups = records.groupby(KEYS, sort=False).agg(
        population=('population', 'sum'), grid_count=('population', 'size'),
        city_count=('city', 'nunique'), weighted_D=('weighted_D', 'sum'),
        weighted_U=('weighted_U', 'sum'), baseline_links=('baseline_links', 'sum'),
        reachable_links=('reachable_links', 'sum'))
    index = pd.MultiIndex.from_frame(pooled[KEYS])
    aligned = groups.reindex(index)
    require(set(groups.index) == set(index[pooled.grid_count.gt(0)]),
            '')
    valid = pooled.grid_count.to_numpy() > 0
    pop = pooled.population.to_numpy(float)
    require(np.array_equal(aligned.grid_count.fillna(0).to_numpy(), pooled.grid_count.to_numpy())
            and np.array_equal(aligned.city_count.fillna(0).to_numpy(), pooled.city_count.to_numpy())
            and np.allclose(aligned.population.fillna(0).to_numpy(), pop, rtol=1e-12, atol=1e-6),
            '')
    weighted_d = np.divide(aligned.weighted_D.to_numpy(), pop,
                           out=np.full(len(pooled), np.nan), where=valid)
    detour_mean = np.divide(aligned.weighted_U.to_numpy(), pop,
                            out=np.full(len(pooled), np.nan), where=valid)
    mean_error = float(np.max(np.abs(weighted_d[valid] + detour_mean[valid]
                                      - pooled.mean_EFAR_population_weighted.to_numpy()[valid])))
    require(mean_error < 1e-9, '')
    baseline_links = aligned.baseline_links.fillna(0).to_numpy(np.int64)
    reachable_links = aligned.reachable_links.fillna(0).to_numpy(np.int64)
    disconnected_links = baseline_links - reachable_links
    require(np.array_equal(baseline_links > 0, valid) and (disconnected_links >= 0).all(),
            '')
    disconnected = np.divide(disconnected_links, baseline_links,
                             out=np.full(len(pooled), np.nan), where=valid)
    reachable = np.divide(reachable_links, baseline_links,
                          out=np.full(len(pooled), np.nan), where=valid)
    result = pooled.copy()
    result['baseline_link_count'] = baseline_links
    result['disconnected_link_count'] = disconnected_links
    result['reachable_link_count'] = reachable_links
    result['disconnected_share'] = disconnected
    result['reachable_share'] = reachable
    result['mean_disconnection_contribution'] = weighted_d
    result['mean_detour_contribution'] = detour_mean
    require(np.array_equal(result.disconnected_share.isna(), result.grid_count.eq(0)),
            '')
    complement_error = float(np.max(np.abs(
        result.loc[valid, 'disconnected_share'] + result.loc[valid, 'reachable_share'] - 1)))
    require(complement_error < 1e-12, '')
    audit = {
        'definitions': {
            'x': 'Resident baseline distance to nearest same-type facility; right-closed 1 km bins',
            'link_unit': 'One baseline-reachable resident-grid--facility-POI pair; multiple POIs in a facility grid count separately',
            'disconnected_share': 'sum(disconnected_link_count) / sum(baseline_link_count)',
            'reachable_share': 'sum(reachable_link_count) / sum(baseline_link_count); includes unchanged and detoured routes',
            'weighting': 'Raw link counts pooled across all five cities; no population or inverse-distance weights',
            'not_isolated_grid_proportion': 'Not a fraction of residents or grids with all facilities disconnected',
            'detoured_link_counts_available': False,
            'detour_limitation': 'The cache stores detour accessibility loss but no detoured link counts; weighted loss cannot identify the number of detoured links',
            'EFAR_component_columns': 'mean_disconnection_contribution and mean_detour_contribution remain population-weighted EFAR components for validation, not raw link shares',
            'count_source_logic': 'resource_distance.py sums POI counts times baseline/reachable link masks, including same-grid POI counts',
            'distinction_from_old_distance_curve': 'The old curve grouped each link by its own distance; this curve groups all links of a resident by the resident nearest-facility distance',
            'same_grid': 'Original 250 m and always-reachable assumption retained',
            'directions': DIRECTIONS,
        },
        'validation': {
            'rows': len(result), 'empty_rows': int((~valid).sum()),
            'raw_grid_rows': len(records), 'source_hashes': 'passed',
            'grid_and_group_population_alignment': 'passed',
            'max_grid_EFAR_reconstruction_error': max_grid_error,
            'max_grid_component_additivity_error': max_additivity_error,
            'max_pooled_component_additivity_error': mean_error,
            'max_connectivity_complement_error': complement_error,
            'integer_link_counts_and_conservation': 'passed',
            'route_calls': 0, 'original_data_changed': False,
        },
        'support': support, 'sources': sources,
    }
    return result, audit


if __name__ == '__main__':
    from prepare_pooled_data import prepare_pooled_data
    pooled, _, _ = prepare_pooled_data()
    data, report = prepare_connectivity(pooled)
    print(json.dumps(report['validation'], ensure_ascii=False, indent=2))
    print(data[['disconnected_share', 'reachable_share', 'mean_detour_contribution']].agg(['min', 'max']).to_string())
