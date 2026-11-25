"Weight resource-grid accessibility by the number of same-function facilities."
import argparse
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import bmat, csr_matrix
from settings import DEFAULT_CITIES, CITIES, DIMENSIONS, DIRECTIONS, PERIODS, routing, scope_name
from network_cache import get_network

def reverse_network(network):
    "Reverse road edges while retaining grid-source outgoing access edges."
    n = network['n_road_nodes']
    m = len(network['source_nodes'])
    graphs = {}
    for scenario, graph in network['graphs'].items():
        graphs[scenario] = bmat([[graph[:n, :n].T, csr_matrix((n, m))],
                                [graph[n:, :n], csr_matrix((m, m))]], format='csr')
    return dict(network, graphs=graphs)

def get_counts(city, output, grids):
    table = pd.read_csv(output / city / f'{city}_grid_facility_counts.csv', dtype={'grid_id': str})
    if table.grid_id.duplicated().any() or set(table.grid_id) != set(grids.grid_id.astype(str)):
        raise ValueError()
    return table.set_index('grid_id').loc[grids.grid_id.astype(str),
        [d + '数量' for d in DIMENSIONS]].to_numpy(np.int64)

def aggregate(city, output, network, grids, counts, batch_size):
    "Route resource grids in batches and aggregate statistics without storing a complete OD matrix."
    cache = output / city / f'{city}_facility_weighted_statistics.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if not np.array_equal(data['counts'], counts) or not np.array_equal(
                    data['grid_ids'], grids.grid_id.astype(str).to_numpy()):
                raise ValueError()
            return {key: data[key].copy() for key in data.files}
    n = len(grids)
    result = {'counts': counts, 'grid_ids': np.asarray(grids.grid_id, dtype=str),
        'base_count': np.zeros((n, 3)), 'scene_count': np.zeros((5, n, 3)),
        'base_inverse': np.zeros((n, 3)), 'scene_inverse': np.zeros((5, n, 3))}
    start_time = time.monotonic()
    # Rescue travels facility-to-resident; reversed routing gives resident-to-facility distances for other functions.
    for dims, current_network in [([0], network), ([1, 2], reverse_network(network))]:
        resources = np.flatnonzero(counts[:, dims].sum(axis=1) > 0)
        function_names = ("Emergency rescue", "Healthcare", "Emergency shelter")
        print(f'{city}: {"/".join(function_names[k] for k in dims)} {len(resources)} resource grids', flush=True)
        last_report = time.monotonic()
        for start in range(0, len(resources), batch_size):
            current = resources[start:start + batch_size]
            _, baseline = next(routing.compute_distances(current_network, current, 'baseline', batch_size))
            diagonal = np.arange(n)[None, :] == current[:, None]
            baseline_mask = np.isfinite(baseline) & (baseline > 0) & ~diagonal
            weights = counts[current][:, dims].astype(float)
            inverse = np.zeros_like(baseline)
            np.divide(1.0, baseline, out=inverse, where=baseline_mask)
            # Aggregate small dimension batches without BLAS to avoid delayed Windows DLL loading.
            result['base_count'][:, dims] += np.einsum('bn,bd->nd', baseline_mask, weights, optimize=False)
            result['base_inverse'][:, dims] += np.einsum('bn,bd->nd', inverse, weights, optimize=False)
            for s, period in enumerate(PERIODS):
                _, scene = next(routing.compute_distances(current_network, current, f'T{period}', batch_size))
                reachable = baseline_mask & np.isfinite(scene) & (scene > 0)
                if np.any(scene[reachable] + 1e-6 < baseline[reachable]):
                    raise AssertionError()
                inverse.fill(0)
                np.divide(1.0, scene, out=inverse, where=reachable)
                result['scene_count'][s][:, dims] += np.einsum('bn,bd->nd', reachable, weights, optimize=False)
                result['scene_inverse'][s][:, dims] += np.einsum('bn,bd->nd', inverse, weights, optimize=False)
            if time.monotonic() - last_report >= 30 or start + batch_size >= len(resources):
                print(f'{city}: Current direction completed {min(start + batch_size, len(resources))}/{len(resources)}, '
                      f'elapsed {time.monotonic() - start_time:.1f}s', flush=True)
                last_report = time.monotonic()
    # Store same-grid statistics separately so the local-distance convention can be evaluated without rerouting.
    result['elapsed_seconds'] = np.array(time.monotonic() - start_time)
    np.savez_compressed(cache, **result)
    return result

def export_city(city, output, grids, data, metric, local, local_distance_m):
    counts = data['counts']
    denominator = data['base_count'].copy()
    scene_counts = data['scene_count'].copy()
    if local == 'include':
        denominator += counts
        scene_counts += counts[None, :, :]
    if metric == 'count':
        base_sum, scene_sum = denominator, scene_counts
    else:
        base_sum = data['base_inverse'].copy()
        scene_sum = data['scene_inverse'].copy()
        if local == 'include':
            # Assign a representative distance only to same-grid resources; retain other network distances.
            base_sum += counts / local_distance_m
            scene_sum += counts[None, :, :] / local_distance_m
    valid = (denominator > 0) & (base_sum > 0)
    risks = np.full(scene_sum.shape, np.nan)
    np.divide(scene_sum, base_sum[None, :, :], out=risks, where=valid[None, :, :])
    risks = 1.0 - risks
    if np.any(risks[np.isfinite(risks)] < -1e-9) or np.any(risks[np.isfinite(risks)] > 1 + 1e-9):
        raise AssertionError()
    risks = np.clip(risks, 0, 1)
    directory = output / city
    directory.mkdir(parents=True, exist_ok=True)
    wide = grids.copy()
    for k, code in enumerate(['rescue', 'medical', 'shelter']):
        wide[f'{code}_count'] = counts[:, k]
        wide[f'{code}_base_reachable'] = denominator[:, k].astype(np.int64)
        for s, period in enumerate(PERIODS):
            wide[f'{code}_risk_T{period:03d}'] = risks[s, :, k]
    wide.to_file(directory / f'{city}_three_function_accessibility_risk.gpkg', layer='resource_risk_500m',
                 driver='GPKG', index=False)
    rows, summaries = [], []
    for k, dimension in enumerate(DIMENSIONS):
        table = pd.DataFrame({'city': city, '城市': CITIES[city],
            'grid_id': grids.grid_id.astype(str), '评估维度': dimension, '可达方向': DIRECTIONS[k],
            '格网内设施数量': counts[:, k], '基准可达设施数量': denominator[:, k].astype(np.int64),
            '状态': np.where(valid[:, k], '有效', '基准无可达资源'),
            '指标': metric, '同网格处理': local,
            '同网格距离m': local_distance_m if local == 'include' and metric == 'inverse' else np.nan,
            '基准加权可达水平': np.divide(base_sum[:, k], denominator[:, k],
                out=np.full(len(grids), np.nan), where=valid[:, k]),
            'centroid_x': grids.centroid_x, 'centroid_y': grids.centroid_y})
        for s, period in enumerate(PERIODS):
            risk = risks[s, :, k]
            table[f'风险_T{period:03d}'] = risk
            table[f'可达设施数_T{period:03d}'] = scene_counts[s, :, k].astype(np.int64)
            table[f'加权可达水平_T{period:03d}'] = np.divide(scene_sum[s, :, k], denominator[:, k],
                out=np.full(len(grids), np.nan), where=valid[:, k])
            values = risk[np.isfinite(risk)]
            summaries.append(dict(city=city, 城市=CITIES[city], 评估维度=dimension,
                可达方向=DIRECTIONS[k], 重现期=period, 指标=metric, 同网格处理=local,
                同网格距离m=local_distance_m if local == 'include' and metric == 'inverse' else np.nan,
                汇总权重='有效格网等权，未人口加权',
                总格网数=len(grids), 有效格网数=len(values), 基准无可达资源格网数=int((~valid[:, k]).sum()),
                格网内设施数=int(counts[:, k].sum()), 有设施格网数=int((counts[:, k] > 0).sum()),
                格网等权平均风险=float(values.mean()) if len(values) else np.nan,
                风险中位数=float(np.median(values)) if len(values) else np.nan,
                最大风险=float(values.max()) if len(values) else np.nan,
                完全不可达格网数=int(np.sum(scene_counts[s, valid[:, k], k] == 0)),
                有风险格网数=int(np.sum(values > 1e-10))))
        rows.append(table)
    pd.concat(rows, ignore_index=True).to_csv(directory / f'{city}_three_function_accessibility_risk.csv',
        index=False, encoding='utf-8-sig')
    return summaries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help="Directory containing per-city facility counts.")
    parser.add_argument('--result-dir', type=Path, help="Risk output directory, which may differ from facility and network-cache directories.")
    parser.add_argument('--cities', nargs='+', choices=list(CITIES), default=DEFAULT_CITIES)
    parser.add_argument('--metric', choices=['inverse', 'count'], default='inverse')
    parser.add_argument('--local', choices=['exclude', 'include'], default='include')
    parser.add_argument('--local-distance-m', type=float, default=250.0,
                        help="Representative same-grid facility distance in metres; other network distances are unchanged (default 250).")
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    args.cities = list(dict.fromkeys(args.cities))
    if args.batch_size <= 0:
        parser.error('')
    if not np.isfinite(args.local_distance_m) or args.local_distance_m <= 0:
        parser.error('')
    result_dir = args.result_dir or args.output
    result_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    grid_total = 0
    for city in args.cities:
        network, grids, metadata = get_network(city, args.output)
        counts = get_counts(city, args.output, grids)
        data = aggregate(city, args.output, network, grids, counts, args.batch_size)
        summaries.extend(export_city(city, result_dir, grids, data, args.metric, args.local,
                                     args.local_distance_m))
        grid_total += len(grids)
        print(f'{city}: Three-function risk results exported', flush=True)
    pd.DataFrame(summaries).to_csv(result_dir / f'{scope_name(args.cities)}_three_function_accessibility_risk_summary.csv',
                                 index=False, encoding='utf-8-sig')
    (result_dir / 'risk_calculation_methods.json').write_text(json.dumps(dict(
        input_cache_dir=str(args.output.resolve()), result_dir=str(result_dir.resolve()),
        cities=args.cities, total_grid_count=grid_total, grid_dimension_rows=grid_total * 3,
        risk_values_including_missing=grid_total * 3 * len(PERIODS),
        metric=args.metric, local=args.local,
        same_grid_distance_m=args.local_distance_m if args.metric == 'inverse' and args.local == 'include' else None,
        other_grid_distance='原路网距离不变；不加250米',
        same_grid_availability='本格资源在基准及所有五情景下均可达' if args.local == 'include' else '排除本格资源',
        distance_decay='1/d，只有本格资源d采用代表距离',
        direction=dict(zip(DIMENSIONS, DIRECTIONS)), periods=PERIODS,
        target_weights='同维度目的地格网内设施数量；救援按设施发出方向计权',
        baseline_resource_set='每个居民格网基准可达的固定资源集合',
        baseline_unreachable='风险为NaN，不能直接认定风险为1',
        within_grid='到达设施所在格网即认为设施可达，不计算设施点道路接入',
        summary='均值为有效格网等权均值，未使用人口权重',
        resource_failure='仅计道路通行变化，未判定资源自身淹没或失效',
        shelter_scope='有明确洪涝安置依据的补充点；候选地震场所未纳入',
        coverage='官方名单为部分城市部分年份公开记录，不代表完整最新名录或安置容量'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'OUTPUT={result_dir}', flush=True)
