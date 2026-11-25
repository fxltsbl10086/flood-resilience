"Build reusable baseline and five-scenario networks without calculating the risk formula."
import argparse
import pickle
import time
from pathlib import Path
from settings import DEFAULT_CITIES, CITIES, city_config, input_functions, routing

def get_network(city, output):
    cfg = city_config(city)
    cache = output / city / f'{city}_network_cache.pkl'
    # Load only local caches generated in this output directory.
    if cache.exists():
        with cache.open('rb') as file:
            stored = pickle.load(file)
        if stored['input_signatures'] == signatures(cfg):
            return stored['network'], stored['grids'], stored['metadata']
        raise ValueError()
    start = time.monotonic()
    print(f'{city}: Reading existing grids, roads, directions and five-scenario states', flush=True)
    grids, roads, edges, boundary, _ = input_functions.read_inputs(cfg)
    print(f'{city}: Generating grid-boundary access candidates', flush=True)
    candidates = input_functions.build_candidates(grids, roads, boundary, cfg)
    print(f'{city}: Building directed network with {len(candidates)} candidates', flush=True)
    network = routing.build_network(roads, edges, candidates, boundary, len(grids), cfg['PERIODS'])
    metadata = dict(city=city, elapsed_seconds=time.monotonic() - start,
                    candidates_qa=candidates.attrs.get('qa', {}), network_qa=network['qa'],
                    inputs={k: str(v) for k, v in cfg.items() if k.endswith('_PATH')})
    stored = dict(network=network, grids=grids, metadata=metadata, input_signatures=signatures(cfg))
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(stored, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache)
    print(f'{city}: Network complete: {network["n_nodes"]} nodes in {metadata["elapsed_seconds"]:.1f}s', flush=True)
    return network, grids, metadata

def signatures(cfg):
    result = {}
    for key in ['GRID_PATH', 'ROAD_PATH', 'EDGE_PATH', 'BOUNDARY_PATH', 'RAW_ROAD_PATH']:
        path = Path(cfg[key])
        stat = path.stat()
        result[key] = [str(path), stat.st_size, stat.st_mtime_ns]
        if path.suffix.lower() == '.shp':
            for suffix in ['.dbf', '.shx', '.prj']:
                sidecar = path.with_suffix(suffix)
                stat = sidecar.stat()
                result[key + suffix] = [str(sidecar), stat.st_size, stat.st_mtime_ns]
    for name in ['config.py', 'functions.py', 'reachability.py']:
        from settings import DEFAULT_CITIES, ENGINE
        path = ENGINE / name
        stat = path.stat()
        result[name] = [str(path), stat.st_size, stat.st_mtime_ns]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cities', nargs='+', choices=list(CITIES), default=DEFAULT_CITIES)
    args = parser.parse_args()
    for city in args.cities:
        get_network(city, args.output)
