"""Rebuild the road restoration network from the published inputs."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys
_RESULT4_ROOT = Path(__file__).resolve().parents[1]
if str(_RESULT4_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESULT4_ROOT))
from _support.analysis.restoration.paths import read_json, resolve_path
import time


SOURCE_MODULE = Path(__file__).resolve().parent / "core"
PERIODS = (10, 20, 50, 100, 200)
CITY_NAMES = {"chengdu": "成都", "zhengzhou": "郑州", "hangzhou": "杭州",
              "wuhan": "武汉", "fuzhou": "福州"}


def _require(condition, message):
    if not condition:
        raise ValueError("")


def _payload_key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _load_api():
    names = ("config", "cache_io", "prepare_inputs", "network_engine", "run_support", "road_priority", "foundation")
    for name in names:
        if name in sys.modules:
            _require(Path(sys.modules[name].__file__).resolve().parent == SOURCE_MODULE.resolve(),
                     "")
    if str(SOURCE_MODULE) not in sys.path:
        sys.path.insert(0, str(SOURCE_MODULE))
    modules = {name: importlib.import_module(name) for name in names}
    # Use the five manuscript cities with the original numerical definitions.
    modules["config"].SUPPORTED_CITIES = tuple(CITY_NAMES)
    modules["config"].CITY_NAMES = {**modules["config"].CITY_NAMES, **CITY_NAMES}
    cache, support = modules["cache_io"], modules["run_support"]
    return {**modules, "engine": modules["network_engine"],
            "make_priority": support.make_priority, "make_result": support.make_result,
            "check_no_governance": support.check_no_governance,
            "build_road_priority": modules["road_priority"].build_road_priority,
            "signature": cache.signature, "dump_json": cache.dump_json,
            "road_hash": support.road_hash}


class _Reader:
    def __init__(self, signature):
        self.signature = signature
        self.records = {}

    def verify(self, path, expected=None):
        path = resolve_path(path)
        key = str(path)
        record = self.records.get(key)
        if record is None:
            record = self.signature(path)
            self.records[key] = record
        if expected is not None:
            _require(path == resolve_path(expected["path"]) and
                     record["bytes"] == expected["bytes"] and record["sha256"] == expected["sha256"],
                     "")
        return record

    def json(self, path):
        before = self.verify(path)
        value = read_json(path)
        _require(self.signature(path) == before, "")
        return value

    def finish(self):
        for old in self.records.values():
            _require(self.signature(old["path"]) == old, "")
        return list(self.records.values())


def _fresh_length_inventory(pack, api, reader, period, progress):
    """Call the original matching algorithm and pure length core, rebuilding all spatial relationships in this run."""
    import importlib.util
    import numpy as np
    import geopandas as gpd
    import shapely
    lengths_api = importlib.import_module("inundation_lengths")
    definitions = api["prepare_inputs"]._definitions()
    batch = Path(definitions["cfg"]["ROAD_PATH"]).resolve().parent
    resolved_path = batch / "metadata/resolved_config.json"
    resolved = reader.json(resolved_path)
    _require(resolved["city"] == pack["metadata"]["city"] and period in resolved["periods"] and
             float(resolved["threshold_m"]) == 0.30, "")
    source_dir = batch / "metadata/code_snapshot/src"
    package_name = "_governance_fresh_road_flood_source_" + hashlib.sha256(
        str(source_dir.resolve()).encode()).hexdigest()[:12]
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(package_name, source_dir / "__init__.py",
                                                     submodule_search_locations=[str(source_dir)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    else:
        _require(Path(sys.modules[package_name].__file__).resolve().parent == source_dir.resolve(),
                 "")
    rasters = importlib.import_module(package_name + ".rasters")
    matching = importlib.import_module(package_name + ".matching")
    depth = resolve_path(resolved["depth_paths"][resolved["periods"].index(period)])
    for path in [depth, Path(lengths_api.__file__), source_dir / "__init__.py",
                 Path(rasters.__file__), Path(matching.__file__)]:
        reader.verify(path)
    progress(f'T{period}Read original depths and recalculate road-raster relationships')
    scenario_id = resolved["scenario_ids"][resolved["periods"].index(period)]
    stack = rasters.read_depths((depth,), (period,), (scenario_id,))
    boundary = gpd.GeoDataFrame({"geometry": [pack["boundary"]]}, crs=pack["roads"].crs)
    tolerance = float(resolved["tolerance_m"])
    spatial = matching.build_spatial_mapping(pack["roads"], boundary, stack,
                    tolerance_m=tolerance, cell_batch_size=int(resolved["cell_batch_size"]))
    pairs = spatial.pairs
    ri, ci = pairs.road_index.to_numpy(np.int64), pairs.cell_id.to_numpy(np.int64)
    wet = stack.valid[0] & (stack.values[0] >= 0.30)
    unknown = ~stack.valid[0]
    wet_count = np.bincount(ri[wet.ravel()[ci]], minlength=len(pack["roads"]))
    missing_count = np.bincount(ri[unknown.ravel()[ci]], minlength=len(pack["roads"]))
    missing = (missing_count > 0) | (spatial.analysis.outside_grid_length_m.to_numpy() > tolerance)
    road_ids = pack["roads"].road_id.astype(str).to_numpy()
    np.testing.assert_array_equal(spatial.analysis.road_id.astype(str).to_numpy(), road_ids)
    np.testing.assert_array_equal(wet_count, pack["roads"][f"wet_cells_T{period:03d}"].to_numpy())
    np.testing.assert_array_equal(missing_count, pack["roads"][f"missing_cells_T{period:03d}"].to_numpy())
    np.testing.assert_array_equal(missing, lengths_api._strict_bool(
        pack["roads"][f"missing_coverage_T{period:03d}"], "原道路缺失覆盖标记"))
    np.testing.assert_array_equal(wet_count > 0, pack["roads"][f"flood_T{period:03d}"].to_numpy() == 1)
    within = shapely.intersection(pack["roads"].geometry.to_numpy(), pack["boundary"])
    within_lengths = shapely.length(within)
    np.testing.assert_allclose(within_lengths, spatial.analysis.analysis_length_m.to_numpy(), atol=1e-6, rtol=0)
    groups = pairs.loc[wet.ravel()[ci]].groupby("road_id", sort=False).cell_id.apply(
        lambda values: values.to_numpy(np.int64))
    inundated = np.zeros(len(road_ids), dtype=float)
    for index, (road_id, geometry) in enumerate(zip(road_ids, within)):
        ids = groups.get(road_id, np.empty(0, dtype=np.int64))
        inundated[index] = lengths_api._union_intersection_length(
            geometry, lengths_api._cells(ids, stack.shape[1], stack.transform))
    _require(np.isfinite(inundated).all() and (inundated >= 0).all() and
             (inundated <= within_lengths + 1e-6).all(), "")
    np.testing.assert_array_equal(inundated > tolerance, wet_count > 0)
    inventory = pack["road_inventory"].copy()
    np.testing.assert_array_equal(inventory.road_id.astype(str).to_numpy(), road_ids)
    inventory["inundated_length_m"] = inundated
    inventory[f"inundated_length_T{period:03d}_m"] = inundated
    inventory["missing_coverage"] = missing
    inventory[f"missing_coverage_T{period:03d}"] = missing
    inventory["flood_vertical_uncertain"] = lengths_api._strict_bool(
        pack["roads"]["flood_vertical_uncertain"], "原道路垂向不确定标记")
    metadata = {"city": pack["metadata"]["city"], "period": period,
        "field": "inundated_length_m", "period_field": f"inundated_length_T{period:03d}_m",
        "threshold_m": 0.30, "unit": "m", "cache_used": False,
        "inundated_length_recomputed": True, "spatial_mapping_recomputed": True,
        "definition": f'原研究区内物理道路与T{period}Intersection length with the union of valid land-depth cells >=0.30 m; do not double-count shared edges.',
        "algorithm": "original_matching.build_spatial_mapping + original_inundation_lengths._cells/_union_intersection_length",
        "missing_coverage": 'Exclude NoData and unknown out-of-raster portions from known above-threshold length; zero does not guarantee dry conditions.',
        "raster": stack.metadata[0], "mapping_qa": spatial.qa,
        "road_count": len(road_ids), "wet_road_count": int((inundated > tolerance).sum()),
        "missing_coverage_road_count": int(missing.sum()),
        "source_signatures": [reader.verify(path) for path in
             [depth, resolved_path, Path(lengths_api.__file__), Path(rasters.__file__), Path(matching.__file__)]]}
    return inventory, metadata


def load_context_fresh(city="fuzhou", district="台江区", periods=(10,), progress=None,
                       evaluation_scope="district"):
    """Run without computation caches: rebuild access, graphs, baseline OD, contact OD/paths and road inundated lengths.

Formal population, facility counts, EFARB and road flood states remain fixed model inputs.
Do not read pickle, npy or parquet computation products or old restoration batches, and do not create/write a cache directory."""
    started = time.perf_counter()
    periods = tuple(map(int, periods))
    _require(city in CITY_NAMES, "")
    _require(evaluation_scope in ("district", "study_area"), "")
    _require(bool(periods) and len(set(periods)) == len(periods) and
             periods == tuple(p for p in PERIODS if p in periods), "")
    if progress is None:
        progress = lambda message: print(message, flush=True)
    api = _load_api()
    import numpy as np
    cfg, inputs_api, engine, foundation = api["config"], api["prepare_inputs"], api["engine"], api["foundation"]
    contact = importlib.import_module("contact_projects")
    reader = _Reader(api["signature"])
    cfg.CITY, cfg.DISTRICT_NAME, cfg.EVALUATION_SCOPE = city, district, evaluation_scope
    active_district = district if evaluation_scope == "district" else None
    cfg.PERIOD = periods[0]
    packs, baselines, selections, period_sources, records = {}, {}, {}, {}, []
    cfg_source_paths = [SOURCE_MODULE / name for name in
         ("config.py", "prepare_inputs.py", "network_engine.py", "run_support.py", "road_priority.py",
          "contact_projects.py", "foundation.py", "inundation_lengths.py", "cache_io.py")]
    cfg_source_paths.append(_RESULT4_ROOT / "paths.py")
    for path in cfg_source_paths:
        reader.verify(path)
    initial_residents = initial_population = initial_baseline = None
    for period in periods:
        cfg.PERIOD = period
        begin = time.perf_counter()
        progress(f'T{period}Uncached: read original GIS and formal model inputs and rebuild access candidates')
        for path in inputs_api.input_paths():
            reader.verify(path)
        pack = inputs_api.load_inputs()
        _require(pack["reference_network"] is None and not pack["metadata"]["source_network_cache"]["reused"],
                 "")
        records.append({"stage": 'Read original GIS/formal tables and regenerate access candidates', "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        begin = time.perf_counter()
        progress(f'T{period}Uncached: rebuild complete roads and directed networks')
        topology = engine.build_network(pack["roads"], pack["edges"], pack["candidates"],
                                         pack["boundary"], len(pack["grids"]), (period,))
        pack["topology"] = topology
        scenario = f"T{period}"
        np.testing.assert_array_equal(topology["road_ids"], pack["roads"].road_id.astype(str).to_numpy())
        np.testing.assert_array_equal(topology["closures"][scenario], pack["roads"][f"flood_T{period:03d}"].to_numpy(bool))
        engine.assert_same_graph(engine.build_scene(topology, (), scenario)["graph"], topology["graphs"][scenario],
                                 "鲜算无治理拓扑内部一致性")
        engine.assert_same_graph(engine.build_scene(topology, np.flatnonzero(topology["closures"][scenario]), scenario)["graph"],
                                 topology["graphs"]["baseline"], "鲜算全恢复与基准拓扑一致性")
        pack["topology_qa"] = {"freshly_built": True, "cached_reference_used": False,
            "ungoverned_internal_graph_matches": True, "full_recovery_matches_baseline": True,
            "network_qa": topology["qa"]}
        records.append({"stage": "完整有向路网重新构建与内部核验", "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        begin = time.perf_counter()
        progress(f"T{period} uncached: recalculate no-flood baseline OD")
        baseline = engine.prepare_baseline(topology, pack["resident_indices"], pack["counts"],
            scenario=scenario, local_distance=cfg.LOCAL_DISTANCE_M, progress=progress, trace_all_projects=False)
        foundation._validate_baseline(pack, baseline)
        records.append({"stage": "无洪涝基准OD重新计算", "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        begin = time.perf_counter()
        progress(f'T{period}Uncached: recalculate unrestored facility-contact OD')
        ungoverned = contact.prepare_ungoverned_od(topology, baseline, progress=progress)
        errors = foundation._validate_ungoverned(pack, baseline, ungoverned)
        records.append({"stage": 'Recalculate unrestored contact OD and verify original EFAR', "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        begin = time.perf_counter()
        progress(f'T{period}Uncached: reselect maximum-loss contacts and trace baseline paths')
        selection = contact.select_loss_projects(topology, baseline, ungoverned,
                            pack["grids"].grid_id.astype(str).to_numpy(), progress=progress)
        selection_qa = foundation._validate_selection(pack, baseline, selection, ungoverned)
        records.append({"stage": "最大损失联系与基准路径重新计算", "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        begin = time.perf_counter()
        pack["road_inventory"], pack["length_metadata"] = _fresh_length_inventory(pack, api, reader, period, progress)
        records.append({"stage": 'Spatially match original depths and recalculate inundated lengths', "period": period,
                        "elapsed_seconds": time.perf_counter() - begin, "cache_used": False})
        residents = pack["resident_indices"]
        population = pack["evaluation_all"].loc[pack["evaluation_all"].efarb_valid, "effective_population"].to_numpy()
        if initial_residents is None:
            initial_residents, initial_population, initial_baseline = residents.copy(), population.copy(), baseline
        else:
            np.testing.assert_array_equal(residents, initial_residents)
            np.testing.assert_array_equal(population, initial_population)
            for key in ("counts", "base_inverse", "base_count", "dimensions_valid"):
                np.testing.assert_array_equal(baseline[key], initial_baseline[key])
        packs[period], baselines[period], selections[period] = pack, baseline, selection
        period_sources[period] = {"city": city, "period": period, "scenario": scenario,
            "evaluation_scope": evaluation_scope, "district": active_district, "cache_used": False,
            "network_rebuilt": True, "baseline_OD_recomputed": True, "ungoverned_OD_recomputed": True,
            "contacts_and_paths_recomputed": True, "road_inundated_lengths_recomputed": True,
            "initial_EFAR_maximum_absolute_error": errors, "selection_qa": selection_qa,
            "length_metadata": pack["length_metadata"], "evaluation_grid_count": len(residents)}
    signatures = reader.finish()
    for period in periods:
        period_sources[period]["scenario_model_key"] = _payload_key({"sources": signatures,
            "city": city, "evaluation_scope": evaluation_scope, "district": active_district,
            "period": period, "preparation": "fresh_no_cache",
            "resident_indices": list(map(int, packs[period]["resident_indices"]))})
    return {"packs": packs, "baselines": baselines, "selections": selections,
        "source_signatures": signatures, "api": api,
        "reuse_metadata": {"status": "PASS", "city": city, "city_name": CITY_NAMES[city],
            "district": active_district, "district_argument_ignored": evaluation_scope == "study_area",
            "periods": list(periods), "evaluation_scope": evaluation_scope,
            "preparation_mode": "fresh_no_cache", "cache_used": False,
            "existing_calculation_cache_read": False, "calculation_cache_written": False,
            "period_sources": period_sources, "timing_records": records,
            "complete_city_network_and_facilities_retained": True,
            "baseline_population_and_denominator_exactly_equal_across_periods": True,
            "formal_population_facility_count_and_EFARB_tables_used_as_fixed_model_inputs": True,
            "city_mapping_overrides": CITY_NAMES,
            "elapsed_seconds": time.perf_counter() - started}}
