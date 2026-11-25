"""Frozen-input context for population/random road-order comparisons.

This module never selects facility contacts or recalculates inundated lengths.
The reference inputs fix the mapping. The routing engine constructs the network
and the unflooded accessibility denominator.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import runtime
from _support.analysis.restoration.paths import resolve_path
from plan import extend_inundated_priority, road_hash, validate_inundated_inventory

SCHEMA = "frozen_road_mapping_comparison_context_v1"
CITIES = ("zhengzhou", "hangzhou", "chengdu", "wuhan", "fuzhou")
PERIODS = (10, 20, 50, 100, 200)


def _require(condition, message):
    if not condition:
        raise ValueError("")


def _signature(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _key(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class _Sources:
    def __init__(self):
        self.records = {}

    def add(self, path, expected=None):
        observed = _signature(path)
        if expected is not None:
            _require(observed["bytes"] == int(expected["bytes"])
                     and observed["sha256"] == expected["sha256"],
                     "")
        old = self.records.get(observed["path"])
        _require(old is None or old == observed, "")
        self.records[observed["path"]] = observed
        return observed

    def finish(self):
        for observed in self.records.values():
            _require(_signature(observed["path"]) == observed,
                     "")
        return sorted(self.records.values(), key=lambda value: value["path"])


def _read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip",
                       dtype={"grid_id": str, "road_id": str,
                              "selected_facility_grid_id": str}, low_memory=False)


def _frozen_tables(source_dir, period, sources, api):
    completion_path = source_dir / "completion.json"
    sources.add(completion_path)
    completion = json.loads(completion_path.read_text(encoding="utf-8-sig"))
    _require(completion.get("status") == "PASS", "")
    _require(completion.get("strategy") == "scenario_priority_inundated_length_targets_v1",
             "")
    paths = {
        "inventory": source_dir / "inputs" / f"T{period}_road_inventory.csv",
        "original_evaluation": source_dir / "inputs" / f"T{period}_evaluation_grids.csv",
        "original_priority": source_dir / "plans" / f"T{period}_road_priority_order.csv",
        "contacts": source_dir / "plans" / f"T{period}_road_link_burden_details.csv",
        "selection": source_dir / "plans" / f"T{period}_maximum_loss_facility_links.csv",
        "links": source_dir / "plans" / f"T{period}_facility_link_road_lookup.csv",
    }
    for path in paths.values():
        matches = [record for record in completion["output_signatures"]
                   if Path(record["path"]).name == path.name]
        _require(len(matches) == 1, "")
        sources.add(path, matches[0])
    frames = {name: _read_csv(path) for name, path in paths.items()}
    inventory, evaluation = frames["inventory"], frames["original_evaluation"]
    original_priority, contacts = frames["original_priority"], frames["contacts"]
    denominator = validate_inundated_inventory(inventory)
    _require(set(original_priority.road_id) == set(inventory.loc[inventory.inundated_length_m.gt(0), "road_id"]),
             "")
    _require(original_priority.road_id.is_unique and inventory.road_id.is_unique,
             "")
    # Pure bookkeeping rebuild of the original score, without contact/path search.
    demand, checked_contacts, priority_audit = api["build_road_priority"](
        evaluation, frames["selection"], frames["links"], inventory)
    rebuilt = extend_inundated_priority(demand, inventory)
    np.testing.assert_array_equal(rebuilt.road_id.to_numpy(), original_priority.road_id.to_numpy())
    for column in ("road_rank", "road_position", "length_m", "inundated_length_m"):
        np.testing.assert_array_equal(rebuilt[column], original_priority[column])
    np.testing.assert_allclose(rebuilt.associated_risk_burden, original_priority.associated_risk_burden,
                               atol=1e-10, rtol=1e-12)
    pd.testing.assert_frame_equal(checked_contacts.reset_index(drop=True), contacts.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-10)
    return frames, {
        "original_priority_rebuilt_without_contact_reselection": True,
        "original_risk_scoring": priority_audit,
        "physical_pool_sha256": road_hash(original_priority.road_id),
        "physical_pool_count": len(original_priority),
        "zero_risk_score_road_count": int(original_priority.associated_risk_burden.eq(0).sum()),
        "scenario_total_inundated_length_m": denominator,
        "maximum_supported_target_pct": 100.0,
        "frozen_mapping_files": {name: str(path) for name, path in paths.items()},
    }


def _verify_reference_inputs(source_dir, city, period, sources, api):
    path = source_dir / "inputs" / f"T{period}_network_calculation_fingerprint.json"
    sources.add(path)
    original = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(original["city"] == city and int(original["period"]) == period
             and original["evaluation_scope"] == "study_area", "")
    _require(original["facility_rule"] == api["config"].FACILITY_RULE
             and float(original["local_distance_m"]) == api["config"].LOCAL_DISTANCE_M,
             "")
    resolved = []
    for record in original["sources"]:
        actual_path = resolve_path(record["path"])
        actual = sources.add(actual_path, record)
        resolved.append({"original_path": record["path"], "verified_path": actual["path"],
                         "sha256": actual["sha256"]})
    return original, resolved


def _validate_current_pack(pack, frames, baseline=None):
    original = frames["original_evaluation"]
    evaluation = pack["evaluation_all"]
    np.testing.assert_array_equal(evaluation.grid_id, original.grid_id)
    np.testing.assert_array_equal(evaluation.grid_index, original.grid_index)
    fields = ["effective_population", "population_exposed", "initial_EFARB", "initial_EFAR_mean"]
    for dimension in ("rescue", "medical", "shelter"):
        fields.extend([f"initial_EFAR_{dimension}", f"initial_reachable_count_{dimension}"])
    np.testing.assert_allclose(evaluation[fields].to_numpy(float), original[fields].to_numpy(float),
                               rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_array_equal(evaluation.efarb_valid.to_numpy(bool),
                                  original.efarb_valid.to_numpy(bool))
    inventory = frames["inventory"]
    np.testing.assert_array_equal(pack["road_inventory"].road_id, inventory.road_id)
    np.testing.assert_array_equal(pack["road_inventory"].road_position, inventory.road_position)
    for field in ("length_m", "inside_evaluation_area_length_m", "outside_evaluation_area_length_m"):
        np.testing.assert_allclose(pack["road_inventory"][field], inventory[field], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(pack["road_inventory"].flood_scenario, inventory.flood_scenario)
    if baseline is not None:
        valid = original.efarb_valid.to_numpy(bool)
        np.testing.assert_array_equal(baseline["resident_indices"], original.loc[valid, "grid_index"])


def load_comparison_context(city: str, period: int, source_root: Path, cache_root: Path,
                            progress=print) -> dict:
    """Load a signed frozen mapping and original model context for one city-period.

    Return pack, baseline, original API, inventory, original_priority, contacts,
    model_key, source_signatures, preparation_seconds and cache_used. No final
    governance effects are calculated here; the caller evaluates actual road sets.
    """
    started = time.perf_counter()
    _require(city in CITIES and period in PERIODS, "")
    api = runtime._load_api()
    cfg, engine, inputs = api["config"], api["engine"], api["prepare_inputs"]
    cfg.CITY, cfg.PERIOD, cfg.EVALUATION_SCOPE = city, period, "study_area"
    sources = _Sources()
    source_dir = Path(source_root).resolve() / city
    frames, audit = _frozen_tables(source_dir, period, sources, api)
    historical, historical_resolved = _verify_reference_inputs(source_dir, city, period, sources, api)
    for path in [Path(__file__), ROOT / "runtime.py", ROOT / "plan.py", ROOT.parent / "paths.py",
                 *sorted((ROOT / "core").glob("*.py")), *inputs.input_paths()]:
        sources.add(path)
    import scipy
    import geopandas
    import shapely
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
                "geopandas": geopandas.__version__, "shapely": shapely.__version__}
    parameters = {"schema": SCHEMA, "city": city, "period": int(period),
        "evaluation_scope": "study_area", "facility_rule": cfg.FACILITY_RULE,
        "local_distance_m": cfg.LOCAL_DISTANCE_M, "facility_weights": [1 / 3] * 3,
        "population_source": "original effective_population, including disconnected residents",
        "road_pool_sha256": audit["physical_pool_sha256"], "versions": versions}
    signatures = sources.finish()
    manifest = {"parameters": parameters, "sources": signatures}
    model_key = _key(manifest)
    # The full digest is verified in context.json. A short directory name
    # leaves room for temporary files when the package is nested on Windows.
    cache_dir = Path(cache_root).resolve() / "contexts" / city / f"T{period}" / model_key[:16]
    payload_path, meta_path = cache_dir / "context.pkl", cache_dir / "context.json"
    cache_used = False
    if payload_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _require(meta.get("complete") is True and meta.get("model_key") == model_key
                 and meta.get("manifest") == manifest, "")
        _require(_signature(payload_path)["sha256"] == meta["payload_sha256"],
                 "")
        with payload_path.open("rb") as stream:
            payload = pickle.load(stream)
        pack, baseline = payload["pack"], payload["baseline"]
        cache_used = True
        progress(f"{city} T{period}: verified context cache reused")
        preparation_records = payload["preparation_records"]
    else:
        preparation_records = []
        begin = time.perf_counter()
        progress(f"{city} T{period}: loading original inputs and frozen contact mapping")
        pack = inputs.load_inputs()
        _validate_current_pack(pack, frames)
        ranked, _ = api["make_priority"](pack["evaluation_all"])
        np.testing.assert_array_equal(ranked["rank"], frames["original_evaluation"]["rank"])
        np.testing.assert_array_equal(ranked.ranking_eligible, frames["original_evaluation"].ranking_eligible)
        pack["evaluation_all"] = ranked
        preparation_records.append({"stage": "load_original_inputs", "elapsed_seconds": time.perf_counter() - begin})
        begin = time.perf_counter()
        progress(f"{city} T{period}: constructing original directed road network")
        topology = engine.build_network(pack["roads"], pack["edges"], pack["candidates"],
                                        pack["boundary"], len(pack["grids"]), (period,))
        scenario = f"T{period}"
        np.testing.assert_array_equal(topology["road_ids"], pack["roads"].road_id.astype(str))
        np.testing.assert_array_equal(topology["closures"][scenario], frames["inventory"].flood_scenario.astype(bool))
        engine.assert_same_graph(engine.build_scene(topology, (), scenario)["graph"],
                                 topology["graphs"][scenario], "comparison ungoverned network")
        engine.assert_same_graph(engine.build_scene(topology, np.flatnonzero(topology["closures"][scenario]), scenario)["graph"],
                                 topology["graphs"]["baseline"], "comparison restoration of all closures")
        reference = pack.get("reference_network")
        topology_qa = engine.validate_topology(topology, reference, scenario) if reference is not None else {
            "original_reference_available": False, "ungoverned_internal_graph_matches": True,
            "full_recovery_matches_baseline": True}
        pack["topology"], pack["topology_qa"] = topology, topology_qa
        # The old candidate cache is used only after its original input validation.
        # Its duplicate network is unnecessary in our own persistent context.
        pack["reference_network"] = None
        del reference
        preparation_records.append({"stage": "construct_validate_network", "elapsed_seconds": time.perf_counter() - begin})
        begin = time.perf_counter()
        progress(f"{city} T{period}: computing no-flood denominator, no contact reselection")
        baseline = engine.prepare_baseline(topology, pack["resident_indices"], pack["counts"],
                    scenario=scenario, local_distance=cfg.LOCAL_DISTANCE_M, progress=progress,
                    trace_all_projects=False)
        api["foundation"]._validate_baseline(pack, baseline)
        preparation_records.append({"stage": "baseline_OD", "elapsed_seconds": time.perf_counter() - begin})
        _validate_current_pack(pack, frames, baseline)
        pack["road_inventory"] = frames["inventory"].copy()
        signatures = sources.finish()
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = payload_path.with_name(payload_path.name + "." + uuid.uuid4().hex + ".tmp")
        payload = {"pack": pack, "baseline": baseline, "preparation_records": preparation_records}
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(payload_path)
        _atomic_json(meta_path, {"complete": True, "model_key": model_key, "manifest": manifest,
            "payload_sha256": _signature(payload_path)["sha256"], "schema": SCHEMA,
            "original_preparation_seconds": time.perf_counter() - started,
            "preparation_records": preparation_records})
    _validate_current_pack(pack, frames, baseline)
    api["foundation"]._validate_baseline(pack, baseline)
    np.testing.assert_array_equal(baseline["resident_indices"], np.asarray(historical["residents"], dtype=np.int64))
    signatures = sources.finish()
    audit.update({"historical_inputs_verified": historical_resolved,
                  "original_evaluation_matches": True, "no_contact_reselection": True,
                  "no_inundated_length_recomputation": True, "versions": versions,
                  "preparation_records": preparation_records, "topology": pack["topology_qa"]})
    return {"pack": pack, "baseline": baseline, "api": api, "inventory": frames["inventory"],
            "original_priority": frames["original_priority"], "contacts": frames["contacts"],
            "original_evaluation": frames["original_evaluation"], "model_key": model_key,
            "source_signatures": signatures, "preparation_seconds": time.perf_counter() - started,
            "cache_used": cache_used, "source_dir": source_dir, "audit": audit,
            "context_cache_path": payload_path}
