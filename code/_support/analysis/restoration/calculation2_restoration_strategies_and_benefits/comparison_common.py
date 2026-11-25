"""Shared I/O, scores and content-addressed caching for two ordering controls; preserve the verified numerical engine."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from _release import ANALYSIS
from _support.analysis.restoration.paths import DEFAULT_BATCH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from plan import TARGET_PCTS, road_hash, validate_inundated_inventory

CITY_CODES = {"zhengzhou": 1, "hangzhou": 2, "chengdu": 3, "wuhan": 4, "fuzhou": 5}
CITY_NAMES = {"zhengzhou": "郑州", "hangzhou": "杭州", "chengdu": "成都", "wuhan": "武汉", "fuzhou": "福州"}
PERIODS = (10, 20, 50, 100, 200)
SOURCE_ROOT = DEFAULT_BATCH
STRATEGIES = ("population_priority", "random_priority")
SCHEMA_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def state_key(model_key, road_ids):
    return canonical_hash({"model_key": model_key, "road_set_hash": road_hash(road_ids),
                           "array_schema": SCHEMA_VERSION})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def wet_pool(inventory):
    validate_inundated_inventory(inventory)
    pool = inventory.loc[inventory.inundated_length_m.gt(0)].copy()
    pool["road_id"] = pool.road_id.astype(str)
    if not pool.flood_scenario.eq(1).all():
        raise ValueError("")
    return pool.sort_values("road_id", kind="stable").reset_index(drop=True)


def population_priority(inventory, contacts, evaluation):
    pool = wet_pool(inventory)
    required = {"road_id", "grid_id", "effective_population"}
    if not required <= set(contacts):
        raise ValueError("")
    links = contacts.copy()
    if links[list(required)].isna().any().any():
        raise ValueError("")
    for column in ("road_id", "grid_id"):
        links[column] = links[column].astype(str)
    pop = links.effective_population.to_numpy(float)
    if not np.isfinite(pop).all() or (pop <= 0).any():
        raise ValueError("")
    if not set(links.road_id) <= set(pool.road_id):
        raise ValueError("")
    if (links.groupby("grid_id").effective_population.nunique() != 1).any():
        raise ValueError("")
    if "road_position" in links:
        reference = pool.set_index("road_id").road_position
        if not np.array_equal(reference.loc[links.road_id].to_numpy(), links.road_position.to_numpy()):
            raise ValueError("")
    # Original city/scenario exposed-person counts are frozen in the evaluation table; do not use population shares,
    # or replace missing values with effective_population. Keep original contact mappings and the common road pool fixed.
    if not {"grid_id", "population_exposed"} <= set(evaluation):
        raise ValueError("")
    exposures = evaluation[["grid_id", "population_exposed"]].copy()
    if exposures.grid_id.isna().any():
        raise ValueError("")
    exposures["grid_id"] = exposures.grid_id.astype(str)
    if exposures.grid_id.duplicated().any():
        raise ValueError("")
    if not set(links.grid_id) <= set(exposures.grid_id):
        raise ValueError("")
    unique = links.drop_duplicates(["road_id", "grid_id"]).copy()
    unique["population_exposed"] = unique.grid_id.map(exposures.set_index("grid_id").population_exposed)
    exposed_values = unique.population_exposed.to_numpy(float)
    if not np.isfinite(exposed_values).all() or (exposed_values < 0).any():
        raise ValueError("")
    scores = unique.groupby("road_id", sort=False).population_exposed.agg(lambda x: math.fsum(map(float, x)))
    counts = unique.groupby("road_id", sort=False).grid_id.size()
    exposed_counts = unique.loc[unique.population_exposed.gt(0)].groupby("road_id", sort=False).grid_id.size()
    pool["associated_exposed_population"] = pool.road_id.map(scores).fillna(0.0)
    pool["associated_grid_count"] = pool.road_id.map(counts).fillna(0).astype(int)
    pool["associated_exposed_grid_count"] = pool.road_id.map(exposed_counts).fillna(0).astype(int)
    pool = pool.sort_values(["associated_exposed_population", "road_id"], ascending=[False, True], kind="stable")
    pool.insert(0, "road_rank", np.arange(1, len(pool) + 1))
    return pool.reset_index(drop=True)


def random_priority(inventory, city, period, replicate_id, seed=20261107):
    if city not in CITY_CODES or period not in PERIODS or replicate_id < 0 or seed < 0:
        raise ValueError("")
    pool = wet_pool(inventory)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [int(seed), CITY_CODES[city], int(period), int(replicate_id)])))
    result = pool.iloc[rng.permutation(len(pool))].copy().reset_index(drop=True)
    result.insert(0, "road_rank", np.arange(1, len(result) + 1))
    return result


def validate_arrays(arrays, n):
    for field in ("risk", "scene_inverse", "reachable_count", "nearest_distance"):
        if field not in arrays or arrays[field].shape != (n, 3):
            raise ValueError("")
    if not np.isfinite(arrays["risk"]).all() or not np.isfinite(arrays["scene_inverse"]).all():
        raise ValueError("")
    if (arrays["risk"] < -1e-10).any() or (arrays["risk"] > 1 + 1e-10).any():
        raise ValueError("")
    if (arrays["scene_inverse"] < 0).any() or (arrays["reachable_count"] < 0).any():
        raise ValueError("")
    count = arrays["reachable_count"]
    if not np.isfinite(count).all() or not np.equal(count, np.floor(count)).all():
        raise ValueError("")
    if not np.array_equal(count > 0, arrays["scene_inverse"] > 0):
        raise ValueError("")
    if np.isnan(arrays["nearest_distance"]).any() or (arrays["nearest_distance"] <= 0).any():
        raise ValueError("")


def assert_monotone(previous, current):
    if (current["risk"] > previous["risk"] + 1e-10).any():
        raise ValueError("")
    if (current["reachable_count"] < previous["reachable_count"]).any():
        raise ValueError("")
    if (current["nearest_distance"] > previous["nearest_distance"] + 1e-6).any():
        raise ValueError("")


def load_state(cache_root, model_key, road_ids, n):
    key = state_key(model_key, road_ids)
    path = Path(cache_root) / "states" / model_key[:16] / f"{key[:24]}.npz"
    metadata_path = path.with_suffix(".json")
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.exists() or not metadata_path.exists():
        # Uncommitted orphan files are not successful caches; retain them for traceability.
        return None
    metadata = read_json(metadata_path)
    if metadata["state_key"] != key or metadata["model_key"] != model_key or metadata["sha256"] != sha256(path):
        raise ValueError("")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    validate_arrays(arrays, n)
    return arrays, metadata


def save_state(cache_root, model_key, road_ids, result, n):
    key = state_key(model_key, road_ids)
    path = Path(cache_root) / "states" / model_key[:16] / f"{key[:24]}.npz"
    arrays = {name: np.asarray(result[name]) for name in
              ("risk", "scene_inverse", "reachable_count", "nearest_distance")}
    validate_arrays(arrays, n)
    atomic_npz(path, **arrays)
    metadata = {"state_key": key, "model_key": model_key, "road_set_hash": road_hash(road_ids),
                "schema_version": SCHEMA_VERSION, "path": str(path.resolve()), "sha256": sha256(path),
                "resident_count": n, "created_at": datetime.now().isoformat(),
                "timings": result.get("timings", {})}
    atomic_json(path.with_suffix(".json"), metadata)
    return arrays, metadata
