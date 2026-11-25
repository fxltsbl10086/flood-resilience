"Persist complete depth-independent road-grid matches and reuse geometry-fingerprinted caches."

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import shapely

from .matching import SpatialMapping, build_spatial_mapping


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_descriptor(roads, boundary, stack, tolerance_m: float) -> dict:
    digest = hashlib.sha256()
    digest.update(roads.crs.to_wkt().encode("utf-8"))
    for road_id, geometry in zip(roads.road_id, shapely.to_wkb(roads.geometry.to_numpy())):
        identity = json.dumps(road_id, ensure_ascii=False, default=str).encode("utf-8")
        digest.update(len(identity).to_bytes(8, "little"))
        digest.update(identity)
        digest.update(len(geometry).to_bytes(8, "little"))
        digest.update(geometry)
    study = boundary.to_crs(stack.crs).geometry.union_all()
    return {
        "version": "complete_study_grid_positive_length_v1",
        "algorithm_code_sha256": file_digest(Path(__file__).with_name("matching.py")),
        "roads_geometry_and_order_sha256": digest.hexdigest(),
        "boundary_geometry_sha256": hashlib.sha256(shapely.to_wkb(study)).hexdigest(),
        "grid_shape": list(stack.shape), "grid_transform": list(stack.transform),
        "grid_crs": stack.crs.to_wkt(),
        "positive_length_tolerance_m": tolerance_m,
    }


def get_spatial_mapping(config, roads, boundary, stack) -> SpatialMapping:
    started = perf_counter()
    descriptor = mapping_descriptor(roads, boundary, stack, config.tolerance_m)
    key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()
    folder = config.matching_root / config.city / key
    manifest_path = folder / "matching_manifest.json"
    hit = folder.exists()
    if hit:
        if not manifest_path.is_file():
            raise RuntimeError()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("descriptor") != descriptor:
            raise RuntimeError()
        for filename, digest in manifest["files_sha256"].items():
            path = folder / filename
            if not path.is_file() or file_digest(path) != digest:
                raise RuntimeError()
        mapping = SpatialMapping(
            pd.read_parquet(folder / "road_grid_matches.parquet"),
            pd.read_parquet(folder / "road_analysis.parquet"),
            np.load(folder / "study_cell_ids.npy", allow_pickle=False),
            descriptor, manifest["qa"],
        )
    else:
        mapping = build_spatial_mapping(roads, boundary, stack, config.tolerance_m,
                                        config.cell_batch_size, descriptor)
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {"status": "writing", "key": key, "descriptor": descriptor,
                    "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    "scope": "完整研究区格网的道路正长度匹配；包含未淹没和NoData格网，独立于水深/阈值",
                    "qa": mapping.qa}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        mapping.pairs.to_parquet(folder / "road_grid_matches.parquet", index=False)
        mapping.analysis.to_parquet(folder / "road_analysis.parquet", index=False)
        with (folder / "study_cell_ids.npy").open("xb") as file:
            np.save(file, mapping.study_cell_ids, allow_pickle=False)
        manifest["files_sha256"] = {name: file_digest(folder / name) for name in
                                   ("road_grid_matches.parquet", "road_analysis.parquet", "study_cell_ids.npy")}
        manifest["status"] = "completed"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    mapping.qa = {**mapping.qa, "mapping_cache_hit": hit, "mapping_cache_dir": str(folder),
                  "mapping_cache_key": key, "mapping_acquisition_seconds": perf_counter() - started,
                  "mapping_build_seconds_this_run": 0.0 if hit else mapping.qa["build_timings_seconds"]["mapping_total"]}
    print(f"[{config.city}] {'Reused' if hit else 'Built'} complete mapping cache: {len(mapping.pairs)} road-grid pairs", flush=True)
    return mapping
