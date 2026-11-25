"Validate city inputs, export formal results and record provenance."

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
from time import perf_counter
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely

from .config import CityConfig
from .cache import get_spatial_mapping
from .matching import identify_roads, validate_roads
from .missing_policy import apply_missing_policy, review_missing
from .rasters import read_depths


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def prepare_inputs(config: CityConfig):
    roads = gpd.read_file(config.road_path, layer=config.facility_layer)
    boundary = gpd.read_file(config.boundary_path)
    validate_roads(roads)
    stack = read_depths(config.depth_paths, config.periods, config.scenario_ids)
    if boundary.crs is None or boundary.empty or not boundary.geometry.is_valid.all():
        raise ValueError()
    edges = None
    if config.edge_path:
        edges = gpd.read_file(config.edge_path, layer=config.edge_layer)
        if (edges.crs is None or "road_id" not in edges or "edge_id" not in edges
                or edges.edge_id.isna().any() or not edges.edge_id.is_unique
                or not edges.road_id.isin(roads.road_id).all()):
            raise ValueError()
    return roads, boundary, stack, edges


def verify_vector(original: gpd.GeoDataFrame, path: Path, layer: str, id_column: str,
                  expected_status: pd.DataFrame) -> None:
    restored = gpd.read_file(path, layer=layer)
    if len(restored) != len(original) or restored.crs != original.crs:
        raise RuntimeError()
    if not np.array_equal(original[id_column].to_numpy(), restored[id_column].to_numpy()):
        raise RuntimeError()
    if not np.array_equal(shapely.to_wkb(original.geometry.to_numpy()), shapely.to_wkb(restored.geometry.to_numpy())):
        raise RuntimeError()
    old_columns = [c for c in original.columns if c != original.geometry.name]
    before = original[old_columns].reset_index(drop=True)
    after = restored[old_columns].reset_index(drop=True)
    equal = before.eq(after) | (before.isna() & after.isna())
    if not equal.to_numpy().all():
        raise RuntimeError()
    labels = expected_status.set_index("road_id")
    for column in (c for c in labels if c.startswith("flood_T")):
        expected = labels.loc[restored.road_id, column].to_numpy()
        if not np.array_equal(expected, restored[column].to_numpy()):
            raise RuntimeError()


def run_city(config: CityConfig, batch_name: str | None = None) -> dict:
    started = perf_counter()
    now = datetime.now(timezone(timedelta(hours=8)))
    batch_name = batch_name or (now.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8])
    if Path(batch_name).name != batch_name or "/" in batch_name or "\\" in batch_name or batch_name in {".", ".."}:
        raise ValueError()
    destination = config.output_root / config.city / batch_name
    if destination.exists():
        raise FileExistsError()
    roads, boundary, stack, edges = prepare_inputs(config)
    print(f"[{config.city}] {len(roads)} roads; scenarios {list(config.scenario_ids)}; threshold >= {config.threshold_m} m", flush=True)
    spatial_mapping = get_spatial_mapping(config, roads, boundary, stack)
    result = identify_roads(roads, boundary, stack, config.threshold_m,
                            config.tolerance_m, config.cell_batch_size, spatial_mapping)
    review = review_missing(result, spatial_mapping, stack, config.tolerance_m)
    result = apply_missing_policy(result, review, config.missing_policy)
    input_road_count = len(roads)
    input_edge_count = len(edges) if edges is not None else 0
    retained = roads.road_id.isin(result.status.road_id)
    roads = roads.loc[retained].reset_index(drop=True)
    if edges is not None:
        edges = edges.loc[edges.road_id.isin(result.status.road_id)].reset_index(drop=True)
    scope_counts = {"input_road_count": input_road_count, "output_road_count": len(roads),
                    "excluded_outside_study_road_count": input_road_count - len(roads),
                    "input_edge_count": input_edge_count,
                    "output_edge_count": len(edges) if edges is not None else 0,
                    "excluded_edge_count": input_edge_count - (len(edges) if edges is not None else 0)}
    new_columns = [c for c in result.status if c != "road_id"]
    conflicts = set(new_columns).intersection(roads.columns)
    if conflicts or (edges is not None and set(new_columns).intersection(edges.columns)):
        raise ValueError()
    # Preserve input provenance, effective settings and mapping tables in each new batch.
    destination.mkdir(parents=True, exist_ok=False)
    intermediate = destination / "intermediate"
    metadata_dir = destination / "metadata"
    intermediate.mkdir()
    metadata_dir.mkdir()
    manifest_path = destination / "run_record.json"
    manifest = {"city": config.city, "batch_name": batch_name,
                "created_at": now.isoformat(), "status": "running",
                "output_dir": str(destination), "scenario_ids": config.scenario_ids,
                "threshold_m": config.threshold_m,
                "depth_product": config.depth_file,
                "geometry_rule": "study-area road line intersects a cell with positive length",
                "status_meaning": {"1": "有效水深达到阈值或按已确认规则判为淹没",
                                   "0": "有效覆盖未达阈值或按已确认规则判为未淹没"},
                "missing_policy": config.missing_policy,
                **scope_counts,
                "vertical_policy": "几何匹配照常标记；桥隧/高架/混合设施另标不确定性，不据此判断实际关闭"}
    write_json(manifest_path, manifest)
    try:
        source_paths = [config.road_path, *config.depth_paths]
        for record in review.auxiliary_rasters:
            source_paths.extend(Path(record[key]) for key in ("study_path", "domain_path") if record[key])
        if config.edge_path:
            source_paths.append(config.edge_path)
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"):
            path = config.boundary_path.with_suffix(suffix)
            if path.is_file():
                source_paths.append(path)
        if config.boundary_path.suffix.lower() != ".shp":
            source_paths.append(config.boundary_path)
        input_fingerprints = [fingerprint(p) for p in dict.fromkeys(source_paths)]
        write_json(metadata_dir / "input_manifest.json", {"files": input_fingerprints, "rasters": stack.metadata})
        write_json(metadata_dir / "resolved_config.json", asdict(config))
        code_root = Path(__file__).resolve().parents[1]
        snapshot = metadata_dir / "code_snapshot"
        code_files = sorted([*code_root.rglob("*.py"), *code_root.rglob("*.toml")])
        for path in code_files:
            target = snapshot / path.relative_to(code_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        write_json(metadata_dir / "code_manifest.json", {"files": [fingerprint(p) for p in code_files]})
        write_json(metadata_dir / "environment.json", {"python": platform.python_version(),
                   "geopandas": gpd.__version__, "shapely": shapely.__version__,
                   "rasterio": rasterio.__version__, "numpy": np.__version__, "pandas": pd.__version__})

        facilities_output = destination / "01-01__road_flood_status_five_scenarios.gpkg"
        facilities = roads.copy()
        for column in new_columns:
            facilities[column] = result.status[column].to_numpy()
        facilities.to_file(facilities_output, layer="road_flood_status", driver="GPKG", index=False)
        verify_vector(roads, facilities_output, "road_flood_status", "road_id", result.status)
        result.status.to_csv(destination / "01-02__road_flood_status_five_scenarios.csv", index=False, encoding="utf-8-sig")
        if edges is not None:
            edge_output = destination / "01-03__directed_edge_flood_status_five_scenarios.gpkg"
            labelled_edges = edges.copy()
            aligned = result.status.set_index("road_id").loc[edges.road_id]
            for column in new_columns:
                labelled_edges[column] = aligned[column].to_numpy()
            labelled_edges.to_file(edge_output, layer="routing_edge_flood_status", driver="GPKG", index=False)
            verify_vector(edges, edge_output, "routing_edge_flood_status", "edge_id", result.status)
        long_tables = []
        for period, scenario in zip(config.periods, config.scenario_ids):
            code = f"T{period:03d}"
            long_tables.append(pd.DataFrame({
                "road_id": result.status.road_id,
                "city": config.city, "scenario_id": scenario, "return_period_year": period,
                "flood_status": result.status[f"flood_{code}"],
                "missing_coverage": result.status[f"missing_coverage_{code}"],
                "wet_cell_count": result.status[f"wet_cells_{code}"],
                "missing_cell_count": result.status[f"missing_cells_{code}"],
                "policy_applied": result.status[f"policy_applied_{code}"],
                "policy_reasons": result.status[f"policy_reasons_{code}"],
                "analysis_length_m": result.status.analysis_length_m,
                "vertical_uncertain": result.status.flood_vertical_uncertain,
            }))
        pd.concat(long_tables, ignore_index=True).to_parquet(destination / "02-01__road_scenario_status_long.parquet", index=False)
        result.summary["city"] = config.city
        result.summary.to_csv(destination / "02-02__scenario_summary.csv", index=False, encoding="utf-8-sig")
        # Save an independently auditable table copy while retaining reusable shared caches.
        shutil.copy2(Path(result.qa["mapping_cache_dir"]) / "road_grid_matches.parquet",
                     intermediate / "road_grid_matches.parquet")
        write_json(intermediate / "mapping_scope.json", {
            "scope": "完整研究区格网的道路正长度匹配，包含未受淹和NoData格网",
            "reuse": "换水深、阈值或同格网情景时自动复用；道路/边界/格网/几何容差或算法改变时按新指纹重建。",
            "matching_cache_dir": result.qa["mapping_cache_dir"],
            "matching_cache_key": result.qa["mapping_cache_key"],
            "cache_hit": result.qa["mapping_cache_hit"],
            "input_files": input_fingerprints,
        })
        manifest.update(status="completed", elapsed_seconds=perf_counter() - started,
                        facility_output=str(facilities_output),
                        matching_cache_dir=result.qa["mapping_cache_dir"],
                        mapping_cache_hit=result.qa["mapping_cache_hit"],
                        scenario_summary=result.summary.to_dict(orient="records"))
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=perf_counter() - started)
        write_json(manifest_path, manifest)
        raise
    print(f"[{config.city}] Completed; output={destination}", flush=True)
    return manifest
