"Match roads and candidate grids once, then summarize threshold exceedance and unknown coverage by scenario."

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import geometry_mask
import shapely
from shapely.geometry import box, mapping

from .rasters import DepthStack


@dataclass
class MatchResult:
    status: pd.DataFrame
    pairs: pd.DataFrame
    summary: pd.DataFrame
    qa: dict


@dataclass
class SpatialMapping:
    pairs: pd.DataFrame
    analysis: pd.DataFrame
    study_cell_ids: np.ndarray
    descriptor: dict
    qa: dict


def validate_roads(roads: gpd.GeoDataFrame) -> None:
    if roads.crs is None or roads.empty:
        raise ValueError()
    if "road_id" not in roads or roads.road_id.isna().any() or not roads.road_id.is_unique:
        raise ValueError()
    if not roads.geometry.geom_type.isin(["LineString", "MultiLineString"]).all():
        raise ValueError()
    if roads.geometry.is_empty.any() or not roads.geometry.is_valid.all():
        raise ValueError()


def build_spatial_mapping(roads: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
                          stack: DepthStack, tolerance_m: float = 1e-6,
                          cell_batch_size: int = 20000, descriptor: dict | None = None) -> SpatialMapping:
    started = perf_counter()
    validate_roads(roads)
    if boundary.crs is None or boundary.empty:
        raise ValueError()
    if not boundary.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all() or not boundary.geometry.is_valid.all():
        raise ValueError()
    if tolerance_m < 0 or cell_batch_size <= 0:
        raise ValueError()
    work = roads.to_crs(stack.crs)
    study = boundary.to_crs(stack.crs).geometry.union_all()
    if study.is_empty:
        raise ValueError()
    analysis_geometry = shapely.intersection(work.geometry.to_numpy(), study)
    analysis_length = shapely.length(analysis_geometry)
    in_study = analysis_length > tolerance_m
    n = len(roads)
    height, width = stack.shape
    transform = stack.transform
    footprint = box(transform.c, transform.f + height * transform.e,
                    transform.c + width * transform.a, transform.f)
    outside_length = shapely.length(shapely.difference(analysis_geometry, footprint))
    outside = outside_length > tolerance_m

    # Include boundary-touching grids, then require clipped roads with positive intersection length.
    study_cells = geometry_mask([mapping(study)], stack.shape, transform,
                               all_touched=True, invert=True)
    # Store all in-study-area geometric matches independently of depth, thresholds and NoData.
    rows, cols = np.nonzero(study_cells)
    cell_ids = rows.astype(np.int64) * width + cols
    prep_seconds = perf_counter() - started
    match_started = perf_counter()
    tree = shapely.STRtree(analysis_geometry)
    road_indices, matched_cells, matched_lengths = [], [], []
    for begin in range(0, len(rows), cell_batch_size):
        end = min(begin + cell_batch_size, len(rows))
        r, c = rows[begin:end], cols[begin:end]
        polygons = shapely.box(
            transform.c + c * transform.a,
            transform.f + (r + 1) * transform.e,
            transform.c + (c + 1) * transform.a,
            transform.f + r * transform.e,
        )
        pairs = tree.query(polygons, predicate="intersects")
        if not pairs.shape[1]:
            continue
        lengths = shapely.length(shapely.intersection(
            polygons[pairs[0]], analysis_geometry[pairs[1]],
        ))
        keep = lengths > tolerance_m
        road_indices.append(pairs[1][keep])
        matched_cells.append(cell_ids[begin:end][pairs[0][keep]])
        matched_lengths.append(lengths[keep])
    ri = np.concatenate(road_indices) if road_indices else np.empty(0, dtype=np.int64)
    ci = np.concatenate(matched_cells) if matched_cells else np.empty(0, dtype=np.int64)
    il = np.concatenate(matched_lengths) if matched_lengths else np.empty(0, dtype=float)
    match_seconds = perf_counter() - match_started
    pairs_table = pd.DataFrame({
        "road_index": ri, "road_id": roads.road_id.to_numpy()[ri],
        "cell_id": ci, "row": ci // width, "col": ci % width,
        "intersection_length_m": il,
    })
    qa = {
        "road_count": n, "roads_entering_study_area": int(in_study.sum()),
        "roads_outside_study_area": int((~in_study).sum()),
        "roads_with_parts_outside_raster": int(outside.sum()),
        "study_cells": len(rows), "matched_road_cell_pairs": len(ci),
        "duplicate_road_cell_pairs": int(pairs_table.duplicated(["road_id", "cell_id"]).sum()),
        "point_only_contacts_excluded": True,
        "pair_lengths_are_not_additive_at_shared_grid_edges": True,
        "build_timings_seconds": {"preparation": prep_seconds, "spatial_mapping": match_seconds,
                                  "mapping_total": perf_counter() - started},
    }
    if "target_length_m" in roads:
        differences = np.abs(roads.target_length_m.to_numpy(dtype=float) - analysis_length)
        qa["max_analysis_vs_original_target_length_difference_m"] = float(np.nanmax(differences))
    if qa["duplicate_road_cell_pairs"]:
        raise RuntimeError()
    analysis = pd.DataFrame({"road_id": roads.road_id.to_numpy(),
                             "analysis_length_m": analysis_length,
                             "outside_grid_length_m": outside_length})
    descriptor = descriptor or {"grid_shape": list(stack.shape), "grid_transform": list(stack.transform),
                                "grid_crs": stack.crs.to_wkt()}
    return SpatialMapping(pairs_table, analysis, cell_ids, descriptor, qa)


def identify_roads(roads: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
                   stack: DepthStack, threshold_m: float = 0.30,
                   tolerance_m: float = 1e-6, cell_batch_size: int = 20000,
                   spatial_mapping: SpatialMapping | None = None) -> MatchResult:
    started = perf_counter()
    validate_roads(roads)
    if not np.isfinite(threshold_m) or threshold_m <= 0 or tolerance_m < 0 or cell_batch_size <= 0:
        raise ValueError()
    if (stack.values.shape != stack.valid.shape or len(stack.periods) != len(stack.values)
            or len(set(stack.periods)) != len(stack.periods)):
        raise ValueError()
    spatial_mapping = spatial_mapping or build_spatial_mapping(
        roads, boundary, stack, tolerance_m, cell_batch_size)
    if (spatial_mapping.descriptor.get("grid_shape") != list(stack.shape)
            or spatial_mapping.descriptor.get("grid_transform") != list(stack.transform)
            or spatial_mapping.descriptor.get("grid_crs") != stack.crs.to_wkt()):
        raise ValueError()
    analysis = spatial_mapping.analysis
    if not np.array_equal(analysis.road_id.to_numpy(), roads.road_id.to_numpy()):
        raise ValueError()
    n = len(roads)
    analysis_length = analysis.analysis_length_m.to_numpy()
    outside_length = analysis.outside_grid_length_m.to_numpy()
    in_study = analysis_length > tolerance_m
    outside = outside_length > tolerance_m
    ri = spatial_mapping.pairs.road_index.to_numpy(dtype=np.int64)
    ci = spatial_mapping.pairs.cell_id.to_numpy(dtype=np.int64)
    if len(ci) and (ci.min() < 0 or ci.max() >= np.prod(stack.shape)):
        raise ValueError()
    wet = stack.valid & (stack.values >= threshold_m)
    flat_wet = wet.reshape(len(stack.periods), -1)
    flat_unknown = (~stack.valid).reshape(len(stack.periods), -1)
    vertical = np.zeros(n, dtype=bool)
    for column in ("contains_bridge", "is_elevated", "is_underpass", "mixed_vertical_signature"):
        if column in roads:
            vertical |= roads[column].fillna(False).to_numpy(dtype=bool)
    status = pd.DataFrame({
        "road_id": roads.road_id.to_numpy(),
        "analysis_length_m": analysis_length,
        "outside_grid_length_m": outside_length,
        "flood_vertical_uncertain": vertical,
    })
    summary = []
    for k, (period, scenario) in enumerate(zip(stack.periods, stack.scenario_ids)):
        code = f"T{period:03d}"
        hit_count = np.bincount(ri[flat_wet[k, ci]], minlength=n)
        unknown_count = np.bincount(ri[flat_unknown[k, ci]], minlength=n)
        hit = hit_count > 0
        uncertain = (unknown_count > 0) | outside | ~in_study
        # Any known inundated part confirms exposure; a non-hit with unknown coverage cannot be classified as zero.
        flood_status = np.where(hit, 1, np.where(uncertain, -1, 0)).astype(np.int8)
        status[f"flood_{code}"] = flood_status
        status[f"unknown_{code}"] = uncertain
        status[f"wet_cells_{code}"] = hit_count.astype(np.int32)
        status[f"unknown_cells_{code}"] = unknown_count.astype(np.int32)
        summary.append({
            "city": "", "return_period_year": period, "scenario_id": scenario,
            "threshold_m": threshold_m, "road_count": n,
            "flooded_road_count": int(hit.sum()),
            "known_below_threshold_road_count": int((flood_status == 0).sum()),
            "unknown_status_road_count": int((flood_status == -1).sum()),
            "roads_with_unknown_coverage": int(uncertain.sum()),
            "flooded_roads_with_vertical_uncertainty": int((hit & vertical).sum()),
            # This is the within-area length of affected facility segments, not the length of inundated fragments.
            "flooded_facility_analysis_length_m": float(analysis_length[hit].sum()),
            "valid_study_cells": int(stack.valid[k].ravel()[spatial_mapping.study_cell_ids].sum()),
            "wet_study_cells": int(wet[k].ravel()[spatial_mapping.study_cell_ids].sum()),
        })
    qa = {
        **spatial_mapping.qa, "road_id_unique": True,
        "vertical_uncertain_road_count": int(vertical.sum()),
        "shared_mapping_for_all_scenarios": True,
        "grid_identical_across_scenarios": True,
        "status_values": [-1, 0, 1],
        "aggregation_seconds": perf_counter() - started,
    }
    return MatchResult(status, spatial_mapping.pairs, pd.DataFrame(summary), qa)
