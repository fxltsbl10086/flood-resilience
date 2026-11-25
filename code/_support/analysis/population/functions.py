"Calculate exposure on complete 100 m population pixels, then aggregate to existing 500 m grids."

import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from pyproj import CRS


def check_metric_crs(crs, label):
    "Calculate areas in a metre-based projected CRS."
    if crs is None:
        raise ValueError()
    crs = CRS.from_user_input(crs)
    if not crs.is_projected or any(
        abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]
    ):
        raise ValueError()
    return crs


def read_grids(config, grid_limit=None):
    "Use the original centroid rule for district selection and retain complete selected grids."
    grids = gpd.read_file(config["GRID_PATH"]).reset_index(drop=True)
    check_metric_crs(grids.crs, "500 m格网")
    if "grid_id" not in grids or grids.empty:
        raise ValueError()
    if grids.grid_id.isna().any() or grids.grid_id.duplicated().any():
        raise ValueError()
    if grids.geometry.isna().any() or (~grids.geometry.is_valid).any():
        raise ValueError()
    if (grids.geometry.area <= 0).any():
        raise ValueError()
    geometry = grids.geometry.to_numpy()
    left, right = shapely.STRtree(geometry).query(geometry, predicate="intersects")
    pairs = left < right
    if np.any(shapely.area(shapely.intersection(geometry[left[pairs]], geometry[right[pairs]])) > 1e-5):
        raise ValueError()

    scope = config.get("ANALYSIS_SCOPE", "city")
    if scope == "district":
        districts = gpd.read_file(config["DISTRICT_PATH"]).to_crs(grids.crs)
        boundary = gpd.read_file(config["BOUNDARY_PATH"]).to_crs(grids.crs).geometry.union_all()
        name_field = "name" if "name" in districts else "NAME"
        district_name = config["DISTRICT_NAME"]
        source_name = "\u53f0\u6c5f\u533a" if district_name.casefold() == "taijiang" else district_name
        chosen = districts.loc[districts[name_field].isin((district_name, source_name))]
        if chosen.empty:
            raise ValueError()
        district = chosen.geometry.union_all().intersection(boundary)
        # Reuse the SAR district-centroid rule without clipping grid or population-pixel geometry.
        centres = shapely.points(grids.centroid_x, grids.centroid_y)
        selected = shapely.covers(boundary, centres) & shapely.covers(district, centres)
        grids = grids.loc[selected].copy()
    elif scope != "city":
        raise ValueError()
    grids["grid_id"] = grids.grid_id.astype(str)
    grids = grids.sort_values("grid_id").reset_index(drop=True)
    if grid_limit is not None:
        if grid_limit <= 0:
            raise ValueError()
        grids = grids.iloc[:grid_limit].copy()
    if grids.empty:
        raise ValueError()
    return grids


def raster_cell_polygons(transform, rows, cols):
    "Construct native pixel corners from the affine transform, including rotation, without resampling."
    cols = np.asarray(cols)[:, None] + np.array([0, 1, 1, 0, 0])
    rows = np.asarray(rows)[:, None] + np.array([0, 0, 1, 1, 0])
    x = transform.a * cols + transform.b * rows + transform.c
    y = transform.d * cols + transform.e * rows + transform.f
    return shapely.polygons(np.stack([x, y], axis=-1))


def bincount_by_grid(indices, weights, grid_count):
    "Accumulate grid-index weights using the existing grid-exposure method."
    return np.bincount(indices, weights=weights, minlength=grid_count)


def prepare_population_pixels(grids, population_path, city, batch_size=20000):
    "Read complete 100 m pixels intersecting selected grids. Store each source pixel once and use overlap/source-pixel area as alpha. Zero population is valid; record NoData area without estimating missing population."
    count = len(grids)
    grid_geometry = grids.geometry.to_numpy()
    tree = shapely.STRtree(grid_geometry)
    pixel_geometry, pixel_rows, pixel_cols, pixel_population = [], [], [], []
    mapping_pixel, mapping_grid, mapping_fraction = [], [], []
    known_area = np.zeros(count)
    valid_count, invalid_count = np.zeros(count, dtype=int), np.zeros(count, dtype=int)
    offset = 0
    with rasterio.open(population_path) as src:
        crs = check_metric_crs(src.crs, "人口栅格")
        if crs != CRS.from_user_input(grids.crs):
            raise ValueError()
        if not np.allclose(src.res, (100, 100), atol=1e-8, rtol=0):
            raise ValueError()
        band = src.read(1, masked=True)
        values = np.asarray(band.data)
        invalid = np.ma.getmaskarray(band) | ~np.isfinite(values) | (values < 0)
        cell_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
        for start in range(0, src.width * src.height, batch_size):
            positions = np.arange(start, min(start + batch_size, src.width * src.height))
            rows, cols = np.divmod(positions, src.width)
            cells = raster_cell_polygons(src.transform, rows, cols)
            cell_index, grid_index = tree.query(cells, predicate="intersects")
            if not len(cell_index):
                continue
            areas = shapely.area(shapely.intersection(cells[cell_index], grid_geometry[grid_index]))
            positive = areas > 0
            cell_index, grid_index, areas = cell_index[positive], grid_index[positive], areas[positive]
            if not len(cell_index):
                continue
            unique_cells, local_index = np.unique(cell_index, return_inverse=True)
            selected_rows, selected_cols = rows[unique_cells], cols[unique_cells]
            weights = values[selected_rows, selected_cols].astype(float)
            weights[invalid[selected_rows, selected_cols]] = np.nan
            # Retain the complete source pixel when calculating its inundation fraction.
            pixel_geometry.append(cells[unique_cells])
            pixel_rows.append(selected_rows)
            pixel_cols.append(selected_cols)
            pixel_population.append(weights)
            mapping_pixel.append(offset + local_index)
            mapping_grid.append(grid_index)
            mapping_fraction.append(areas / cell_area)
            valid = ~invalid[rows[cell_index], cols[cell_index]]
            known_area += bincount_by_grid(grid_index[valid], areas[valid], count)
            valid_count += np.bincount(grid_index[valid], minlength=count)
            invalid_count += np.bincount(grid_index[~valid], minlength=count)
            offset += len(unique_cells)
        metadata = {
            "population_path": str(population_path), "population_crs": str(src.crs),
            "population_resolution_m": list(src.res), "population_cell_area_m2": cell_area,
            "population_nodata": src.nodata, "population_source_tags": src.tags(),
            "population_raster_transform": list(src.transform)[:6],
            "population_raster_width": src.width, "population_raster_height": src.height,
            "population_resampled": False,
        }
        transform = src.transform
    def concatenate(parts, dtype):
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)
    rows, cols = concatenate(pixel_rows, int), concatenate(pixel_cols, int)
    population = concatenate(pixel_population, float)
    geometry = concatenate(pixel_geometry, object)
    pixel_index = concatenate(mapping_pixel, int)
    grid_index = concatenate(mapping_grid, int)
    fraction = concatenate(mapping_fraction, float)
    selected_fraction = np.bincount(pixel_index, weights=fraction, minlength=len(population))
    if np.any(selected_fraction > 1 + 1e-9):
        raise ValueError()
    grid_area = grids.geometry.area.to_numpy()
    unknown_area = grid_area - known_area
    if np.any(unknown_area < -1e-5):
        raise ValueError()
    tiny_negative = (unknown_area < 0) & (unknown_area >= -1e-5)
    unknown_area[tiny_negative] = 0
    status = np.full(count, "partial", dtype=object)
    status[unknown_area <= 1e-5] = "complete"
    status[known_area == 0] = "no_data"
    x = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
    y = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
    table = pd.DataFrame({
        "pixel_id": [f"{city}_P{row}_{col}" for row, col in zip(rows, cols)],
        "city": city, "population_row": rows, "population_col": cols,
        "x": x, "y": y, "population": population,
        "population_data_valid": np.isfinite(population),
        "selected_grid_area_fraction": selected_fraction,
        "population_in_selected_grids": population * selected_fraction,
    })
    metadata.update({
        "source_pixel_count": len(table), "valid_source_pixel_count": int(np.isfinite(population).sum()),
        "full_source_pixel_known_population_sum": float(np.nansum(population)),
        "known_population_sum": float(np.nansum(population * selected_fraction)),
        "population_complete_grid_count": int((status == "complete").sum()),
        "population_partial_grid_count": int((status == "partial").sum()),
        "population_no_data_grid_count": int((status == "no_data").sum()),
        "population_unknown_area_m2": float(unknown_area.sum()),
        "floating_area_residual_corrected_count": int(tiny_negative.sum()),
    })
    return {
        "table": table, "geometry": geometry, "cell_area": cell_area,
        "pixel_index": pixel_index, "grid_index": grid_index, "fraction": fraction,
        "known_area": known_area, "unknown_area": unknown_area, "status": status,
        "known_pixel_count": valid_count, "missing_pixel_count": invalid_count,
    }, metadata


def calculate_pixel_flood_population(pixels, flood_path, target_crs, threshold, batch_size=20000):
    "Stage 1: calculate inundated area fractions and exposed population on complete 100 m pixels. Count only valid depth >=0.30 m; treat other areas, including NoData and uncovered cells, as not inundated."
    tree = shapely.STRtree(pixels["geometry"])
    wet_area = np.zeros(len(pixels["table"]))
    with rasterio.open(flood_path) as src:
        if check_metric_crs(src.crs, "水深栅格") != CRS.from_user_input(target_crs):
            raise ValueError()
        band = src.read(1, masked=True)
        values = np.asarray(band.data)
        valid = ~np.ma.getmaskarray(band) & np.isfinite(values)
        if np.any(values[valid] < 0):
            raise ValueError()
        rows, cols = np.where(valid & (values >= threshold))
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            cells = raster_cell_polygons(src.transform, rows[start:stop], cols[start:stop])
            cell_index, pixel_index = tree.query(cells, predicate="intersects")
            if not len(cell_index):
                continue
            areas = shapely.area(shapely.intersection(cells[cell_index], pixels["geometry"][pixel_index]))
            wet_area += np.bincount(pixel_index, weights=areas, minlength=len(wet_area))
        metadata = {
            "flood_path": str(flood_path), "flood_crs": str(src.crs),
            "flood_resolution_m": list(src.res), "flood_nodata": src.nodata,
            "flood_valid_cell_count": int(valid.sum()), "flood_affected_cell_count": len(rows),
            "flood_resampled": False, "uncovered_treated_as_nonexposed": True,
        }
    if np.any(wet_area > pixels["cell_area"] + 1e-5):
        raise ValueError()
    # Correct only geometric floating-point error; retain substantive validation failures.
    tiny_excess = (wet_area > pixels["cell_area"]) & (wet_area <= pixels["cell_area"] + 1e-5)
    wet_area[tiny_excess] = pixels["cell_area"]
    wet_fraction = wet_area / pixels["cell_area"]
    population = pixels["table"].population.to_numpy()
    exposed = population * wet_fraction
    nonexposed = population - exposed
    metadata["floating_area_excess_corrected_count"] = int(tiny_excess.sum())
    return {"exposed": exposed, "nonexposed": nonexposed,
            "wet_area": wet_area, "wet_fraction": wet_fraction}, metadata


def aggregate_to_grids(pixels, values, grid_count):
    "Stage 2: aggregate 100 m population attributes to 500 m grids using fixed alpha weights."
    valid = pixels["table"].population_data_valid.to_numpy()
    if np.any(~np.isfinite(values[valid])):
        raise ValueError()
    known_values = np.where(valid, values, 0)
    return bincount_by_grid(pixels["grid_index"],
                           known_values[pixels["pixel_index"]] * pixels["fraction"], grid_count)



def calculate_100m_pixels(config, grid_limit=None):
    "Stage-1 entry: compute 100 m results and save geometry information for stage 2."
    started = time.perf_counter()
    batch_size, threshold = config["GEOMETRY_BATCH_SIZE"], config["DEPTH_THRESHOLD_M"]
    if batch_size <= 0 or not np.isfinite(threshold) or threshold < 0:
        raise ValueError()
    grids = read_grids(config, grid_limit)
    pixels, population_metadata = prepare_population_pixels(
        grids, config["POPULATION_PATH"], config["CITY"], batch_size)
    table = pixels["table"]
    area = config["DISTRICT_NAME"] if config.get("ANALYSIS_SCOPE", "city") == "district" else "全研究区"
    record = {
        "stage": "100m", "status": "completed", "city": config["CITY"],
        "test_area": area, "grid_path": str(config["GRID_PATH"]), "grid_limit": grid_limit,
        "grid_count": len(grids), "selected_grid_ids": grids.grid_id.tolist(),
        "pixel_count": len(table),
        "algorithm": "完整100 m像元的人口乘以原像元受淹面积比例",
        "depth_threshold_m": threshold, "threshold_operator": ">=",
        "uncovered_flood_rule": "水深NoData、非有限或栅格未覆盖均按未淹没",
        "population": population_metadata, "scenarios": {},
    }
    for scenario, path in config["FLOOD_PATHS"].items():
        print(f"[{config['CITY']}] {scenario}: calculating exposure for {len(table)} complete 100 m pixels", flush=True)
        groups, metadata = calculate_pixel_flood_population(
            pixels, path, grids.crs, threshold, batch_size)
        table[f"inundated_area_m2_{scenario}"] = groups["wet_area"]
        table[f"inundated_fraction_{scenario}"] = groups["wet_fraction"]
        for group in ["exposed", "nonexposed"]:
            table[f"population_{group}_{scenario}"] = groups[group]
            table[f"population_{group}_in_selected_grids_{scenario}"] = (
                groups[group] * table.selected_grid_area_fraction)
        record["scenarios"][scenario] = metadata
    record["elapsed_seconds"] = time.perf_counter() - started
    return table, record


def restore_population_mapping(grids, table, source_record, batch_size):
    "Recover pixels and alpha from saved rows, columns and affine transforms without rereading population rasters."
    from affine import Affine

    metadata = source_record["population"]
    if check_metric_crs(metadata["population_crs"], "保存的人口坐标系") != CRS.from_user_input(grids.crs):
        raise ValueError()
    required = ["pixel_id", "city", "population_row", "population_col", "population",
                "population_data_valid", "selected_grid_area_fraction"]
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError()
    if len(table) != source_record["pixel_count"] or table.pixel_id.duplicated().any():
        raise ValueError()
    rows, cols = table.population_row.to_numpy(), table.population_col.to_numpy()
    if (np.any(~np.isfinite(rows)) or np.any(~np.isfinite(cols))
            or np.any(rows != np.floor(rows)) or np.any(cols != np.floor(cols))):
        raise ValueError()
    rows, cols = rows.astype(int), cols.astype(int)
    if (np.any(rows < 0) or np.any(cols < 0)
            or np.any(rows >= metadata["population_raster_height"])
            or np.any(cols >= metadata["population_raster_width"])
            or table.duplicated(["population_row", "population_col"]).any()):
        raise ValueError()
    expected_ids = [f"{source_record['city']}_P{row}_{col}" for row, col in zip(rows, cols)]
    if table.pixel_id.tolist() != expected_ids or not table.city.eq(source_record["city"]).all():
        raise ValueError()
    population = table.population.to_numpy(float)
    if not table.population_data_valid.isin([True, False]).all():
        raise ValueError()
    valid = table.population_data_valid.to_numpy(bool)
    if not np.array_equal(valid, np.isfinite(population)) or np.any(population[valid] < 0):
        raise ValueError()
    transform = Affine(*metadata["population_raster_transform"])
    cell_area = abs(transform.a * transform.e - transform.b * transform.d)
    if not np.isclose(cell_area, metadata["population_cell_area_m2"], atol=1e-8, rtol=0):
        raise ValueError()
    geometry = raster_cell_polygons(transform, rows, cols)
    tree = shapely.STRtree(grids.geometry.to_numpy())
    pixel_parts, grid_parts, fraction_parts = [], [], []
    for start in range(0, len(table), batch_size):
        stop = min(start + batch_size, len(table))
        local_pixel, grid_index = tree.query(geometry[start:stop], predicate="intersects")
        areas = shapely.area(shapely.intersection(
            geometry[start:stop][local_pixel], grids.geometry.to_numpy()[grid_index]))
        positive = areas > 0
        pixel_parts.append(start + local_pixel[positive])
        grid_parts.append(grid_index[positive])
        fraction_parts.append(areas[positive] / cell_area)
    pixel_index = np.concatenate(pixel_parts) if pixel_parts else np.array([], dtype=int)
    grid_index = np.concatenate(grid_parts) if grid_parts else np.array([], dtype=int)
    fraction = np.concatenate(fraction_parts) if fraction_parts else np.array([], dtype=float)
    scope_fraction = np.bincount(pixel_index, weights=fraction, minlength=len(table))
    if not np.allclose(scope_fraction, table.selected_grid_area_fraction, atol=1e-9, rtol=1e-10):
        raise ValueError()
    pair_valid = valid[pixel_index]
    known_area = bincount_by_grid(grid_index[pair_valid],
                                 fraction[pair_valid] * cell_area, len(grids))
    unknown_area = grids.geometry.area.to_numpy() - known_area
    if np.any(unknown_area < -1e-5):
        raise ValueError()
    tiny_negative = (unknown_area < 0) & (unknown_area >= -1e-5)
    unknown_area[tiny_negative] = 0
    status = np.full(len(grids), "partial", dtype=object)
    status[unknown_area <= 1e-5] = "complete"
    status[known_area == 0] = "no_data"
    return {
        "table": table, "geometry": geometry, "cell_area": cell_area,
        "pixel_index": pixel_index, "grid_index": grid_index, "fraction": fraction,
        "known_area": known_area, "unknown_area": unknown_area, "status": status,
        "known_pixel_count": np.bincount(grid_index[pair_valid], minlength=len(grids)),
        "missing_pixel_count": np.bincount(grid_index[~pair_valid], minlength=len(grids)),
    }


def aggregate_saved_pixels(config, table, source_record):
    "Stage-2 entry: aggregate completed 100 m results to original 500 m grids by alpha. Read no population, depth or district raster; use the grid IDs saved by stage 1."
    started = time.perf_counter()
    if source_record.get("stage") != "100m" or source_record.get("status") != "completed":
        raise ValueError()
    if source_record["city"] != config["CITY"] or source_record["threshold_operator"] != ">=":
        raise ValueError()
    # Select original grids using the stage-1 IDs without repeating district selection.
    grid_config = dict(config, ANALYSIS_SCOPE="city")
    all_grids = read_grids(grid_config)
    selected_ids = source_record["selected_grid_ids"]
    if (len(set(selected_ids)) != len(selected_ids)
            or not set(selected_ids).issubset(set(all_grids.grid_id))):
        raise ValueError()
    grids = all_grids.set_index("grid_id", drop=False).loc[selected_ids].reset_index(drop=True)
    pixels = restore_population_mapping(grids, table, source_record, config["GEOMETRY_BATCH_SIZE"])
    population = aggregate_to_grids(pixels, table.population.to_numpy(float), len(grids))
    result = pd.DataFrame(grids.drop(columns=grids.geometry.name)).copy()
    result["city"] = config["CITY"]
    result["full_grid_area_m2"] = grids.geometry.area.to_numpy()
    result["population"] = population
    result["population_status"] = pixels["status"]
    result["population_known_area_m2"] = pixels["known_area"]
    result["population_unknown_area_m2"] = pixels["unknown_area"]
    result["population_data_coverage_ratio"] = pixels["known_area"] / result.full_grid_area_m2
    result["population_valid_pixel_count"] = pixels["known_pixel_count"]
    result["population_invalid_pixel_count"] = pixels["missing_pixel_count"]
    no_data = pixels["status"] == "no_data"
    result.loc[no_data, "population"] = np.nan
    record = {
        "stage": "500m", "status": "completed", "city": config["CITY"],
        "test_area": source_record["test_area"], "grid_path": str(config["GRID_PATH"]),
        "grid_count": len(grids), "pixel_count": len(table),
        "depth_threshold_m": source_record["depth_threshold_m"], "threshold_operator": ">=",
        "algorithm": "读取100 m原人数和已计算的淹没人数，各乘一次alpha汇总至500 m",
        "reread_population_raster": False, "reread_flood_raster": False,
        "scenarios": {},
    }
    valid = table.population_data_valid.to_numpy(bool)
    source_population = table.population.to_numpy(float)
    summary = []
    for scenario in source_record["scenarios"]:
        exposed_column = f"population_exposed_{scenario}"
        unexposed_column = f"population_nonexposed_{scenario}"
        if exposed_column not in table or unexposed_column not in table:
            raise ValueError()
        source_exposed = table[exposed_column].to_numpy(float)
        source_unexposed = table[unexposed_column].to_numpy(float)
        tolerance = np.maximum(1e-7, source_population[valid] * 1e-10)
        if (np.any(~np.isfinite(source_exposed[valid]))
                or np.any(~np.isfinite(source_unexposed[valid]))
                or np.any(source_exposed[valid] < 0)
                or np.any(source_exposed[valid] > source_population[valid] + tolerance)
                or np.any(np.abs(source_exposed[valid] + source_unexposed[valid] - source_population[valid]) > tolerance)
                or np.any(np.isfinite(source_exposed[~valid]))
                or np.any(np.isfinite(source_unexposed[~valid]))):
            raise ValueError()
        # Read complete-pixel population, avoiding a second multiplication by the selected-area fraction.
        exposed = aggregate_to_grids(pixels, source_exposed, len(grids))
        nonexposed = population - exposed
        for group, values in [("exposed", exposed), ("nonexposed", nonexposed)]:
            result[f"population_{group}_{scenario}"] = values
            result.loc[no_data, f"population_{group}_{scenario}"] = np.nan
        result[f"exposed_ratio_{scenario}"] = np.divide(
            exposed, population, out=np.full(len(grids), np.nan), where=population > 0)
        error = exposed + nonexposed - population
        record["scenarios"][scenario] = {"maximum_population_partition_error": float(np.abs(error).max())}
        summary.append({
            "city": config["CITY"], "city_name": config["CITY_NAME"], "test_area": source_record["test_area"],
            "scenario": scenario, "grid_count": len(grids),
            "depth_threshold_m": source_record["depth_threshold_m"], "threshold_operator": ">=",
            "population": float(population.sum()), "population_exposed": float(exposed.sum()),
            "population_nonexposed": float(nonexposed.sum()),
            "population_unknown_area_m2": float(pixels["unknown_area"].sum()),
            "population_partial_grid_count": int((pixels["status"] == "partial").sum()),
            "population_no_data_grid_count": int(no_data.sum()),
        })
    record["elapsed_seconds"] = time.perf_counter() - started
    return result, pd.DataFrame(summary), record


def calculate_city(config, grid_limit=None):
    "Retain the legacy API while the two formal scripts use the independent stages."
    table, source_record = calculate_100m_pixels(config, grid_limit)
    result, summary, record = aggregate_saved_pixels(config, table, source_record)
    record["population"] = source_record["population"]
    for scenario, metadata in source_record["scenarios"].items():
        record["scenarios"][scenario].update(metadata)
    record["elapsed_seconds"] += source_record["elapsed_seconds"]
    return result, summary, record, table
