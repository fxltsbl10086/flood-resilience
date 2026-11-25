"""Summarize exported depths without changing models or native results."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

if __package__:
    from .export_depth import DEPTH_PRODUCTS, _model_input, _result_directory, load_config, read_static
else:
    from export_depth import DEPTH_PRODUCTS, _model_input, _result_directory, load_config, read_static


CITY_NAMES = {
    "fuzhou": "福州", "wuhan": "武汉", "hangzhou": "杭州",
    "chengdu": "成都", "zhengzhou": "郑州",
}
THRESHOLDS = (0.1, 0.3, 0.5, 1.0, 2.0)


def _write_text(path: Path, text: str, overwrite: bool) -> bool:
    """Preserve existing statistics by default instead of replacing historical conclusions."""
    if path.exists() and not path.is_file():
        raise FileExistsError()
    if path.is_file() and not overwrite:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _write_json(path: Path, data: dict, overwrite: bool) -> bool:
    return _write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       overwrite)


def _mark_validation_failure(result_root: Path, notes: list[str],
                             directories: Sequence[Path] = ()) -> None:
    """Persist validation failure and stop the pipeline while preserving invalid status."""
    paths = [result_root / "batch_run_status.json"]
    paths.extend(directory / "run_status.json" for directory in directories)
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
        if not str(record.get("status", "")).startswith("invalid"):
            record["status"] = "failed_forcing_reception"
        record["scientific_use_valid"] = False
        record["forcing_reception_validation_status"] = "failed"
        record["forcing_reception_failure_notes"] = notes
        record["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(path, record, overwrite=True)


def _read_product(path: Path, grid: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """Calculate area-weighted statistics including zero depths within the valid mask."""
    with rasterio.open(path) as src:
        if (src.count != 1 or src.crs != grid["crs"] or src.transform != grid["transform"]
                or src.shape != grid["mask"].shape):
            raise ValueError()
        if not src.crs.is_projected:
            raise ValueError()
        unit_factor = float(src.crs.linear_units_factor[1])
        cell_area_m2 = abs(src.transform.a * src.transform.e
                           - src.transform.b * src.transform.d) * unit_factor ** 2
        raw = src.read(1, masked=True)
        selected = ~np.ma.getmaskarray(raw)
        depth = np.asarray(raw.data, dtype=np.float32)
        values = depth[selected]
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError()
        stats = {
            "path": str(path.resolve()), "units": "m", "valid_cells": int(selected.sum()),
            "valid_area_km2": float(selected.sum() * cell_area_m2 / 1e6),
            "minimum_depth_m": float(values.min()) if values.size else None,
            "maximum_depth_m": float(values.max()) if values.size else None,
            "mean_depth_m_including_zeros": float(values.mean(dtype=np.float64)) if values.size else None,
            "area_at_least_depth_km2": {
                str(threshold): float(np.count_nonzero(values >= threshold) * cell_area_m2 / 1e6)
                for threshold in THRESHOLDS
            },
            "grid_resolution_x_m": float(abs(src.transform.a) * unit_factor),
            "grid_resolution_y_m": float(abs(src.transform.e) * unit_factor),
            "tif_tags": src.tags(),
        }
    return stats, depth, selected


def _forcing_reception(ds: xr.Dataset, grid: dict, input_path: Path,
                       start: pd.Timestamp, stop: pd.Timestamp) -> dict:
    """Check received CN runoff without claiming a complete water-budget closure.

SFINCS 2.4 cumprcp is cumulative; do not sum separate timemax snapshots.
Integrate forcing frame by frame and compare at saved output times."""
    internal = grid["mask"] == 1
    cells = int(internal.sum())
    if not cells:
        raise ValueError()
    settings = grid["config"]
    if int(float(settings.get("amprblock", 0))) != 1:
        raise ValueError()
    forcing_path = (input_path.parent / settings["netamprfile"]).resolve()
    if "cumprcp" not in ds or "timemax" not in ds.cumprcp.dims:
        raise ValueError()
    if str(ds.cumprcp.attrs.get("units", "")).strip().lower() != "m":
        raise ValueError()
    native_times = pd.DatetimeIndex(ds.timemax.values)
    if (not native_times.is_monotonic_increasing or native_times.has_duplicates
            or not len(native_times) or native_times[-1] != stop):
        raise ValueError()
    if tuple(ds.cumprcp.dims[-2:]) != tuple(ds.msk.dims):
        raise ValueError()
    # Flip north-up static arrays to match south-to-north solver/forcing coordinates.
    unit_factor = float(grid["crs"].linear_units_factor[1])
    transform = grid["transform"]
    area = abs(transform.a * transform.e - transform.b * transform.d) * unit_factor ** 2
    expected = np.zeros(internal.shape, dtype=np.float64)
    checkpoints, notes, previous_actual, previous_expected = [], [], None, None
    last_positive_end, dry_intervals_checked = None, 0
    # Use h5netcdf for NETCDF4 inputs and non-ASCII Windows paths.
    with xr.open_dataset(forcing_path, engine="h5netcdf") as forcing:
        times = pd.DatetimeIndex(forcing.time.values)
        if (len(times) < 2 or not times.is_monotonic_increasing or times.has_duplicates
                or times[0] > start or times[-1] < stop):
            raise ValueError()
        variable = forcing["Precipitation"]
        if variable.dims != ("time", "y", "x") or variable.shape[1:] != internal.shape:
            raise ValueError()
        forcing_units = str(variable.attrs.get("units", "")).strip().lower()
        if forcing_units not in {"mm/hr", "mm/h", "mm/hour", "mm hour-1", "mm h-1"}:
            raise ValueError()
        rows, columns = internal.shape
        wanted_x = transform.c + np.arange(columns) * transform.a
        wanted_y = transform.f + rows * transform.e - np.arange(rows) * transform.e
        if (not np.allclose(forcing.x.values, wanted_x, rtol=0, atol=1e-5)
                or not np.allclose(forcing.y.values, wanted_y, rtol=0, atol=1e-5)):
            raise ValueError()
        raw_time_units = str(forcing.time.encoding.get("units", ""))
        cursor = start
        index = int(times.searchsorted(start, side="right") - 1)
        frame, frame_index = None, None
        for native_index, checkpoint in enumerate(native_times):
            if checkpoint < start or checkpoint > stop:
                raise ValueError()
            # Use decoded dates to calculate actual intervals, irrespective of seconds/minutes encoding.
            while cursor < checkpoint:
                if frame_index != index:
                    frame = np.flipud(variable.isel(time=index).values).astype(np.float64)
                    values = frame[internal]
                    if not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError()
                    frame_index = index
                end = min(checkpoint, times[index + 1])
                seconds = float((end - cursor).total_seconds())
                expected[internal] += frame[internal] * seconds / 3600.0
                if np.any(frame[internal] > 0):
                    last_positive_end = end
                cursor = end
                if cursor == times[index + 1] and cursor < stop:
                    index += 1
            actual_m = np.flipud(ds.cumprcp.isel(timemax=native_index).values).astype(np.float64)
            actual = actual_m[internal] * 1000.0
            anticipated = expected[internal]
            if not np.isfinite(actual).all() or (actual < -0.1).any():
                raise ValueError()
            error = actual - anticipated
            # Allow bounded cumulative precision and sampling differences near forcing nodes.
            tolerance = np.maximum(0.1, 0.01 * anticipated)
            bad = int(np.count_nonzero(np.abs(error) > tolerance))
            expected_mean, actual_mean = float(anticipated.mean()), float(actual.mean())
            record = {
                "time": checkpoint.strftime("%Y-%m-%d %H:%M:%S"),
                "expected_cumulative_mean_mm": expected_mean,
                "actual_cumulative_mean_mm": actual_mean,
                "expected_cumulative_volume_m3": float(anticipated.sum() * area / 1000.0),
                "actual_cumulative_volume_m3": float(actual.sum() * area / 1000.0),
                "actual_to_expected_ratio": actual_mean / expected_mean if expected_mean > 0 else None,
                "mean_error_mm": actual_mean - expected_mean,
                "maximum_absolute_cell_error_mm": float(np.max(np.abs(error))),
                "cells_exceeding_tolerance": bad,
            }
            if bad:
                notes.append(f"At {record['time']}, native cumulative runoff differs from integrated input in {bad} cells by more than "
                             "the tolerance max(0.1 mm, 1% of expected cumulative input).")
            if previous_actual is not None:
                increment = actual - previous_actual
                increment_expected = anticipated - previous_expected
                record.update(
                    interval_expected_mean_mm=float(increment_expected.mean()),
                    interval_actual_mean_mm=float(increment.mean()),
                    interval_expected_volume_m3=float(increment_expected.sum() * area / 1000.0),
                    interval_actual_volume_m3=float(increment.sum() * area / 1000.0),
                )
                if not np.any(increment_expected > 1e-9):
                    dry_intervals_checked += 1
                    # Compare cumulative differences rather than treating a later cumulative window as new inflow.
                    tail_bad = int(np.count_nonzero(np.abs(increment) > tolerance))
                    record["dry_interval_cells_exceeding_tolerance"] = tail_bad
                    record["dry_interval_maximum_absolute_increment_mm"] = float(np.max(np.abs(increment)))
                    if tail_bad:
                        notes.append(f"After rainfall stops, cumulative runoff changes beyond the event tolerance in {tail_bad} cells.")
            checkpoints.append(record)
            previous_actual, previous_expected = actual, anticipated.copy()
    if last_positive_end is not None and last_positive_end < stop and not dry_intervals_checked:
        notes.append("The native cumulative snapshots do not isolate the dry interval; save snapshots at rainfall cessation and simulation end to check post-rainfall input.")
    final = checkpoints[-1]
    return {
        "forcing_reception_validation_status": "failed" if notes else "passed",
        "forcing_reception_failure_notes": notes,
        "forcing_reception": {
            "forcing_netcdf": str(forcing_path), "forcing_time_units": raw_time_units,
            "forcing_intensity_units": forcing_units, "amprblock": 1,
            "integration_rule": "Left-endpoint step intensity multiplied by decoded time intervals; internal msk=1 cells only",
            "internal_cells": cells, "internal_area_km2": float(cells * area / 1e6),
            "last_positive_forcing_interval_end": (last_positive_end.strftime("%Y-%m-%d %H:%M:%S")
                                                    if last_positive_end is not None else None),
            "native_cumprcp_interpretation": "Cumulative from simulation start without reset; the final snapshot is total input and snapshots must not be summed",
            "cell_tolerance_mm": "max(0.1 mm, 1% of expected cumulative runoff at that time)",
            "checkpoints": checkpoints,
            "dry_intervals_checked": dry_intervals_checked,
            "dry_interval_reception_verified": bool(dry_intervals_checked and not any(
                checkpoint.get("dry_interval_cells_exceeding_tolerance", 0) for checkpoint in checkpoints)),
            "expected_event_mean_mm": final["expected_cumulative_mean_mm"],
            "actual_event_mean_mm": final["actual_cumulative_mean_mm"],
            "expected_event_volume_m3": final["expected_cumulative_volume_m3"],
            "actual_event_volume_m3": final["actual_cumulative_volume_m3"],
            "actual_to_expected_event_ratio": final["actual_to_expected_ratio"],
            "event_mean_error_mm": final["mean_error_mm"],
            "input_reception_only_not_complete_mass_balance": True,
        },
    }


def _native_diagnostics(path: Path, grid: dict, maximum_land: np.ndarray,
                        final_land: np.ndarray, land: np.ndarray, input_path: Path) -> dict:
    """Inspect native saved frames without diagnostic experiments or spatial post-processing."""
    active = grid["mask"] > 0
    start = pd.to_datetime(grid["config"]["tstart"], format="%Y%m%d %H%M%S")
    stop = pd.to_datetime(grid["config"]["tstop"], format="%Y%m%d %H%M%S")
    # Use the installed HDF5 backend for non-ASCII result paths.
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        if not ds.time.size or pd.Timestamp(ds.time.values[-1]) != stop:
            raise ValueError()
        status = float(ds.status.values[-1])
        if not np.isfinite(status) or status != 0:
            raise ValueError()
        if not np.array_equal(np.flipud(ds.msk.values), grid["mask"]):
            raise ValueError()
        if not np.allclose(np.flipud(ds.zb.values)[active], grid["terrain"][active],
                           rtol=0, atol=1e-5):
            raise ValueError()
        native_max = np.flipud(ds.hmax.max("timemax", skipna=True).values)
        native_final = np.flipud(ds.h.isel(time=-1).values)
        if np.isinf(native_max[active]).any() or not np.isfinite(native_final[active]).all():
            raise ValueError()
        display_max = np.maximum(np.nan_to_num(native_max, nan=0), 0)
        display_final = np.maximum(native_final, 0)
        if not np.allclose(display_max[land], maximum_land[land], rtol=0, atol=1e-5):
            raise ValueError()
        if not np.allclose(display_final[land], final_land[land], rtol=0, atol=1e-5):
            raise ValueError()
        negative_any = np.zeros(active.shape, dtype=bool)
        minimum, nonfinite = None, 0
        # Read frames individually instead of loading the full 12-hour series.
        for index in range(ds.time.size):
            frame = np.flipud(ds.h.isel(time=index).values)
            values = frame[active]
            finite = np.isfinite(values)
            nonfinite += int(np.count_nonzero(~finite))
            if finite.any():
                local_min = float(values[finite].min())
                minimum = local_min if minimum is None else min(minimum, local_min)
            negative_any |= active & np.isfinite(frame) & (frame < -0.01)
        times = pd.DatetimeIndex(ds.time.values)
        earlier = np.flatnonzero(times <= stop - pd.Timedelta(hours=1))
        rising = near_peak = None
        compare_time = None
        if earlier.size:
            index = int(earlier[-1])
            previous = np.flipud(ds.h.isel(time=index).values)
            valid = land & np.isfinite(previous) & np.isfinite(native_final)
            # The 1 mm threshold diagnoses late rises without changing inundation results.
            rising_cells = valid & ((native_final - previous) > 0.001)
            rising = int(np.count_nonzero(rising_cells))
            near_peak = int(np.count_nonzero(rising_cells & (maximum_land >= 0.3)
                                             & ((maximum_land - display_final) <= 0.001)))
            compare_time = times[index].strftime("%Y-%m-%d %H:%M:%S")
        windows = int(ds.sizes.get("timemax", 0))
        forcing_reception = _forcing_reception(ds, grid, input_path, start, stop)
    return {
        "native_netcdf": str(path.resolve()), "native_final_status": status,
        "simulation_start": start.strftime("%Y-%m-%d %H:%M:%S"),
        "simulation_stop": stop.strftime("%Y-%m-%d %H:%M:%S"),
        "simulation_duration_hours": float((stop - start).total_seconds() / 3600),
        "hmax_time_window_count": windows, "native_mask_matches_input": True,
        "native_terrain_matches_input": True, "native_hmax_matches_land_tif": True,
        "native_final_h_matches_land_tif": True,
        "native_minimum_depth_m_at_saved_times": minimum,
        "native_nonfinite_active_values_at_saved_times": nonfinite,
        "domain_cells_below_minus_0p01_at_any_saved_time": int(np.count_nonzero(negative_any)),
        "land_negative_cells_below_minus_0p01": int(np.count_nonzero(negative_any & land)),
        "late_comparison_start": compare_time,
        "late_rising_land_cells_above_1mm": rising,
        "late_rising_near_peak_cells": near_peak,
        "diagnostic_time_sampling_note": "Checks use saved NetCDF depth frames, not every internal solver step.",
        **forcing_reception,
    }


def _rainfall_info(config: Mapping[str, Any], city: str, period: int) -> dict:
    """Retain rainfall values and applicability flags without claiming to recheck source standards."""
    file = Path(config["rainfall_table"])
    table = pd.read_csv(file).fillna("")
    rows = table[(table.city_code == city) & (table.return_period_year == period)]
    if rows.empty:
        raise ValueError()
    records = rows.to_dict(orient="records")
    totals = {str(row["formula_zone_id"]): float(row["rainfall_6h_mm"]) for row in records}
    notes = []
    for row in records:
        # An absent validity range does not establish interpolation or extrapolation.
        range_fields = ("valid_duration_min", "valid_duration_max",
                        "valid_return_period_min", "valid_return_period_max")
        if any(not str(row.get(key, "")).strip() for key in range_fields):
            note = "Formula validity ranges for duration/return period are unrecorded and were not verified here."
            if note not in notes:
                notes.append(note)
        # Retain filled rainfall/parameter values and report incomplete provenance fields.
        parameter_fields = ("a", "c", "b", "n", "rainfall_6h_mm")
        source_status_fields = ("source_standard", "source_page_table",
                                "parameter_status", "calculation_status")
        pending_markers = ("待用户填写", "待填写", "待填", "missing_parameters")
        parameters_recorded = all(str(row.get(key, "")).strip()
                                  for key in parameter_fields)
        pending_source_status = any(marker in str(row.get(key, ""))
                                    for key in source_status_fields
                                    for marker in pending_markers)
        if parameters_recorded and pending_source_status:
            note = ("Rainfall and formula values are present, but provenance/status fields still indicate incomplete records; "
                    "the original formula source was not verified here.")
            if note not in notes:
                notes.append(note)
        if row.get("return_period_application_status") == "extrapolated":
            limit = row.get("valid_return_period_max")
            if limit:
                notes.append(f"T{period:03d} exceeds the recorded formula limit of {limit} years.")
            else:
                notes.append(f"T{period:03d} is recorded as extrapolated, with no numeric upper bound in the rainfall table.")
        duration_status = str(row.get("duration_application_status", ""))
        if duration_status not in {"within_standard", ""}:
            status_names = {
                "\u5f85\u6838\u5b9e": "pending verification",
                "\u5f85\u7528\u6237\u586b\u5199": "pending documentation",
            }
            notes.append(f"Recorded duration applicability: {status_names.get(duration_status, duration_status)}.")
    # Show a shared rainfall value when all zones have the same total.
    unique_totals = set(totals.values())
    return {
        "gross_rainfall_mm": next(iter(unique_totals)) if len(unique_totals) == 1 else None,
        "gross_rainfall_by_zone_mm": totals, "rainfall_table": str(file.resolve()),
        "rainfall_table_records": records, "rainfall_application_notes": notes,
        "source_verification_note": "Values and applicability flags come from the project rainfall table; source-standard PDFs were not independently rechecked.",
    }


def summarize_city(city: str, model_root: str | Path, result_root: str | Path,
                   periods: Sequence[int], config: Mapping[str, Any] | None = None,
                   overwrite: bool = False) -> dict[str, Any]:
    """Summarize cities after exporting all six rasters, preserving existing statistics by default.

Return summaries and written/reused files; incomplete scenarios fail before completion is recorded."""
    config = load_config(city) if config is None else config
    if config.get("city") != city:
        raise ValueError()
    selected = sorted(set(int(period) for period in periods))
    if not selected or any(period not in config["return_periods"] for period in selected):
        raise ValueError()
    model_root, result_root = Path(model_root).resolve(), Path(result_root).resolve()
    if not result_root.is_dir():
        raise FileNotFoundError()
    summaries, directories = [], []
    # Check every requested scenario before creating a completion summary.
    for period in selected:
        directory = _result_directory(result_root, period)
        input_path = _model_input(model_root, directory.name)
        grid = read_static(input_path)
        products, arrays, masks = {}, {}, {}
        for name, _, _, _ in DEPTH_PRODUCTS:
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError()
            products[name], arrays[name], masks[name] = _read_product(path, grid)
        for scope in ("domain", "study_area", "land_study_area"):
            if not np.array_equal(masks[f"max_depth_{scope}_m.tif"], masks[f"final_depth_{scope}_m.tif"]):
                raise ValueError()
        land = masks["max_depth_land_study_area_m.tif"]
        study = masks["max_depth_study_area_m.tif"]
        domain = masks["max_depth_domain_m.tif"]
        if (land & ~study).any() or (study & ~domain).any() or not np.array_equal(domain, grid["mask"] > 0):
            raise ValueError()
        try:
            native = _native_diagnostics(directory / "sfincs_map.nc", grid,
                                         arrays["max_depth_land_study_area_m.tif"],
                                         arrays["final_depth_land_study_area_m.tif"], land, input_path)
        except (OSError, ValueError, KeyError) as error:
            note = f"{directory.name}原生结果/强迫接收核对不能完成：{error}"
            _mark_validation_failure(result_root, [note], [directory])
            raise RuntimeError() from error
        maximum = products["max_depth_land_study_area_m.tif"]
        final = products["final_depth_land_study_area_m.tif"]
        summary = {
            "city": city, "scenario": directory.name, "return_period_year": period,
            "summary_scope": "Exported depth/area statistics and native saved-frame diagnostics",
            **native, **_rainfall_info(config, city, period), "products": products,
            "study_land_area_km2": maximum["valid_area_km2"],
            "land_area_at_least_0p3_km2": maximum["area_at_least_depth_km2"]["0.3"],
            "maximum_land_depth_m": maximum["maximum_depth_m"],
            "mean_maximum_land_depth_m": maximum["mean_depth_m_including_zeros"],
            "final_land_area_at_least_0p3_km2": final["area_at_least_depth_km2"]["0.3"],
            "final_maximum_land_depth_m": final["maximum_depth_m"],
            "main_geotiff": maximum["path"], "final_land_geotiff": final["path"],
            "rain_duration_hours": float(config["rain_duration_min"]) / 60,
            "rainfall_label": str(config.get("rainfall_label", "均匀降雨")),
            "drainage_mm_h": float(config["drainage_mm_h"]),
            "drainage_codes": list(config["drainage_codes"]),
            "terrain_source": str(Path(config["sources"]["dem"]).resolve()),
            "input_parameter_source_note": "Rainfall duration/profile, drainage and terrain paths come from this input configuration; time, grid and elevation are checked against INP/NetCDF.",
            "physical_accuracy_observation_validated": False,
            "mass_balance_independently_verified": False,
            "complete_event_peak_convergence_verified": False,
            "spatial_smoothing": False, "isolated_patch_filtering": False,
            "result_quality_status": ("failed_forcing_reception"
                                      if native["forcing_reception_validation_status"] == "failed"
                                      else "input_reception_verified"),
            "expected_event_net_runoff_mean_mm": native["forcing_reception"]["expected_event_mean_mm"],
            "actual_event_net_runoff_mean_mm": native["forcing_reception"]["actual_event_mean_mm"],
            "expected_event_net_runoff_volume_m3": native["forcing_reception"]["expected_event_volume_m3"],
            "actual_event_net_runoff_volume_m3": native["forcing_reception"]["actual_event_volume_m3"],
            "actual_to_expected_net_runoff_ratio": native["forcing_reception"]["actual_to_expected_event_ratio"],
        }
        summaries.append(summary)
        directories.append(directory)
    period_count = len(selected)
    period_label = {1: "single", 2: "two", 3: "three", 4: "four", 5: "five"}.get(
        period_count, str(period_count)
    )
    csv_path = result_root / f"{period_label}_period_result_summary.csv"
    json_name = "single_period_result_metadata.json" if period_count == 1 else f"{period_label}_period_result_summary.json"
    json_path = result_root / json_name
    city_name = str(config.get("city_name") or CITY_NAMES.get(city, city))
    written, reused = [], []
    for directory, summary in zip(directories, summaries):
        path = directory / "result_summary.json"
        (written if _write_json(path, summary, overwrite) else reused).append(str(path))
    if csv_path.is_file() and not overwrite:
        reused.append(str(csv_path))
    else:
        columns = {
            "城市": "city", "重现期_年": "return_period_year", "气象总雨量_mm": "gross_rainfall_mm",
            "模拟开始": "simulation_start", "模拟结束": "simulation_stop",
            "模拟时长_h": "simulation_duration_hours", "研究区陆地面积_km2": "study_land_area_km2",
            "陆地最大水深_m": "maximum_land_depth_m", "陆地最大水深均值_m": "mean_maximum_land_depth_m",
            "陆地最大水深_大于等于0.3m面积_km2": "land_area_at_least_0p3_km2",
            "结束陆地最大水深_m": "final_maximum_land_depth_m",
            "结束陆地水深_大于等于0.3m面积_km2": "final_land_area_at_least_0p3_km2",
            "陆地曾出现低于负0.01m水深格数": "land_negative_cells_below_minus_0p01",
            "末段上涨超过1mm陆地格数": "late_rising_land_cells_above_1mm",
            "末段仍上涨且接近峰值格数": "late_rising_near_peak_cells",
            "强迫接收核对": "forcing_reception_validation_status",
            "计算域内部预计净产流_mm": "expected_event_net_runoff_mean_mm",
            "计算域内部实际累计接收净产流_mm": "actual_event_net_runoff_mean_mm",
            "计算域内部预计净产流_m3": "expected_event_net_runoff_volume_m3",
            "计算域内部实际累计接收净产流_m3": "actual_event_net_runoff_volume_m3",
            "实际接收与预计产流比例": "actual_to_expected_net_runoff_ratio",
            "主要最大水深TIF": "main_geotiff", "主要结束水深TIF": "final_land_geotiff",
        }
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns))
            writer.writeheader()
            writer.writerows({label: summary[key] for label, key in columns.items()} for summary in summaries)
        written.append(str(csv_path))
    overview = {
        "status": ("failed_forcing_reception"
                   if any(s["forcing_reception_validation_status"] == "failed" for s in summaries)
                   else "completed_basic_summary"), "city": city, "city_name": city_name,
        "updated_utc": datetime.now(timezone.utc).isoformat(), "return_periods": selected,
        "model_root": str(model_root), "result_root": str(result_root), "summaries": summaries,
        "physical_accuracy_observation_validated": False, "mass_balance_independently_verified": False,
        "complete_event_peak_convergence_verified": False,
    }
    (written if _write_json(json_path, overview, overwrite) else reused).append(str(json_path))
    failed = [summary for summary in summaries if summary["forcing_reception_validation_status"] == "failed"]
    if failed:
        notes = [f"{summary['scenario']}：{note}" for summary in failed
                 for note in summary["forcing_reception_failure_notes"]]
        failed_directories = [result_root / summary["scenario"] for summary in failed]
        _mark_validation_failure(result_root, notes, failed_directories)
        raise RuntimeError()
    return {**overview, "written_files": written, "reused_files": reused,
            "summary_csv": str(csv_path), "summary_json": str(json_path)}
