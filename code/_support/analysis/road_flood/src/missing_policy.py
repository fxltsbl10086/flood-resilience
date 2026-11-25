"Classify missing-coverage sources and apply explicit binary road-state rules."

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .matching import MatchResult, SpatialMapping
from .rasters import DepthStack


REASONS = {
    "water_mask": ("A", "陆地成果屏蔽水域，含水域的研究区成果有效"),
    "study_scope_mask": ("B", "研究区成果屏蔽边缘或出流单元，计算域成果有效"),
    "missing_domain": ("C", "计算域成果也没有有效水深"),
    "unclassified_nodata": ("D", "辅助成果缺失、掩膜不一致或不能细分的NoData"),
    "outside_raster": ("E", "研究区内道路有部分超出栅格范围"),
    "outside_study": ("F", "没有研究区内有效分析长度"),
}


@dataclass
class MissingReview:
    details: pd.DataFrame
    summary: pd.DataFrame
    auxiliary_rasters: tuple[dict, ...]


def validate_policy(raw: dict) -> dict[str, int | str]:
    extra = set(raw) - set(REASONS)
    if extra:
        raise ValueError()
    policy = {name: raw.get(name, "pending") for name in REASONS}
    for name, value in policy.items():
        allowed = (type(value) is int and value in (0, 1)) or value == "pending"
        allowed |= name == "outside_study" and value == "exclude"
        if not allowed:
            raise ValueError()
    return policy


def _auxiliary_mask(path: Path, stack: DepthStack) -> np.ndarray | None:
    if not path.is_file():
        return None
    with rasterio.open(path) as source:
        if (source.count != 1 or source.shape != stack.shape
                or tuple(source.transform) != tuple(stack.transform)
                or source.crs is None or source.crs != stack.crs):
            raise ValueError()
        array = source.read(1, masked=True)
        return (~np.ma.getmaskarray(array) & np.isfinite(array.data)).ravel()


def review_missing(result: MatchResult, spatial_mapping: SpatialMapping,
                   stack: DepthStack, tolerance_m: float) -> MissingReview:
    status = result.status
    n = len(status)
    ri = spatial_mapping.pairs.road_index.to_numpy(dtype=np.int64)
    ci = spatial_mapping.pairs.cell_id.to_numpy(dtype=np.int64)
    matched_counts = np.bincount(ri, minlength=n)
    inside = status.analysis_length_m.to_numpy() > tolerance_m
    outside = status.outside_grid_length_m.to_numpy() > tolerance_m
    if (inside & ~outside & (matched_counts == 0)).any():
        raise RuntimeError()
    tables, summaries, auxiliary = [], [], []
    for k, (period, scenario) in enumerate(zip(stack.periods, stack.scenario_ids)):
        code = f"T{period:03d}"
        path = Path(stack.metadata[k]["path"])
        missing = ~stack.valid[k].ravel()
        study = domain = None
        record = {"scenario_id": scenario, "study_path": None, "domain_path": None}
        if path.name.endswith("_depth_land_study_area_m.tif"):
            study_path = path.with_name(path.name.replace("land_study_area", "study_area"))
            domain_path = path.with_name(path.name.replace("land_study_area", "domain"))
            study = _auxiliary_mask(study_path, stack)
            domain = _auxiliary_mask(domain_path, stack)
            record.update(study_path=str(study_path) if study is not None else None,
                          domain_path=str(domain_path) if domain is not None else None)
        water = missing & study if study is not None else np.zeros_like(missing)
        scope = (missing & ~study & domain if study is not None and domain is not None
                 else np.zeros_like(missing))
        absent = (missing & ~domain if domain is not None else np.zeros_like(missing))
        inconsistent = (study & ~domain if study is not None and domain is not None
                        else np.zeros_like(missing))
        water &= ~inconsistent
        absent &= ~inconsistent
        residual = missing & ~(water | scope | absent)
        record["inconsistent_parent_mask_cells"] = int(inconsistent.sum())
        auxiliary.append(record)
        masks = {"water_mask": water, "study_scope_mask": scope,
                 "missing_domain": absent, "unclassified_nodata": residual}
        data_hit = status[f"flood_{code}"].to_numpy() == 1
        table = pd.DataFrame({
            "road_id": status.road_id, "scenario_id": scenario,
            "return_period_year": period, "observed_flooded": data_hit,
            "analysis_length_m": status.analysis_length_m,
            "outside_grid_length_m": status.outside_grid_length_m,
            "matched_cell_count": matched_counts,
            "missing_cell_count": status[f"unknown_cells_{code}"],
        })
        for name, mask in masks.items():
            counts = np.bincount(ri[mask[ci]], minlength=n)
            table[name] = counts > 0
            table[f"{name}_cell_count"] = counts
        table["outside_raster"] = outside
        table["outside_study"] = ~inside
        reason_matrix = table[list(REASONS)].to_numpy(dtype=bool)
        affected = reason_matrix.any(axis=1)
        # Only non-hit roads with missing coverage require a configured missing-data rule.
        table["needs_policy"] = affected & ~data_hit
        table["reason_codes"] = [";".join(REASONS[name][0] for name, active in zip(REASONS, row) if active)
                                  for row in reason_matrix]
        table["nodata_extent"] = np.where(
            table.missing_cell_count == 0, "none",
            np.where(table.missing_cell_count == matched_counts, "all_matched_cells", "partial_matched_cells"))
        tables.append(table)
        for name, (letter, description) in REASONS.items():
            reason = table[name].to_numpy()
            needs = reason & table.needs_policy.to_numpy()
            summaries.append({
                "scenario_id": scenario, "return_period_year": period,
                "reason_code": letter, "reason_key": name, "description": description,
                "affected_road_count": int(reason.sum()),
                "needs_policy_road_count": int(needs.sum()),
                "already_observed_flooded_road_count": int((reason & data_hit).sum()),
                "needs_policy_all_nodata_road_count": int((needs & (table.nodata_extent == "all_matched_cells")).sum()),
                "needs_policy_partial_nodata_road_count": int((needs & (table.nodata_extent == "partial_matched_cells")).sum()),
            })
    return MissingReview(pd.concat(tables, ignore_index=True), pd.DataFrame(summaries), tuple(auxiliary))


def apply_missing_policy(result: MatchResult, review: MissingReview,
                         configured_policy: dict) -> MatchResult:
    policy = validate_policy(configured_policy)
    first_period = result.summary.return_period_year.iloc[0]
    first_table = review.details[review.details.return_period_year == first_period].reset_index(drop=True)
    if not np.array_equal(first_table.road_id.to_numpy(), result.status.road_id.to_numpy()):
        raise RuntimeError()
    excluded = (first_table.outside_study.to_numpy() if policy["outside_study"] == "exclude"
                else np.zeros(len(result.status), dtype=bool))
    retained_ids = result.status.loc[~excluded, "road_id"]
    retained_review = review.details[review.details.road_id.isin(retained_ids)]
    pending_rows = []
    for name, (letter, _) in REASONS.items():
        if policy[name] == "pending":
            counts = retained_review[retained_review.needs_policy & retained_review[name]].groupby("return_period_year").size()
            if len(counts):
                pending_rows.append(f"{letter}/{name}（单情景最多{int(counts.max())}条）")
    if pending_rows:
        raise ValueError()
    status = result.status.loc[~excluded].drop(columns=[c for c in result.status if c.startswith(("unknown_T", "unknown_cells_T"))]).reset_index(drop=True)
    if status.empty:
        raise ValueError()
    summary = result.summary.drop(columns=["unknown_status_road_count", "roads_with_unknown_coverage"]).copy()
    for period in summary.return_period_year:
        code = f"T{period:03d}"
        table = retained_review[retained_review.return_period_year == period].reset_index(drop=True)
        if not np.array_equal(table.road_id.to_numpy(), status.road_id.to_numpy()):
            raise RuntimeError()
        observed = table.observed_flooded.to_numpy()
        resolved_flood = np.zeros(len(status), dtype=bool)
        for name, value in policy.items():
            if value == 1:
                resolved_flood |= table[name].to_numpy()
        final = observed | resolved_flood
        status[f"flood_{code}"] = final.astype(np.int8)
        status[f"missing_coverage_{code}"] = table[list(REASONS)].any(axis=1).to_numpy()
        status[f"missing_cells_{code}"] = table.missing_cell_count.to_numpy(dtype=np.int32)
        status[f"policy_applied_{code}"] = table.needs_policy.to_numpy()
        status[f"policy_reasons_{code}"] = table.reason_codes.to_numpy()
        row = summary.return_period_year == period
        summary.loc[row, "input_road_count"] = len(result.status)
        summary.loc[row, "excluded_outside_study_road_count"] = int(excluded.sum())
        summary.loc[row, "road_count"] = len(status)
        summary.loc[row, "known_below_threshold_road_count"] = int((result.status.loc[~excluded, f"flood_{code}"] == 0).sum())
        summary.loc[row, "flooded_road_count"] = int(final.sum())
        summary.loc[row, "not_flooded_road_count"] = int((~final).sum())
        summary.loc[row, "observed_flooded_road_count"] = int(observed.sum())
        summary.loc[row, "policy_flooded_road_count"] = int((final & ~observed).sum())
        summary.loc[row, "policy_resolved_road_count"] = int(table.needs_policy.sum())
        summary.loc[row, "roads_with_missing_coverage"] = int(status[f"missing_coverage_{code}"].sum())
        summary.loc[row, "flooded_roads_with_vertical_uncertainty"] = int((final & status.flood_vertical_uncertain.to_numpy()).sum())
        summary.loc[row, "flooded_facility_analysis_length_m"] = float(status.analysis_length_m.to_numpy()[final].sum())
    count_columns = [column for column in summary if column.endswith("_count") or column == "roads_with_missing_coverage"]
    summary[count_columns] = summary[count_columns].astype(np.int64)
    qa = {**result.qa, "road_count": len(status), "input_road_count": len(result.status),
          "excluded_outside_study_road_count": int(excluded.sum()),
          "vertical_uncertain_road_count": int(status.flood_vertical_uncertain.sum()),
          "status_values": [0, 1], "missing_policy": policy,
          "missing_reason_descriptions": REASONS,
          "multiple_reason_rule": "任一适用原因配置为1则最终状态为1；已知超阈值命中始终为1",
          "scope_rule": "outside_study=exclude时没有区内有效长度的道路不赋0/1、不导出道路状态、不纳入统计",
          "needs_policy_road_scenario_count": int(retained_review.needs_policy.sum())}
    return MatchResult(status, result.pairs, summary, qa)
