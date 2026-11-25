"""Generate fixed road priorities and cumulative whole-road restoration stages from associated risk burden.

Operate only on existing maximum-loss contacts and their obstructed roads, without shortest paths or benefit statistics.
Road scores are associated burden, not independent single-road benefits; one contact's full burden can be associated with several roads."""
import json
import math

import numpy as np
import pandas as pd


DIMENSIONS = ("rescue", "medical", "shelter")
ROAD_METRIC_COLUMNS = [
    "road_rank", "associated_risk_burden", "best_grid_rank",
    "associated_grid_count", "associated_contact_count",
    *[f"{dimension}_associated_risk_burden" for dimension in DIMENSIONS],
]
CONTACT_COLUMNS = [
    "grid_id", "dimension", "road_id", "road_position", "effective_population",
    "initial_EFARB", "grid_rank", "selected_EFAR_contribution",
    "selected_loss_share", "contact_risk_burden",
]
SCENARIO_COLUMNS = [
    "scenario_index", "restored_road_count", "candidate_road_count",
    "candidate_road_fraction", "actual_road_coverage_pct",
    "requested_road_coverage_pcts", "from_absolute_steps",
]


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError("")


def _bool_values(series, label):
    """Parse Boolean flags explicitly so that the string 'False' does not become True."""
    mapping = {True: True, False: False, "True": True, "False": False,
               "true": True, "false": False, "1": True, "0": False}
    result = series.map(mapping)
    if result.isna().any():
        raise ValueError("")
    return result.astype(bool)


def _identifier(frame, column, label):
    if frame[column].isna().any():
        raise ValueError("")
    frame[column] = frame[column].astype(str)
    if frame[column].eq("").any():
        raise ValueError("")


def build_road_priority(evaluation, selection_table, link_roads, inventory):
    """Return fixed road rankings, deduplicated road-contact burdens and method/check metadata.

Use only ranking_eligible=True cells with valid positive population and positive pre-restoration EFARB.
b_ik = p_i / 3 * selected_EFAR_contribution; count each contact once per physical road_id.
S_e = fsum(b_ik), without dividing by the number of roads on the path.
Break exact score ties by the best original grid rank, then lexicographic road_id."""
    _require(evaluation, "", "评价格网")
    _require(selection_table, "", "设施联系")
    _require(link_roads, "", "联系道路")
    _require(inventory, "", "道路清单")
    ev, selected, links, roads = (frame.copy() for frame in
                                 (evaluation, selection_table, link_roads, inventory))
    for frame, columns, label in (
        (ev, ["grid_id"], "评价格网"), (selected, ["grid_id", "dimension"], "设施联系"),
        (links, ["grid_id", "dimension", "road_id"], "联系道路"),
        (roads, ["road_id"], "道路清单"),
    ):
        for column in columns:
            _identifier(frame, column, label)
    if ev.grid_id.duplicated().any():
        raise ValueError("")
    if roads.road_id.duplicated().any() or roads.road_position.duplicated().any():
        raise ValueError("")
    if roads.road_position.isna().any():
        raise ValueError("")
    if selected.duplicated(["grid_id", "dimension"]).any():
        raise ValueError("")
    if set(selected.dimension) - set(DIMENSIONS) or set(links.dimension) - set(DIMENSIONS):
        raise ValueError("")
    if set(selected.grid_id) - set(ev.grid_id) or set(links.grid_id) - set(ev.grid_id):
        raise ValueError("")

    valid = _bool_values(ev.efarb_valid, "efarb_valid")
    eligible = _bool_values(ev.ranking_eligible, "ranking_eligible")
    expected = valid & ev.effective_population.gt(0) & ev.initial_EFARB.gt(0)
    if not eligible.equals(expected):
        raise ValueError("")
    if not np.isfinite(ev.loc[eligible, ["effective_population", "initial_EFARB", "rank"]]).all().all():
        raise ValueError("")
    ranking = ev.loc[eligible, "rank"].to_numpy()
    if not np.array_equal(np.sort(ranking), np.arange(1, len(ranking) + 1)):
        raise ValueError("")
    if not ev.loc[~eligible, "rank"].eq(0).all():
        raise ValueError("")
    eligible_ids = set(ev.loc[eligible, "grid_id"])
    eligible_selection = selected.loc[selected.grid_id.isin(eligible_ids)].copy()
    actual_pairs = set(zip(eligible_selection.grid_id, eligible_selection.dimension))
    expected_pairs = {(grid_id, dimension) for grid_id in eligible_ids for dimension in DIMENSIONS}
    if actual_pairs != expected_pairs:
        raise ValueError("")
    if "ranking_eligible" in selected:
        selection_eligible = _bool_values(selected.ranking_eligible, "联系ranking_eligible")
        if not np.array_equal(selection_eligible, selected.grid_id.isin(eligible_ids)):
            raise ValueError("")
    if "rank" in selected:
        if not np.array_equal(selected["rank"], selected.grid_id.map(ev.set_index("grid_id")["rank"])):
            raise ValueError("")
    positive = eligible_selection.selection_status.eq("selected")
    contribution = eligible_selection.selected_EFAR_contribution
    if not np.isfinite(contribution).all() or (contribution < 0).any():
        raise ValueError("")
    if (positive & contribution.le(0)).any() or (~positive & contribution.ne(0)).any():
        raise ValueError("")
    shares = eligible_selection.loc[positive, "selected_loss_share"]
    if not np.isfinite(shares).all() or shares.le(0).any() or shares.gt(1 + 1e-12).any():
        raise ValueError("")

    if not links.empty:
        mapped_positions = links.road_id.map(roads.set_index("road_id").road_position)
        if mapped_positions.isna().any() or not np.array_equal(mapped_positions, links.road_position):
            raise ValueError("")
    original_link_count = len(links)
    # Count a road once per contact even when directional edges or path fragments repeat it.
    links = links.drop_duplicates(["grid_id", "dimension", "road_id"]).copy()
    duplicate_link_count = original_link_count - len(links)
    ineligible_link_count = int((~links.grid_id.isin(eligible_ids)).sum())
    links = links.loc[links.grid_id.isin(eligible_ids)].copy()
    if not set(zip(links.grid_id, links.dimension)).issubset(actual_pairs):
        raise ValueError("")
    payload = eligible_selection[["grid_id", "dimension", "selection_status",
                                  "selected_EFAR_contribution", "selected_loss_share"]]
    details = links[["grid_id", "dimension", "road_id", "road_position"]].merge(
        payload, on=["grid_id", "dimension"], how="left", validate="many_to_one")
    if not details.selection_status.eq("selected").all():
        raise ValueError("")
    if "selected_facility_grid_id" in links and "selected_facility_grid_id" in eligible_selection:
        actual_targets = links.merge(eligible_selection[["grid_id", "dimension", "selected_facility_grid_id"]],
                                     on=["grid_id", "dimension"], suffixes=("_road", "_selection"),
                                     validate="many_to_one")
        if not actual_targets.selected_facility_grid_id_road.eq(
                actual_targets.selected_facility_grid_id_selection).all():
            raise ValueError("")
    if "project_road_count" in eligible_selection:
        counts = details.groupby(["grid_id", "dimension"]).size()
        actual = [int(counts.get((row.grid_id, row.dimension), 0))
                  for row in eligible_selection.itertuples(index=False)]
        if not np.array_equal(actual, eligible_selection.project_road_count):
            raise ValueError("")
    details = details.merge(ev[["grid_id", "effective_population", "initial_EFARB", "rank"]],
                            on="grid_id", how="left", validate="many_to_one").rename(columns={"rank": "grid_rank"})
    details["contact_risk_burden"] = (details.effective_population / 3.0 *
                                      details.selected_EFAR_contribution)
    if not np.isfinite(details.contact_risk_burden).all() or details.contact_risk_burden.le(0).any():
        raise ValueError("")
    details = details[CONTACT_COLUMNS].sort_values(
        ["grid_id", "dimension", "road_id"], kind="stable").reset_index(drop=True)
    candidate_ids = set(details.road_id)
    candidate_inventory = roads.loc[roads.road_id.isin(candidate_ids)].copy()
    if not candidate_inventory.flood_scenario.eq(1).all():
        raise ValueError("")
    records = []
    for road_id, group in details.groupby("road_id", sort=True):
        record = {
            "road_id": road_id,
            "associated_risk_burden": math.fsum(group.contact_risk_burden),
            "best_grid_rank": int(group.grid_rank.min()),
            "associated_grid_count": int(group.grid_id.nunique()),
            "associated_contact_count": len(group),
        }
        for dimension in DIMENSIONS:
            record[f"{dimension}_associated_risk_burden"] = math.fsum(
                group.loc[group.dimension.eq(dimension), "contact_risk_burden"])
        records.append(record)
    score_columns = ["road_id", *[name for name in ROAD_METRIC_COLUMNS if name != "road_rank"]]
    scores = pd.DataFrame(records, columns=score_columns)
    priority = candidate_inventory.merge(scores, on="road_id", how="left", validate="one_to_one")
    priority = priority.sort_values(["associated_risk_burden", "best_grid_rank", "road_id"],
                                    ascending=[False, True, True], kind="stable").reset_index(drop=True)
    priority.insert(0, "road_rank", np.arange(1, len(priority) + 1, dtype=int))
    selected_unique = details.drop_duplicates(["grid_id", "dimension"])
    grid_with_projects = set(details.grid_id)
    metadata = {
        "score_name": "道路关联风险负担",
        "contact_burden_formula": "b_ik=effective_population_i/3*selected_EFAR_contribution_ik",
        "road_score_formula": "S_e=sum_ik(b_ik*1[e in selected-contact blocked physical roads])",
        "road_score_interpretation": 'Risk burden associated with fixed contacts, not independent single-road reduction; repeated associations across roads cannot be summed as total burden.',
        "score_accumulation": "math.fsum; each grid-dimension-road_id counted once; no rounding or length normalization",
        "sort_order": ["associated_risk_burden descending (raw precision)",
                       "best_grid_rank ascending", "road_id lexicographic ascending"],
        "scoring_grid_count": len(eligible_ids),
        "scoring_contact_count": int(positive.sum()),
        "contact_with_project_count": len(selected_unique),
        "grid_with_project_count": len(grid_with_projects),
        "grid_without_project_count": len(eligible_ids - grid_with_projects),
        "candidate_road_count": len(priority),
        "road_contact_row_count": len(details),
        "duplicate_contact_road_rows_removed": duplicate_link_count,
        "ineligible_contact_road_rows_ignored": ineligible_link_count,
        "distinct_selected_contact_risk_burden": math.fsum(
            (eligible_selection.loc[positive, "grid_id"].map(ev.set_index("grid_id").effective_population)
             / 3.0 * eligible_selection.loc[positive, "selected_EFAR_contribution"]).tolist()),
        "sum_road_association_scores_nonadditive": math.fsum(priority.associated_risk_burden),
        "candidate_road_blockage_column": "flood_scenario",
        "checks": "PASS",
    }
    return priority, details, metadata


def build_road_scenarios(M):
    """Cumulative restoration counts: 0..10 and floors of 5%..100% of candidate roads, deduplicated."""
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)) or M < 0:
        raise ValueError("")
    M = int(M)
    absolute_steps = set(range(min(10, M) + 1))
    percentage_steps = {}
    for pct in range(5, 101, 5):
        percentage_steps.setdefault(pct * M // 100, []).append(pct)
    quantities = sorted(absolute_steps | set(percentage_steps))
    return pd.DataFrame([
        {
            "scenario_index": index,
            "restored_road_count": number,
            "candidate_road_count": M,
            "candidate_road_fraction": number / M if M else float("nan"),
            "actual_road_coverage_pct": 100 * number / M if M else float("nan"),
            "requested_road_coverage_pcts": json.dumps(percentage_steps.get(number, []), separators=(",", ":")),
            "from_absolute_steps": number in absolute_steps,
        }
        for index, number in enumerate(quantities)
    ], columns=SCENARIO_COLUMNS)
