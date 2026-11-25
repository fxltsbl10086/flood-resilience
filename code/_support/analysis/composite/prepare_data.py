"""Join the original population counts, established EFAR and 500 m geometry."""
import hashlib
import json
import geopandas as gpd
import numpy as np
import pandas as pd
from config import CITIES, PERIODS, DIMENSIONS, DIMENSION_KEYS, input_paths
from _support.analysis_paths import city_selection


def signature(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_city(city):
    paths = input_paths(city)
    metadata = json.loads(paths["service_metadata"].read_text(encoding="utf-8"))
    if not (metadata.get("metric") == "inverse" and metadata.get("local") == "include"
            and metadata.get("same_grid_distance_m") == 250 and metadata.get("periods") == PERIODS):
        raise ValueError()
    grid = gpd.read_file(paths["grid"])
    boundary = gpd.read_file(paths["boundary"])
    population = pd.read_csv(paths["population"], encoding="utf-8-sig", dtype={"grid_id": str})
    risk = pd.read_csv(paths["services"], encoding="utf-8-sig", dtype={"grid_id": str})
    if not grid.grid_id.is_unique or not population.grid_id.is_unique:
        raise ValueError()
    if set(grid.grid_id) != set(population.grid_id):
        raise ValueError()
    if grid.crs is None or boundary.crs is None or not grid.crs.is_projected:
        raise ValueError()
    if not grid.geometry.is_valid.all() or not boundary.geometry.is_valid.all():
        raise ValueError()
    if not all(axis.unit_name.lower() in {"metre", "meter"} for axis in grid.crs.axis_info):
        raise ValueError()
    bounds = grid.geometry.bounds
    if not (np.allclose(bounds.maxx-bounds.minx, 500, atol=1e-6, rtol=0)
            and np.allclose(bounds.maxy-bounds.miny, 500, atol=1e-6, rtol=0)):
        raise ValueError()
    pop = population.set_index("grid_id").loc[grid.grid_id].reset_index()
    centers = grid.geometry.centroid
    for coord, expected in [("centroid_x", centers.x), ("centroid_y", centers.y)]:
        if not np.allclose(pop[coord], expected, atol=1e-5, rtol=0):
            raise ValueError()
    if not pop.city.eq(city).all() or not risk.city.eq(city).all():
        raise ValueError()
    if set(risk["评估维度"]) != set(DIMENSIONS) or risk.duplicated(["grid_id", "评估维度"]).any():
        raise ValueError()
    if not (risk["指标"].eq("inverse").all() and risk["同网格处理"].eq("include").all()
            and risk["同网格距离m"].eq(250).all()):
        raise ValueError()
    data = grid[["grid_id", "row", "col", "geometry"]].copy()
    data["city"] = city
    data["centroid_x"], data["centroid_y"] = centers.x.to_numpy(), centers.y.to_numpy()
    for field in ["population", "population_status", "population_data_coverage_ratio"]:
        data[field] = pop[field].to_numpy()
    known = pop.population.notna().to_numpy()
    if not np.isfinite(pop.population.to_numpy(float)[known]).all() or (pop.population[known] < 0).any():
        raise ValueError()
    common = known & pop.population.gt(0).to_numpy()
    for dim, key in zip(DIMENSIONS, DIMENSION_KEYS):
        part = risk.loc[risk["评估维度"] == dim].set_index("grid_id")
        if set(part.index) != set(grid.grid_id):
            raise ValueError()
        part = part.loc[grid.grid_id]
        for coord in ["centroid_x", "centroid_y"]:
            if not np.allclose(part[coord], data[coord], atol=1e-5, rtol=0):
                raise ValueError()
        if not part["状态"].isin(["有效", "基准无可达资源"]).all():
            raise ValueError()
        valid = part["状态"].eq("有效").to_numpy()
        values = part[[f"风险_T{t:03d}" for t in PERIODS]].to_numpy(float)
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError()
        if np.any(values[valid] < 0) or np.any(values[valid] > 1):
            raise ValueError()
        data[f"baseline_valid_{key}"] = valid.astype(int)
        common &= valid
        for j, t in enumerate(PERIODS):
            data[f"A_{key}_T{t}"] = values[:, j]
    for t in PERIODS:
        exposed = pop[f"population_exposed_T{t}"].to_numpy(float)
        if not np.isfinite(exposed[known]).all() or not np.isnan(exposed[~known]).all():
            raise ValueError()
        p = pop.population.to_numpy(float)
        if np.any(exposed[known] < -1e-7) or np.any(exposed[known] > p[known] + 1e-7):
            raise ValueError()
        data[f"exposed_people_T{t}"] = exposed
    data["analysis_included"] = common.astype(int)
    data["exclusion_reason"] = np.select(
        [~known, known & pop.population.eq(0).to_numpy(), ~common],
        ["population_unknown", "zero_known_population", "one_or_more_baseline_services_unavailable"], default="included")
    if common.sum() < 30:
        raise ValueError()
    # Verify queen adjacency through the actual geometry centres, not row labels alone.
    for field, center_values in [("col", data.centroid_x), ("row", data.centroid_y)]:
        grouped = pd.DataFrame({"index": data[field], "coordinate": center_values}).groupby("index").coordinate
        if np.any((grouped.max()-grouped.min()).to_numpy() > 1e-5):
            raise ValueError()
        coords = grouped.mean().sort_index()
        diffs = np.diff(coords.to_numpy()) / np.diff(coords.index.to_numpy())
        if not np.allclose(np.abs(diffs), 500, atol=1e-5, rtol=0):
            raise ValueError()
    boundary = boundary.to_crs(data.crs).geometry.union_all()
    if boundary.difference(data.geometry.union_all()).area > 1:
        raise ValueError()
    source = {key: signature(path) for key, path in paths.items()}
    return data, boundary, source


def prepare_all(cities=CITIES):
    cities = city_selection(cities)
    datasets, sources = {}, {}
    ref = {"people": -1.0}
    for city in cities:
        data, boundary, source = read_city(city)
        datasets[city] = (data, boundary)
        sources[city] = source
        for t in PERIODS:
            column = data[f"exposed_people_T{t}"]
            index = column.idxmax()
            value = float(column.loc[index])
            if value > ref["people"]:
                ref = {"people": value, "city": city, "period": t, "grid_id": data.loc[index, "grid_id"]}
        print(f"{city}: {int(data.analysis_included.sum()):,}/{len(data):,} populated cells with three valid service baselines", flush=True)
    if ref["people"] <= 0:
        raise ValueError()
    for city, (data, _) in datasets.items():
        included = data.analysis_included.eq(1).to_numpy()
        for t in PERIODS:
            data[f"E_T{t}"] = data[f"exposed_people_T{t}"] / ref["people"]
            parts = data[[f"A_{key}_T{t}" for key in DIMENSION_KEYS]].to_numpy(float)
            risk = data[f"E_T{t}"].to_numpy(float) / 2 + parts.sum(axis=1) / 6
            risk[~included] = np.nan
            if not np.isfinite(risk[included]).all() or np.any(risk[included] < 0) or np.any(risk[included] > 1):
                raise ValueError()
            data[f"R_T{t}"] = risk
            affected = (parts > 1e-10).sum(axis=1) + (data[f"exposed_people_T{t}"].to_numpy(float) > 0)
            data[f"affected_dimension_count_T{t}"] = np.where(included, affected, np.nan)
    print(f"Fixed exposure reference: {ref['people']:.6f} people ({ref['city']}, T{ref['period']}, {ref['grid_id']})", flush=True)
    return datasets, sources, ref
