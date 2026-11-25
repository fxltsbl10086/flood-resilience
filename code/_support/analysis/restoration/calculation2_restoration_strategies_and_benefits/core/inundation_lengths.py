"""Geometry helpers for the intersection of road segments and flooded cells."""
import numpy as np
import pandas as pd
import shapely


def _require(condition, message):
    if not condition:
        raise ValueError("")


def _strict_bool(values, name):
    values = pd.Series(values)
    if pd.api.types.is_bool_dtype(values):
        _require(not values.isna().any(), "")
        return values.to_numpy(bool)
    text = values.astype(str).str.lower()
    _require(text.isin(["true", "false", "0", "1"]).all(), "")
    return text.isin(["true", "1"]).to_numpy(bool)


def _union_intersection_length(road, cells):
    """Union wet cells first; count roads on shared boundaries once, with zero length for point contacts."""
    if not len(cells):
        return 0.0
    wet_area = shapely.union_all(cells)
    return float(shapely.length(shapely.intersection(road, wet_area)))


def _cells(cell_ids, width, transform):
    ids = np.asarray(cell_ids, dtype=np.int64)
    row, col = ids // width, ids % width
    return shapely.box(transform.c + col * transform.a,
                       transform.f + (row + 1) * transform.e,
                       transform.c + (col + 1) * transform.a,
                       transform.f + row * transform.e)
