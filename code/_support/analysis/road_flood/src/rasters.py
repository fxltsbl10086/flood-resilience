"Read scenario depth rasters on one computational grid with separate validity masks."

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS


@dataclass(frozen=True)
class DepthStack:
    values: np.ndarray
    valid: np.ndarray
    transform: Affine
    crs: CRS
    periods: tuple[int, ...]
    scenario_ids: tuple[str, ...]
    metadata: tuple[dict, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape[1:]


def read_depths(paths: tuple[Path, ...], periods: tuple[int, ...], scenario_ids: tuple[str, ...]) -> DepthStack:
    if not paths or len(paths) != len(periods) or len(paths) != len(scenario_ids):
        raise ValueError()
    values, masks, metadata = [], [], []
    reference = None
    for path in paths:
        with rasterio.open(path) as src:
            if src.crs is None or src.count != 1:
                raise ValueError()
            current = (src.shape, tuple(src.transform), src.crs)
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError()
            if src.transform.b != 0 or src.transform.d != 0 or src.transform.a <= 0 or src.transform.e >= 0:
                raise ValueError()
            band = src.read(1, masked=True)
            data = np.asarray(band.data, dtype=np.float32)
            valid = ~np.ma.getmaskarray(band) & np.isfinite(data)
            values.append(data)
            masks.append(valid)
            metadata.append({
                "path": str(path), "height": src.height, "width": src.width,
                "crs": str(src.crs), "transform": list(src.transform),
                "resolution_m": list(src.res), "nodata": src.nodata,
                "valid_cells": int(valid.sum()), "dtype": src.dtypes[0],
            })
    crs = CRS(reference[2])
    if not crs.is_projected or any(abs(axis.unit_conversion_factor - 1) > 1e-9 for axis in crs.axis_info[:2]):
        raise ValueError()
    return DepthStack(np.stack(values), np.stack(masks), Affine(*reference[1][:6]),
                      crs, periods, scenario_ids, tuple(metadata))
