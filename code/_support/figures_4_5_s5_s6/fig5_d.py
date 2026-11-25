"""Drawing functions adapted from signed original sources; no experiment code runs."""
import math
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, MaxNLocator, StrMethodFormatter
from matplotlib.transforms import Bbox
def degree_label(value, axis):
    degrees = int(abs(value))
    minutes = int(round((abs(value) - degrees) * 60))
    if minutes == 60:
        degrees += 1
        minutes = 0
    suffix = ("E" if value >= 0 else "W") if axis == "lon" else ("N" if value >= 0 else "S")
    return f"{degrees}°{minutes:02d}′{suffix}"
def geographic_ticks(low, high):
    candidates = np.array([1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120, 180])
    step = candidates[candidates >= (high - low) * 60 / 3][0] / 60
    inset = (high - low) * 0.08
    return np.arange(np.ceil((low + inset) / step) * step, high - inset + 1e-10, step)
def draw_case_map(ax, layers):
    display = {name: layer.to_crs("EPSG:4326") for name, layer in layers.items()}
    grids, study = display["resident_grids"], display["study_area"]
    cmap = colors.LinearSegmentedColormap.from_list("improvement", ["#FAF8FB", "#C4A8D2", "#6C4385"])
    norm = colors.Normalize(vmin=0, vmax=1)
    study.plot(ax=ax, facecolor="white", edgecolor="#9A9A9A", linewidth=.35, zorder=0)
    grids.plot(ax=ax, column="delta_risk", cmap=cmap, norm=norm, edgecolor="none", linewidth=0, zorder=1)
    if len(display["inundation"]):
        display["inundation"].plot(ax=ax, color="#65BFCB", edgecolor="none", alpha=.42, zorder=2)
    highlighted = grids.loc[grids.confirmed_unflooded_improved.astype(bool)]
    if len(highlighted):
        highlighted.boundary.plot(ax=ax, color="#238477", linewidth=.45, zorder=3)
    if len(display["restored_roads"]):
        display["restored_roads"].plot(ax=ax, color="#D96F3A", linewidth=.55, alpha=.95, zorder=4)
    study.boundary.plot(ax=ax, color="#555555", linewidth=.45, zorder=5)
    a, b = grids.total_bounds, study.total_bounds
    xmin, ymin = min(a[0], b[0]), min(a[1], b[1])
    xmax, ymax = max(a[2], b[2]), max(a[3], b[3])
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    aspect = 1 / np.cos(np.deg2rad(cy))
    fig = ax.figure
    position = ax.get_position()
    physical_aspect = position.width * fig.get_figwidth() / (position.height * fig.get_figheight())
    sy = max((ymax - ymin) / .94, (xmax - xmin) / (physical_aspect * aspect * .94))
    sx = sy * physical_aspect * aspect
    ax.set_xlim(cx - sx / 2, cx + sx / 2)
    ax.set_ylim(cy - sy / 2, cy + sy / 2)
    ax.set_aspect(aspect, adjustable="box")
    ax.set_axis_on()
    ax.set_xticks(geographic_ticks(*ax.get_xlim()))
    ax.set_yticks(geographic_ticks(*ax.get_ylim()))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: degree_label(value, "lon")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: degree_label(value, "lat")))
    ax.tick_params(axis="both", length=2.5, width=.65, pad=2, labelsize=6.5)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=90)
    for label in ax.get_yticklabels():
        label.set_va("center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap), len(highlighted), display
