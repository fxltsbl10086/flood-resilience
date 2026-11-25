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
AXIS_FONT_SIZE = 7.0
MAP_FRAME_LINEWIDTH_PT = 0.65
STUDY_BOUNDARY_LINEWIDTH_PT = 0.45
TICK_LENGTH_PT = 3.0
TICK_PAD_PT = 2.0
HATCH_LINEWIDTH_PT = 0.25
HOTSPOT_WHITE_LINEWIDTH_PT = 0.85
HOTSPOT_OUTLINE_LINEWIDTH_PT = 0.30
COORDINATE_TICK_INSETS = {"longitude": 0.08, "latitude": 0.14}
BIVARIATE_PALETTE = [
    ["#E8E8E8", "#B6DDDA", "#69C2BF"],
    ["#D8B4D7", "#AAA5CD", "#6B8EAE"],
    ["#B767B9", "#865D9B", "#3F4E7C"],
]
LEGEND_FONT_SIZE=7.0
def geographic_ticks(low, high, edge_fraction=0.08):
    minute_steps = np.array([1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120])
    span = high - low
    step_minutes = minute_steps[minute_steps >= span * 60 / 3][0]
    step = step_minutes / 60
    start = int(np.ceil((low + span * edge_fraction) / step))
    end = int(np.floor((high - span * edge_fraction) / step))
    return np.arange(start, end + 1) * step
def degree_label(value, direction):
    degrees, minute = divmod(int(round(abs(value) * 60)), 60)
    return f"{degrees}°{minute:02d}′{direction}"
def fit_extent(ax, bounds, physical_aspect):
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    geographic_aspect = 1 / np.cos(np.deg2rad(cy))
    span_y = max(height / 0.94,
                 width / (physical_aspect * geographic_aspect * 0.94))
    span_x = span_y * physical_aspect * geographic_aspect
    ax.set_xlim(cx - span_x / 2, cx + span_x / 2)
    ax.set_ylim(cy - span_y / 2, cy + span_y / 2)
    ax.set_aspect(geographic_aspect, adjustable="box")
    ax.set_axis_on()
    ax.set_xticks(geographic_ticks(*ax.get_xlim(), COORDINATE_TICK_INSETS["longitude"]))
    ax.set_yticks(geographic_ticks(*ax.get_ylim(), COORDINATE_TICK_INSETS["latitude"]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _: degree_label(value, "E" if value >= 0 else "W")))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: degree_label(value, "N" if value >= 0 else "S")))
    ax.tick_params(
        axis="both", which="major", labelsize=AXIS_FONT_SIZE,
        length=TICK_LENGTH_PT, width=MAP_FRAME_LINEWIDTH_PT,
        pad=TICK_PAD_PT, top=False, right=False)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=90)
    for label in ax.get_xticklabels():
        label.set_ha("center")
        label.set_va("top")
    for label in ax.get_yticklabels():
        label.set_ha("right")
        label.set_va("center")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(MAP_FRAME_LINEWIDTH_PT)
        spine.set_color("black")
    ax.grid(False)
