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
PERIODS = (10, 20, 50, 200)
FIGURE_MM = (183, 127)
PANEL_MM = 48
LEFT_MM = 13
TOP_MM = 7
COLUMN_GAP_MM = 10
ROW_GAP_MM = 16
def panel_slot(row: int, column: int) -> tuple[float, float, float, float]:
    width, height = FIGURE_MM
    x = LEFT_MM + column * (PANEL_MM + COLUMN_GAP_MM)
    y = height - TOP_MM - PANEL_MM - row * (PANEL_MM + ROW_GAP_MM)
    return x / width, y / height, PANEL_MM / width, PANEL_MM / height
def draw_bivariate_legend(fig):
    # Reuse the exact Fig. 4 legend, then place it in the lower-right slot.
    scenario_ax, _, legend = body.add_legends(fig)
    scenario_ax.remove()
    x, y, w, h = panel_slot(1, 2)
    figure_w, figure_h = FIGURE_MM
    size_mm = body.LEGEND_LAYOUT["bivariate_size_mm"]
    legend_w, legend_h = size_mm / figure_w, size_mm / figure_h
    legend.set_position((x + (w - legend_w) / 2 + 4 / figure_w,
                         y + (h - legend_h) / 2 + 1 / figure_h,
                         legend_w, legend_h))
    return legend
def validate_layout(fig, maps, legend) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [ax.get_tightbbox(renderer) for ax in [*maps, legend]]
    for index, box in enumerate(bounds):
        if not (fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(box.x1, box.y1)):
            raise ValueError()
        if any(box.overlaps(other) for other in bounds[index + 1:]):
            raise ValueError()
    for ax in maps:
        label = ax._combined_panel_label.get_window_extent(renderer)
        if any(label.overlaps(tick.get_window_extent(renderer))
               for tick in ax.get_xticklabels() + ax.get_yticklabels() if tick.get_visible()):
            raise ValueError()
        if len(ax.get_xticklabels()) < 2 or len(ax.get_yticklabels()) < 2:
            raise ValueError()
