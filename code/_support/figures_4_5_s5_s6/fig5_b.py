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
sys.dont_write_bytecode = True
CITY_ORDER = ["zhengzhou", "hangzhou", "fuzhou", "chengdu", "wuhan"]
CITY_LABELS = {"zhengzhou": "Zhengzhou", "hangzhou": "Hangzhou", "fuzhou": "Fuzhou",
               "chengdu": "Chengdu", "wuhan": "Wuhan"}
PERIODS = [10, 20, 50, 100, 200]
TARGET_PCT = 30.0
EPSILON = 1e-8
WIDTH_MM, HEIGHT_MM, DPI = 130.0, 85.0, 600
X_LIMITS, Y_LIMITS = (93.0, 100.0), (0.0, 80.0)
MARKER_SIZE_PT = 4.5
FIT_COLOR = "#555555"
CI_ALPHA = 0.18
BOOTSTRAP_REPETITIONS = 1000
BOOTSTRAP_SEED = 2026
METHOD_REFERENCES = [
    "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html",
]
def draw_panel(ax, points, city_colors, markers, letter="b", curve=None):
    "Draw one point per archived city-scenario result."
    if curve is None:
        curve, _, _ = fit_ols(points)
    ax.fill_between(curve.improved_population_pct, curve.CI95_lower, curve.CI95_upper,
                    color=FIT_COLOR, alpha=CI_ALPHA, linewidth=0, zorder=1)
    ax.plot(curve.improved_population_pct, curve.fitted_reconnected_population_pct,
            color=FIT_COLOR, linewidth=1.0, linestyle="-", zorder=2, label="OLS fit")
    artists = []
    for row in points.itertuples():
        artist, = ax.plot(row.improved_population_pct, row.reconnected_population_pct,
                          linestyle="none", marker=markers[row.period],
                          markersize=MARKER_SIZE_PT, markerfacecolor="none",
                          markeredgecolor=city_colors[row.city], markeredgewidth=0.85,
                          zorder=3, clip_on=False)
        artists.append(artist)
    labels = []
    ax.set_xlim(*X_LIMITS)
    # Set the displayed lower axis limit to zero, retaining the fitted values and intervals.
    y_lower, y_upper = Y_LIMITS
    ax.set_ylim(y_lower, y_upper)
    ax.set_xticks(np.arange(93, 101, 1))
    ax.set_yticks(np.arange(0, y_upper+1, 20))
    ax.set_xlabel("Population with improved accessibility (%)", labelpad=4, color="black")
    ax.set_ylabel("Population with restored connectivity (%)", labelpad=4, color="black")
    ax.tick_params(axis="both", colors="black", pad=2.5, length=2.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.65)
    panel_letter = ax.text(-0.12, 1.055, letter, transform=ax.transAxes, fontsize=10,
                            fontweight="bold", color="black", ha="left", va="top")
    return artists, labels, panel_letter
def check_layout(fig, ax, points, labels, panel_letter, legends):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = [ax.xaxis.label, ax.yaxis.label, panel_letter, *labels,
                    *ax.get_xticklabels(), *ax.get_yticklabels()]
    items = [*text_artists, *legends]
    for artist in items:
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
            raise ValueError()
    legend_boxes = [legend.get_window_extent(renderer) for legend in legends]
    if legend_boxes[0].overlaps(legend_boxes[1]):
        raise ValueError()
    if any(box.overlaps(ax.xaxis.label.get_window_extent(renderer)) for box in legend_boxes):
        raise ValueError()
    axes_box = ax.get_window_extent(renderer)
    for box in legend_boxes:
        if box.x0 < axes_box.x0 or box.y0 < axes_box.y0 or box.x1 > axes_box.x1 or box.y1 > axes_box.y1:
            raise ValueError()
    xy = points[["improved_population_pct", "reconnected_population_pct"]].to_numpy()
    if not ((xy[:, 0] >= X_LIMITS[0]) & (xy[:, 0] <= X_LIMITS[1]) &
            (xy[:, 1] >= Y_LIMITS[0]) & (xy[:, 1] <= Y_LIMITS[1])).all():
        raise ValueError()
    display_points = ax.transData.transform(xy)
    if any(box.contains(px,py) for box in legend_boxes for px,py in display_points):
        raise ValueError()
    return {"all_labels_inside_canvas": True, "legend_overlap": False,
            "legend_inside_axes": True, "legend_covers_points": False,
            "all_points_inside_axes": True, "point_count": len(points), "jitter": False}
