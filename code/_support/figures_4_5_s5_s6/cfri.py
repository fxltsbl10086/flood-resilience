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
STYLE = {
    "body_font_pt": 8.0, "axis_font_pt": 7.0, "tick_font_pt": 7.0,
    "city_font_pt": 8.5, "panel_font_pt": 10.0,
    "legend_font_pt": 7.0,
    "inset_tick_font_pt": 5.5, "inset_value_font_pt": 5.0,
    "frame_linewidth_pt": 0.65, "inset_frame_linewidth_pt": 0.45,
    "curve_linewidth_pt": 1.2, "grid_linewidth_pt": 0.4,
    "left_fraction": 0.10, "right_fraction": 0.97,
    "column_gap_ratio": 0.28,
    # Keep a 10.5 mm row gap for the measured text extents.
    "row_gap_mm": 10.5, "top_margin_mm": 8.0, "panel_box_aspect": 1.0,
    "minimum_interrow_text_gap_mm": 2.5,
    # Crop only the unused bottom margin, retaining figure width and physical panel dimensions.
    "trim_bottom_whitespace": True,
    "export_bottom_padding_mm": 0.5,
    "city_position_axes": [0.96, 0.05], "axis_labelpad_pt": 2.5,
    "xlabel_to_legend_gap_mm": 5.0,  # Retained for individual-city figures.
    # Align to the lower Figure 1 legend's left edge and vertical centre.
    "legend_loc": "center left",
    "legend_anchor_axes": [0.0650290093, 0.3939966968],
    # Place panel letters near the frame and abbreviations inside the lower-right corner.
    "panel_label_offset_pt": [-6.0, 3.0],
    "gini_inset_bounds": [0.10, 0.69, 0.54, 0.26],
}
SCENARIO_COLORS = {10: "#4C78A8", 20: "#59A14F", 50: "#E3B34C", 100: "#E17C45", 200: "#A05C99"}
EXCEEDANCE_LINESTYLES = {10: ":", 20: "--", 50: "-.", 100: "--", 200: "-"}
FORMAL_CITY_ABBREVIATIONS = {"fuzhou": "FZ", "chengdu": "CD", "wuhan": "WH",
                             "hangzhou": "HZ", "zhengzhou": "ZZ"}
PERIODS=[10,20,50,100,200]
SCENARIO_STYLES=EXCEEDANCE_LINESTYLES
CITY_ABBREVIATIONS=FORMAL_CITY_ABBREVIATIONS
def city_panel(ax, city, summary, curves, letter=None, show_xlabel=True, show_ylabel=True):
    ax.set_box_aspect(STYLE["panel_box_aspect"])
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=0.7, zorder=1)
    for period in PERIODS:
        x, y = curves[(city, period)]
        ax.plot(x, y, color=SCENARIO_COLORS[period], linestyle=SCENARIO_STYLES[period],
                linewidth=STYLE["curve_linewidth_pt"], zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(color="#E8EBED", linewidth=STYLE["grid_linewidth_pt"])
    ax.set_axisbelow(True)
    ax.set_xlabel("Cumulative population share" if show_xlabel else "",
                  labelpad=STYLE["axis_labelpad_pt"])
    ax.set_ylabel("Cumulative risk share" if show_ylabel else "",
                  labelpad=STYLE["axis_labelpad_pt"])
    ax.set_title("")
    ax._cfri_city_caption = ax.text(
        *STYLE["city_position_axes"], CITY_ABBREVIATIONS[city],
        transform=ax.transAxes, fontsize=STYLE["city_font_pt"],
        ha="right", va="bottom", color="black", zorder=5)
    if letter is not None:
        ax._cfri_panel_label = ax.annotate(
            letter, xy=(0, 1), xycoords="axes fraction",
            xytext=STYLE["panel_label_offset_pt"], textcoords="offset points",
            fontsize=STYLE["panel_font_pt"], fontweight="bold",
            ha="left", va="baseline", annotation_clip=False, clip_on=False)
    inset = ax.inset_axes(STYLE["gini_inset_bounds"])
    values = summary.loc[summary.city.eq(city)].set_index(
        "return_period").loc[PERIODS, "gini"].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError()
    inset.bar(np.arange(5), values, color=[SCENARIO_COLORS[t] for t in PERIODS],
              width=0.75, edgecolor="white", linewidth=0.3)
    inset.set_xlim(-0.6, 4.6)
    inset.set_ylim(0, 1.10)
    inset.set_xticks(np.arange(5), [str(t) for t in PERIODS])
    inset.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    inset.tick_params(axis="both", labelsize=STYLE["inset_tick_font_pt"],
                      length=1.5, width=0.4, pad=0.8)
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(STYLE["inset_frame_linewidth_pt"])
    for index, value in enumerate(values):
        inset.text(index, value + 0.025, f"{value:.2f}", ha="center", va="bottom",
                   fontsize=STYLE["inset_value_font_pt"])
    inset.grid(False)
