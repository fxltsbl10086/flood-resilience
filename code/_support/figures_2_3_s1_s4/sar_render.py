"""Plot the original 500 m SAR for five cities in the Supplementary Figure 2 layout."""

from __future__ import annotations

import sys
from pathlib import Path

from sar_data import load_data, set_style
from sar_config import (
    CITY_INPUTS, DPI, EXCLUDED_COLOR, PERIODS, RISK_COLORS, RISK_COLUMN, RISK_LIMITS,
    STUDY_BOUNDARY_LINEWIDTH,
)
from s2_render import (
    CITIES, FRAME_COLOR, draw_coordinates, draw_north_arrow, draw_scale_bar,
    map_limits, validate_coordinates,
)

import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

SAR_CMAP = LinearSegmentedColormap.from_list("all_grid_SAR", RISK_COLORS)
SAR_NORM = Normalize(*RISK_LIMITS)
SAR_TICKS = (0, 0.25, 0.50, 0.75, 1)


def draw_map(ax, included, excluded, outline, crs, period, limits) -> tuple:
    if not excluded.empty:
        excluded.plot(ax=ax, color=EXCLUDED_COLOR, edgecolor="none", aspect=None, zorder=1)
    included.plot(ax=ax, column=RISK_COLUMN.format(period=period), cmap=SAR_CMAP,
                  norm=SAR_NORM, edgecolor="none", aspect=None, zorder=2)
    outline.plot(ax=ax, color=FRAME_COLOR, linewidth=STUDY_BOUNDARY_LINEWIDTH,
                 aspect=None, zorder=3)
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal", facecolor="white")
    coordinate_axes = draw_coordinates(ax, crs)
    ax.text(0.025, 0.018, f"{period}-year", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color=FRAME_COLOR, zorder=4)
    return coordinate_axes


def draw_legend(ax, city_name, map_ax, crs, show_invalid):
    ax.set_axis_off()
    ax.text(0.09, 0.901, city_name, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="center")

    color_ax = ax.inset_axes((0.09, 0.645, 0.60, 0.056))
    colorbar = ax.figure.colorbar(
        ScalarMappable(norm=SAR_NORM, cmap=SAR_CMAP), cax=color_ax,
        orientation="horizontal", ticks=SAR_TICKS, format="%.2f",
    )
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("SAR", fontsize=8, labelpad=3)
    colorbar.ax.tick_params(labelsize=7, length=2.5, pad=1.8, width=0.5,
                            colors=FRAME_COLOR)
    colorbar.outline.set_linewidth(0.4)

    boundary_y = 0.440
    ax.plot([0.09, 0.185], [boundary_y, boundary_y], transform=ax.transAxes,
            color=FRAME_COLOR, linewidth=STUDY_BOUNDARY_LINEWIDTH)
    ax.text(0.22, boundary_y, "Study boundary", transform=ax.transAxes,
            va="center", fontsize=7.7)
    if show_invalid:
        swatch = Rectangle((0.09, 0.330), 0.095, 0.044, transform=ax.transAxes,
                           facecolor=EXCLUDED_COLOR, edgecolor=FRAME_COLOR, linewidth=0.35)
        ax.add_patch(swatch)
        ax.text(0.22, 0.352, "No baseline", transform=ax.transAxes,
                va="center", fontsize=7.7)

    draw_north_arrow(ax, map_ax, crs)
    draw_scale_bar(ax, map_ax, crs.axis_info[0].unit_conversion_factor)
    return colorbar


def plot_city(name, data, boundary, output):
    included = data.loc[data.analysis_included]
    excluded = data.loc[~data.analysis_included].geometry
    grid_bounds = data.total_bounds
    study_bounds = boundary.total_bounds
    bounds = np.r_[np.minimum(grid_bounds[:2], study_bounds[:2]),
                   np.maximum(grid_bounds[2:], study_bounds[2:])]
    crs = data.crs
    outline = boundary.geometry.boundary

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 126 / 25.4))
    try:
        fig.subplots_adjust(left=0.049, right=0.951, top=0.936, bottom=0.060,
                            wspace=0.26, hspace=0.245)
        limits = map_limits(bounds, axes.flat[0])
        panels = [
            (ax, draw_map(ax, included, excluded, outline, crs, period, limits))
            for ax, period in zip(axes.flat, PERIODS)
        ]
        legend_ax = axes.flat[-1]
        colorbar = draw_legend(legend_ax, name, axes.flat[0], crs, not excluded.empty)
        validate_coordinates(fig, panels)
        renderer = fig.canvas.get_renderer()
        boxes = [colorbar.ax.get_tightbbox(renderer)]
        boxes.extend(label.get_window_extent(renderer) for label in legend_ax.texts if label.get_text())
        for index, box in enumerate(boxes):
            if not (legend_ax.bbox.contains(box.x0, box.y0)
                    and legend_ax.bbox.contains(box.x1, box.y1)):
                raise ValueError()
            if any(box.overlaps(other) for other in boxes[index + 1:]):
                raise ValueError()

        path = output / f"fig_s3_{name.lower()}.png"
        fig.savefig(path, dpi=DPI, facecolor="white")
        print(f"{name}: {len(included):,} of {len(data):,} cells with valid baseline. {path}", flush=True)
    finally:
        plt.close(fig)


