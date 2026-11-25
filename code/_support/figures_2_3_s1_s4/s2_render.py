"""Draw the five Supplementary Figure 2 PNGs from existing maximum-depth rasters."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Patch
from pyproj import Geod, Transformer

from shared_context import ROOT, DATA, settings
SETTINGS=settings('S2')
RESULTS=ROOT/'results/flood_simulation'
from sfincs_plotting import COLORS, LEVELS, _common_context, _context, _load_depth_maps
from s1_render import CITIES, FRAME_COLOR, RETURN_PERIODS, draw_coordinates, validate_coordinates
MAP_FONT=SETTINGS['font_family']
import geopandas as gpd

SOURCE_PRODUCT = "max_depth_land_study_area_m.tif"
DEPTH_COLORS = ["#ffffff", *COLORS[1:]]
DEPTH_CLASSES = [
    f"<{LEVELS[1]:.2f}",
    *(f"{lower:.2f}–<{upper:.2f}" for lower, upper in zip(LEVELS[1:-2], LEVELS[2:-1])),
    f"≥{LEVELS[-2]:.2f}",
]
DEPTH_CMAP = ListedColormap(DEPTH_COLORS, name="supplementary_maximum_depth")
DEPTH_CMAP.set_bad((0, 0, 0, 0))
DEPTH_NORM = BoundaryNorm(LEVELS, DEPTH_CMAP.N, clip=True)
BOUNDARY_COLOR = "#333333"
WGS84 = Geod(ellps="WGS84")


def map_limits(bounds, map_ax) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fit a common equal-aspect window, reserving space for lower-left period labels."""
    left, bottom, right, top = bounds
    position = map_ax.get_position(original=True)
    fig_width, fig_height = map_ax.figure.get_size_inches()
    aspect = (position.width * fig_width) / (position.height * fig_height)
    width = max(1.07 * (right - left), 1.14 * (top - bottom) * aspect)
    height = width / aspect
    x_mid = (left + right) / 2
    y_mid = (bottom + top) / 2 - 0.038 * height
    return (x_mid - width / 2, x_mid + width / 2), (y_mid - height / 2, y_mid + height / 2)


def draw_map(ax, raster, outline, limits) -> tuple:
    ax.imshow(raster.depth, extent=raster.extent, origin="upper",
              cmap=DEPTH_CMAP, norm=DEPTH_NORM, interpolation="nearest", zorder=1)
    outline.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=0.55, zorder=2)
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal", facecolor="white")
    coord_axes = draw_coordinates(ax, raster.crs)
    ax.text(0.025, 0.018, f"{raster.period}-year", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7, color=FRAME_COLOR, zorder=3)
    return coord_axes


def draw_north_arrow(ax, map_ax, crs) -> None:
    """Show true north at the shared map-window centre in projected map orientation."""
    center_x, center_y = np.mean(map_ax.get_xlim()), np.mean(map_ax.get_ylim())
    longitude, latitude = Transformer.from_crs(crs, 4326, always_xy=True).transform(center_x, center_y)
    north_lon, north_lat, _ = WGS84.fwd(longitude, latitude, 0, 1000)
    north_x, north_y = Transformer.from_crs(4326, crs, always_xy=True).transform(north_lon, north_lat)
    delta_x, delta_y = north_x - center_x, north_y - center_y
    base_x, base_y = 0.83, 0.682
    rise = 0.105
    bounds = ax.get_position()
    width, height = ax.figure.get_size_inches()
    box_aspect = (bounds.width * width) / (bounds.height * height)
    tip_x, tip_y = base_x + rise * (delta_x / delta_y) / box_aspect, base_y + rise
    ax.annotate("", xy=(tip_x, tip_y), xytext=(base_x, base_y),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={"arrowstyle": "-|>", "color": FRAME_COLOR,
                            "lw": 0.85, "mutation_scale": 10, "shrinkA": 0, "shrinkB": 0})
    ax.text(tip_x, tip_y + 0.026, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8)


def draw_scale_bar(ax, map_ax, units_to_m: float) -> None:
    """Place a distance bar in the legend at the maps' actual physical scale."""
    x_min, x_max = map_ax.get_xlim()
    width_m = (x_max - x_min) * units_to_m
    distance_km = max(km for km in (1, 2, 5, 10, 20, 50, 100) if km * 1000 <= 0.25 * width_m)
    map_to_legend_width = map_ax.get_position().width / ax.get_position().width
    length = distance_km * 1000 / width_m * map_to_legend_width
    start, y = 0.10, 0.095
    end = start + length
    ax.plot([start, end], [y, y], color=FRAME_COLOR, linewidth=0.9,
            solid_capstyle="butt", transform=ax.transAxes)
    for x in (start, (start + end) / 2, end):
        ax.plot([x, x], [y - 0.006, y + 0.008], color=FRAME_COLOR,
                linewidth=0.7, transform=ax.transAxes)
    for x, label in ((start, "0"), (end, f"{distance_km} km")):
        ax.text(x, y + 0.022, label, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7)


def draw_legend(ax, city_name: str, map_ax, crs, units_to_m: float) -> None:
    ax.set_axis_off()
    ax.text(0.09, 0.901, city_name, transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="center")
    handles = [
        Patch(facecolor=color, edgecolor=FRAME_COLOR, linewidth=0.38, label=label)
        for color, label in zip(DEPTH_COLORS, DEPTH_CLASSES, strict=True)
    ]
    ax.legend(handles=handles,
              loc="upper left", bbox_to_anchor=(0.09, 0.793), alignment="left",
              fontsize=7.7, handlelength=1.55, handletextpad=0.65, labelspacing=0.46,
              borderaxespad=0, borderpad=0, frameon=False)
    boundary_y = 0.235
    ax.plot([0.09, 0.185], [boundary_y, boundary_y], transform=ax.transAxes,
            color=BOUNDARY_COLOR, linewidth=0.55)
    ax.text(0.22, boundary_y, "Study-area boundary", transform=ax.transAxes,
            va="center", fontsize=7.7)
    draw_north_arrow(ax, map_ax, crs)
    draw_scale_bar(ax, map_ax, units_to_m)


def plot_city(city: str, city_name: str, output: Path) -> None:
    maps = _load_depth_maps(RESULTS / city, RETURN_PERIODS, SOURCE_PRODUCT)
    config={'city_name':city_name}
    context = _common_context([_context(city, config, raster, []) for raster in maps])
    if not np.isclose(context.simulation_hours, 12):
        raise ValueError()
    crs = maps[0].crs
    boundary = gpd.read_file(DATA/f'plot_study_{city}.gpkg').to_crs(crs)
    outline = boundary.geometry.boundary

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 126 / 25.4))
    try:
        fig.subplots_adjust(left=0.049, right=0.951, top=0.936, bottom=0.060,
                            wspace=0.26, hspace=0.245)
        example_ax = axes.flat[0]
        limits = map_limits(boundary.total_bounds, example_ax)
        panels = [(ax, draw_map(ax, raster, outline, limits)) for ax, raster in zip(axes.flat, maps)]
        draw_legend(axes.flat[-1], city_name, example_ax, crs, context.units_to_m)
        validate_coordinates(fig, panels)
        legend_ax = axes.flat[-1]
        legend = legend_ax.get_legend()
        renderer = fig.canvas.get_renderer()
        boxes = [legend.get_window_extent(renderer)]
        boxes.extend(label.get_window_extent(renderer) for label in legend_ax.texts if label.get_text())
        for i, bounds in enumerate(boxes):
            if not (legend_ax.bbox.contains(bounds.x0, bounds.y0)
                    and legend_ax.bbox.contains(bounds.x1, bounds.y1)):
                raise ValueError()
            if any(bounds.overlaps(other) for other in boxes[i + 1:]):
                raise ValueError()
        output_file = output / f"fig_s2_{city}.png"
        fig.savefig(output_file, dpi=int(SETTINGS["dpi"]), facecolor="white")
        print(f"{city_name}: {output_file}", flush=True)
    finally:
        plt.close(fig)


