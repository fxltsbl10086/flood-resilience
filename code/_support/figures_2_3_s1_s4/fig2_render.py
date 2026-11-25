"Formal five-city layout with map insets above five exceedance curves."

import argparse
import csv
from datetime import datetime
from importlib import import_module
from math import ceil
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, PercentFormatter
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union

from fig2_config import (
    PERIODS, PLOT_COLORS, EXCEEDANCE_COLORS, EXCEEDANCE_LINESTYLES, SCENARIO_COLORS, FORMAL_FONT,
    FORMAL_LEGEND_FONT_SIZE,
    FORMAL_CURVE_CSV, FORMAL_OUTPUT_DIR, FORMAL_FIGSIZE, FORMAL_DPI,
    FORMAL_HEIGHT_RATIOS, FORMAL_OUTER_GAP, FORMAL_CURVE_ASPECT,
    FORMAL_CURVE_ROW_GAP, FORMAL_CURVE_COL_GAP,
    FORMAL_PANEL_CITIES, FORMAL_CURVE_POSITIONS, FORMAL_CURVE_LEGEND_POSITION,
    FORMAL_CURVE_LEGEND_LOC, FORMAL_CURVE_LEGEND_ANCHOR,
    FORMAL_CITY_NAMES, FORMAL_BOUNDARY_PATHS,
    FORMAL_CITY_ABBREVIATIONS, FORMAL_CITY_EDGE_COLOR, FORMAL_STUDY_EDGE_COLOR,
    FORMAL_COUNTRY_FILL_COLOR,
    FORMAL_CITY_BOUNDARY_PATHS, FORMAL_EXPOSURE_CSV, FORMAL_MAP_EXTENT, FORMAL_MAP_INSETS,
    FORMAL_PROVINCE_PATH, FORMAL_MAP_LEGEND_ANCHOR,
)

# Reuse the population plotting helper with the formal font sizes and two group colors.
population_plot = import_module("fig2_population")


def set_style():
    "Set the configured font, common frames and scenario colors."
    for filename in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        
        if FORMAL_FONT.with_name(filename).is_file():
            font_manager.fontManager.addfont(str(FORMAL_FONT.with_name(filename)))
    name = font_manager.FontProperties(fname=str(FORMAL_FONT)).get_name()
    plt.rcParams.update({
        "font.family": name, "font.size": 8, "axes.unicode_minus": False, "axes.linewidth": 0.65,
        "axes.spines.top": True, "axes.spines.right": True, "hatch.linewidth": 0.25,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def draw_panel_label(ax, label):
    "Place panel letters at a fixed offset in points."
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=(-6, 5 if label != "a" else 3), textcoords="offset points",
                fontsize=10, fontweight="bold", ha="left", va="baseline",
                annotation_clip=False, clip_on=False, zorder=20)


def read_curves(path):
    "Read the frozen step-curve table without recalculating population or SAR."
    required = {"city", "scenario", "exceedance_population_share",
                "city_valid_population"}
    groups = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError()
        # Accept the historical NAR threshold alias.
        threshold_field = "SAR_threshold" if "SAR_threshold" in reader.fieldnames else "NAR_threshold"
        if threshold_field not in reader.fieldnames:
            raise ValueError()
        for row in reader:
            if "SAR_threshold" in row and "NAR_threshold" in row:
                if float(row["SAR_threshold"]) != float(row["NAR_threshold"]):
                    raise ValueError()
            if row["city"] in FORMAL_PANEL_CITIES:
                key = (row["city"], row["scenario"])
                groups.setdefault(key, []).append((float(row[threshold_field]),
                                                    float(row["exceedance_population_share"]),
                                                    float(row["city_valid_population"])))
    curves, populations = {}, {}
    cities = [city for city in FORMAL_PANEL_CITIES if any(key[0] == city for key in groups)]
    if cities != list(FORMAL_PANEL_CITIES):
        raise ValueError()
    for city in cities:
        for period in PERIODS:
            values = np.array(groups[(city, f"T{period}")])
            values = values[np.argsort(values[:, 0])]
            x, y, population = values.T
            if not np.isfinite(values).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) > 1e-12):
                raise ValueError()
            if x[0] != 0 or x[-1] != 1 or np.any((y < -1e-12) | (y > 1 + 1e-12)):
                raise ValueError()
            if not np.all(population == population[0]) or population[0] <= 0:
                raise ValueError()
            if city in populations and populations[city] != population[0]:
                raise ValueError()
            populations[city] = population[0]
            curves[(city, period)] = (x, y)
    return cities, curves, populations


def read_study_lines(cities, paths=FORMAL_BOUNDARY_PATHS, as_boundary=True):
    "Read existing spatial layers, returning outlines or polygons as requested."
    lines = {}
    for city in cities:
        data = gpd.read_file(paths[city], encoding="UTF-8")
        if data.crs is None:
            raise ValueError()
        if as_boundary:
            data.geometry = data.geometry.map(
                lambda geometry: geometry.boundary if geometry.geom_type in ("Polygon", "MultiPolygon") else geometry
            )
        lines[city] = data
    return lines


def draw_population_inset(ax, city, values, population_limit):
    "Draw the dual-axis inset with a transparent background and full tick labels."
    ax.set_gid(f"population_{city}")
    population_ax = population_plot.draw_city(ax, city, values, population_limit, "", compact=True,
                                              english=True, font_size=7.5)
    # Restore the inset above the administrative polygon fill.
    ax.set_zorder(8)
    # Reserve the upper axis margin for the city abbreviation.
    ax.set_ylim(0, 1.2)
    # Show both axis names only in the upper-left inset.
    show_axis_names = city == "zhengzhou"
    ax.set_ylabel("SAR" if show_axis_names else "", fontsize=7, labelpad=2)
    population_ax.set_ylabel("Pop (M)" if show_axis_names else "", fontsize=7, labelpad=2)
    ax.set_xlabel("")
    # Align population ticks with the risk-axis ticks.
    population_ax.set_yticks([0, population_limit / 2, population_limit])
    population_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    # Reduce only the outer tick-label font in the narrow inset.
    ax.tick_params(axis="both", labelsize=6.5)
    population_ax.tick_params(axis="y", labelsize=6.5)
    ax.set_facecolor("none")
    for axis in (ax, population_ax):
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
    # Restore the original bar-chart horizontal margins.
    ax.set_xlim(-0.6, len(PERIODS) - 0.4)
    ax.text(0.03, 0.96, FORMAL_CITY_ABBREVIATIONS[city], transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, fontweight="normal", color="black",
            zorder=6)


def draw_map_legend(ax):
    "Use exposure groups as legend columns and SAR/population as rows."
    widths = (30, 52, 64)

    def cell(column, text="", group=None, symbol=None):
        width, height = widths[column], 8
        area = DrawingArea(width, height, 0, 0)
        if text:
            area.add_artist(Text(width / 2, height / 2, text, fontsize=6.5,
                                 ha="center", va="center"))
        if symbol == "line":
            color = PLOT_COLORS[group]
            area.add_artist(Line2D([width / 2 - 7, width / 2, width / 2 + 7],
                                  [height / 2] * 3, color=color, linewidth=1.0,
                                  linestyle="-" if group == "exposed" else "--",
                                  marker="o" if group == "exposed" else "s",
                                  markevery=[1], markersize=2.5))
        if symbol == "bar":
            area.add_artist(Rectangle((width / 2 - 7, height / 2 - 2), 14, 4,
                                      facecolor=PLOT_COLORS[group], alpha=0.55,
                                      edgecolor="none"))
        return area

    header = [cell(0), cell(1, "Directly exposed"), cell(2, "Not directly exposed")]
    risk = [cell(0, "SAR"), cell(1, group="exposed", symbol="line"),
            cell(2, group="nonexposed", symbol="line")]
    population = [cell(0, "Population"), cell(1, group="exposed", symbol="bar"),
                  cell(2, group="nonexposed", symbol="bar")]
    rows = [HPacker(children=row, align="center", pad=0, sep=1.5)
            for row in (header, risk, population)]
    legend = AnchoredOffsetbox(loc="lower left", child=VPacker(children=rows, align="left", pad=0, sep=1),
                              bbox_to_anchor=FORMAL_MAP_LEGEND_ANCHOR,
                              bbox_transform=ax.transAxes, pad=0, borderpad=0, frameon=False)
    legend.set_zorder(10)
    ax.add_artist(legend)


def fit_map_extent(ax):
    "Match the geographic extent to the map frame without stretching boundaries."
    west, east, south, north = FORMAL_MAP_EXTENT
    aspect = 1 / np.cos(np.deg2rad((south + north) / 2))
    rectangle = ax.get_window_extent()
    ratio = rectangle.width / rectangle.height * aspect
    width = max(east - west, (north - south) * ratio)
    height = width / ratio
    center_x, center_y = (west + east) / 2, (south + north) / 2
    ax.set_xlim(center_x - width / 2, center_x + width / 2)
    ax.set_ylim(center_y - height / 2, center_y + height / 2)
    ax.set_aspect(aspect, adjustable="box")
    ax.apply_aspect()


def draw_study_map(ax, cities, study_areas, city_lines, exposure_data):
    "Place equal-sized statistical insets around the full-width geographic map."
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fit_map_extent(ax)
    west, east = ax.get_xlim()
    south, north = ax.get_ylim()
    window = box(west, south, east, north)
    provinces = gpd.read_file(FORMAL_PROVINCE_PATH, encoding="UTF-8").to_crs(4326)
    # Fill the existing administrative polygons, leaving other areas white.
    land = provinces.geometry.intersection(window)
    land.loc[~land.is_empty].plot(ax=ax, aspect=None, facecolor=FORMAL_COUNTRY_FILL_COLOR,
                                edgecolor="none", linewidth=0, zorder=0)
    # Clip existing boundary lines without adding rectangular clipping edges.
    provinces.boundary.intersection(window).plot(ax=ax, aspect=None, color="#DADEE1",
                                                linewidth=0.3, zorder=1)
    largest = max(number / 1000000 for city in cities for group in ("exposed", "nonexposed")
                  for number in exposure_data[city][f"{group}_valid_population"])
    population_limit = ceil(max(largest * 1.08, 1) / 2) * 2
    for city in cities:
        city_geographic = city_lines[city].to_crs(4326)
        city_geographic.plot(ax=ax, aspect=None,
                             facecolor=FORMAL_CITY_EDGE_COLOR,  # Yellow fill
                             edgecolor=FORMAL_CITY_EDGE_COLOR,  # Yellow outline
                             linewidth=0.35, zorder=2)
        geographic = study_areas[city].to_crs(4326)
        # Draw the study-area outline and the separate administrative boundary.
        geographic.boundary.plot(ax=ax, aspect=None, color=FORMAL_STUDY_EDGE_COLOR,
                                 linewidth=0.45, zorder=3)
        # Place abbreviations outside the administrative boundary.
        xmin, ymin, xmax, ymax = city_geographic.total_bounds
        # Place the Chengdu abbreviation southwest of its administrative polygon.
        label_points = {"fuzhou": (xmax, (ymin + ymax) / 2), "chengdu": (xmin, ymin),
                        "wuhan": (xmax, ymax), "hangzhou": (xmax, ymin),
                        "zhengzhou": (xmin, ymax)}
        # (horizontal offset, vertical offset)
        offsets = {"fuzhou": (-10, 8), "chengdu": (-6, 0), "wuhan": (5, -15),
                   "hangzhou": (-21, 1), "zhengzhou": (1, -10)}
        offset = offsets[city]
        # Anchor each abbreviation near the actual administrative boundary.
        city_boundary = unary_union(city_geographic.geometry.to_numpy()).boundary
        label_point = nearest_points(city_boundary, Point(label_points[city]))[0]
        ax.annotate(FORMAL_CITY_ABBREVIATIONS[city], (label_point.x, label_point.y), xytext=offset,
                    textcoords="offset points", ha="right" if offset[0] < 0 else "left",
                    va="bottom" if offset[1] >= 0 else "top",
                    fontsize=8, fontweight="normal", color="black", zorder=6)
        bounds = FORMAL_MAP_INSETS[city]
        inset = ax.inset_axes(bounds, facecolor="none", zorder=8)
        if city in cities:
            draw_population_inset(inset, city, exposure_data[city], population_limit)
        else:
            inset.set_xticks([])
            inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("#39434A" if city in cities else "#AAB3B9")
            spine.set_linewidth(0.45)
    # Keep ticks within the current geographic extent.
    ax.set_xticks([value for value in (100, 110, 120, 130) if west <= value <= east])
    ax.set_yticks([value for value in (22, 28, 34) if south <= value <= north])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}°N"))
    ax.tick_params(labelsize=7, length=2.5, pad=2)
    draw_panel_label(ax, "a")
    draw_map_legend(ax)



def draw_curve_panel(ax, city, curves, population, label):
    "Retain the frozen inclusive step convention and original endpoints."
    ax.set_gid(f"curve_{city}")
    for period in PERIODS:
        x, y = curves[(city, period)]
        color = EXCEEDANCE_COLORS[period]
        ax.step(x, y, where="pre", color=color, linestyle=EXCEEDANCE_LINESTYLES[period], linewidth=1.2)
        ax.plot([0, 1], [y[0], y[-1]], color=color, linestyle="none", marker="o",
                markersize=2.6, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("SAR", fontsize=7, labelpad=2.5)
    ax.set_ylabel("Pop share", fontsize=7, labelpad=2.5)
    draw_panel_label(ax, label)
    # Place the city abbreviation inside the lower-left corner.
    ax.text(0.045, 0.06, FORMAL_CITY_ABBREVIATIONS[city], transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.5, color="black", zorder=5)
    ax.tick_params(labelsize=7, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.set_axisbelow(True)
    ax.grid(color="#E8EBED", linewidth=0.4)


def make_figure(cities, curves, populations, study_areas, city_lines, exposure_data):
    "Arrange five map insets above a 3-plus-2 curve layout with shared outer margins."
    set_style()
    fig = plt.figure(figsize=FORMAL_FIGSIZE, dpi=FORMAL_DPI)
    # Retain the map extent and aspect within the manuscript figure dimensions.
    outer = fig.add_gridspec(2, 1, height_ratios=FORMAL_HEIGHT_RATIOS,
                            left=0.10, right=0.97, bottom=0.08, top=0.965, hspace=FORMAL_OUTER_GAP)
    # Derive the lower-row height from the physical axes aspect.
    panel_aspect = FORMAL_CURVE_ASPECT
    row_gap, col_gap = FORMAL_CURVE_ROW_GAP, FORMAL_CURVE_COL_GAP
    lower_area = outer[1].get_position(fig)
    panel_width = lower_area.width / (3 + 2 * col_gap)
    panel_height = panel_width * panel_aspect * fig.get_figwidth() / fig.get_figheight()
    lower_bottom = lower_area.y1 - panel_height * (2 + row_gap)
    lower = fig.add_gridspec(2, 3, left=lower_area.x0, right=lower_area.x1,
                            top=lower_area.y1, bottom=lower_bottom,
                            hspace=row_gap, wspace=col_gap)
    map_ax = fig.add_subplot(outer[0])
    map_ax.set_gid("study_map")
    draw_study_map(map_ax, cities, study_areas, city_lines, exposure_data)
    slots = list(FORMAL_CURVE_POSITIONS.values()) + [FORMAL_CURVE_LEGEND_POSITION]
    if (set(FORMAL_CURVE_POSITIONS) != set(cities) or len(set(slots)) != 6
            or set(slots) != {(row, col) for row in range(2) for col in range(3)}):
        raise ValueError()
    for index, city in enumerate(cities):
        row, col = FORMAL_CURVE_POSITIONS[city]
        ax = fig.add_subplot(lower[row, col])
        ax.set_box_aspect(panel_aspect)
        ax.set_anchor("N")
        draw_curve_panel(ax, city, curves, populations[city], chr(98 + index))
        ax.set_ylabel("Pop share" if col == 0 else "", fontsize=7, labelpad=2.5)
        ax.set_xlabel("SAR" if row == 1 else "", fontsize=7, labelpad=2.5)
    legend_row, legend_col = FORMAL_CURVE_LEGEND_POSITION
    legend_ax = fig.add_subplot(lower[legend_row, legend_col])
    legend_ax.set_gid("scenario_legend")
    legend_ax.set_box_aspect(panel_aspect)
    legend_ax.set_anchor("N")
    legend_ax.axis("off")
    scenario_handles = [Line2D([], [], color=SCENARIO_COLORS[period],
                               linestyle=EXCEEDANCE_LINESTYLES[period], linewidth=1.4,
                               label=f"{period}-year") for period in PERIODS]
    legend_ax.legend(handles=scenario_handles, loc=FORMAL_CURVE_LEGEND_LOC,
                     bbox_to_anchor=FORMAL_CURVE_LEGEND_ANCHOR,
                     bbox_transform=legend_ax.transAxes, borderaxespad=0, ncol=1,
                     fontsize=FORMAL_LEGEND_FONT_SIZE, handlelength=2.0,
                     labelspacing=0.7, frameon=False)
    fig.canvas.draw()
    print("Lower layout: Zhengzhou, Hangzhou, Fuzhou / Chengdu, Wuhan, legend; legend font: "
          f"{FORMAL_LEGEND_FONT_SIZE:g} pt", flush=True)
    return fig


