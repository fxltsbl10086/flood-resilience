"""Draw Supplementary Fig. 1 from the production SFINCS and EFAR inputs."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import rowcol
from shapely.geometry import shape

from shared_context import ROOT as PROJECT_ROOT, DATA, settings
SETTINGS=settings('S1')

CITIES = (
    ("zhengzhou", "Zhengzhou"),
    ("hangzhou", "Hangzhou"),
    ("fuzhou", "Fuzhou"),
    ("chengdu", "Chengdu"),
    ("wuhan", "Wuhan"),
)
RETURN_PERIODS = (10, 20, 50, 100, 200)
DOMAIN_EDGE = "#617888"
ROAD_COLOR = "#c7cdd2"
WATER_COLOR = "#7aaac8"
ROAD_FILTER = "fclass IN ('motorway', 'trunk', 'primary')"
WATER_BODY_MIN_AREA_M2 = 500_000
MINUTE_STEPS = np.array([1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120])
MAP_FONT = SETTINGS["font_family"]
FRAME_COLOR = "#000000"
STUDY_EDGE = "#303b44"
STUDY_DASH = (0, (3, 2))
FIGURE_NAME = "Supplementary_Figure_1"


@dataclass(frozen=True)
class FacilityStyle:
    key: str
    source: str
    label: str
    color: str
    marker: str
    size: float
    opacity: float


RESCUE = FacilityStyle("rescue", "应急救援", "Emergency rescue", "#315e92", "^", 3.3, 0.79)
HEALTH = FacilityStyle("medical", "医疗资源", "Healthcare", "#bd642d", "o", 1.9, 0.64)
SHELTER = FacilityStyle("shelter", "避险转移", "Emergency shelter", "#15735f", "D", 9.0, 0.93)
FACILITIES = (RESCUE, HEALTH, SHELTER)


def load_expected_counts():
    return {}

def load_city(code, expected):
    path=DATA/f's1_layers_{code}.gpkg'
    read=lambda name:gpd.read_file(path,layer=name).geometry
    return read('domain'),read('study'),{f.key:read(f.key) for f in FACILITIES},{'city':code}

def read_reference_layers(city, study):
    path=DATA/f's1_layers_{city}.gpkg'
    return tuple(gpd.read_file(path,layer=name).geometry for name in ('roads','rivers','water_outlines'))

def set_map_extent(ax, domain: gpd.GeoSeries, study: gpd.GeoSeries, points: dict) -> None:
    layers = [domain, study, *points.values()]
    bounds = Bbox.union([Bbox.from_extents(*layer.total_bounds) for layer in layers])
    position = ax.get_position(original=True)
    figure_width, figure_height = ax.figure.get_size_inches()
    aspect = (position.width * figure_width) / (position.height * figure_height)
    width = max(1.04 * bounds.width, 1.18 * bounds.height * aspect)
    height = width / aspect
    center_x = (bounds.xmin + bounds.xmax) / 2
    center_y = (bounds.ymin + bounds.ymax) / 2 - 0.038 * height
    ax.set(xlim=(center_x - width / 2, center_x + width / 2),
           ylim=(center_y - height / 2, center_y + height / 2), aspect="equal")


def geographic_ticks(first: np.ndarray, opposite: np.ndarray, inset: float) -> np.ndarray:
    """Use the Result 3 minute intervals over the range common to both edges."""
    low = max(first[0], opposite[0])
    high = min(first[-1], opposite[-1])
    span = high - low
    step = MINUTE_STEPS[MINUTE_STEPS >= span * 60 / 3][0]
    start = int(np.ceil((low + span * inset) * 60 / step))
    end = int(np.floor((high - span * inset) * 60 / step))
    return np.arange(start, end + 1) * step / 60


def degree_labels(values: np.ndarray, positive: str, negative: str) -> list[str]:
    result = []
    for value in values:
        degrees, minutes = divmod(round(abs(value) * 60), 60)
        hemisphere = positive if value >= 0 else negative
        result.append(f"{degrees}°{minutes:02d}′{hemisphere}")
    return result


def draw_coordinates(ax, crs) -> tuple:
    """Locate each WGS84 coordinate where it intersects the specific UTM frame edge."""
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    x = np.linspace(left, right, 129)
    y = np.linspace(bottom, top, 129)
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_bottom, _ = to_wgs84.transform(x, np.full_like(x, bottom))
    lon_top, _ = to_wgs84.transform(x, np.full_like(x, top))
    _, lat_left = to_wgs84.transform(np.full_like(y, left), y)
    _, lat_right = to_wgs84.transform(np.full_like(y, right), y)
    longitudes = geographic_ticks(lon_bottom, lon_top, 0.08)
    latitudes = geographic_ticks(lat_left, lat_right, 0.14)
    long_labels = degree_labels(longitudes, "E", "W")
    lat_labels = degree_labels(latitudes, "N", "S")

    top_axis = ax.secondary_xaxis("top")
    right_axis = ax.secondary_yaxis("right")
    # Use a separate axis on each opposite edge so its ticks use its own intersections.
    ax.set_xticks(np.interp(longitudes, lon_bottom, x), long_labels)
    top_axis.set_xticks(np.interp(longitudes, lon_top, x), long_labels)
    ax.set_yticks(np.interp(latitudes, lat_left, y), lat_labels)
    right_axis.set_yticks(np.interp(latitudes, lat_right, y), lat_labels)
    for coordinate_axis, axis in ((ax, "x"), (top_axis, "x"), (ax, "y"), (right_axis, "y")):
        coordinate_axis.tick_params(
            axis=axis, length=2.5, width=0.55, pad=1.6, labelsize=7, color=FRAME_COLOR,
            labelcolor=FRAME_COLOR, labelrotation=0 if axis == "x" else 90,
            labelfontfamily=MAP_FONT, direction="out",
        )
    for label in ax.get_yticklabels() + right_axis.get_yticklabels():
        label.set_va("center")
    for spine in ax.spines.values():
        spine.set(linewidth=0.55, edgecolor=FRAME_COLOR)
    top_axis.spines["top"].set_visible(False)
    right_axis.spines["right"].set_visible(False)
    return top_axis, right_axis


def draw_scale_bar(ax) -> None:
    """Draw a lower-right scale in the map's metre-based UTM coordinates."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x, span_y = xmax - xmin, ymax - ymin
    length_km = next(km for km in (100, 50, 20, 10, 5, 2, 1) if km * 1000 <= span_x * 0.25)
    end = xmax - span_x * 0.09
    start = end - length_km * 1000
    y = ymin + span_y * 0.038
    ax.plot([start, end], [y, y], color=FRAME_COLOR, linewidth=0.9,
            solid_capstyle="butt", zorder=10)
    ax.vlines([start, (start + end) / 2, end], y - span_y * 0.006, y + span_y * 0.008,
              colors=FRAME_COLOR, linewidth=0.7, zorder=10)
    for x, label in ((start, "0"), (end, f"{length_km} km")):
        ax.text(x, y + span_y * 0.014, label, ha="center", va="bottom",
                color=FRAME_COLOR, fontsize=7, fontfamily=MAP_FONT, zorder=11)


def draw_map(ax, panel: str, name: str, domain: gpd.GeoSeries,
             study: gpd.GeoSeries, points: dict, context: tuple[gpd.GeoSeries, ...]) -> tuple:
    roads, rivers, water_outlines = context
    for layer, color, linewidth in (
        (roads, ROAD_COLOR, 0.26), (rivers, WATER_COLOR, 0.34), (water_outlines, WATER_COLOR, 0.34)
    ):
        if not layer.empty:
            layer.plot(ax=ax, color=color, linewidth=linewidth, zorder=1)
    domain.boundary.plot(ax=ax, color=DOMAIN_EDGE, linewidth=0.55, zorder=2)
    study.boundary.plot(ax=ax, color=STUDY_EDGE, linewidth=0.65,
                        linestyle=STUDY_DASH, zorder=3)

    for zorder, style in enumerate((HEALTH, RESCUE, SHELTER), start=4):
        geometry = points[style.key]
        outline = "white" if style is SHELTER else "none"
        ax.scatter(geometry.x, geometry.y, s=style.size, marker=style.marker,
                   c=style.color, alpha=style.opacity, linewidths=0.28,
                   edgecolors=outline, zorder=zorder)

    set_map_extent(ax, domain, study, points)
    coordinate_axes = draw_coordinates(ax, domain.crs)
    draw_scale_bar(ax)

    # Match the main Result 3 map typography, with space for the additional top ticks.
    ax.annotate(panel, (0, 1), xycoords="axes fraction",
                xytext=(-6, 16), textcoords="offset points", va="baseline",
                fontsize=10, fontfamily=MAP_FONT, fontweight="bold", color="black",
                annotation_clip=False, clip_on=False)
    ax.text(0.025, 0.015, name, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, fontfamily=MAP_FONT, color="black", zorder=10)
    return coordinate_axes


def draw_legend(ax) -> None:
    ax.set_axis_off()
    handles = [
        Line2D([], [], color=DOMAIN_EDGE, linewidth=0.75, label="SFINCS domain boundary"),
        Line2D([], [], color=STUDY_EDGE, linestyle=STUDY_DASH, linewidth=0.85,
               label="Study-area boundary"),
        *[
            Line2D([], [], marker=style.marker, color=style.color, linestyle="none",
                   markersize=4.2, markeredgewidth=0, label=style.label)
            for style in FACILITIES
        ],
        Line2D([], [], color=ROAD_COLOR, linewidth=0.65, label="Major roads"),
        Line2D([], [], color=WATER_COLOR, linewidth=0.65, label="Major water bodies"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.015, 0.55),
              fontsize=7.15, handletextpad=0.70, labelspacing=0.66, handlelength=1.9,
              borderaxespad=0, frameon=False, labelcolor=FRAME_COLOR)
    ax.annotate("", xy=(0.05, 0.17), xytext=(0.05, 0.083),
                arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": FRAME_COLOR})
    ax.text(0.11, 0.125, "Grid north", va="center", color=FRAME_COLOR, fontsize=6.6)


def validate_coordinates(fig, panels: list[tuple]) -> None:
    """Reject coordinate labels that are empty, overlap, or extend off the canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for ax, (top_axis, right_axis) in panels:
        edge_labels = (ax.get_xticklabels(), ax.get_yticklabels(),
                       top_axis.get_xticklabels(), right_axis.get_yticklabels())
        for labels in edge_labels:
            if len(labels) < 2:
                raise ValueError()
            boxes.extend(label.get_window_extent(renderer) for label in labels)
        boxes.extend(label.get_window_extent(renderer) for label in ax.texts)
    for i, current in enumerate(boxes):
        if not (fig.bbox.contains(current.x0, current.y0)
                and fig.bbox.contains(current.x1, current.y1)):
            raise ValueError()
        if any(current.overlaps(other) for other in boxes[i + 1:]):
            raise ValueError()


