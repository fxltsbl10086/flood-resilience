"Population bars and population-weighted SAR curves, one panel per city."

import argparse
import csv
from math import ceil, isfinite
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import matplotlib

# Use a noninteractive backend for command-line and batch exports.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from fig2_config import (
    CITIES, CITY_INPUTS, PERIODS, PLOT_COLORS, PLOT_DATA_CSV,
    PLOT_DPI, PLOT_FIGSIZE, PLOT_FILENAME, PLOT_FONT,
)


def set_style():
    "Set the configured font, white background and line widths."
    font_manager.fontManager.addfont(str(PLOT_FONT))
    font_name = font_manager.FontProperties(fname=str(PLOT_FONT)).get_name()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def read_plot_data(path, cities):
    "Read the frozen plotting table by city and return period."
    columns = ["exposed_valid_population", "nonexposed_valid_population",
               "exposed_population_weighted_SAR", "nonexposed_population_weighted_SAR"]
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        # Map the historical NAR column alias to SAR at the input boundary.
        names = {name.replace("_NAR", "_SAR") for name in (reader.fieldnames or [])}
        missing = {"city", "scenario", *columns} - names
        if missing:
            raise ValueError()
        for row in reader:
            for group in ("exposed", "nonexposed"):
                old_name = f"{group}_population_weighted_NAR"
                new_name = f"{group}_population_weighted_SAR"
                if new_name not in row and old_name in row:
                    row[new_name] = row.pop(old_name)
            key = (row["city"], row["scenario"])
            if key in records:
                raise ValueError()
            records[key] = row

    data = {}
    for city in cities:
        data[city] = {column: [] for column in columns}
        for period in PERIODS:
            key = (city, f"T{period}")
            if key not in records:
                raise ValueError()
            for column in columns:
                text = records[key][column]
                # Preserve missing risk as a line gap.
                if text == "" and column.endswith("SAR"):
                    value = float("nan")
                else:
                    value = float(text)
                    if not isfinite(value) or value < -1e-10:
                        raise ValueError()
                    if column.endswith("SAR") and value > 1 + 1e-10:
                        raise ValueError()
                data[city][column].append(value)
    return data


def draw_city(ax, city, values, population_limit, panel_label, compact=False,
              english=False, scenario_colors=None, font_size=None):
    "Use SAR on the left and baseline-valid population on the right; compact panels are map insets."
    tick_size = font_size if font_size is not None else (5.2 if compact else 9)
    label_size = tick_size if compact else 10
    label_pad = 2 if compact else 5
    positions = list(range(len(PERIODS)))
    if compact:
        # Use a separate tick axis and place bars within the same plot area.
        population_ax = ax.secondary_yaxis("right", functions=(
            lambda value: value * population_limit,
            lambda value: value / population_limit,
        ))
    else:
        population_ax = ax.twinx()
    width = 0.32
    for group, offset in [("exposed", -0.18), ("nonexposed", 0.18)]:
        unit = 1000000 if english else 10000
        population = [number / unit for number in values[f"{group}_valid_population"]]
        bar_ax = ax if compact else population_ax
        heights = [value / population_limit for value in population] if compact else population
        colors = [scenario_colors[period] for period in PERIODS] if scenario_colors else PLOT_COLORS[group]
        alpha = (0.65 if group == "exposed" else 0.22) if scenario_colors else (0.55 if compact else 0.45)
        hatch = "///" if scenario_colors and group == "nonexposed" else None
        bar_ax.bar([x + offset for x in positions], heights, width=width,
                   color=colors, alpha=alpha, hatch=hatch,
                   edgecolor=colors, linewidth=0.4 if compact else 0.7, zorder=2)
    if not compact:
        population_ax.set_ylim(0, population_limit)
    population_ax.yaxis.set_major_locator(MaxNLocator(nbins=2 if compact else 4))
    population_ax.set_ylabel("Population (M)" if english else "Population (10,000)",
                             fontsize=label_size, labelpad=label_pad)
    population_ax.tick_params(axis="y", labelsize=tick_size, length=1.5 if compact else 3, pad=1 if compact else 3.5)
    population_ax.spines["top"].set_visible(False)
    ax.set_xticks(positions, [str(period) for period in PERIODS])
    ax.set_xlim(-0.6, len(PERIODS) - 0.4)
    ax.set_xlabel("Return period (yr)" if english else "Return period (yr)",
                 fontsize=label_size, labelpad=label_pad)
    if not compact:
        ax.set_title(f"{panel_label}  {CITY_INPUTS[city]['CITY_NAME']}", loc="left",
                     fontsize=12, fontweight="bold", pad=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8EBED", linewidth=0.35 if compact else 0.6)
    ax.tick_params(axis="both", labelsize=tick_size, length=1.5 if compact else 3, pad=1 if compact else 3.5)

    for group, marker, linestyle in [("exposed", "o", "-"), ("nonexposed", "s", "--")]:
        risks = values[f"{group}_population_weighted_SAR"]
        ax.plot(positions, risks,
                color="#454C51" if scenario_colors else PLOT_COLORS[group],
                marker=None if scenario_colors else marker, linestyle=linestyle,
                markersize=3 if compact else 4.6, linewidth=1.15 if compact else 1.8,
                markeredgecolor="white", markeredgewidth=0.25 if compact else 0.6, zorder=4)
        if scenario_colors:
            # Distinguish exposure groups with line and marker styles.
            ax.scatter(positions, risks, c=[scenario_colors[period] for period in PERIODS],
                       marker=marker, s=5, edgecolors="white", linewidths=0.2, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1] if compact else [0, 0.25, 0.5, 0.75, 1])
    ax.set_ylabel("SAR" if english else "Population-weighted SAR",
                   fontsize=label_size, labelpad=label_pad)
    ax.spines["top"].set_visible(english)
    ax.spines["right"].set_visible(english or not compact)
    # Place the risk axis above the population bars with a transparent background.
    ax.set_zorder(population_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    return population_ax


def draw_preview(data, cities):
    "Use common limits for the five city panels and the sixth slot for the legend."
    set_style()
    # Reserve the sixth slot of the 2-by-3 layout for the shared legend.
    columns = min(3, len(cities) + 1)
    rows = ceil((len(cities) + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=PLOT_FIGSIZE, squeeze=False)
    axes = list(axes.flat)
    largest = max(number / 10000 for city in cities for group in ("exposed", "nonexposed")
                  for number in data[city][f"{group}_valid_population"])
    population_limit = ceil(max(largest * 1.08, 1) / 200) * 200
    for index, city in enumerate(cities):
        draw_city(axes[index], city, data[city], population_limit, chr(97 + index))

    legend_ax = axes[len(cities)]
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=PLOT_COLORS["exposed"], alpha=0.35, label="Valid directly exposed population"),
        Patch(facecolor=PLOT_COLORS["nonexposed"], alpha=0.35, label="Valid population without direct exposure*"),
        Line2D([], [], color=PLOT_COLORS["exposed"], marker="o", linewidth=1.8,
               label="Directly exposed: population-weighted SAR"),
        Line2D([], [], color=PLOT_COLORS["nonexposed"], marker="s", linestyle="--",
               linewidth=1.8, label="Without direct exposure*: population-weighted SAR"),
    ]
    legend_ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.02),
                     fontsize=10, handlelength=2.7, labelspacing=1.2, borderaxespad=0)
    legend_ax.text(0, 0.42,
                   "* Includes valid population without direct exposure,\n  without an SAR>0 filter.\n"
                   "Bars: known group population in baseline-valid grids.\n"
                   "Lines: population-weighted SAR for valid baseline grids.\n"
                   "Exposure threshold: depth >= 0.30 m.",
                   transform=legend_ax.transAxes, va="top", fontsize=9,
                   color="#545B61", linespacing=1.8)
    for ax in axes[len(cities) + 1:]:
        ax.axis("off")
    fig.suptitle(f"Population exposure and SAR across return periods in {len(cities)} cities", fontsize=15, y=0.97)
    fig.subplots_adjust(left=0.065, right=0.94, bottom=0.09, top=0.88,
                        wspace=0.68, hspace=0.52)
    return fig


