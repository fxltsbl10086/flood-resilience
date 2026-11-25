"Five-city, five-scenario, three-function population-weighted violin figure."
import argparse
from datetime import datetime
from pathlib import Path
import json
from s4_config import (HERE, OUTPUT_DIR, CITIES, CITY_ABBREVIATIONS, DIMENSIONS, FUNCTION_NAMES,
                    FUNCTION_COLORS, PERIODS, FONT_FAMILY, FONT_PATH, FIGURE_SIZE_MM, PANEL_ASPECT, DPI,
                    RESULT1_STYLE_CONFIG, RESULT1_STYLE_SCRIPT,
                    FIGURE_STEM, KDE_BANDWIDTH, FUNCTION_OFFSETS, MAX_HALF_WIDTH, COVERAGE_CSV,
                    DENSITY_WIDTH_POWER, VIOLIN_ALPHA, LAYOUT_MM,
                    LEGEND_FONT_SIZE, PANEL_POSITIONS, LEGEND_POSITION, LEGEND_LOC, LEGEND_ANCHOR,
                    BASE_FONT_SIZE, TICK_FONT_SIZE,
                    AXIS_FONT_SIZE, CITY_FONT_SIZE, PANEL_LABEL_FONT_SIZE, AXIS_LABEL_PAD)
from shared_manifest import signature
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

def draw(y, densities, stats, output):
    for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
        
        if FONT_PATH.with_name(name).is_file():
            font_manager.fontManager.addfont(str(FONT_PATH.with_name(name)))
    font_path = str(FONT_PATH)
    plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': BASE_FONT_SIZE, 'axes.linewidth': 0.65,
        'axes.edgecolor': 'black', 'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.labelsize': TICK_FONT_SIZE, 'ytick.labelsize': TICK_FONT_SIZE, 'axes.unicode_minus': False,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figure_width, figure_height = FIGURE_SIZE_MM
    panel_width = (figure_width - LAYOUT_MM['left_margin'] - LAYOUT_MM['right_margin']
                   - 2 * LAYOUT_MM['column_gap']) / 3
    panel_height = panel_width * PANEL_ASPECT
    fig = plt.figure(figsize=tuple(mm / 25.4 for mm in FIGURE_SIZE_MM), dpi=DPI)
    axes = {}
    for row in range(2):
        bottom = figure_height - LAYOUT_MM['row_top_offsets'][row] - panel_height
        for col in range(3):
            left = LAYOUT_MM['left_margin'] + col * (panel_width + LAYOUT_MM['column_gap'])
            axes[(row, col)] = fig.add_axes([left / figure_width, bottom / figure_height,
                                            panel_width / figure_width, panel_height / figure_height])
    positions = list(PANEL_POSITIONS.values())
    if (set(PANEL_POSITIONS) != set(CITIES) or len(set(positions)) != len(CITIES)
            or LEGEND_POSITION in positions
            or set(positions) | {LEGEND_POSITION} != set(axes)):
        raise ValueError()
    global_peak = max(float(d.max()) for d in densities.values())
    indexed = stats.set_index(['city', 'function', 'return_period'])
    city_titles = {}
    for panel, city in enumerate(CITIES):
        row, col = PANEL_POSITIONS[city]
        ax = axes[(row, col)]
        ax.set_box_aspect(PANEL_ASPECT)
        ax.set_anchor('N')
        city_stats = stats.loc[stats.city == city]
        lower = float(city_stats.display_y_min.iloc[0])
        upper = float(city_stats.display_y_max.iloc[0])
        visible = (y >= lower) & (y <= upper)
        for group, period in enumerate(PERIODS):
            for dimension, offset in zip(DIMENSIONS, FUNCTION_OFFSETS):
                center = group + offset
                density = densities[(city, dimension, period)]
                width = MAX_HALF_WIDTH * (density / global_peak) ** DENSITY_WIDTH_POWER
                color = FUNCTION_COLORS[dimension]
                # Fill the true density width without outlining very small tails.
                ax.fill_betweenx(y[visible], center - width[visible], center + width[visible],
                                 facecolor=color, edgecolor='none', linewidth=0, alpha=VIOLIN_ALPHA, zorder=2)
                q = indexed.loc[(city, FUNCTION_NAMES[dimension], period)]
                ax.plot([center, center], [q.P25, q.P75], color='#222222', linewidth=0.85,
                        solid_capstyle='butt', zorder=3)
                ax.scatter([center], [q['median']], s=6, color='white', edgecolor='#222222',
                           linewidth=0.5, zorder=4, clip_on=False)
        ax.set_xlim(-0.55, 4.55)
        ax.set_ylim(lower, upper)
        ax.set_xticks(range(5), [f'T{period}' for period in PERIODS])
        ax.set_yticks([lower, (lower + upper) / 2, upper])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.tick_params(direction='out', pad=3.5)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E8EBED', linewidth=0.4)
        ax.set_xlabel('')
        ax.set_ylabel('EFAR' if col == 0 else '', fontsize=AXIS_FONT_SIZE, labelpad=AXIS_LABEL_PAD)
        # Keep city labels equally spaced below the ticks in both rows.
        title_y = -LAYOUT_MM['city_title_gap'][row] / panel_height
        city_titles[city] = ax.text(0.5, title_y, CITY_ABBREVIATIONS[city], transform=ax.transAxes,
                fontsize=CITY_FONT_SIZE, ha='center', va='top', linespacing=1.0)
        ax.annotate(chr(ord('a') + panel), xy=(0, 1), xycoords='axes fraction',
                    xytext=(-6, 5), textcoords='offset points', fontsize=PANEL_LABEL_FONT_SIZE,
                    fontweight='bold', ha='left', va='baseline', annotation_clip=False,
                    clip_on=False, zorder=20)
    handles = [Patch(facecolor=FUNCTION_COLORS[d], edgecolor=FUNCTION_COLORS[d], alpha=VIOLIN_ALPHA,
                     linewidth=0.5, label=FUNCTION_NAMES[d]) for d in DIMENSIONS]
    legend_ax = axes[LEGEND_POSITION]
    legend_ax.set_axis_off()
    legend = legend_ax.legend(handles=handles, loc=LEGEND_LOC,
                              bbox_to_anchor=LEGEND_ANCHOR, bbox_transform=legend_ax.transAxes,
                              borderaxespad=0, ncol=1, fontsize=LEGEND_FONT_SIZE,
                              handlelength=2.0, handletextpad=0.8, labelspacing=0.7, frameon=False)
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    legend_bounds_mm = [float(v) / fig.dpi * 25.4 for v in legend_box.bounds]
    print(f'Legend in lower-right slot; size: {LEGEND_FONT_SIZE:g} pt; anchor: {LEGEND_ANCHOR}', flush=True)
    path = output / f'{FIGURE_STEM}.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, {'font_family': FONT_FAMILY, 'font_file': font_path, 'size_mm': FIGURE_SIZE_MM,
        'dpi': DPI, 'formats': ['png'], 'panel_order': CITIES, 'function_colors': FUNCTION_COLORS,
        'layout': '2 rows × 3 columns', 'panel_aspect': PANEL_ASPECT,
        'layout_mm': LAYOUT_MM, 'panel_size_mm': [panel_width, panel_height],
        'panel_positions': PANEL_POSITIONS,
        'legend_font_size_pt': LEGEND_FONT_SIZE, 'legend_position': LEGEND_POSITION,
        'legend_loc': LEGEND_LOC, 'legend_anchor_axes_fraction': LEGEND_ANCHOR,
        'legend_columns': 1, 'legend_bounds_mm': legend_bounds_mm,
        'result1_style_source': signature(RESULT1_STYLE_CONFIG),
        'result1_style_script': signature(RESULT1_STYLE_SCRIPT),
        'font_sizes_pt': {'base': BASE_FONT_SIZE, 'ticks': TICK_FONT_SIZE, 'axis_labels': AXIS_FONT_SIZE,
                          'city_labels': CITY_FONT_SIZE, 'panel_letters': PANEL_LABEL_FONT_SIZE,
                          'legend': LEGEND_FONT_SIZE},
        'axis_label': 'EFAR', 'y_tick_format': 'raw EFAR values, not population percentages',
        'axis_label_policy': 'EFAR on left column only; no x-axis title; numeric and scenario ticks on all panels',
        'global_peak_density': global_peak, 'maximum_half_width_category_units': MAX_HALF_WIDTH,
        'width_rule': 'width=MAX_HALF_WIDTH*(density/global_peak_density)**DENSITY_WIDTH_POWER across all displayed 75 distributions',
        'density_width_power': DENSITY_WIDTH_POWER, 'fill_alpha': VIOLIN_ALPHA,
        'width_interpretation': 'Common square-root density scale; widths are not linear density ratios',
        'city_title_range_annotation': False, 'point_line_note_in_figure': False,
        'population_distribution_count': len(densities), 'titles': 'below each subplot',
        'city_title_abbreviations': {city: CITY_ABBREVIATIONS[city] for city in CITIES},
        'city_y_ranges': {c: [float(stats.loc[stats.city == c, 'display_y_min'].iloc[0]),
                               float(stats.loc[stats.city == c, 'display_y_max'].iloc[0])] for c in CITIES},
        'quantile_interval': 'weighted empirical P25–P75, not a confidence interval'}

