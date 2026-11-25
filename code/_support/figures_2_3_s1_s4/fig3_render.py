"""Formal Result 2: distance-response panels above full-population loss decomposition.

Render the frozen source tables into one six-panel PNG, without editing either
original figure directory or recalculating the model.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
_dll_handles = []
_library = Path(sys.prefix) / 'Library/bin'
if os.name == 'nt' and _library.is_dir():
    os.environ['PATH'] = str(_library) + os.pathsep + os.environ.get('PATH', '')
    _dll_handles.append(os.add_dll_directory(str(_library)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import fig3_radial as radial
from shared_context import DATA, settings, font_path
SETTINGS=settings("fig3")
OUTPUT_DIR=None

FIGURE_MM = (183.0, 130.0)
DPI = int(SETTINGS["dpi"])
COLUMN_CENTRES_MM = (34.0, 93.0, 152.0)
UPPER_BOTTOM_MM = 75.0
UPPER_WIDTH_MM = 46.0
UPPER_HEIGHT_MM = 43.0
UPPER_XLABEL_Y_MM = 68.8
LOWER_CENTRE_Y_MM = 37.0
LOWER_LETTER_Y_MM = 65.0
LEGEND_Y_MM = 5.5
LEGEND_CONTENT_GAP_MM = 2.5
EFAR_LABEL_X_MM = 4.4
INSET_LEFT_OFFSET_MM = 24.0
INSET_BOTTOM_OFFSET_MM = 3.0
INSET_WIDTH_MM = 20.0
INSET_HEIGHT_MM = 19.0
SCENARIO_COLORS = {10: '#4C78A8', 20: '#59A14F', 50: '#E3B34C',
                   100: '#E17C45', 200: '#A05C99'}
SCENARIO_STYLES = {10: ':', 20: '--', 50: '-.', 100: '--', 200: '-'}
STATUS_COLORS = {'Disconnected': radial.LOSS_COLORS['disconnection'], 'Reachable': '#E5E5E5'}
FIGURE_NAME = 'fig3.png'
SOURCE_NAMES = ['EFAR_link_balance_source_data.csv',
                'EFAR_stacked_connectivity_source_data.csv',
                'Radial_figure_source_data.csv']
require = radial.require


def signature(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load_data():
    provenance_path = DATA/'fig3_input_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    hashes = {row['file']: row['sha256'] for row in provenance['source_files']}
    frames = []
    for name in SOURCE_NAMES:
        path = DATA/('fig3_'+name)
        require(signature(path)['sha256'] == hashes[name], '')
        frames.append(pd.read_csv(path, float_precision='round_trip'))
    curves, inset, loss = frames
    require(not curves.duplicated(['function', 'period', 'bin_id']).any(), '')
    require(len(inset) == 15 and not inset.duplicated(['function', 'period']).any(), '')
    counts = ['baseline_link_count', 'disconnected_link_count', 'reachable_link_count']
    grouped = curves.groupby(['function', 'period'])[counts].sum().sort_index()
    reference = inset.set_index(['function', 'period']).sort_index()
    require(np.array_equal(grouped.to_numpy(), reference[counts].to_numpy()), '')
    require(np.array_equal(inset.disconnected_link_count + inset.reachable_link_count,
                           inset.baseline_link_count), '')
    for state in ('disconnected', 'reachable'):
        expected = inset[f'{state}_link_count']/inset.baseline_link_count
        require(np.allclose(expected, inset[f'{state}_share'], rtol=0, atol=1e-12), '')
    require(len(loss) == 75 and not loss.duplicated(['city', 'function', 'period']).any(),
            '')
    require(np.array_equal(loss.grid_count, loss.analysis_grid_count)
            and np.allclose(loss.population, loss.analysis_population, rtol=0, atol=1e-6),
            '')
    require((loss.groupby('city')[['population', 'grid_count']].nunique() == 1).all().all(),
            '')
    require(np.array_equal(loss.reachable_grid_count + loss.disconnected_grid_count, loss.grid_count)
            and np.allclose(loss.reachable_population + loss.disconnected_population,
                            loss.population, rtol=0, atol=1e-6),
            '')
    require(np.allclose(loss.disconnection_loss + loss.detour_loss, loss.mean_EFAR,
                        rtol=0, atol=1e-9), '')
    require(np.allclose(loss.disconnection_share,
                        loss.disconnection_loss/(loss.disconnection_loss + loss.detour_loss),
                        rtol=0, atol=1e-12), '')
    return curves, inset, loss, provenance


def style():
    for name in ('times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'):
        pass # Matplotlib supplies the configured portable font.
    font_path(SETTINGS['font_family'])
    plt.rcParams.update({'font.family': SETTINGS['font_family'], 'font.size': 7.5,
                         'axes.labelsize': 7.5, 'axes.linewidth': .6,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'xtick.major.width': .6, 'ytick.major.width': .6,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'xtick.major.pad': 2, 'ytick.major.pad': 2,
                         'legend.fontsize': 7.5, 'legend.frameon': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'lines.solid_capstyle': 'butt'})


def require_clear_inset(x, y, xmax):
    xlo = xmax*INSET_LEFT_OFFSET_MM/UPPER_WIDTH_MM
    xhi = xmax*(INSET_LEFT_OFFSET_MM+INSET_WIDTH_MM)/UPPER_WIDTH_MM
    ylo = -.015+1.03*INSET_BOTTOM_OFFSET_MM/UPPER_HEIGHT_MM
    yhi = -.015+1.03*(INSET_BOTTOM_OFFSET_MM+INSET_HEIGHT_MM)/UPPER_HEIGHT_MM
    for i in range(len(x)-1):
        if not np.isfinite(y[i:i+2]).all():
            continue
        lo, hi = max(x[i], xlo), min(x[i+1], xhi)
        if lo <= hi:
            levels = np.interp([lo, hi], x[i:i+2], y[i:i+2])
            require(levels.min() > yhi or levels.max() < ylo, '')


def add_inset(fig, data, function, main_left, text_mm, *,
              figure_mm=FIGURE_MM, main_bottom_mm=UPPER_BOTTOM_MM):
    width, height = figure_mm
    left = main_left + INSET_LEFT_OFFSET_MM
    bottom = main_bottom_mm + INSET_BOTTOM_OFFSET_MM
    fig.add_artist(Rectangle((left/width, bottom/height), INSET_WIDTH_MM/width,
                            INSET_HEIGHT_MM/height, transform=fig.transFigure,
                            facecolor='white', edgecolor='black', linewidth=.35, zorder=4))
    text_mm(left+INSET_WIDTH_MM/2, bottom+INSET_HEIGHT_MM-1.5,
            'Link status (%)', fontsize=5.8, ha='center', va='center', zorder=6)
    ax = fig.add_axes([(left+5)/width, (bottom+2.5)/height,
                       (INSET_WIDTH_MM-7)/width, (INSET_HEIGHT_MM-5.2)/height],
                      facecolor='white', zorder=5)
    ax.set(xlim=(0, 100), ylim=(-.5, 4.5))
    ax.set_yticks([4, 3, 2, 1, 0], [f'T{p}' for p in radial.PERIODS])
    ax.set_xticks([0, 50, 100], ['0', '50', '100'])
    ax.tick_params(axis='both', labelsize=5.5, pad=1, width=.5, length=1.5, colors='black')
    ax.tick_params(axis='y', length=0)
    ax.spines[['left', 'top', 'right']].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(.5)
    checks = []
    for i, period in enumerate(radial.PERIODS):
        part = data.loc[data.function.eq(function) & data.period.eq(period)].iloc[0]
        d, r = 100*float(part.disconnected_share), 100*float(part.reachable_share)
        require(abs(d+r-100) < 1e-9, '')
        b1 = ax.barh(4-i, d, height=.63, color=STATUS_COLORS['Disconnected'], edgecolor='none')[0]
        b2 = ax.barh(4-i, r, left=d, height=.63, color=STATUS_COLORS['Reachable'], edgecolor='none')[0]
        require(np.allclose([b1.get_width(), b2.get_x(), b2.get_width()], [d, d, r], rtol=0, atol=1e-12),
                '')
        checks.append({'function': function, 'period': period, 'disconnected_pct': d, 'reachable_pct': r})
    return ax, checks


def draw_upper_panel(fig, curves, inset, function, left, bottom, text_mm, *,
                     figure_mm=FIGURE_MM, show_left=False):
    """Draw the shared distance curves and connectivity inset without changing values."""
    width, height = figure_mm
    ax = fig.add_axes([left/width, bottom/height,
                       UPPER_WIDTH_MM/width, UPPER_HEIGHT_MM/height])
    part = curves.loc[curves.function.eq(function)]
    max_km = float(part.bin_upper_m.max()/1000)
    ticks = ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, max_km)
    xmax = float(ticks[-1])
    ax.set(xlim=(0, xmax), ylim=(-.015, 1.015))
    ax.set_xticks(ticks)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E5E5E5', linewidth=.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(.6)
    ax.tick_params(top=False, right=False, labelleft=show_left)
    curve_checks = []
    for period in radial.PERIODS:
        frame = part.loc[part.period.eq(period)].sort_values('bin_id')
        x = frame.distance_midpoint_km.to_numpy(float)
        y = frame.mean_EFAR_population_weighted.to_numpy(float)
        require_clear_inset(x, y, xmax)
        line, = ax.plot(x, y, color=SCENARIO_COLORS[period], linestyle=SCENARIO_STYLES[period],
                        linewidth=1.2, zorder=3)
        require(np.array_equal(line.get_xdata(), x) and np.array_equal(line.get_ydata(), y, equal_nan=True), '')
        valid = np.isfinite(y)
        endpoints = valid & (~np.r_[False, valid[:-1]] | ~np.r_[valid[1:], False])
        ax.plot(x[endpoints], y[endpoints], linestyle='none', marker='o', markersize=2.6,
                markeredgewidth=.35, color=SCENARIO_COLORS[period], clip_on=False)
        curve_checks.append({'function': function, 'period': period, 'points': int(valid.sum())})
    inset_ax, checks = add_inset(fig, inset, function, left, text_mm,
                                 figure_mm=figure_mm, main_bottom_mm=bottom)
    return ax, inset_ax, curve_checks, checks


def draw(curves, inset, loss):
    style()
    width, height = FIGURE_MM
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    texts, tick_axes, curve_checks, inset_checks, radial_checks = [], [], [], [], []
    lower_texts = []

    def text_mm(x, y, label, **kwargs):
        artist = fig.text(x/width, y/height, label, **kwargs)
        texts.append(artist)
        return artist

    for i, function in enumerate(radial.FUNCTIONS):
        cx = COLUMN_CENTRES_MM[i]
        left = cx-UPPER_WIDTH_MM/2
        ax, inset_ax, curves_checked, checks = draw_upper_panel(
            fig, curves, inset, function, left, UPPER_BOTTOM_MM, text_mm, show_left=i == 0)
        tick_axes.extend([ax, inset_ax])
        curve_checks.extend(curves_checked)
        inset_checks.extend(checks)
        text_mm(left, UPPER_BOTTOM_MM+UPPER_HEIGHT_MM+2.2, chr(97+i),
                fontsize=10, fontweight='bold', ha='left', va='bottom')
        text_mm(cx, UPPER_BOTTOM_MM+UPPER_HEIGHT_MM+6.5, function,
                fontsize=8.5, ha='center', va='center')
        text_mm(cx, UPPER_XLABEL_Y_MM, 'Distance (km)', ha='center', va='center')

        half = radial.PANEL_HALF_MM
        lower = fig.add_axes([(cx-half)/width, (LOWER_CENTRE_Y_MM-half)/height,
                              2*half/width, 2*half/height])
        panel_texts, panel_checks = radial.draw_panel(lower, loss, function, radial.CENTRE_LABELS[i])
        texts.extend(panel_texts)
        lower_texts.extend(panel_texts)
        radial_checks.extend(panel_checks)
        text_mm(left, LOWER_LETTER_Y_MM, chr(100+i), fontsize=10,
                fontweight='bold', ha='left', va='center')

    text_mm(EFAR_LABEL_X_MM, UPPER_BOTTOM_MM+UPPER_HEIGHT_MM/2, 'EFAR', rotation=90,
            rotation_mode='anchor', ha='center', va='center')
    scenario_handles = [Line2D([], [], color=SCENARIO_COLORS[p], linestyle=SCENARIO_STYLES[p],
                               linewidth=1.4, label=f'{p}-year') for p in radial.PERIODS]
    status_handles = [Patch(facecolor=color, edgecolor='none', label=label)
                      for label, color in STATUS_COLORS.items()]
    loss_handles = [Patch(facecolor=radial.LOSS_COLORS[key], edgecolor='none', label=label)
                    for key, label in [('disconnection', 'Disconnection'), ('detour', 'Detour')]]
    leg = fig.legend(handles=scenario_handles+status_handles+loss_handles, loc='center',
                     bbox_to_anchor=(COLUMN_CENTRES_MM[1]/width, LEGEND_Y_MM/height), ncol=9,
                     columnspacing=.85, handlelength=1.6, handletextpad=.45, borderaxespad=0)
    texts.extend(leg.get_texts())
    for ax in tick_axes:
        texts.extend(t for t in ax.get_xticklabels()+ax.get_yticklabels() if t.get_visible())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Place the legend by the rendered lower labels, keeping a physical text gap.
    lower_text_bottom = min(radial.text_corners(t, renderer)[:, 1].min() for t in lower_texts)
    legend_box = leg.get_window_extent(renderer)
    target_top = lower_text_bottom - LEGEND_CONTENT_GAP_MM*fig.dpi/25.4
    legend_y_mm = LEGEND_Y_MM + (target_top-legend_box.y1)*25.4/fig.dpi
    leg.set_bbox_to_anchor((COLUMN_CENTRES_MM[1]/width, legend_y_mm/height))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t.get_text(), radial.text_corners(t, renderer)) for t in texts]
    for label, box in boxes:
        require(box[:, 0].min() >= 0 and box[:, 1].min() >= 0
                and box[:, 0].max() <= fig.bbox.x1 and box[:, 1].max() <= fig.bbox.y1,
                '')
    overlapping_labels = []
    for i, (label, box) in enumerate(boxes):
        for label2, box2 in boxes[i+1:]:
            if radial.overlaps(box, box2):
                overlapping_labels.append(f'{label}/{label2}')
    require(not overlapping_labels, '')
    require(len(curve_checks) == len(inset_checks) == 15, '')
    require(len(radial_checks) == 75, '')
    require(sum(c['disconnection_share_label'] is not None for c in radial_checks) == 30,
            '')
    path = OUTPUT_DIR/FIGURE_NAME
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, {'figure_mm': list(FIGURE_MM), 'dpi': DPI,
                  'font': SETTINGS['font_family'], 'main_font_pt': 7.5,
                  'column_centres_mm': list(COLUMN_CENTRES_MM),
                  'upper_axis_size_mm': [UPPER_WIDTH_MM, UPPER_HEIGHT_MM],
                  'upper_xlabel': 'Distance (km)', 'upper_xlabel_count': 3,
                  'upper_xlabel_y_mm': UPPER_XLABEL_Y_MM,
                  'lower_centre_y_mm': LOWER_CENTRE_Y_MM,
                  'legend_placement': 'One row at the bottom of the whole figure',
                  'legend_columns': 9, 'legend_y_mm': legend_y_mm,
                  'legend_content_gap_mm': LEGEND_CONTENT_GAP_MM,
                  'efar_label_x_mm': EFAR_LABEL_X_MM,
                  'panels': ['a', 'b', 'c', 'd', 'e', 'f'],
                  'upper_curves': curve_checks, 'inset_stacked_bars': inset_checks,
                  'radial_bars': radial_checks, 'top_explanatory_header': False,
                  'radial_population_scope': 'All common-valid positive-population analysis grids, including fully disconnected grids',
                  'text_bounds_and_overlap': 'passed', 'inset_occlusion': 'passed',
                  'data_to_glyph_values': 'passed'}


def draw_panel_b(curves, inset):
    """Export the healthcare distance panel with its own axes and relevant legends."""
    style()
    figure_mm = (80.0, 88.0)
    width, height = figure_mm
    centre, bottom = width/2, 33.0
    left = centre-UPPER_WIDTH_MM/2
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    texts = []

    def text_mm(x, y, label, **kwargs):
        artist = fig.text(x/width, y/height, label, **kwargs)
        texts.append(artist)
        return artist

    ax, inset_ax, curve_checks, inset_checks = draw_upper_panel(
        fig, curves, inset, 'Healthcare', left, bottom, text_mm,
        figure_mm=figure_mm, show_left=True)
    text_mm(left, bottom+UPPER_HEIGHT_MM+2.2, 'b', fontsize=10,
            fontweight='bold', ha='left', va='bottom')
    text_mm(centre, bottom+UPPER_HEIGHT_MM+6.5, 'Healthcare',
            fontsize=8.5, ha='center', va='center')
    text_mm(centre, bottom-6.2, 'Distance (km)', ha='center', va='center')
    text_mm(6.0, bottom+UPPER_HEIGHT_MM/2, 'EFAR', rotation=90,
            rotation_mode='anchor', ha='center', va='center')
    scenario_handles = [Line2D([], [], color=SCENARIO_COLORS[p], linestyle=SCENARIO_STYLES[p],
                               linewidth=1.4, label=f'{p}-year') for p in radial.PERIODS]
    status_handles = [Patch(facecolor=color, edgecolor='none', label=label)
                      for label, color in STATUS_COLORS.items()]
    scenario_legend = fig.legend(handles=scenario_handles, loc='center',
        bbox_to_anchor=(.5, 17.0/height), ncol=3, columnspacing=1.0,
        handlelength=1.6, handletextpad=.45, borderaxespad=0)
    status_legend = fig.legend(handles=status_handles, loc='center',
        bbox_to_anchor=(.5, 6.0/height), ncol=2, columnspacing=1.0,
        handlelength=1.6, handletextpad=.45, borderaxespad=0)
    texts.extend(scenario_legend.get_texts())
    texts.extend(status_legend.get_texts())
    for ticks in [ax, inset_ax]:
        texts.extend(t for t in ticks.get_xticklabels()+ticks.get_yticklabels() if t.get_visible())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(text.get_text(), radial.text_corners(text, renderer)) for text in texts]
    for label, box in boxes:
        require(box[:, 0].min() >= 0 and box[:, 1].min() >= 0
                and box[:, 0].max() <= fig.bbox.x1 and box[:, 1].max() <= fig.bbox.y1, '')
    require(not any(radial.overlaps(a, b) for i, (_, a) in enumerate(boxes)
                    for _, b in boxes[i+1:]), '')
    require(len(curve_checks) == len(inset_checks) == 5, '')
    path = OUTPUT_DIR/'fig3b.png'
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path, {'figure_mm': list(figure_mm), 'dpi': DPI, 'font': SETTINGS['font_family'],
                  'panels': ['b'], 'function': 'Healthcare',
                  'upper_axis_size_mm': [UPPER_WIDTH_MM, UPPER_HEIGHT_MM],
                  'upper_curves': curve_checks, 'inset_stacked_bars': inset_checks,
                  'legends': ['10-year', '20-year', '50-year', '100-year', '200-year',
                              'Disconnected', 'Reachable'],
                  'text_bounds_and_overlap': 'passed', 'inset_occlusion': 'passed',
                  'data_to_glyph_values': 'passed'}


