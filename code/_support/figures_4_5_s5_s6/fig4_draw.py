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
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import release_io as _release_io
map_style = _release_io.module('map_style')
cfri = _release_io.module('cfri')
map_config=map_style
CITY_ORDER = ['zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan']
CITY_NAMES = {'zhengzhou': 'Zhengzhou', 'hangzhou': 'Hangzhou', 'fuzhou': 'Fuzhou',
              'chengdu': 'Chengdu', 'wuhan': 'Wuhan'}
PERIODS = [10, 20, 50, 100, 200]
MAP_PERIOD = 50
FIGURE_MM = (183.0, 155.0)
REFERENCE_BOTTOM_TRIM_MM = 3.982057020399308
DPI, PREVIEW_DPI = 600, 180
LAYOUT_MM = {
    'left': 13.0, 'top': 7.0, 'panel_width': 34.5, 'panel_height': 34.5,
    # Column gaps; each city occupies two columns.
    'column_gaps': [9.0, 11.0, 9.0], 'row_gap': 16.0,
    'minimum_interrow_clearance': 2.5, 'bottom_padding': 0.5,
}
PANEL_LABEL_OFFSET_PT = (-6.0, 6.0)
MAP_CITY_LABEL_POSITION = (0.025, 0.015)
MAP_CITY_LABEL_FONT_PT = 7.0
LEGEND_LAYOUT = {
    # Reserve the last two slots for the bivariate and scenario legends.
    'scenario_anchor_axes': (0.10, 0.40), 'scenario_loc': 'center left',
    'bivariate_size_mm': 18.0, 'bivariate_left_inset_mm': 14.0,
    'bivariate_bottom_inset_mm': 8.0,
}
def slot_bounds(row, col):
    w, h = FIGURE_MM
    x = LAYOUT_MM['left'] + col * LAYOUT_MM['panel_width'] + sum(LAYOUT_MM['column_gaps'][:col])
    y = h - LAYOUT_MM['top'] - LAYOUT_MM['panel_height'] - row * (
        LAYOUT_MM['panel_height'] + LAYOUT_MM['row_gap'])
    return [x / w, y / h, LAYOUT_MM['panel_width'] / w, LAYOUT_MM['panel_height'] / h]
def panel_letter(ax, text):
    ax._combined_panel_label = ax.annotate(text, (0, 1), xycoords='axes fraction',
                       xytext=PANEL_LABEL_OFFSET_PT, textcoords='offset points',
                       fontsize=cfri.STYLE['panel_font_pt'], fontweight='bold',
                       ha='left', va='baseline', annotation_clip=False, clip_on=False)
    return ax._combined_panel_label
def draw_map(ax, city, datasets, period, letter):
    data, boundary = datasets[city]
    display = data.to_crs('EPSG:4326')
    area = gpd.GeoSeries([boundary], crs=data.crs).to_crs('EPSG:4326')
    field = f'bivariate_class_T{period}'
    excluded = display[field].isna()
    if excluded.any():
        display.loc[excluded].plot(ax=ax, facecolor='#D2D5D7', edgecolor='none', zorder=1)
    palette = np.asarray(map_config.BIVARIATE_PALETTE).reshape(9)
    selected = ~excluded
    display.loc[selected].plot(ax=ax, color=palette[display.loc[selected, field].to_numpy(int)],
                                edgecolor='none', zorder=2)
    hot = display[f'hotspot_T{period}'].eq(1)
    if hot.any():
        union = display.loc[hot].geometry.union_all()
        layer = gpd.GeoSeries([union], crs=display.crs)
        layer.plot(ax=ax, facecolor='none', edgecolor='white', hatch='///',
                   linewidth=map_config.HOTSPOT_WHITE_LINEWIDTH_PT, zorder=4)
        layer.boundary.plot(ax=ax, color='black', linewidth=map_config.HOTSPOT_OUTLINE_LINEWIDTH_PT, zorder=5)
    area.boundary.plot(ax=ax, color='black', linewidth=map_config.STUDY_BOUNDARY_LINEWIDTH_PT, zorder=6)
    map_style.fit_extent(ax, display.total_bounds, LAYOUT_MM['panel_width'] / LAYOUT_MM['panel_height'])
    panel_letter(ax, letter)
    ax.text(*MAP_CITY_LABEL_POSITION, f'{cfri.CITY_ABBREVIATIONS[city]} · T{period}',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=MAP_CITY_LABEL_FONT_PT, zorder=10)
    return {'city': city, 'period': period, 'display_crs': 'EPSG:4326',
            'analysis_grid_count': int(selected.sum()), 'hotspot_count': int(hot.sum()),
            'longitude_limits': list(ax.get_xlim()), 'latitude_limits': list(ax.get_ylim()),
            'cached_class_field': field, 'cached_hotspot_field': f'hotspot_T{period}',
            'geometry': 'Original complete cells; reprojected, not clipped or simplified.'}
def add_legends(fig):
    # Preserve the scenario-style legend position in slot 12.
    scenario_ax = fig.add_axes(slot_bounds(2, 3))
    scenario_ax.axis('off')
    handles = [Line2D([], [], color=cfri.SCENARIO_COLORS[t], linestyle=cfri.SCENARIO_STYLES[t],
                      linewidth=1.4, label=f'{t}-year') for t in PERIODS]
    scenario_legend = scenario_ax.legend(handles=handles, loc=LEGEND_LAYOUT['scenario_loc'],
        bbox_to_anchor=LEGEND_LAYOUT['scenario_anchor_axes'], borderaxespad=0, frameon=False,
        fontsize=cfri.STYLE['legend_font_pt'], handlelength=2.0, labelspacing=.7)
    # Show the bivariate color legend in slot 11; explain other layers in the caption.
    bx, by, _, _ = slot_bounds(2, 2)
    w, h = FIGURE_MM
    size = LEGEND_LAYOUT['bivariate_size_mm']
    color_ax = fig.add_axes([bx + LEGEND_LAYOUT['bivariate_left_inset_mm'] / w,
                            by + LEGEND_LAYOUT['bivariate_bottom_inset_mm'] / h,
                            size / w, size / h])
    for y in range(3):
        for x in range(3):
            color_ax.add_patch(Rectangle((x-.5, y-.5), 1, 1,
                facecolor=map_config.BIVARIATE_PALETTE[y][x], edgecolor='white', linewidth=.5))
    color_ax.set(xlim=(-.5, 2.5), ylim=(-.5, 2.5), aspect='equal')
    color_ax.set_xticks([0, 1, 2], ['Low', 'Mid', 'High'], fontsize=map_config.LEGEND_FONT_SIZE)
    color_ax.set_yticks([0, 1, 2], ['Low', 'Mid', 'High'], fontsize=map_config.LEGEND_FONT_SIZE)
    color_ax.tick_params(length=0, pad=2)
    color_ax.set_xlabel('Exposed people', fontsize=map_config.LEGEND_FONT_SIZE, labelpad=2)
    color_ax.set_ylabel('Service risk', fontsize=map_config.LEGEND_FONT_SIZE, labelpad=2)
    for spine in color_ax.spines.values():
        spine.set_visible(False)
    return scenario_ax, scenario_legend, color_ax
def visible_texts(ax, renderer):
    return [(text.get_text(), text.get_window_extent(renderer))
            for text in ax.findobj(Text) if text.get_visible() and text.get_text()]
def audit(fig, panels, legends, summary, curves):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pxmm = fig.dpi / 25.4
    failures, row_gaps = [], []
    main_axes = [p['ax'] for p in panels]
    for p in panels:
        ax, city = p['ax'], p['city']
        texts = visible_texts(ax, renderer)
        letter_box = ax._combined_panel_label.get_window_extent(renderer)
        if any(letter_box.overlaps(t.get_window_extent(renderer))
               for t in ax.get_xticklabels() + ax.get_yticklabels() if t.get_visible()):
            failures.append(f'{city}: panel letter overlaps its own axis tick label.')
        if p['kind'] == 'inequality':
            for line, period in zip(ax.lines[1:], PERIODS):
                x, y = curves[(city, period)]
                if not np.array_equal(line.get_xdata(), x) or not np.array_equal(line.get_ydata(), y):
                    failures.append('Lorenz vertices changed.')
            inset = ax.child_axes[0]
            expected = summary.loc[summary.city.eq(city)].set_index('return_period').loc[PERIODS, 'gini'].to_numpy()
            if not np.array_equal([b.get_height() for b in inset.patches], expected):
                failures.append('Gini bar values changed.')
            boxes = [t.get_window_extent(renderer) for t in inset.texts]
            if any((b.x0-a.x1)/pxmm < .4 for a, b in zip(boxes, boxes[1:])):
                failures.append(f'{city}: Gini value labels have less than 0.4 mm clearance.')
            inset_ticks = inset.get_xticklabels()
            if any(a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
                   for a, b in zip(inset_ticks, inset_ticks[1:])):
                failures.append(f'{city}: Gini tick labels overlap.')
        else:
            if any(t.get_rotation() != 0 for t in ax.get_xticklabels()):
                failures.append('Longitude labels must be horizontal.')
            if any(t.get_rotation() != 90 for t in ax.get_yticklabels()):
                failures.append('Latitude labels must be vertical.')
            xticks = ax.get_xticklabels()
            if any(a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
                   for a, b in zip(xticks, xticks[1:])):
                failures.append(f'{city}: longitude labels overlap.')
        for value, box in texts:
            if any(box.overlaps(other.bbox) for other in main_axes if other is not ax):
                failures.append(f'{city}: text {value} overlaps another panel.')
        for q in panels:
            if q['row'] != p['row'] + 1:
                continue
            lower = visible_texts(q['ax'], renderer)
            gaps = [(a.y0-b.y1)/pxmm for _, a in texts for _, b in lower
                    if a.x0 < b.x1 and a.x1 > b.x0]
            if gaps:
                gap = float(min(gaps))
                row_gaps.append({'upper': p['letter'], 'lower': q['letter'], 'gap_mm': gap})
                if gap < LAYOUT_MM['minimum_interrow_clearance'] - 1e-6:
                    failures.append(f"Panels {p['letter']}/{q['letter']}: row text gap {gap:.3f} mm < 2.5 mm.")
    # Check text collisions across adjacent panels in the same row.
    for index, p in enumerate(panels):
        for q in panels[index+1:]:
            if p['row'] == q['row'] and any(a.overlaps(b)
                    for _, a in visible_texts(p['ax'], renderer)
                    for _, b in visible_texts(q['ax'], renderer)):
                failures.append(f"Panels {p['letter']}/{q['letter']}: text overlap.")
    _, scenario_legend, color_ax = legends
    legend_boxes = [scenario_legend.get_window_extent(renderer), color_ax.get_tightbbox(renderer)]
    if legend_boxes[0].overlaps(legend_boxes[1]):
        failures.append('The two legends overlap.')
    for box in legend_boxes:
        if any(box.overlaps(ax.get_tightbbox(renderer)) for ax in main_axes):
            failures.append('Legend overlaps panel content.')
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
                failures.append(f'Visible text outside canvas: {text.get_text()}')
    boxes = [ax.get_tightbbox(renderer) for ax in main_axes] + legend_boxes
    low_mm = min(box.y0 for box in boxes) / pxmm
    trim_mm = max(0., min(REFERENCE_BOTTOM_TRIM_MM, low_mm-LAYOUT_MM['bottom_padding']))
    return {'failures': sorted(set(failures)), 'row_text_clearances': row_gaps,
            'curves_verified': 25, 'gini_bars_verified': 25,
            'canvas_mm': FIGURE_MM, 'export_size_mm': [FIGURE_MM[0], FIGURE_MM[1]-trim_mm],
            'bottom_trim_mm': trim_mm, 'bottom_padding_mm': low_mm-trim_mm,
            'main_panel_count': len(main_axes), 'legend_count': 2,
            'legend_positions_zero_based': {'bivariate': [2, 2], 'scenario': [2, 3]},
            'rows': 3, 'columns': 4, 'layout_mm': LAYOUT_MM, 'legend_layout': LEGEND_LAYOUT,
            'font': 'Times New Roman', 'dpi': DPI,
            'panel_order': [{k: v for k, v in p.items() if k != 'ax'} for p in panels]}
