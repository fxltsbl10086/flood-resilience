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
WIDTH_MM, HEIGHT_MM, DPI = 130.0, 85.0, 600
CITY_LABELS = {'zhengzhou':'Zhengzhou', 'hangzhou':'Hangzhou', 'fuzhou':'Fuzhou',
               'chengdu':'Chengdu', 'wuhan':'Wuhan'}
SCENARIO_SPACING = 0.15
AXES_POSITION = [0.13, 0.25, 0.85, 0.69]  # left, bottom, width, height
LEGEND_SCENARIOS_Y, LEGEND_INTERVALS_Y = 0.115, 0.047
MEDIAN_LINEWIDTH_PT = 0.7
OUTER_LINEWIDTH_PT = 0.75
BOX_WIDTH = 0.034
BOX_ALPHA = 0.20
SHOW_GRID_POINTS = True  # Show resident-grid points alongside the weighted box and violin.
RAIN_X_OFFSETS = (-0.058, -0.029)
RAIN_ALPHA = 0.08
RAIN_POPULATION_PER_PT2 = 4000.0  # Use the same population-to-area scale across all cities.
JITTER_SEED = 20261109
CLOUD_X_OFFSET, CLOUD_MAX_WIDTH, CLOUD_ALPHA = 0.025, 0.050, 0.30
KDE_BANDWIDTH = 'scott'
KDE_GRID_SIZE = 601
CITY_ORDER=['zhengzhou','hangzhou','fuzhou','chengdu','wuhan']
PERIODS=[10,20,50,100,200]
QUANTILE_COLUMNS=['P10','P25','P50','P75','P90']
PROBABILITIES=np.array([.1,.25,.5,.75,.9])
def weighted_quantiles(values, population):
    "Weighted inverse empirical CDF, including zero values and ties."
    values, population = np.asarray(values, dtype=float), np.asarray(population, dtype=float)
    if len(values) == 0 or values.shape != population.shape:
        raise ValueError()
    if not (np.isfinite(values).all() and np.isfinite(population).all() and (population > 0).all()):
        raise ValueError()
    order = np.argsort(values, kind='stable')
    v, w = values[order], population[order]
    cumulative = np.cumsum(w, dtype=np.float64)
    positions = np.searchsorted(cumulative, PROBABILITIES * cumulative[-1], side='left')
    quantiles = v[positions]
    total = math.fsum(map(float, w))
    for probability, quantile in zip(PROBABILITIES, quantiles):
        before = math.fsum(map(float, w[v < quantile])) / total
        at = math.fsum(map(float, w[v <= quantile])) / total
        if before > probability + 1e-12 or at < probability - 1e-12:
            raise ValueError()
    if not np.all(np.diff(quantiles) >= 0):
        raise ValueError()
    return quantiles
def draw_panel(fig, data, colors, markers, groups):
    "Draw the combined panel using frozen quantiles, density and grid values."
    ax = fig.add_axes(AXES_POSITION)
    positions, artists = [], []
    indexed = data.set_index(['city','period'])
    for i, city in enumerate(CITY_ORDER):
        for j, period in enumerate(PERIODS):
            row = indexed.loc[(city, period)]
            x = i + (j - 2) * SCENARIO_SPACING
            color = colors[city]
            group = groups[(city,period)]
            cloud_width = group['density'] / group['density'].max() * CLOUD_MAX_WIDTH
            cloud = ax.fill_betweenx(group['density_y'], x+CLOUD_X_OFFSET,
                                    x+CLOUD_X_OFFSET+cloud_width,color=color,
                                    alpha=CLOUD_ALPHA,linewidth=0,zorder=1)
            rain = None
            if SHOW_GRID_POINTS:
                rain = ax.scatter(x+group['jitter'],group['values'],
                                  s=group['population']/RAIN_POPULATION_PER_PT2,
                                  color=color,alpha=RAIN_ALPHA,edgecolors='none',zorder=1.5)
            outer = ax.plot([x,x,np.nan,x,x], [row.P10,row.P25,np.nan,row.P75,row.P90], color=color,
                            lw=OUTER_LINEWIDTH_PT, solid_capstyle='butt', zorder=2)[0]
            box = Rectangle((x-BOX_WIDTH/2,row.P25),BOX_WIDTH,row.P75-row.P25,
                            facecolor=matplotlib.colors.to_rgba(color,BOX_ALPHA),
                            edgecolor=color,linewidth=0.7,zorder=3)
            ax.add_patch(box)
            for percentile in (row.P10,row.P90):
                ax.plot([x-BOX_WIDTH*.32,x+BOX_WIDTH*.32],[percentile,percentile],
                        color=color,lw=0.6,solid_capstyle='butt',zorder=3)
            median = ax.plot([x-BOX_WIDTH/2,x+BOX_WIDTH/2],[row.P50,row.P50],
                             color=color,lw=MEDIAN_LINEWIDTH_PT,marker='None',
                             solid_capstyle='butt',zorder=4)[0]
            positions.append({'city':city, 'period':period, 'x':x,
                              'P10':float(row.P10), 'P25':float(row.P25), 'P50':float(row.P50),
                              'P75':float(row.P75), 'P90':float(row.P90)})
            artists.append({'outer':outer,'box':box,'median':median,'rain':rain,'cloud':cloud})
    ax.set_xlim(-0.55,4.55)
    ax.set_ylim(0,1)
    ax.set_yticks(np.linspace(0,1,6))
    ax.set_yticklabels([f'{value:.1f}' for value in np.linspace(0,1,6)])
    ax.set_ylabel('Reduction in EFAR', labelpad=5)
    ax.set_xticks(np.arange(len(CITY_ORDER)))
    ax.set_xticklabels([CITY_LABELS[city] for city in CITY_ORDER])
    ax.tick_params(axis='x', length=0, pad=6)
    ax.tick_params(axis='y', direction='out', length=3, pad=3)
    ax.grid(False)
    fig.text(0.025,0.95,'c',fontsize=10,fontweight='bold',ha='left',va='center',color='black')
    scenario_handles = [Line2D([],[],linestyle='none',marker='None',
                              label='Within each city, left to right: T10, T20, T50, T100, T200')]
    interval_handles = [
        Line2D([],[],color='#555555',lw=MEDIAN_LINEWIDTH_PT,marker='None',label='Median'),
        Patch(facecolor='#DDDDDD',edgecolor='#555555',lw=0.7,label='P25–P75'),
        Line2D([],[],color='#555555',lw=OUTER_LINEWIDTH_PT,label='P10–P90')]
    legends = [fig.legend(handles=scenario_handles, loc='center',
                          bbox_to_anchor=(0.55,LEGEND_SCENARIOS_Y),ncol=1,
                          frameon=False,fontsize=6.5,handlelength=0,handletextpad=0),
               fig.legend(handles=interval_handles, loc='center',
                          bbox_to_anchor=(0.55,LEGEND_INTERVALS_Y),ncol=3,
                          frameon=False,fontsize=6.5,handlelength=1.7,columnspacing=1.8)]
    return ax, artists, legends, positions
def check_layout(fig, ax, artists, legends, positions, groups):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    texts = [ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *fig.texts]
    boxes = [text.get_window_extent(renderer) for text in texts if text.get_text()]
    boxes += [legend.get_window_extent(renderer) for legend in legends]
    if any(b.x0 < bounds.x0 or b.y0 < bounds.y0 or b.x1 > bounds.x1 or b.y1 > bounds.y1 for b in boxes):
        raise ValueError()
    legend_boxes = [legend.get_window_extent(renderer) for legend in legends]
    if legend_boxes[0].overlaps(legend_boxes[1]):
        raise ValueError()
    city_boxes = [text.get_window_extent(renderer) for text in ax.get_xticklabels()]
    if any(a.overlaps(b) for i,a in enumerate(city_boxes) for b in city_boxes[i+1:]):
        raise ValueError()
    if any(b.overlaps(lb) for b in city_boxes for lb in legend_boxes):
        raise ValueError()
    if len(artists) != 25:
        raise ValueError()
    scatter_count = 0
    for row, group_artists in zip(positions, artists):
        group = groups[(row['city'],row['period'])]
        outer, box, median, rain = [group_artists[k] for k in ('outer','box','median','rain')]
        if not (np.array_equal(outer.get_ydata(), [row['P10'],row['P25'],np.nan,row['P75'],row['P90']],equal_nan=True) and
                np.allclose([box.get_y(),box.get_y()+box.get_height()], [row['P25'],row['P75']],rtol=0,atol=1e-15) and
                np.array_equal(median.get_ydata(), [row['P50'],row['P50']]) and
                np.array_equal(median.get_xdata(), [row['x']-BOX_WIDTH/2,row['x']+BOX_WIDTH/2])):
            raise ValueError()
        if median.get_marker() not in (None,'None','none',''):
            raise ValueError()
        if SHOW_GRID_POINTS:
            offsets = np.asarray(rain.get_offsets())
            if not (np.array_equal(offsets[:,1],group['values']) and
                    np.array_equal(rain.get_sizes(),group['population']/RAIN_POPULATION_PER_PT2)):
                raise ValueError()
            if len(offsets) != len(group['values']):
                raise ValueError()
            scatter_count += len(offsets)
        elif rain is not None:
            raise ValueError()
    if not SHOW_GRID_POINTS and any(isinstance(item,PathCollection) for item in ax.collections):
        raise ValueError()
    if CLOUD_X_OFFSET+CLOUD_MAX_WIDTH >= SCENARIO_SPACING+RAIN_X_OFFSETS[0]:
        raise ValueError()
    return {'status':'PASS','quantile_intervals':25,'median_symbols':0,'median_horizontal_lines':25,
            'weighted_half_violins':25,'original_grid_scatter_points':scatter_count,
            'grid_points_shown':SHOW_GRID_POINTS,
            'original_grids_used_for_distribution':sum(len(g['values']) for g in groups.values()),
            'text_inside_canvas':True,'legends_nonoverlapping':True,
            'artist_coordinates_match_data':True,'y_limits':[0,1]}
