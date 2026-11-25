"""Frozen radial renderer for the formal Result 2 figure; source values are unchanged."""

import copy
import numpy as np
from matplotlib.patches import Arc

from shared_context import settings
FONT = settings('fig3')['font_family']

TEXT_PT = 7.5

TITLE_PT = 7.5

CENTRE_LABEL_Y_MM = 0.0

PANEL_HALF_MM = 26.5

INNER_RADIUS_MM = 6.5

EFAR_RADIUS_MM = 20.0

CITY_LABEL_RADIUS_MM = 24.0

BAR_WIDTH_PT = 3.0

VALUE_PT = 5.3

VALUE_OFFSET_MM = .65

# Keep the updated tall bar's value clear of the fixed city percentage label.
VALUE_OFFSET_OVERRIDES_MM = {('zhengzhou', 'Emergency shelter', 50): .45}

LABEL_PERIODS = (10, 200)

SHARE_PAIR_PT = 4.8

CITY_SHARE_GAP_MM = 3.0

SIDE_CITY_SHARE_GAP_MM = 5.0

TOP_CITY_LABEL_SHIFT_MM = 3.5

CITY_GAP_DEG = 12.0

RADIAL_GRID_VALUES = (0, .25, .5, .75, 1)

GRID_COLOR = '#000000'

SECTOR_EDGE_COLOR = '#000000'

CITIES = ['zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan']

CITY_LABELS = ['ZZ', 'HZ', 'FZ', 'CD', 'WH']

FUNCTIONS = ['Emergency rescue', 'Healthcare', 'Emergency shelter']

CENTRE_LABELS = ['Emergency\nrescue', 'Healthcare', 'Emergency\nshelter']

PERIODS = [10, 20, 50, 100, 200]

LOSS_COLORS = {'disconnection': '#426987', 'detour': '#D69B46'}

def require(condition, message):
    if not condition:
        raise RuntimeError()

def xy(radius, angle_deg):
    angle = np.deg2rad(angle_deg)
    return np.array([radius * np.cos(angle), radius * np.sin(angle)])

def text_corners(text, renderer):
    """Use oriented text rectangles so diagonal labels are not falsely rejected."""
    unrotated = copy.copy(text)
    unrotated.set_rotation(0)
    box = unrotated.get_window_extent(renderer)
    points = np.array([[box.x0, box.y0], [box.x1, box.y0],
                       [box.x1, box.y1], [box.x0, box.y1]])
    angle = float(text.get_rotation())
    if angle:
        require(text.get_rotation_mode() == 'anchor', '')
        anchor = text.get_transform().transform(text.get_unitless_position())
        rad = np.deg2rad(angle)
        rotation = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        points = (points - anchor) @ rotation.T + anchor
    return points

def overlaps(a, b):
    for points in (a, b):
        edges = np.roll(points, -1, axis=0) - points
        for edge in edges[:2]:
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() or pb.max() <= pa.min():
                return False
    return True

def draw_panel(ax, data, function, centre_label):
    """Return label artists and exact data-to-glyph checks for one facility type."""
    ax.set(xlim=(-PANEL_HALF_MM, PANEL_HALF_MM),
           ylim=(-PANEL_HALF_MM, PANEL_HALF_MM), aspect='equal')
    ax.set_axis_off()
    texts, checks = [], []
    radial_scale = EFAR_RADIUS_MM - INNER_RADIUS_MM
    # The twelve-o'clock city gap holds one shared radial scale per panel.
    for value in [0, .5, 1]:
        radius = INNER_RADIUS_MM + radial_scale * value
        texts.append(ax.text(0, radius, f'{value:g}', ha='center', va='center',
                             fontsize=6.5, color='#000000', zorder=6,
                             bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .30}))

    texts.append(ax.text(0, CENTRE_LABEL_Y_MM, centre_label, ha='center', va='center',
                         multialignment='center', fontsize=TITLE_PT, linespacing=1.15,
                         color='#20262C', zorder=6))

    city_width = 360 / len(CITIES)
    sector_width = city_width - CITY_GAP_DEG
    slot_width = sector_width / len(PERIODS)
    for ci, (city, short_name) in enumerate(zip(CITIES, CITY_LABELS)):
        sector_start = 90 - ci * city_width - CITY_GAP_DEG / 2
        sector_end = sector_start - sector_width
        sector_middle = (sector_start + sector_end) / 2
        # Each city's scale is an independent annular sector. Grid arcs stop
        # at its two radial edges, leaving the inter-city gaps fully open.
        for value in RADIAL_GRID_VALUES:
            radius = INNER_RADIUS_MM + radial_scale * value
            ax.add_patch(Arc((0, 0), 2*radius, 2*radius,
                             theta1=sector_end, theta2=sector_start,
                             linewidth=.35, color=GRID_COLOR, zorder=0))
        for edge_angle in (sector_start, sector_end):
            inner, outer = xy(INNER_RADIUS_MM, edge_angle), xy(EFAR_RADIUS_MM, edge_angle)
            ax.plot([inner[0], outer[0]], [inner[1], outer[1]],
                    color=SECTOR_EDGE_COLOR, linewidth=.4,
                    solid_capstyle='butt', zorder=1)
        lx, ly = xy(CITY_LABEL_RADIUS_MM, sector_middle)
        # Lift the two upper city labels enough to keep their lower percentage
        # line outside the bar-tip labels; circle geometry stays unchanged.
        if ly > 0:
            ly += TOP_CITY_LABEL_SHIFT_MM
        texts.append(ax.text(lx, ly, short_name, ha='center', va='center',
                             fontsize=TEXT_PT))
        endpoints = data.loc[data.city.eq(city) & data.function.eq(function)
                             & data.period.isin(LABEL_PERIODS)].set_index('period')
        require(len(endpoints) == 2, '')
        pair_values = []
        for endpoint in LABEL_PERIODS:
            endpoint_row = endpoints.loc[endpoint]
            endpoint_total = float(endpoint_row.disconnection_loss + endpoint_row.detour_loss)
            require(endpoint_total > 0, '')
            pair_values.append(float(endpoint_row.disconnection_loss) / endpoint_total)
        # Pair percentages with the city name, separate from all EFAR bar tips.
        share_gap = SIDE_CITY_SHARE_GAP_MM if city in ('hangzhou', 'chengdu') else CITY_SHARE_GAP_MM
        pair_y = ly - share_gap
        texts.append(ax.text(lx, pair_y, f'{pair_values[0]:.1%} → {pair_values[1]:.1%}',
                             ha='center', va='center', fontsize=SHARE_PAIR_PT,
                             color='#000000', zorder=6))

        for si, period in enumerate(PERIODS):
            selected = data.loc[data.city.eq(city) & data.function.eq(function)
                                & data.period.eq(period)]
            require(len(selected) == 1, '')
            row = selected.iloc[0]
            d, u, mean = (float(row.disconnection_loss), float(row.detour_loss),
                          float(row.mean_EFAR))
            require(d >= 0 and u >= 0 and 0 <= mean <= 1, '')
            require(row.grid_count > 0 and row.population > 0, '')
            require(abs(d + u - mean) < 1e-8, '')
            angle = sector_start - (si + .5) * slot_width
            r0, r1, r2 = (INNER_RADIUS_MM,
                          INNER_RADIUS_MM + radial_scale * d,
                          INNER_RADIUS_MM + radial_scale * (d + u))
            for start, end, color in [(r0, r1, LOSS_COLORS['disconnection']),
                                      (r1, r2, LOSS_COLORS['detour'])]:
                a, b = xy(start, angle), xy(end, angle)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                        linewidth=BAR_WIDTH_PT, solid_capstyle='butt', zorder=3)

            # Every bar tip shows total EFAR; endpoint shares sit by the city name.
            share = d / (d + u) if d + u > 0 else np.nan
            label = f'{share:.1%}' if period in LABEL_PERIODS else None
            normal_angle = (angle + 180) % 360 - 180
            left_side = abs(normal_angle) > 90
            rotation = normal_angle + (180 if left_side else 0)
            value_offset = VALUE_OFFSET_OVERRIDES_MM.get((city, function, period), VALUE_OFFSET_MM)
            tx, ty = xy(r2 + value_offset, angle)
            texts.append(ax.text(
                tx, ty, f'{mean:.2f}', fontsize=VALUE_PT,
                ha='right' if left_side else 'left', va='center',
                rotation=rotation, rotation_mode='anchor', zorder=5,
                color='#000000',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .15}))

            require(abs((r2-r0) / radial_scale - mean) < 1e-8,
                    '')
            checks.append({'city': city, 'function': function, 'period': period,
                           'angle_degrees': angle, 'mean_EFAR': mean,
                           'disconnection_loss': d, 'detour_loss': u,
                           'analysis_grid_count': int(row.grid_count),
                           'analysis_population': float(row.population),
                           'reachable_grid_count': int(row.reachable_grid_count),
                           'reachable_population': float(row.reachable_population),
                           'disconnected_grid_count': int(row.disconnected_grid_count),
                           'disconnected_population': float(row.disconnected_population),
                           'disconnection_share': float(share) if np.isfinite(share) else None,
                           'disconnection_share_label': label,
                           'total_EFAR_label': f'{mean:.2f}',
                           'value_offset_mm': value_offset,
                           'radial_length_mm': r2-r0})
    return texts, checks
