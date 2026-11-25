"""Formal Fig. 2 style constants and package-relative frozen sources.

Values retain the V1113 formal renderer; the public YAML controls font and dpi.
"""
from shared_context import ROOT, DATA, FIGURES, settings, font_path
_settings=settings('fig2')
CITIES=['zhengzhou','hangzhou','fuzhou','chengdu','wuhan']
CITY_INPUTS={c:{'CITY_NAME':c.title()} for c in CITIES}
EXCEEDANCE_COLORS = {10: '#4C78A8', 20: '#59A14F', 50: '#E3B34C', 100: '#E17C45', 200: '#A05C99'}
EXCEEDANCE_LINESTYLES = {10: ':', 20: '--', 50: '-.', 100: '--', 200: '-'}
FORMAL_CITY_ABBREVIATIONS = {'fuzhou': 'FZ', 'chengdu': 'CD', 'wuhan': 'WH', 'hangzhou': 'HZ', 'zhengzhou': 'ZZ'}
FORMAL_CITY_EDGE_COLOR = '#FDF6E6'
FORMAL_CITY_NAMES = {'fuzhou': 'Fuzhou',
 'chengdu': 'Chengdu',
 'wuhan': 'Wuhan',
 'hangzhou': 'Hangzhou',
 'zhengzhou': 'Zhengzhou'}
FORMAL_COUNTRY_FILL_COLOR = '#F5F5F5'
FORMAL_CURVE_ASPECT = 0.78
FORMAL_CURVE_COL_GAP = 0.35
FORMAL_CURVE_LEGEND_ANCHOR = (0.6, 0.0)
FORMAL_CURVE_LEGEND_LOC = 'lower right'
FORMAL_CURVE_LEGEND_POSITION = (1, 2)
FORMAL_CURVE_POSITIONS = {'zhengzhou': (0, 0), 'hangzhou': (0, 1), 'fuzhou': (0, 2), 'chengdu': (1, 0), 'wuhan': (1, 1)}
FORMAL_CURVE_ROW_GAP = 0.36
FORMAL_FIGSIZE = (5.511811023622047, 6.496062992125984)
FORMAL_HEIGHT_RATIOS = (3.0, 2.8)
FORMAL_LEGEND_FONT_SIZE = 7
FORMAL_MAP_EXTENT = (103.0, 126.0, 22.1, 35.5)
FORMAL_MAP_INSETS = {'zhengzhou': [0.12, 0.73, 0.21, 0.22],
 'hangzhou': [0.6, 0.72, 0.21, 0.22],
 'chengdu': [0.085, 0.3, 0.21, 0.22],
 'wuhan': [0.39, 0.23, 0.21, 0.22],
 'fuzhou': [0.75, 0.22, 0.21, 0.22]}
FORMAL_MAP_LEGEND_ANCHOR = (0.02, 0.01)
FORMAL_OUTER_GAP = 0.15
FORMAL_PANEL_CITIES = ['zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan']
FORMAL_STUDY_EDGE_COLOR = '#000000'
PERIODS = [10, 20, 50, 100, 200]
PLOT_COLORS = {'exposed': '#C65D57', 'nonexposed': '#377E89'}
PLOT_DPI = 240
PLOT_FIGSIZE = (12.4, 7.6)
PLOT_FILENAME = 'five_cities_population_exposure_and_weighted_risk_preview.png'
SCENARIO_COLORS = {10: '#4C78A8', 20: '#59A14F', 50: '#E3B34C', 100: '#E17C45', 200: '#A05C99'}
FORMAL_DPI=int(_settings['dpi'])
FORMAL_FONT=font_path(_settings['font_family'])
PLOT_FONT=FORMAL_FONT
FORMAL_OUTPUT_DIR=OUTPUT_DIR=FIGURES
FORMAL_EXPOSURE_CSV=PLOT_DATA_CSV=DATA/'fig2_exposure.csv'
FORMAL_CURVE_CSV=DATA/'fig2_exceedance.csv'
FORMAL_PROVINCE_PATH=DATA/'fig2_provinces.gpkg'
FORMAL_BOUNDARY_PATHS={c:DATA/f'plot_study_{c}.gpkg' for c in FORMAL_PANEL_CITIES}
FORMAL_CITY_BOUNDARY_PATHS={c:DATA/f'fig2_city_{c}.gpkg' for c in FORMAL_PANEL_CITIES}
