"Population-weighted EFAR violins with common bandwidth, axes and width scaling."
from pathlib import Path
import os
import sys
import importlib.util

RUNTIME_BIN = Path(sys.prefix) / 'Library/bin'
DLL_DIRECTORY_HANDLES = []
if os.name == 'nt' and RUNTIME_BIN.is_dir():
    os.environ['PATH'] = str(RUNTIME_BIN) + os.pathsep + os.environ.get('PATH', '')
    DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(RUNTIME_BIN)))
HERE=Path(__file__).resolve().parent
from shared_context import DATA, settings, font_path
import fig2_config as result1_style
_settings=settings('S4')
RESULT1_STYLE_CONFIG=HERE/'fig2_config.py'
RESULT1_STYLE_SCRIPT=HERE/'fig2_render.py'
SOURCE_DIR=DATA
GRID_CSV=DATA/'s4_grid_population_efar.csv'
SOURCE_METADATA=DATA/'s4_grid_provenance.json'
COVERAGE_CSV=DATA/'s4_coverage.csv'
OUTPUT_DIR=DATA.parent/'figures'
# Follow the Figure 2 city order: Zhengzhou, Hangzhou, Fuzhou / Chengdu, Wuhan, legend.
# Use the manuscript city order for the supplementary figure.
CITIES = list(result1_style.FORMAL_PANEL_CITIES)
PANEL_POSITIONS = {city: tuple(result1_style.FORMAL_CURVE_POSITIONS[city]) for city in CITIES}
LEGEND_POSITION = tuple(result1_style.FORMAL_CURVE_LEGEND_POSITION)
# Align the legend with the Figure 2 lower-panel legend.
# Account for the different legend text and number of entries.
# Keep the measured reference legend position in the lower-right panel.
LEGEND_LOC = 'center left'
LEGEND_ANCHOR = (0.06, 0.5)
CITY_NAMES = dict(result1_style.FORMAL_CITY_NAMES)
CITY_ABBREVIATIONS = dict(result1_style.FORMAL_CITY_ABBREVIATIONS)
DIMENSIONS = ['应急救援', '医疗资源', '避险转移']
FUNCTION_NAMES = {'应急救援': 'Emergency rescue', '医疗资源': 'Healthcare',
                  '避险转移': 'Emergency shelter'}
FUNCTION_COLORS = {'应急救援': result1_style.SCENARIO_COLORS[10],
                   '医疗资源': result1_style.SCENARIO_COLORS[100],
                   '避险转移': result1_style.SCENARIO_COLORS[20]}
PERIODS = [10, 20, 50, 100, 200]
FONT_FAMILY = _settings['font_family']
FONT_PATH = font_path(FONT_FAMILY)
# Retain the Figure 2 width and panel aspect with figure-specific gaps.
FIGURE_SIZE_MM = (result1_style.FORMAL_FIGSIZE[0] * 25.4, 90.27)
PANEL_ASPECT = result1_style.FORMAL_CURVE_ASPECT
COLUMN_GAP_MM = 8.0
ROW_GAP_MM = 12.0
PANEL_WIDTH_MM = (FIGURE_SIZE_MM[0] * (0.97 - 0.10) - 2 * COLUMN_GAP_MM) / 3
# Font sizes follow the formal Figure 2 settings, in points.
BASE_FONT_SIZE = 8
TICK_FONT_SIZE = 7
AXIS_FONT_SIZE = 7
CITY_FONT_SIZE = 8.5
PANEL_LABEL_FONT_SIZE = 10
AXIS_LABEL_PAD = 2.5
# Express the physical layout in millimetres.
LAYOUT_MM = {
    'left_margin': FIGURE_SIZE_MM[0] * 0.10,
    'right_margin': FIGURE_SIZE_MM[0] * 0.03,
    'column_gap': COLUMN_GAP_MM,
    'row_top_offsets': [7.0, 7.0 + PANEL_WIDTH_MM * PANEL_ASPECT + ROW_GAP_MM],
    'city_title_gap': [5.8, 5.8],
}
DPI = int(_settings['dpi'])
LEGEND_FONT_SIZE = result1_style.FORMAL_LEGEND_FONT_SIZE
# Place a single-column legend in the lower-right slot.
FIGURE_STEM = 'fig_s4'
# Use one absolute EFAR bandwidth across all 75 groups.
KDE_BANDWIDTH = 0.03
KDE_Y_POINTS = 401
KDE_CHUNK_SIZE = 512
FUNCTION_OFFSETS = [-0.33, 0.0, 0.33]
MAX_HALF_WIDTH = 0.16
# Use one common square-root density-to-width mapping.
DENSITY_WIDTH_POWER = 0.5
VIOLIN_ALPHA = 0.87
DISPLAY_TAIL_QUANTILE = 0.01
DISPLAY_ROUNDING_STEP = 0.10
