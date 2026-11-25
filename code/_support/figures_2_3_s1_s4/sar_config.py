"Original unweighted SAR spatial values and plotting parameters."

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


from shared_context import DATA, settings, font_path
_settings=settings('S3')
OUTPUT_DIR=DATA.parent/'figures'
PERIODS=[10,20,50,100,200]
CITY_ORDER=['zhengzhou','hangzhou','fuzhou','chengdu','wuhan']
CITY_NAMES={c:c.title() for c in CITY_ORDER}
CITY_ABBR={'zhengzhou':'ZZ','hangzhou':'HZ','fuzhou':'FZ','chengdu':'CD','wuhan':'WH'}
CITY_INPUTS={c:{'RISK_CSV':DATA/f's3_risk_{c}.csv','GRID_PATH':DATA/f's3_grid_{c}.gpkg',
                'BOUNDARY_PATH':DATA/f'plot_study_{c}.gpkg'} for c in CITY_ORDER}

# Display original SAR on 0–1 without population weighting or filtering.
RISK_COLUMN = "SAR_T{period}"

# Include coordinate labels, city names and the common colorbar in the physical layout.
FIGURE_MM = (160.0, 194.0)
ROWS, COLUMNS = 3, 2
DPI = int(_settings["dpi"])
MAP_LAYOUT_MM = {"left": 12.0, "right": 4.0, "top": 7.0,
                 "column_gap": 7.0, "panel_height": 48.0, "row_gap": 9.5}
COLORBAR_WIDTH_MM = 110.0
COLORBAR_BOTTOM_MM = 8.0
COLORBAR_HEIGHT_MM = 3.0

# Reuse the formal map font, frame and coordinate styles.
FONT_FILE = font_path(_settings["font_family"])
BODY_FONT_SIZE = 8.0
AXIS_FONT_SIZE = 7.0
CITY_FONT_SIZE = 8.5
PANEL_FONT_SIZE = 10.0
LEGEND_FONT_SIZE = 7.0
MAP_FRAME_LINEWIDTH = 0.65
STUDY_BOUNDARY_LINEWIDTH = 0.45
TICK_LENGTH = 3.0
TICK_PAD = 2.0
PANEL_LABEL_OFFSET = (-6.0, 2.0)
CITY_LABEL_OFFSET = 12.0
COORDINATE_TICK_INSETS = {"longitude": 0.08, "latitude": 0.14}
EXTENT_FILL_FRACTION = 0.94

# Use one continuous 0–1 color scale for all five cities and scenarios.
RISK_COLORS = ["#FFFFFF", "#D8B4D7", "#B767B9", "#865D9B", "#3F4E7C"]
RISK_LIMITS = (0.0, 1.0)
EXCLUDED_COLOR = "#D2D5D7"
