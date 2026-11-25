"""Render manuscript Fig. 2 using the released frozen plot data."""
from pathlib import Path
import argparse
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_support/figures_2_3_s1_s4'))
import shared_context  # Initialize the portable GDAL/PROJ environment first.
_dll_handles = []
_library = Path(sys.prefix) / 'Library/bin'
if os.name == 'nt' and _library.is_dir():
    os.environ['PATH'] = str(_library) + os.pathsep + os.environ.get('PATH', '')
    _dll_handles.append(os.add_dll_directory(str(_library)))


def run(output: Path) -> Path:
    import fig2_render as plot
    import fig2_config as cfg
    from shared_context import settings
    from PIL import Image
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cities, curves, population = plot.read_curves(cfg.FORMAL_CURVE_CSV)
    study = plot.read_study_lines(cfg.FORMAL_PANEL_CITIES, as_boundary=False)
    administrative = plot.read_study_lines(cfg.FORMAL_PANEL_CITIES, cfg.FORMAL_CITY_BOUNDARY_PATHS,
                                           as_boundary=False)
    exposure = plot.population_plot.read_plot_data(cfg.FORMAL_EXPOSURE_CSV, cities)
    fig = plot.make_figure(cities, curves, population, study, administrative, exposure)
    path = output / 'fig2.png'
    fig.savefig(path, dpi=cfg.FORMAL_DPI, facecolor='white')
    plt.close(fig)
    # Manuscript crop is 208 pixels at the original 600 dpi, with no data changes.
    bottom = round(float(settings('fig2').get('crop_bottom_px', 208)) * cfg.FORMAL_DPI / 600)
    if bottom:
        with Image.open(path) as im:
            if not 0 <= bottom < im.height:
                raise ValueError()
            cropped = im.crop((0, 0, im.width, im.height - bottom))
        cropped.save(path, dpi=(cfg.FORMAL_DPI, cfg.FORMAL_DPI))
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', '--output', type=Path,
                        default=HERE.parent / 'results/figures')
    run(parser.parse_args().output_dir)
