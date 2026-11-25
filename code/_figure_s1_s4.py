"""Internal S1–S4 renderers used by plot_supplementary.py.

S1 uses geometry snapshots of the original formal plot inputs; S2 reads the
released 25 maximum-depth results; S3 retains unweighted SAR; S4 renders the
frozen population-weighted density and empirical-quantile tables.
"""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_support/figures_2_3_s1_s4'))
_dll_handles = []
_library = Path(sys.prefix) / 'Library/bin'
if os.name == 'nt' and _library.is_dir():
    os.environ['PATH'] = str(_library) + os.pathsep + os.environ.get('PATH', '')
    _dll_handles.append(os.add_dll_directory(str(_library)))


def run(figure: str, output: Path) -> list[Path]:
    from shared_context import DATA, settings
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = settings(figure)
    plt.rcParams.update({'font.family': cfg['font_family'], 'text.color': '#000000',
                         'axes.unicode_minus': False})
    if figure == 'S1':
        import s1_render as plot
        fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 132 / 25.4))
        fig.subplots_adjust(left=.049, right=.951, top=.901, bottom=.052,
                            wspace=.26, hspace=.34)
        panels = []
        for ax, letter, (city, name) in zip(axes.flat, 'abcde', plot.CITIES):
            domain, study, points, _ = plot.load_city(city, {})
            context = plot.read_reference_layers(city, study)
            panels.append((ax, plot.draw_map(ax, letter, name, domain, study, points, context)))
        plot.draw_legend(axes.flat[-1])
        plot.validate_coordinates(fig, panels)
        path = output / 'fig_s1.png'
        fig.savefig(path, dpi=int(cfg['dpi']), facecolor='white')
        plt.close(fig)
        return [path]
    if figure == 'S2':
        import s2_render as plot
        plot.SETTINGS = cfg
        plot.MAP_FONT = cfg['font_family']
        for city, name in plot.CITIES:
            plot.plot_city(city, name, output)
        return [output / f'fig_s2_{c}.png' for c, _ in plot.CITIES]
    if figure == 'S3':
        import sar_data
        import sar_render as plot
        import pyogrio
        from sar_config import CITY_INPUTS, PERIODS
        sar_data.set_style()
        plt.rcParams['font.family'] = cfg['font_family']
        plot.DPI = int(cfg['dpi'])
        datasets = sar_data.load_data(PERIODS)
        for city, name in plot.CITIES:
            grid, boundary = datasets[city]
            native = pyogrio.read_info(CITY_INPUTS[city]['GRID_PATH'])['crs']
            plot.plot_city(name, grid.to_crs(native), boundary.to_crs(native), output)
        return [output / f'fig_s3_{c}.png' for c, _ in plot.CITIES]
    if figure == 'S4':
        import numpy as np
        import pandas as pd
        import s4_render as plot
        from s4_config import FUNCTION_NAMES
        density = pd.read_csv(DATA / 's4_density.csv', float_precision='round_trip')
        stats = pd.read_csv(DATA / 's4_summary.csv', float_precision='round_trip')
        reverse = {v: k for k, v in FUNCTION_NAMES.items()}
        y = np.sort(density.EFAR.unique())
        densities = {}
        for (city, function, period), group in density.groupby(['city', 'function', 'return_period']):
            group = group.sort_values('EFAR')
            if not np.array_equal(group.EFAR.to_numpy(), y):
                raise ValueError()
            densities[(city, reverse[function], int(period))] = group.population_weighted_density.to_numpy()
        if len(densities) != 75 or len(stats) != 75:
            raise ValueError()
        plot.FONT_FAMILY = cfg['font_family']
        plot.DPI = int(cfg['dpi'])
        path, _ = plot.draw(y, densities, stats, output)
        return [path]
    raise ValueError()
