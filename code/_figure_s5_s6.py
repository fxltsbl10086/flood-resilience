"""Package-local S5/S6 renderers used by plot_supplementary.py."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support/figures_4_5_s5_s6'))
import release_io as io

def run(figure: str, output: Path):
    figure=figure.upper()
    if figure=='S5':return _run_s5(output)
    if figure=='S6':return _run_s6(output)
    raise ValueError()

def _run_s5(output):
    from plot_fig4 import configured_body, configured_map_period
    body,config,_=configured_body('S5')
    supplement=io.module('s5_draw')
    supplement.body=body
    supplement.FIGURE_MM=(config.get('width_mm',183),config.get('height_mm',127))
    supplement.PANEL_MM=config.get('panel_mm',48)
    supplement.LEFT_MM=config.get('left_mm',13)
    supplement.TOP_MM=config.get('top_mm',7)
    supplement.COLUMN_GAP_MM=config.get('column_gap_mm',10)
    supplement.ROW_GAP_MM=config.get('row_gap_mm',16)
    data,metadata=io.maps()
    output=io.output_directory(output)
    paths=[]
    main_period = configured_map_period(body, io.settings('fig4'))
    for period in body.PERIODS:
        if period == main_period:
            continue
        w,h=supplement.FIGURE_MM
        fig=body.plt.figure(figsize=(w/25.4,h/25.4),dpi=body.DPI)
        panels=[]
        for index,city in enumerate(body.CITY_ORDER):
            ax=fig.add_axes(supplement.panel_slot(index//3,index%3))
            body.draw_map(ax,city,data,period,chr(ord('a')+index))
            panels.append(ax)
        legend=supplement.draw_bivariate_legend(fig)
        supplement.validate_layout(fig,panels,legend)
        path=output/f'fig_s5_t{period:03d}.png'
        fig.savefig(path,dpi=body.DPI,facecolor='white')
        body.plt.close(fig)
        paths.append(path)
    return paths

def _run_s6(output):
    drawing=io.module('s6_draw')
    config=io.settings('S6')
    drawing.SIZE_MM=(float(config.get('width_mm',155)),float(config.get('height_mm',149)))
    drawing.DPI=int(config.get('dpi',600))
    drawing.INSET_BOUNDS=config.get('inset_bounds',drawing.INSET_BOUNDS)
    drawing.BAND_ALPHA=config.get('band_alpha',drawing.BAND_ALPHA)
    font=io.configure_style(config,drawing.STYLE)
    drawing.STYLE['font.family']=font
    curves=io.csv('fig_s6_city_curves.csv',dtype={'return_periods':str})
    fig=drawing.build_figure(curves)
    output=io.output_directory(output)
    png=output/'fig_s6.png'
    fig.savefig(png,format='png',dpi=drawing.DPI,metadata={'Title':'Supplementary Fig. S6',
        'Description':'Mean cumulative EFARB reduction across matched rainfall return periods. '
        'Shading shows pointwise 95% percentile intervals from paired resampling of return-period curves.'})
    drawing.plt.close(fig)
    return [png]
