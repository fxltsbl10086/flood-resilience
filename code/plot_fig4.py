"""Redraw Figure 4 from signed, frozen package inputs (no network calculation)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

SUPPORT=Path(__file__).resolve().parent/'_support/figures_4_5_s5_s6'
sys.path.insert(0,str(SUPPORT))
import release_io as io

def configured_body(figure='fig4'):
    body=io.module('fig4_draw')
    config=io.settings(figure)
    body.DPI=int(config.get('dpi',600))
    body.FIGURE_MM=(float(config.get('width_mm',183)),float(config.get('height_mm',155)))
    body.LAYOUT_MM.update(config.get('layout_mm',{}))
    body.LEGEND_LAYOUT.update(config.get('legend_layout',{}))
    body.PANEL_LABEL_OFFSET_PT=config.get('panel_label_offset_pt',[-6,6])
    body.MAP_CITY_LABEL_POSITION=config.get('map_city_label_position',[.025,.015])
    body.MAP_CITY_LABEL_FONT_PT=config.get('map_city_label_font_pt',7)
    body.cfri.STYLE.update(config.get('cfri_style',{}))
    if 'bivariate_palette' in config:body.map_config.BIVARIATE_PALETTE=config['bivariate_palette']
    if 'scenario_colors' in config:body.cfri.SCENARIO_COLORS={int(k):v for k,v in config['scenario_colors'].items()}
    if 'scenario_linestyles' in config:body.cfri.SCENARIO_STYLES={int(k):v for k,v in config['scenario_linestyles'].items()}
    font=io.configure_style(config,{'font.size':8,'axes.labelsize':7,'axes.linewidth':.65,
        'xtick.labelsize':7,'ytick.labelsize':7,'xtick.major.size':3,'ytick.major.size':3,
        'xtick.major.width':.65,'ytick.major.width':.65,'legend.frameon':False,
        'figure.facecolor':'white','savefig.facecolor':'white','path.simplify':False})
    body.plt.rcParams['hatch.linewidth']=body.map_config.HATCH_LINEWIDTH_PT
    return body,config,font

def configured_map_period(body, config):
    period = int(config.get('map_period', body.MAP_PERIOD))
    if period not in body.PERIODS:
        raise ValueError(f'Unsupported Figure 4 map period: {period}')
    return period


def read_inputs(datasets,metadata):
    import numpy as np
    summary=io.csv('fig4_gini.csv')
    vertices=io.csv('fig4_lorenz.csv.gz')
    method=io.json_data('fig4_method.json')
    expected={(c,t) for c in datasets for t in [10,20,50,100,200]}
    summary=summary.loc[summary.city.isin(datasets)].copy()
    vertices=vertices.loc[vertices.city.isin(datasets)].copy()
    if len(summary)!=25 or summary.duplicated(['city','return_period']).any():
        raise ValueError()
    if set(zip(summary.city,summary.return_period))!=expected:raise ValueError()
    reference = metadata['exposure_reference']
    if (reference != method['exposure_reference']
            or reference['city'] not in metadata['sources']
            or not np.isfinite(reference['people']) or reference['people'] <= 0):
        raise ValueError()
    curves={}
    for (city,period),group in vertices.groupby(['city','return_period'],sort=False):
        group=group.sort_values('vertex')
        x=group.cumulative_population_share.to_numpy()
        y=group.cumulative_population_weighted_risk_share.to_numpy()
        row=summary.loc[summary.city.eq(city)&summary.return_period.eq(period)].iloc[0]
        if (x[0],y[0],x[-1],y[-1])!=(0,0,1,1) or (np.diff(x)<0).any() or (np.diff(y)<0).any():
            raise ValueError()
        np.testing.assert_allclose(1-np.sum(np.diff(x)*(y[1:]+y[:-1])),row.gini,rtol=0,atol=1e-11)
        valid=datasets[city][0].loc[datasets[city][0].analysis_included.eq(1)]
        if len(valid)!=row.analytic_grid_count:raise ValueError()
        np.testing.assert_allclose(valid.population.sum(),row.analytic_population,rtol=1e-11,atol=1e-6)
        np.testing.assert_allclose(np.dot(valid.population,valid[f'R_T{int(period)}']),
            row.population_weighted_CFRI_burden,rtol=1e-11,atol=1e-6)
        curves[(city,int(period))]=(x,y)
    if set(curves)!=expected:raise ValueError()
    return summary,curves

def run(output: Path):
    body,config,_=configured_body()
    output=io.output_directory(output)
    data,metadata=io.maps()
    summary,curves=read_inputs(data,metadata)
    period=configured_map_period(body,config)
    w,h=body.FIGURE_MM
    fig=body.plt.figure(figsize=(w/25.4,h/25.4),dpi=body.DPI)
    panels=[]
    for index,city in enumerate(body.CITY_ORDER):
        row,first_col=index//2,(index%2)*2
        for offset,kind in enumerate(['map','inequality']):
            col,letter=first_col+offset,chr(ord('a')+index*2+offset)
            ax=fig.add_axes(body.slot_bounds(row,col))
            if kind=='inequality':
                body.cfri.city_panel(ax,city,summary,curves,letter=None,show_xlabel=True,show_ylabel=col==1)
                ax._cfri_city_caption.remove()
                body.panel_letter(ax,letter)
            else:body.draw_map(ax,city,data,period,letter)
            panels.append({'ax':ax,'city':city,'kind':kind,'letter':letter,'row':row,'column':col})
    legends=body.add_legends(fig)
    qa=body.audit(fig,panels,legends,summary,curves)
    if qa['failures']:
        body.plt.close(fig)
        raise ValueError()
    trim=qa['bottom_trim_mm']/25.4
    bbox=body.Bbox.from_bounds(0,trim,w/25.4,h/25.4-trim)
    png=output/'fig4.png'
    fig.savefig(png,dpi=body.DPI,facecolor='white',bbox_inches=bbox,pad_inches=0)
    body.plt.close(fig)
    return png

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=io.ROOT/'results/figures')
    print(run(parser.parse_args().output_dir))
