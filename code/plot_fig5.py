"""Redraw the 20261118 Figure 5 from signed package-local display inputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'_support/figures_4_5_s5_s6'))
import release_io as io

def load_inputs():
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    b,c,d,total=[io.module(n) for n in ['fig5_b','fig5_c','fig5_d','fig5_draw']]
    frame=io.csv('fig5_a_curves.csv')
    incremental=io.csv('fig5_a_incremental.csv')
    ci=io.csv('fig5_a_ci95.csv')
    knee=io.json_data('fig5_a_knee.json')
    points=io.csv('fig5_b_points.csv')
    curve=io.csv('fig5_b_ols_ci95.csv')
    if len(points)!=25 or len(curve)!=301:raise ValueError()
    colors={'zhengzhou':'#B99120','hangzhou':'#4DAA91','fuzhou':'#F28E45','chengdu':'#2878B5','wuhan':'#DB668A'}
    markers={10:'o',20:'s',50:'^',100:'D',200:'p'}
    quantiles=pd.read_csv(io.verified('fig5_c_quantiles.csv'),encoding='utf-8-sig')
    grids=pd.read_csv(io.verified('fig5_c_grid.csv'),encoding='utf-8-sig',
        usecols=['city','period','grid_id','target_pct','plan_index','effective_population','delta_risk'])
    densities=io.csv('fig5_c_kde.csv')
    if len(grids)!=75545 or len(densities)!=25*601:raise ValueError()
    if grids.duplicated(['city','period','grid_id']).any():raise ValueError()
    if not (grids.target_pct.eq(30).all() and grids.plan_index.eq(8).all()):
        raise ValueError()
    indexed=quantiles.set_index(['city','period'])
    groups={}
    for i,city in enumerate(c.CITY_ORDER):
        for period in c.PERIODS:
            part=grids.loc[grids.city.eq(city)&grids.period.eq(period)].sort_values('grid_id',kind='stable')
            density=densities.loc[densities.city.eq(city)&densities.period.eq(period)].sort_values('delta_risk')
            values,population=part.delta_risk.to_numpy(float),part.effective_population.to_numpy(float)
            np.testing.assert_allclose(c.weighted_quantiles(values,population),
                indexed.loc[(city,period),c.QUANTILE_COLUMNS].to_numpy(float),rtol=0,atol=1e-12)
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([c.JITTER_SEED,i,period])))
            groups[(city,period)]={'grid_ids':part.grid_id.to_numpy(),'values':values,
                'population':population,'jitter':rng.uniform(*c.RAIN_X_OFFSETS,size=len(part)),
                'density_y':density.delta_risk.to_numpy(float),
                'density':density.population_weighted_density.to_numpy(float)}
    spatial=io.verified('fig5_d_spatial.gpkg')
    layers={name:gpd.read_file(spatial,layer=name) for name in [
        'resident_grids','study_area','inundation','restored_roads','depth_coverage']}
    water_source=io.verified('fig5_d_osm_water.gpkg')
    water={'water_polygon':gpd.read_file(water_source,layer='water_polygon').to_crs('EPSG:4326')}
    return total,b,c,d,frame,incremental,ci,knee,points,curve,colors,markers,quantiles,groups,layers,water

def run(output: Path):
    total,b,c,d,frame,incremental,ci,knee,points,curve,colors,markers,quantiles,groups,layers,water=load_inputs()
    config=io.settings('fig5')
    total.POSITIONS.update(config.get('positions',{}))
    total.B_LEGEND_ANCHORS=config.get('b_legend_anchors',total.B_LEGEND_ANCHORS)
    total.C_LEGEND_ANCHORS=config.get('c_legend_anchors',total.C_LEGEND_ANCHORS)
    width=float(config.get('width_mm',183))
    height=float(config.get('height_mm',150))
    trim=float(config.get('bottom_trim_mm',20))
    dpi=int(config.get('dpi',600))
    if not 0<=trim<height:raise ValueError()
    io.configure_style(config,total.STYLE)
    fig=total.plt.figure(figsize=(width/25.4,height/25.4),dpi=dpi)
    axes={name:fig.add_axes(total.POSITIONS[name]) for name in 'abcd'}
    right,a_legend,annotations,_=total.draw_a(axes['a'],frame,incremental,ci,knee)
    axes['a_right']=right
    b_legends,b_artists=total.draw_b(fig,axes['b'],b,points,curve,colors,markers)
    proxy=total.PanelCCanvas(fig,axes['c'])
    _,c_artists,c_legends,c_positions=c.draw_panel(proxy,quantiles,colors,markers,groups)
    axes['c'].tick_params(labelsize=7)
    axes['c'].yaxis.label.set_fontsize(7)
    c.check_layout(fig,axes['c'],c_artists,c_legends,c_positions,groups)
    for legend in c_legends:legend.remove()
    colorbar,map_label,_=total.draw_d(fig,axes['d'],d,layers,water)
    axes['colorbar']=colorbar
    for x,y,letter in [(.025,.985,'a'),(.540,.985,'b'),(.610,.555,'d')]:
        fig.text(x,y,letter,fontsize=10,fontweight='bold',ha='left',va='top')
    export=total.Bbox.from_bounds(0,trim/25.4,width/25.4,(height-trim)/25.4)
    total.layout_qa(fig,axes,[a_legend,*b_legends],annotations,map_label,b_artists,c_artists,groups,export)
    b.check_layout(fig,axes['b'],points,[],next(t for t in fig.texts if t.get_text()=='b'),b_legends)
    output=io.output_directory(output)
    png=output/'fig5.png'
    fig.savefig(png,dpi=dpi,facecolor='white',format='png',bbox_inches=export,pad_inches=0,
        metadata={'Title':'Figure 5','Software':'matplotlib'})
    total.plt.close(fig)
    return png

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=io.ROOT/'results/figures')
    print(run(parser.parse_args().output_dir))
