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
STYLE = {'font.family':'Times New Roman','font.size':7,'axes.labelsize':7,
         'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,
         'legend.fontsize':6,'legend.frameon':False,'axes.unicode_minus':False,
         'xtick.direction':'out','ytick.direction':'out','xtick.major.width':.65,
         'ytick.major.width':.65,'text.color':'black','axes.labelcolor':'black',
         'xtick.color':'black','ytick.color':'black','figure.facecolor':'white',
         'axes.facecolor':'white','savefig.facecolor':'white','text.usetex':False}
CITY_ORDER = ['zhengzhou','hangzhou','fuzhou','chengdu','wuhan']
PERIODS = [10,20,50,100,200]
POSITIONS = {'a':[.080,.625,.395,.325], 'b':[.595,.625,.382,.325],
             'c':[.080,.180,.500,.340], 'd':[.630,.180,.287,.340],
             'colorbar':[.932,.180,.012,.340]}
B_LEGEND_ANCHORS = ((.76,.97),(.98,.97))
C_LEGEND_ANCHORS = ((.200,.135),(.465,.135))
BOTTOM_TRIM_MM = 20.0
def require(condition, message):
    if not condition: raise ValueError()
class PanelCCanvas:
    "Adapt panel-c axes and text positions while retaining its drawing loop."
    def __init__(self,fig,ax): self.fig,self.ax,self.legends = fig,ax,[]
    def add_axes(self,_): return self.ax
    def text(self,*_,**__): return self.fig.text(.025,.555,'c',fontsize=10,fontweight='bold',ha='left',va='top')
    def legend(self,handles,**kwargs):
        index = len(self.legends)
        if index==0:
            handles[0].set_label('Left to right: T10, T20, T50, T100, T200')
        kwargs.update(bbox_to_anchor=C_LEGEND_ANCHORS[index],bbox_transform=self.fig.transFigure,fontsize=6)
        if index==1:
            kwargs.update(handlelength=1.25,columnspacing=.8,handletextpad=.4)
        legend = self.fig.legend(handles=handles,**kwargs)
        self.legends.append(legend)
        return legend
def draw_a(ax, frame, incremental, ci, knee):
    right = ax.twinx()
    colors = {'risk_priority':'#CC79A7','population_priority':'#D99A5B','random_priority':'#7C858E'}
    names = ['Risk-burden priority','Exposure priority','Random']
    lines,gains = [],[]
    for strategy in colors:
        band = ci.loc[ci.strategy.eq(strategy)].sort_values('restoration_pct')
        ax.fill_between(band.restoration_pct,band.CI95_lower,band.CI95_upper,
                        color=colors[strategy],alpha=.20,linewidth=0,zorder=1)
        curve = frame.loc[frame.strategy.eq(strategy)].sort_values('restoration_pct')
        gain = incremental.loc[incremental.strategy.eq(strategy)].sort_values('restoration_pct')
        lines += ax.plot(curve.restoration_pct,curve.mean_EFARB_reduction,color=colors[strategy],lw=1.15,zorder=3)
        gains += right.plot(gain.restoration_pct,gain.incremental_efficiency_pp_per_pp,
                            color=colors[strategy],lw=.85,ls=(0,(3,2)),zorder=3)
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='Road restoration target (%)',ylabel='EFAR reduction (%)')
    ax.set_xticks(np.arange(0,101,10)); ax.set_yticks(np.arange(0,101,20))
    right.set_ylabel('Incremental EFAR reduction (pp/pp)',fontsize=7,labelpad=4)
    right.set_ylim(0,float(np.nanmax(incremental.incremental_efficiency_pp_per_pp))*1.08)
    right.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
    right.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
    for owner in [ax,right]:
        owner.tick_params(length=2.5,pad=2,labelsize=7)
        owner.grid(False)
    ax.spines['right'].set_visible(False)
    for side in ['left','top','bottom']: right.spines[side].set_visible(False)
    annotations,records = [],[]
    # Label the existing 25/30/35 nodes without interpolation or a new knee calculation.
    positions = {'cumulative':[(-11,3,'right','bottom'),(0,13,'center','bottom'),(12,-8,'left','top')],
                 'incremental':[(-10,-6,'right','top'),(0,6,'center','bottom'),(13,-9,'left','top')]}
    for kind,owner,table,value in [('cumulative',ax,frame,'mean_EFARB_reduction'),
                                   ('incremental',right,incremental,'incremental_efficiency_pp_per_pp')]:
        table = table.loc[table.strategy.eq('risk_priority')]
        for target,placement in zip([25,30,35],positions[kind]):
            row = table.loc[table.restoration_pct.eq(target)].iloc[0]
            x,y = float(row.restoration_pct),float(row[value])
            owner.plot(x,y,marker='o',ms=2.6,ls='none',mfc='white' if kind=='cumulative' else colors['risk_priority'],
                       mec=colors['risk_priority'],mew=.7,zorder=6)
            if kind=='cumulative':
                ax.plot([x,x],[0,y],color='#888888' if target==30 else '#AAAAAA',
                        lw=.6 if target==30 else .45,ls=(0,(3,2)),zorder=.8)
            dx,dy,ha,va = placement
            annotation = owner.annotate(f'{y:.2f}',xy=(x,y),xytext=(dx,dy),textcoords='offset points',
                ha=ha,va=va,fontsize=6,color=colors['risk_priority'],zorder=8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.2},
                arrowprops={'arrowstyle':'-','color':colors['risk_priority'],'linewidth':.4,'shrinkA':2,'shrinkB':2})
            annotations.append(annotation)
            records.append({'curve':kind,'target_pct':target,'value':y,'on_existing_node':True})
    shade = tuple(Patch(facecolor=colors[s],alpha=.20,edgecolor='none') for s in colors)
    legend = ax.legend([*lines,shade],[*names,'95% CI'],loc='lower right',bbox_to_anchor=(.99,.28),
        fontsize=6,handlelength=1.7,labelspacing=.35,handletextpad=.4,borderaxespad=0,
        handler_map={tuple:HandlerTuple(ndivide=None,pad=.1)})
    return right,legend,annotations,{'curves':6,'confidence_bands':3,'annotations':records,
        'knee_target_pct':float(knee['knee_target_pct']),'knee_reduction_pct':float(knee['reduction_at_knee_pct'])}
def draw_b(fig, ax, module, points, curve, colors, markers):
    artists,_,old_letter = module.draw_panel(ax,points,colors,markers,letter='',curve=curve)
    old_letter.remove()
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_fontsize(7); ax.yaxis.label.set_fontsize(7)
    cities = [Patch(facecolor=colors[c],edgecolor='none',label=module.CITY_LABELS[c]) for c in CITY_ORDER]
    scenarios = [Line2D([],[],color='#555555',ls='none',marker=markers[p],mfc='none',ms=3.7,mew=.75,label=f'T{p}') for p in PERIODS]
    scenarios.extend([
                      Patch(facecolor=module.FIT_COLOR,alpha=module.CI_ALPHA,edgecolor='none',label='95% CI')])
    first = ax.legend(handles=cities,loc='upper right',bbox_to_anchor=B_LEGEND_ANCHORS[0],fontsize=6,
        handlelength=.9,handletextpad=.35,labelspacing=.35,borderaxespad=0)
    ax.add_artist(first)
    second = ax.legend(handles=scenarios,loc='upper right',bbox_to_anchor=B_LEGEND_ANCHORS[1],fontsize=6,
        handlelength=1.25,handletextpad=.35,labelspacing=.35,borderaxespad=0)
    return [first,second],artists
def draw_d(fig, ax, original, layers, water):
    mappable,highlighted,display = original.draw_case_map(ax,layers)
    # Fit the full resident-grid and study-area extent to the physical map frame.
    # Set the axes geometry before the plotting helper uses it.
    ax.set_position(POSITIONS['d'])
    bounds = np.vstack([display['resident_grids'].total_bounds,display['study_area'].total_bounds])
    xmin,ymin,xmax,ymax = bounds[:,0].min(),bounds[:,1].min(),bounds[:,2].max(),bounds[:,3].max()
    cx,cy = (xmin+xmax)/2,(ymin+ymax)/2
    aspect = 1/np.cos(np.deg2rad(cy))
    physical_aspect = POSITIONS['d'][2]*fig.get_figwidth()/(POSITIONS['d'][3]*fig.get_figheight())
    sy = max((ymax-ymin)/.94,(xmax-xmin)/(physical_aspect*aspect*.94))
    sx = sy*physical_aspect*aspect
    ax.set_xlim(cx-sx/2,cx+sx/2); ax.set_ylim(cy-sy/2,cy+sy/2)
    ax.set_aspect(aspect,adjustable='box')
    ax.set_xticks(original.geographic_ticks(*ax.get_xlim()))
    ax.set_yticks(original.geographic_ticks(*ax.get_ylim()))
    xlim,ylim,aspect = ax.get_xlim(),ax.get_ylim(),ax.get_aspect()
    water['water_polygon'].plot(ax=ax,facecolor='#96DBF2',edgecolor='none',linewidth=0,zorder=.3)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect(aspect,adjustable='box')
    ax.tick_params(labelsize=7)
    label = ax.text(.025,.025,'Wuhan · T50',transform=ax.transAxes,ha='left',va='bottom',fontsize=7,zorder=10)
    cax = fig.add_axes(POSITIONS['colorbar'])
    bar = fig.colorbar(mappable,cax=cax,orientation='vertical',ticks=[0,.5,1])
    bar.set_label('Composite EFAR improvement',fontsize=7,labelpad=3)
    bar.ax.yaxis.set_label_position('right')
    bar.ax.yaxis.set_ticks_position('right')
    bar.ax.tick_params(labelsize=7,length=2,width=.65,pad=1.5)
    # Describe categorical spatial layers in the caption and show the continuous colorbar in the panel.
    return cax,label,{'case':'Wuhan T50','resident_grids':len(layers['resident_grids']),
        'highlighted_grid_count':highlighted,'restored_roads':len(layers['restored_roads']),
        'color_limits':[mappable.norm.vmin,mappable.norm.vmax],
        'water_fill':'#96DBF2','waterway_lines':0,'extent':[list(ax.get_xlim()),list(ax.get_ylim())],
        'geometry_clipped_for_export':False,'categorical_legend_shown':False,
        'colorbar_orientation':'vertical','colorbar_location':'right_of_map'}
def layout_qa(fig, axes, legends, annotations, map_label, scatter_artists, c_artists, groups, export_bbox):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = export_bbox.transformed(fig.dpi_scale_trans)
    records = []
    texts = [text for text in fig.findobj(match=Text) if text.get_visible() and text.get_text()]
    for text in texts:
        bb = text.get_window_extent(renderer)
        inside = bb.x0>=bounds.x0-.5 and bb.y0>=bounds.y0-.5 and bb.x1<=bounds.x1+.5 and bb.y1<=bounds.y1+.5
        records.append({'text':text.get_text(),'fontsize':text.get_fontsize(),'inside_canvas':bool(inside),
                        'bounds_px':list(map(float,bb.extents))})
    require(all(row['inside_canvas'] for row in records),'')
    for name,ax in axes.items():
        for axis_name,labels in [('x',ax.get_xticklabels()),('y',ax.get_yticklabels())]:
            boxes = [item.get_window_extent(renderer) for item in labels if item.get_visible()]
            require(not any(a.overlaps(b) for a,b in zip(boxes[:-1],boxes[1:])),'')
    legend_boxes = [item.get_window_extent(renderer) for item in legends]
    require(not any(a.overlaps(b) for i,a in enumerate(legend_boxes) for b in legend_boxes[i+1:]),'')
    annotation_boxes = [item.get_bbox_patch().get_window_extent(renderer) for item in annotations]
    require(not any(a.overlaps(b) for i,a in enumerate(annotation_boxes) for b in annotation_boxes[i+1:]),'')
    np.testing.assert_allclose(axes['a'].get_position().x0,axes['c'].get_position().x0,rtol=0,atol=1e-10)
    a_right_label = axes['a_right'].yaxis.label.get_window_extent(renderer)
    b_left_label = axes['b'].yaxis.label.get_window_extent(renderer)
    require(a_right_label.x1 < b_left_label.x0,'')
    colorbar_bounds = axes['colorbar'].get_window_extent(renderer)
    require(colorbar_bounds.x0 > axes['d'].get_window_extent(renderer).x1,'')
    require(colorbar_bounds.height > colorbar_bounds.width,'')
    np.testing.assert_allclose(axes['colorbar'].get_position().bounds[1::2],
                               axes['d'].get_position().bounds[1::2],rtol=0,atol=1e-10)
    require(not fig.legends,'')
    np.testing.assert_allclose(axes['c'].get_position().bounds[1::2],axes['d'].get_position().bounds[1::2],rtol=0,atol=1e-10)
    map_bounds = axes['d'].get_window_extent(renderer)
    np.testing.assert_allclose(axes['d'].get_position().bounds,POSITIONS['d'],rtol=0,atol=1e-10)
    bb = map_label.get_window_extent(renderer)
    require(map_bounds.contains(bb.x0,bb.y0) and map_bounds.contains(bb.x1,bb.y1),'')
    require(len(scatter_artists)==25 and len(c_artists)==25,'')
    count = 0
    for item,group in zip(c_artists,groups.values()):
        rain = item['rain']; offsets = np.asarray(rain.get_offsets())
        require(len(offsets)==len(group['values']),'')
        np.testing.assert_array_equal(offsets[:,1],group['values'])
        np.testing.assert_array_equal(rain.get_sizes(),group['population']/4000)
        count += len(offsets)
    return {'status':'PASS','text':records,'all_text_inside_canvas':True,'tick_labels_nonoverlapping':True,
        'legends_nonoverlapping':True,'a_numeric_annotations_nonoverlapping':True,
        'b_points':25,'c_original_grid_scatter_points':count,'c_groups':25,'map_label_inside':True,
        'map_frame_matches_specified_physical_position':True,
        'outer_left_edges_aligned':True,'top_y_axis_labels_separated':True,
        'c_legends_removed':True,'c_d_frame_bottom_and_height_equal':True,
        'vertical_colorbar_right_of_map':True,'colorbar_top_bottom_aligned_with_map':True,
        'export_bounds_inches':list(export_bbox.bounds),
        'axes_positions':{name:list(ax.get_position().bounds) for name,ax in axes.items()},
        'visual_review_required':True}
