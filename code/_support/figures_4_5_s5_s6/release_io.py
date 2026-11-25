"""Package-local, signed inputs and portable plotting settings."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'results/figure_table_data'
SUPPORT = Path(__file__).resolve().parent
_dll_handles = []
_runtime = Path(sys.prefix)/'Library'
if os.name == 'nt' and (_runtime/'bin').is_dir():
    os.environ['PATH'] = str(_runtime/'bin')+os.pathsep+os.environ.get('PATH','')
    _dll_handles.append(os.add_dll_directory(str(_runtime/'bin')))
    for key,sub in [('GDAL_DATA','share/gdal'),('PROJ_DATA','share/proj')]:
        if (_runtime/sub).is_dir(): os.environ.setdefault(key,str(_runtime/sub))

def module(name):
    key = 'release_figures45_'+name
    if key in sys.modules:return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key,SUPPORT/(name+'.py'))
    result=importlib.util.module_from_spec(spec)
    sys.modules[key]=result
    spec.loader.exec_module(result)
    return result

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def verified(name):
    manifest=json.loads((DATA/'figures_4_5_s5_s6_manifest.json').read_text('utf-8'))
    relative='results/figure_table_data/'+name
    row=next((r for r in manifest['records'] if r['destination']==relative),None)
    if row is None:raise ValueError()
    path=DATA/name
    if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
        raise ValueError()
    return path

def csv(name, **kwargs):
    import pandas as pd
    return pd.read_csv(verified(name),encoding='utf-8-sig',float_precision='round_trip',**kwargs)

def json_data(name):
    return json.loads(verified(name).read_text('utf-8-sig'))

def settings(figure):
    path=ROOT/'code/plotting.yaml'
    if not path.is_file():return {}
    import yaml
    obj=yaml.safe_load(path.read_text('utf-8')) or {}
    defaults=obj.get('figure_defaults',{})
    specific=obj.get('figures',{}).get(figure,{})
    return {**defaults,**specific}

def configure_style(config, style=None):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    requested=config.get('font_family',['Times New Roman','STIXGeneral','Liberation Serif','DejaVu Serif'])
    if isinstance(requested,str):requested=[requested]
    available={entry.name for entry in font_manager.fontManager.ttflist}
    family=next((f for f in requested if f in available),'STIXGeneral')
    chosen=dict(style or {})
    chosen.update({'font.family':family,'text.usetex':False,
        'path.simplify':config.get('path_simplify',chosen.get('path.simplify',True)),
        'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    plt.rcParams.update(chosen)
    return family

def maps():
    import geopandas as gpd
    result={}
    for city in ['zhengzhou','hangzhou','fuzhou','chengdu','wuhan']:
        path=verified(f'fig4_map_{city}.gpkg')
        grid=gpd.read_file(path,layer='grid_composite_risk')
        area=gpd.read_file(path,layer='study_area')
        result[city]=(grid,area.geometry.union_all())
    return result,json_data('fig4_map_metadata.json')

def output_directory(path):
    path=Path(path).resolve()
    path.mkdir(parents=True,exist_ok=True)
    return path
