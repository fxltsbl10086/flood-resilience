"""Package paths, portable GIS runtime and shared plotting settings."""
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'results/figure_table_data'
FIGURES=ROOT/'results/figures'
_handles=[]
_library=Path(sys.prefix)/'Library/bin'
if os.name=='nt' and _library.is_dir():
    os.environ['PATH']=str(_library)+os.pathsep+os.environ.get('PATH','')
    _handles.append(os.add_dll_directory(str(_library)))
for key,rel in [('GDAL_DATA','Library/share/gdal'),('PROJ_DATA','Library/share/proj')]:
    if (Path(sys.prefix)/rel).is_dir():os.environ.setdefault(key,str(Path(sys.prefix)/rel))
def settings(figure):
    import yaml
    from matplotlib.font_manager import findfont,FontProperties
    config_file=ROOT/'code/plotting.yaml'
    config=(yaml.safe_load(config_file.read_text(encoding='utf-8')) or {}) if config_file.exists() else {}
    result={'font_family':'DejaVu Serif','dpi':600,'preview_dpi':180}
    result.update(config.get('figure_defaults',{}))
    result.update(config.get('figures',{}).get(figure,{}) or {})
    if int(result['dpi'])<=0:raise ValueError()
    families=result['font_family']
    if isinstance(families,str):families=[families]
    for family in families:
        try:candidate=findfont(FontProperties(family=family),fallback_to_default=False)
        except ValueError:continue
        result['font_family']=FontProperties(fname=candidate).get_name()
        break
    else:raise ValueError()
    return result
def font_path(family):
    from matplotlib.font_manager import findfont,FontProperties
    return Path(findfont(FontProperties(family=family),fallback_to_default=False))
