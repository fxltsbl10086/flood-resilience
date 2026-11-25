from _support.analysis_paths import CITY_CODES, city_paths, config, city_selection
p=config('accessibility_risk','ACCESS_CONFIG')
CITIES=city_selection(p['composite']['reference_cities'])
SUPPORTED_CITIES=tuple(CITY_CODES)
CITY_SEED_OFFSETS={'chengdu':0,'fuzhou':1000,'hangzhou':2000,
                   'wuhan':4000,'zhengzhou':5000}
MAP_CITY_ORDER=CITIES
PERIODS=p['return_periods']
DIMENSIONS=['应急救援','医疗资源','避险转移']
DIMENSION_KEYS=['rescue','medical','shelter']
FUNCTION_COLORS={'rescue':'#D55E00','medical':'#0072B2','shelter':'#009E73'}
EXPOSURE_COLOR='#C65D57'
FIGURE_MM=(160,120)
DPI=600
FIGURE_STYLE_METADATA={}
BIVARIATE_PALETTE=[
    ["#E8E8E8", "#B6DDDA", "#69C2BF"],
    ["#D8B4D7", "#AAA5CD", "#6B8EAE"],
    ["#B767B9", "#865D9B", "#3F4E7C"],
]

def input_paths(city):
    f=city_paths(city)
    return {'grid':f['grid'],'boundary':f['boundary'],'population':f['pop'],
            'services':f['efar'],'service_metadata':f['efar_metadata']}
