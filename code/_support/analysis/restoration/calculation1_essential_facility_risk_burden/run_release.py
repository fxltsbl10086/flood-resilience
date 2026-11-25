"""Reproduce the baseline population-weighted EFAR burden tables."""
import argparse
from config import CITY_ORDER,PERIODS,SERVICE_DIMENSIONS
from prepare_data import prepare
from _support.analysis_paths import CITY_CODES, access_inputs, city_selection, output_root

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cities',nargs='+',choices=CITY_CODES,default=CITY_ORDER)
    cities=city_selection(parser.parse_args().cities)
    output=output_root('efarb')
    output.mkdir(parents=True,exist_ok=False)
    prepare(output,{city:access_inputs(city) for city in cities},cities,PERIODS,SERVICE_DIMENSIONS)
