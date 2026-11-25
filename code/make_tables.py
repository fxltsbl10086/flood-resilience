"""Generate V1113 Supplementary Tables S1--S4 from frozen input/result tables.

Run from any directory: python code/make_tables.py [--tables s1 s2 s3 s4].
The supplementary tables use the five manuscript cities.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/figure_table_data'
CITIES = ('zhengzhou', 'hangzhou', 'fuzhou', 'chengdu', 'wuhan')
NAMES = dict(zip(CITIES, ('Zhengzhou', 'Hangzhou', 'Fuzhou', 'Chengdu', 'Wuhan')))
PERIODS = (10, 20, 50, 100, 200)
LAND_TYPES = ('Other impervious surfaces', 'Grass and shrubs', 'Trees', 'Bare soil',
              'Buildings', 'Water', 'Roads', 'Agricultural land')
FACILITIES = (('应急救援', 'emergency_rescue'), ('医疗资源', 'healthcare'), ('避险转移', 'emergency_shelter'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError()


def numeric(value):
    result = float(value)
    require(math.isfinite(result), f'Non-finite numeric input: {value!r}')
    return result


def model_config():
    return yaml.safe_load((ROOT / 'code/flood_simulation.yaml').read_text(encoding='utf-8'))


def parameter_path(key):
    path = Path(model_config()[key])
    require(not path.is_absolute(), f'{key} must be relative to the archive root.')
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT), f'{key} must stay within the archive.')
    return resolved


def table_s1():
    path = parameter_path('rainfall_table')
    source = [r for r in read_csv(path) if r['city_code'] in CITIES]
    records = []
    for city in CITIES:
        selected = [r for r in source if r['city_code'] == city and int(r['return_period_year']) in PERIODS]
        require(len(selected) == 5, f'{city}: expected one rainfall record for each of five return periods.')
        by_period = {int(r['return_period_year']): r for r in selected}
        require(set(by_period) == set(PERIODS), f'{city}: duplicate or missing return period.')
        coefficients = {tuple(numeric(r[k]) for k in ('a', 'c', 'b', 'n')) for r in selected}
        require(len(coefficients) == 1, f'{city}: formula coefficients differ across return periods.')
        require(all(numeric(r['duration_min']) == 360 for r in selected), f'{city}: expected 6-hour rainfall.')
        values = next(iter(coefficients))
        row = dict(city=NAMES[city], **dict(zip(('a', 'c', 'b', 'n'), values)))
        row.update({f'rainfall_T{p}_mm': numeric(by_period[p]['rainfall_6h_mm']) for p in PERIODS})
        records.append(row)
    headers = ['City', 'Coefficients (A, C, b, n)', *[f'T{p} (mm)' for p in PERIODS]]
    display = [[r['city'], '(' + ', '.join(f'{r[k]:g}' for k in ('a', 'c', 'b', 'n')) + ')',
                *[r[f'rainfall_T{p}_mm'] for p in PERIODS]] for r in records]
    return dict(title='Table S1. City-specific rainfall intensity formulae and 6-h design rainfall depths.',
                headers=headers, display=display, records=records, source=source,
                note='T denotes design-rainfall return period in years. Values are read from the frozen model rainfall table; no rainfall formula is recomputed here.',
                formats={col: '0.000' for col in range(3, 8)}, widths=[18, 40, 17, 17, 17, 17, 17])


def table_s2():
    config = model_config()
    group = str(config['soil_group']).upper()
    require(group == 'C', 'V1113 Table S2 requires the actual model soil_group C; select a different manuscript table explicitly if the model assumption changes.')
    source = read_csv(parameter_path('cn_table'))
    by_code = {int(r['landuse_code']): r for r in source}
    require(len(source) == 8 and set(by_code) == set(range(1, 9)), 'Expected exactly eight land-cover classes.')
    records = [dict(land_cover_code=i, land_cover_type=LAND_TYPES[i-1],
                    soil_group=group, curve_number=numeric(by_code[i][f'cn_{group.lower()}'])) for i in range(1, 9)]
    return dict(title='Table S2. Land-cover types and SCS curve numbers used in the flood simulations.',
                headers=['Land-cover code', 'Land-cover type', 'SCS curve number (CN)'],
                display=[[r['land_cover_code'], r['land_cover_type'], r['curve_number']] for r in records],
                records=records, source=source,
                note=f'Hydrologic soil group {group} is read from code/flood_simulation.yaml. Curve numbers are dimensionless; mixed cells use class-specific runoff weighted by area fractions.',
                formats={1: '0', 3: '0'}, widths=[22, 38, 28])


def table_s3():
    source = read_csv(SOURCE / 'table_s3_source.csv')
    records = []
    for city in CITIES:
        rows = [r for r in source if r['city'] == city]
        by_type = {r['评估维度']: r for r in rows}
        require(len(rows) == 3 and set(by_type) == {x[0] for x in FACILITIES}, f'{city}: facility categories missing or duplicated.')
        row = dict(city=NAMES[city])
        for label, key in FACILITIES:
            value = numeric(by_type[label]['格网内设施数'])
            require(value >= 0 and value.is_integer(), f'{city}: invalid facility count.')
            row[key] = int(value)
        row['total'] = sum(row[key] for _, key in FACILITIES)
        records.append(row)
    records.append(dict(city='Total', **{key: sum(r[key] for r in records) for key in ('emergency_rescue', 'healthcare', 'emergency_shelter', 'total')}))
    return dict(title='Table S3. Number of essential facilities by city and facility type.',
                headers=['City', 'Emergency rescue', 'Healthcare', 'Emergency shelter', 'Total'],
                display=[[r[k] for k in ('city', 'emergency_rescue', 'healthcare', 'emergency_shelter', 'total')] for r in records],
                records=records, source=source,
                note='Counts use the frozen 20261010 facility-merge batch and the in-grid facility count, within the five manuscript study areas.',
                formats={i: '#,##0' for i in range(2, 6)}, widths=[18, 23, 21, 24, 18])


def table_s4():
    raw = json.loads((SOURCE / 'table_s4_source.json').read_text(encoding='utf-8-sig'))
    require(raw['threshold_m'] == 0.30, 'Expected 0.30 m inundation threshold.')
    source = raw['rows']
    expected = [(NAMES[c], p) for c in CITIES for p in PERIODS]
    require([(r['city'], r['return_period_years']) for r in source] == expected, 'Unexpected Table S4 city/scenario ordering.')
    records = []
    for r in source:
        area, valid, wet, total = [numeric(r[k]) for k in ('inundated_land_area_km2', 'valid_land_area_km2', 'inundated_road_length_km', 'study_road_length_km')]
        require(0 <= area <= valid and valid > 0 and 0 <= wet <= total and total > 0, 'Invalid Table S4 numerator/denominator.')
        records.append(dict(city=r['city'], return_period_years=r['return_period_years'],
                            inundated_land_area_km2=area, inundated_land_percent=100*area/valid,
                            inundated_road_length_km=wet, inundated_road_percent=100*wet/total))
    return dict(title='Table S4. Inundation extent and road exposure under five design rainfall scenarios.',
                headers=['City', 'Rainfall return period (years)', 'Inundated land area (km²)', 'Inundated share of valid land (%)', 'Inundated road length (km)', 'Inundated share of road length (%)'],
                display=[[r['city'], r['return_period_years'], r['inundated_land_area_km2'], r['inundated_land_percent']/100, r['inundated_road_length_km'], r['inundated_road_percent']/100] for r in records],
                records=records, source=source,
                note='Inundation is maximum land water depth ≥0.30 m within 0–12 h. Road exposure counts only inundated portions of physical roads. Return periods describe rainfall inputs.',
                formats={2:'0', 3:'0.00', 4:'0.00%', 5:'#,##0.00', 6:'0.00%'}, widths=[18, 22, 24, 27, 24, 28])


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def write_workbook(path, name, table):
    book = Workbook()
    sheet = book.active
    sheet.title = f'Table {name.upper()}'
    count = len(table['headers'])
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=count)
    sheet.cell(1, 1, table['title']).font = Font(name='Times New Roman', size=12, bold=True)
    sheet.cell(1, 1).alignment = Alignment(wrap_text=True, vertical='center')
    sheet.row_dimensions[1].height = 34
    for row in [table['headers'], *table['display']]: sheet.append(row)
    last = 2+len(table['display'])
    rule = Side(style='thin', color='000000')
    for row in sheet.iter_rows(min_row=2, max_row=last, max_col=count):
        for cell in row:
            cell.font = Font(name='Times New Roman', size=10, bold=(cell.row==2))
            cell.alignment = Alignment(vertical='center', wrap_text=(cell.row==2), horizontal='left' if cell.column==1 else 'center')
            if cell.row == 2: cell.border = Border(top=rule, bottom=rule)
            if cell.row == last: cell.border = Border(bottom=rule)
            if cell.row > 2 and cell.column in table['formats']: cell.number_format=table['formats'][cell.column]
    sheet.row_dimensions[2].height = 44
    for i, width in enumerate(table['widths'], 1): sheet.column_dimensions[get_column_letter(i)].width=width
    note_row=last+2
    sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=count)
    cell=sheet.cell(note_row,1,'Note: '+table['note'])
    cell.font=Font(name='Times New Roman',size=9)
    cell.alignment=Alignment(wrap_text=True,vertical='top')
    sheet.row_dimensions[note_row].height=48
    sheet.freeze_panes='B3'
    sheet.sheet_view.showGridLines=False
    sheet.print_title_rows='1:2'
    sheet.sheet_properties.pageSetUpPr.fitToPage=True
    sheet.page_setup.orientation='landscape'
    sheet.page_setup.paperSize=sheet.PAPERSIZE_A4
    sheet.page_setup.fitToWidth=1
    sheet.page_setup.fitToHeight=0
    sheet.print_options.horizontalCentered=True
    source=book.create_sheet('Source data')
    headers=list(dict.fromkeys(k for r in table['source'] for k in r))
    source.append(headers)
    for row in table['source']: source.append([row.get(k) for k in headers])
    source.freeze_panes='A2'
    source.auto_filter.ref=source.dimensions
    for cell in source[1]:
        cell.font=Font(name='Calibri',size=10,bold=True)
        cell.fill=PatternFill('solid',fgColor='E9EEF2')
        cell.alignment=Alignment(wrap_text=True)
    for i in range(1,len(headers)+1): source.column_dimensions[get_column_letter(i)].width=24
    source.row_dimensions[1].height=32
    book.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables',nargs='+',choices=['s1','s2','s3','s4'],default=['s1','s2','s3','s4'])
    parser.add_argument('--output-root',type=Path,default=ROOT/'results/recomputed/tables',help='Writes figure_table_data/ and tables/ below this directory.')
    args=parser.parse_args()
    data=args.output_root/'figure_table_data'; tables=args.output_root/'tables'
    data.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)
    for name in dict.fromkeys(args.tables):
        result=globals()[f'table_{name}']()
        write_csv(data/f'table_{name}.csv',result['records'])
        (data/f'table_{name}.json').write_text(json.dumps(dict(table=name,records=result['records'],note=result['note']),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        write_workbook(tables/f'table_{name}.xlsx',name,result)
        print(f'{name.upper()}: {len(result["records"])} rows; {tables/f"table_{name}.xlsx"}')


if __name__=='__main__': main()
