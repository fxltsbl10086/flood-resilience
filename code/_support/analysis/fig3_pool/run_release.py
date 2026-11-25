"""Recompute Fig. 3 aggregation from frozen route-decomposition statistics.

No routing or simulation is run. Formal plotting CSVs are comparison references
only. The comparison status is printed after the tables are written.
"""
from pathlib import Path
import sys,os,json
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from _release import ROOT,fresh_output
from prepare_pooled_data import prepare_pooled_data
from prepare_binned_connectivity import prepare_connectivity
from prepare_full_population_radial import prepare_data as prepare_radial

def compare(actual,filename,keys):
    reference=pd.read_csv(ROOT/'results/figure_table_data'/filename,float_precision='round_trip')
    a=actual.set_index(keys).sort_index();b=reference.set_index(keys).sort_index()
    if not a.index.equals(b.index):return {'status':'different_keys','actual_rows':len(a),'reference_rows':len(b)}
    differences={};verified=[]
    for col in a.columns.intersection(b.columns):
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            x=a[col].to_numpy(float);y=b[col].to_numpy(float)
            same=np.isclose(x,y,rtol=0,atol=1e-9,equal_nan=True)
            if same.all():verified.append(col)
            else:
                rows=[]
                for i in np.flatnonzero(~same):
                    key=a.index[i];key=key if isinstance(key,tuple) else (key,)
                    key=tuple(k.item() if isinstance(k,np.generic) else k for k in key)
                    rows.append({**dict(zip(keys,key)), 'recomputed':float(x[i]),'formal':float(y[i])})
                differences[col]=rows
    return {'status':'numeric_match' if not differences else 'differences_recorded',
            'rows':len(a),'verified_numeric_columns':verified,'differences':differences,'atol':1e-9}

def main():
    requested=os.environ.get('RELEASE_ANALYSIS_OUTPUT')
    if requested:
        output=Path(requested).resolve();output.mkdir(parents=True,exist_ok=True)
        stage_files=['pooled_model_distance_data.csv','pooled_city_population_composition.csv',
                     'model_distance_link_balance_data.csv','stacked_connectivity_data.csv',
                     'radial_full_population_data.csv']
        if any((output/name).exists() for name in stage_files):
            raise FileExistsError()
    else:output=fresh_output(None,'fig3')
    pooled,composition,_=prepare_pooled_data()
    contacts,_=prepare_connectivity(pooled)
    radial,_=prepare_radial()
    stacked=contacts.groupby(['function','period'],sort=False)[['baseline_link_count','disconnected_link_count','reachable_link_count']].sum().reset_index()
    stacked['disconnected_share']=stacked.disconnected_link_count/stacked.baseline_link_count
    stacked['reachable_share']=stacked.reachable_link_count/stacked.baseline_link_count
    frames={'pooled_model_distance_data.csv':pooled,'pooled_city_population_composition.csv':composition,
            'model_distance_link_balance_data.csv':contacts,'stacked_connectivity_data.csv':stacked,
            'radial_full_population_data.csv':radial}
    for name,frame in frames.items():frame.to_csv(output/name,index=False,encoding='utf-8-sig')
    comparisons={
        'distance_and_links':compare(contacts,'fig3_EFAR_link_balance_source_data.csv',['function','period','bin_id']),
        'stacked_contacts':compare(stacked,'fig3_EFAR_stacked_connectivity_source_data.csv',['function','period']),
        'radial':compare(radial,'fig3_Radial_figure_source_data.csv',['city','function','period']),
    }
    print(json.dumps({'output':str(output),'comparisons':{k:v['status'] for k,v in comparisons.items()}},indent=2))
if __name__=='__main__':main()
