# Flood resilience replication materials

This repository accompanies a study of flood exposure, access to essential services and road restoration in Zhengzhou, Hangzhou, Fuzhou, Chengdu and Wuhan. It covers rainfall return periods of 10, 20, 50, 100 and 200 years.

## Data and setup

Download the ZIP files you need from the [companion Zenodo deposit (DOI: 10.5281/zenodo.24492701)](https://doi.org/10.5281/zenodo.24492701). Extract each ZIP directly into this repository so that its `results/` directory merges with the others and sits alongside `code/`.

| Archive | Use |
| --- | --- |
| `figure-table-data.zip` | Redraw the figures and rebuild the supplementary tables. |
| `analysis-data.zip` | Run the analyses using the prepared spatial inputs and intermediate results. Extract alongside the figure and table archive. |
| `sfincs-<city>.zip` | Run SFINCS or export water-depth rasters for the selected city. Extract alongside the other two archives. |

### Upstream data sources

| Input | Source |
| --- | --- |
| Elevation | [FABDEM](https://research-information.bris.ac.uk/en/datasets/fabdem-2/), approximately 30 m. |
| Population | [WorldPop](https://www.worldpop.org/), China 2024, R2025A v1, approximately 100 m. |
| Land cover | [EcoVision](https://zenodo.org/records/14921585), approximately 0.5 m, supplemented with [Esri 2020 land cover](https://registry.opendata.aws/io-lulc/) at 10 m. |
| Roads | [OpenStreetMap](https://www.openstreetmap.org/). |
| Facilities | [AMap](https://developer.amap.com/) points of interest and local-government shelter lists, aggregated to 500 m grids in the analysis archive. |
| Design rainfall | Municipal and provincial rainfall intensity formulae and design profiles. The numerical inputs are in `results/flood_simulation/model_inputs/rainfall_parameters.csv`. |

The recorded formula sources are the Zhengzhou Municipal Bureau of Natural Resources and Planning notice on design rainfall intensity and hyetographs (issue details unrecorded), Zhejiang DB33/T 1191-2020 for Hangzhou (Appendix A, Table A.0.1, p. 10), Fujian DBJ/T 13-52-2021 for Fuzhou, the Chengdu Water Affairs notice No. 41 (2026, p. 4), and Wuhan DB4201/T 641-2020.

The recorded Chengdu formula covers 181–1440 min and 2–100-year return periods. Chengdu and Wuhan T200 are recorded as extrapolated. Numeric applicability bounds for Zhengzhou, Hangzhou, Fuzhou and Wuhan remain unrecorded in the shared parameter table.

Upstream products are available under the providers' respective terms.

```sh
conda env create -f environment.yml
conda activate flood-resilience
```

## Figures and tables

Run the scripts from the repository root.

```sh
python -B code/plot_fig2.py
python -B code/plot_fig3.py
python -B code/plot_fig4.py
python -B code/plot_fig5.py
python -B code/plot_supplementary.py
python -B code/make_tables.py
```

Figure 1 is a conceptual illustration. Figure 4 maps T50 and shows inequality across all five return periods. S5 uses the same bivariate thresholds to map T10, T20, T100 and T200. Figure 5 includes the Wuhan T50 spatial example for the 30% restoration target. S6 uses 23 matched city–return-period cases. Wuhan contributes T10, T20 and T50, and each other city contributes all five periods.

Pass figure IDs such as `S2 S6` to select supplementary figures. Use `--output-dir` for plots, `--tables s1 s2` to select tables and `--output-root` for table output. Each workbook includes the formatted table and a source-data sheet.

## Analysis entry points

| Entry point | Purpose |
| --- | --- |
| `code/flood_simulation.py` | Prepare SFINCS inputs, run the solver and export water-depth rasters. |
| `code/exposure_and_roads.py` | Prepare road topology and derive road inundation and population exposure. |
| `code/accessibility_risk.py` | Compute SAR, population-weighted SAR, EFAR, EFARB, composite risk, inequality and Figure 3 aggregates. |
| `code/road_restoration.py` | Simulate three road-restoration priorities and summarize a comparison run. |

The YAML files in `code/` contain the analysis parameters. Run an entry point with `--help` for its arguments. SFINCS input preparation uses `environment-sfincs.yml`. Solver runs use Windows, PowerShell, Docker and the image specified in `code/flood_simulation.yaml`.
