"Read existing 500 m population and SAR tables, merge, summarize and export population-weighted SAR."

import argparse
from datetime import datetime
from time import perf_counter
import sys
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

from config import CITIES, config_for_city
from functions import merge_grid_results, summarize_population_risk, export_results


def run(cfg):
    timer = perf_counter()
    print(f"[{cfg['CITY'].title()}] Merging existing 500 m population and SAR results", flush=True)
    result = merge_grid_results(cfg)
    summary = summarize_population_risk(result, cfg)
    # Keep one city/batch directory structure and retain previous results.
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = cfg["OUTPUT_DIR"] / cfg["CITY"] / f"{stamp}_{cfg['ANALYSIS_AREA_DIR']}"
    grid_path, summary_path = export_results(result, summary, output_dir, cfg["CITY"])
    overall = summary.loc[summary["population_group"].eq("all")]
    for row in overall.itertuples(index=False):
        print(f"  {row.scenario}: population-weighted SAR={row.population_weighted_SAR:.6f}, "
              f"valid known population={row.group_valid_population:,.3f}, "
              f"coverage={row.population_coverage_rate:.2%}", flush=True)
    print(f"[{cfg['CITY'].title()}] Completed {len(result):,} grids in {perf_counter() - timer:.2f}s", flush=True)
    print(f"Grid table: {grid_path}\nSummary table: {summary_path}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cities", nargs="+", default=CITIES, help="City codes configured in config.py.")
    args = parser.parse_args()
    # Resolve all city settings before starting any batch.
    configs = [config_for_city(city) for city in args.cities]
    for cfg in configs:
        run(cfg)


if __name__ == "__main__":
    main()
