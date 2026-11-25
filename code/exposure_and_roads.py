"""Run population exposure and road flooding from packaged water depths."""
import argparse
from pathlib import Path

from _release import ANALYSIS, CITIES, ROOT, SUPPORT, fresh_output, load_config, record_config, run_script
from _support.analysis_paths import city_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("topology", "roads", "population", "all"))
    parser.add_argument("--cities", nargs="+", choices=CITIES)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--road-root", type=Path)
    parser.add_argument("--grid-limit", type=int, help="Small population test only")
    args = parser.parse_args()
    cfg_path, cfg = load_config("exposure_and_roads", args.config)
    cities = city_selection(list(dict.fromkeys(args.cities)) if args.cities else cfg["city_order"])
    root = args.depth_root.resolve() if args.depth_root else ROOT/"results/flood_simulation"
    roads = args.road_root.resolve() if args.road_root else ANALYSIS/"road_topology"
    settings = {"EXPOSURE_CONFIG": cfg_path, "RELEASE_DEPTH_ROOT": root}
    road_cli = SUPPORT/"road_flood/main.py"
    road_args = ["--cities", *cities, "--road-root", roads, "--depth-root", root]
    target = fresh_output(args.output_dir, "exposure_"+args.stage)
    record_config(cfg_path, target)
    settings.update(RELEASE_ANALYSIS_OUTPUT=target)
    if args.stage == "topology":
        run_script(SUPPORT/"topology/01_run_road_preprocessing.py", *cities, settings=settings)
    if args.stage in {"roads","all"}:
        run_script(road_cli, "run", *road_args, "--output-root", target/"roads",
                   "--matching-root", target/"road_matching", settings=settings)
    if args.stage in {"population","all"}:
        options = ["--cities", *cities, "--scope", "city"]
        if args.grid_limit is not None:
            options += ["--grid-limit", args.grid_limit]
        run_script(SUPPORT/"population/01_calculation100mpopulation_flood_exposure.py", *options, settings=settings)
        batches = [p.parent for p in (target/"population").rglob("*_100mspatial_metadata.json")]
        if len(batches) != len(cities):
            raise RuntimeError("")
        run_script(SUPPORT/"population/02_summary500mgrid_population.py", "--input-dir", *sorted(batches), settings=settings)


if __name__ == "__main__":
    main()

