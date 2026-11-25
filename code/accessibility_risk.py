"""Recompute SAR, EFAR, or the compound risk measures from local inputs."""
import argparse
from pathlib import Path
import shutil

from _release import ANALYSIS, CITIES, SUPPORT, fresh_output, load_config, record_config, run_script
from _support.analysis_paths import city_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sar", "weighted-sar", "efar", "composite", "inequality", "efarb", "fig3-data"))
    parser.add_argument("--cities", nargs="+", choices=CITIES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--composite-root", type=Path,
                        help="Composite stage output to use for inequality.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg_path, cfg = load_config("accessibility_risk", args.config)
    cities = city_selection(list(dict.fromkeys(args.cities)) if args.cities else cfg["cities"])
    if args.composite_root is not None and args.stage != "inequality":
        parser.error("--composite-root is for the inequality stage.")
    target = fresh_output(args.output_dir, "risk_"+args.stage)
    record_config(cfg_path, target)
    settings = {"ACCESS_CONFIG": cfg_path, "RELEASE_ANALYSIS_OUTPUT": target}
    if args.stage == "sar":
        for city in cities:
            run_script(SUPPORT/"sar/main.py", settings={**settings,"RELEASE_CITY":city})
    elif args.stage == "weighted-sar":
        run_script(SUPPORT/"weighted_sar/main.py","--cities",*cities,settings=settings)
    elif args.stage == "efar":
        counts = target/"facility_counts"
        for city in cities:
            directory = counts/city
            directory.mkdir(parents=True)
            name = f"{city}_grid_facility_counts.csv"
            shutil.copy2(ANALYSIS/"facility_counts"/city/name,directory/name)
        current = cfg["efar"]
        run_script(SUPPORT/"efar/main.py", "--output", counts, "--result-dir", target/"efar",
                   "--cities", *cities, "--metric", current["metric"], "--local", current["local"],
                   "--local-distance-m", current["same_grid_distance_m"], "--batch-size", current["batch_size"], settings=settings)
    elif args.stage == "composite":
        options = ["--cities", *cities]
        run_script(SUPPORT/"composite/run_release.py", *options, settings=settings)
    elif args.stage == "inequality":
        if args.composite_root is not None:
            settings["RELEASE_COMPOSITE_INPUT"] = args.composite_root.resolve()
        run_script(SUPPORT/"inequality/run_release.py", "--cities", *cities, settings=settings)
    elif args.stage == "fig3-data":
        run_script(SUPPORT/"fig3_pool/run_release.py", settings={**settings, "RELEASE_ANALYSIS_OUTPUT":target/"fig3"})
    else:
        run_script(SUPPORT/"restoration/calculation1_essential_facility_risk_burden/run_release.py", "--cities", *cities, settings=settings)


if __name__ == "__main__":
    main()

