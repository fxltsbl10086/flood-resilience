"""Run road restoration from the packaged network and frozen case definitions."""
import argparse
from pathlib import Path

from _release import PAPER_CITIES, SUPPORT, fresh_output, load_config, package_path, record_config, run_script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=("risk_burden","population_priority","random_priority","summary"))
    parser.add_argument("--cities", nargs="+", choices=PAPER_CITIES)
    parser.add_argument("--periods", nargs="+", type=int)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeat-count", type=int)
    parser.add_argument("--expected-repeats", type=int, help="Number of repeats to include in summary")
    parser.add_argument("--reference-batch", type=Path)
    parser.add_argument("--run-dir", type=Path, help="A completed comparison run for summary")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    cfg_path, cfg = load_config("road_restoration", args.config)
    reference = args.reference_batch.resolve() if args.reference_batch else package_path(cfg["risk_reference"])
    cities = list(dict.fromkeys(args.cities or PAPER_CITIES))
    periods = sorted(set(args.periods or cfg["return_periods"]))
    if not set(periods).issubset(cfg["return_periods"]):
        parser.error("")
    if args.strategy in {"random_priority","population_priority"} and set(cities).difference(PAPER_CITIES):
        parser.error("")
    if args.strategy=="summary" and not args.run_dir:
        parser.error("")
    if args.expected_repeats is not None and (args.strategy != "summary" or args.expected_repeats < 1):
        parser.error("")
    target = fresh_output(args.output_dir, "restoration_"+args.strategy)
    record_config(cfg_path, target)
    directory = SUPPORT / "restoration" / "calculation2_restoration_strategies_and_benefits"
    settings = {"RESTORATION_CONFIG": cfg_path, "RELEASE_ANALYSIS_OUTPUT": target}
    if args.strategy=="risk_burden":
        for city in cities:
            run_script(directory/"main.py", "--city", city, "--evaluation-scope", "study_area",
                       "--periods", *periods, "--output-dir", target/city, settings=settings)
    elif args.strategy=="summary":
        options = ["--run-dir", args.run_dir.resolve(), "--output-dir", target/"statistics"]
        if args.expected_repeats is not None:
            options += ["--expected-repeats", args.expected_repeats]
        run_script(directory/"summarize_comparison.py", *options, settings=settings)
    else:
        planned = cfg["random_planned_repeats"] if args.strategy=="random_priority" else 1
        count = args.repeat_count if args.repeat_count is not None else planned
        if count<1 or args.repeat_start<0 or args.repeat_start+count>planned:
            parser.error("")
        run_script(directory/"run_comparison.py","--strategy",args.strategy,"--run-dir",target/"scenarios",
                   "--source-root",reference,"--cache-root",target/"cache","--cities",*cities,
                   "--periods",*periods,"--targets",*cfg["target_percentages"],"--seed",cfg["seed"],
                   "--planned-repeats",planned,"--repeat-start",args.repeat_start,"--repeat-count",count,settings=settings)


if __name__ == "__main__":
    main()

