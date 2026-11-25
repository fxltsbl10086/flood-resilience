"""Prepare or run SFINCS simulations and export water-depth rasters."""
import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import shutil

from _release import SUPPORT, CITIES, fresh_output, load_config, package_path, record_config, run_script
from _support.analysis_paths import city_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "export", "all"))
    parser.add_argument("--cities", nargs="+", choices=CITIES)
    parser.add_argument("--periods", nargs="+", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-inputs", type=Path, help="Existing SFINCS inputs for run or export")
    parser.add_argument("--native-results", type=Path, help="Read-only native output for export")
    parser.add_argument("--output-dir", type=Path, help="New batch directory for prepare/run/export/all")
    args = parser.parse_args()
    cfg_path, cfg = load_config("flood_simulation", args.config)
    archive = package_path(cfg["archive"])
    default_cities = cfg.get("default_cities", cfg["cities"])
    cities = city_selection(list(dict.fromkeys(args.cities or default_cities)))
    if not set(cities).issubset(cfg["cities"]):
        parser.error("Select cities listed in the simulation configuration.")
    periods = sorted(set(args.periods or cfg["return_periods"]))
    if not set(periods).issubset(cfg["return_periods"]):
        parser.error("")
    model_root = args.model_inputs.resolve() if args.model_inputs else package_path(cfg["model_inputs"])
    scenarios = [f"T{p:03d}_D12mmh" for p in periods]
    model_files = [
        model_root/city/scenario/name for city in cities for scenario in scenarios
        for name in ("sfincs.inp","sfincs_cn_runoff.nc")
    ] + [model_root/city/f"sfincs.{suffix}" for city in cities for suffix in ("dep","ind","msk","man","qinf")]
    if args.action != "prepare":
        missing = [str(p) for p in model_files if not p.is_file()]
        if missing:
            raise FileNotFoundError()
    if args.action in {"run", "all"} and not (shutil.which("docker") and (shutil.which("pwsh") or shutil.which("powershell"))):
        parser.error("")
    if args.action == "prepare" and find_spec("hydromt_sfincs") is None:
        parser.error("")
    target = fresh_output(args.output_dir, "sfincs_"+args.action, archive)
    record_config(cfg_path, target)
    settings = {"FLOOD_CONFIG": cfg_path}
    support = SUPPORT / "sfincs"
    if args.action == "prepare":
        settings["FLOOD_MODEL_OUTPUT"] = target
        settings["NUMBA_CACHE_DIR"] = os.environ.get("NUMBA_CACHE_DIR") or target / "_numba_cache"
        run_script(support/"inputs/build_inputs.py", "--cities", *cities, "--return-periods", *periods, settings=settings)
        return
    if args.action == "export":
        settings["FLOOD_SCHEDULER_ROOT"] = target / "_scheduler"
        src = args.native_results.resolve() if args.native_results else archive
        for city in cities:
            for scenario in scenarios:
                case = target/city/scenario
                case.mkdir(parents=True)
                shutil.copy2(src/city/scenario/"sfincs_map.nc", case/"sfincs_map.nc")
                for name in ("sfincs.inp", "run_status.json"):
                    file = src/city/scenario/name
                    if file.is_file():
                        shutil.copy2(file, case/name)
    run_script(support/"inundation/simulation_code/main.py", args.action, "--cities", *cities,
               "--return-periods", *periods, "--input-root", model_root, "--output-root", target,
               "--image", cfg["image"], "--threads", cfg["threads"], "--max-concurrent", cfg["max_concurrent"],
               "--without-roads", settings=settings)


if __name__ == "__main__":
    main()

