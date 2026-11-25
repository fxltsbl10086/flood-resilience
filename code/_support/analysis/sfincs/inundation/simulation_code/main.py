"""Run, export, plot or collect SFINCS scenarios across cities.

Example: main.py all --city fuzhou --return-periods 10 20 50 100 200
City data and experiment parameters are read from the shared input configuration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

SFINCS = Path(__file__).resolve().parents[2]
INPUT_CODE = SFINCS / "inputs"
CODE = SFINCS.parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
RESULTS = CODE.parent / "results" / "flood_simulation"
DEFAULT_INPUT_ROOT = RESULTS / "model_inputs"
DEFAULT_OUTPUT_ROOT = RESULTS / "recomputed"
DEFAULT_IMAGE = "deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"


def arguments() -> argparse.Namespace:
    """Select actions and cities while retaining physical settings in the input configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "export", "plot", "all", "status"],
                        help="Run, export, plot, execute all three, or collect existing run status")
    parser.add_argument("--city", "--cities", dest="cities", nargs="+", required=True,
                        help="One or more city codes, for example fuzhou wuhan")
    parser.add_argument("--return-periods", nargs="+", type=int,
                        help="Selected return periods; defaults to all configured periods for each city")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help="Parent model-input directory containing one directory per city")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Parent result directory containing one directory per city")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads per solver")
    parser.add_argument("--max-concurrent", type=int, choices=[1, 2], default=2,
                        help="Project-wide limit of one or two active scenarios; paused containers do not occupy slots")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Installed SFINCS Docker image")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-export existing GeoTIFFs without overwriting NetCDF or rerunning the solver")
    parser.add_argument("--dpi", type=int, default=220, help="PNG resolution in dpi")
    parser.add_argument("--roads", type=Path, help="Road vector for a single city; defaults to the configured city roads")
    parser.add_argument("--boundary", type=Path, help="Optional study-area boundary for a single city")
    parser.add_argument("--without-roads", action="store_true", help="Omit the road overlay")
    args = parser.parse_args()
    args.cities = list(dict.fromkeys(args.cities))
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", city) for city in args.cities):
        parser.error("")
    if not 1 <= args.threads <= 64:
        parser.error("")
    if args.dpi < 100:
        parser.error("")
    if len(args.cities) > 1 and (args.roads or args.boundary):
        parser.error("")
    if args.overwrite and args.action not in {"export", "all"}:
        parser.error("")
    return args


def city_config(city: str) -> dict:
    # Reuse the input configuration as the single source of paths, classes and durations.
    if str(INPUT_CODE) not in sys.path:
        sys.path.insert(0, str(INPUT_CODE))
    from build_inputs import load_config
    return load_config(city)


def _write_status(path: Path, data: dict) -> None:
    """Atomically replace status JSON while holding ownership of the city result directory."""
    from src.runtime import _write_json
    _write_json(path, data)


def record_export(result_root: Path, periods: list[int], records: list[dict]) -> None:
    """Record exported files while preserving historical statistics and migration provenance."""
    path = result_root / "batch_run_status.json"
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    exported = sorted(set(data.get("exported_return_periods", [])) | set(periods))
    requested = set(data.get("requested_return_periods", periods))
    data.update(
        result_root=str(result_root), exported_return_periods=exported,
        geotiff_export_complete=requested.issubset(exported),
        latest_depth_export={"updated_utc": datetime.now(timezone.utc).isoformat(),
                             "products_per_scenario": 6, "scenarios": records},
    )
    # Advance execution status only after all requested scenarios are solved and exported.
    finished = set(data.get("solver_finished_return_periods", data.get("solver_finished", [])))
    invalid = (data.get("scientific_use_valid") is False
               or str(data.get("status", "")).startswith("invalid")
               or data.get("status") == "failed_forcing_reception")
    if requested.issubset(exported) and requested.issubset(finished) and not invalid:
        has_notes = (data.get("status") == "completed_with_quality_notes"
                     or any(state.get("status") == "completed_with_quality_notes"
                            for state in data.get("scenario_states", {}).values()))
        data["status"] = "completed_with_quality_notes" if has_notes else "completed"
    # A complete file list does not establish peak convergence or observational validation.
    _write_status(path, data)


def record_summary(result_root: Path, info: dict) -> None:
    """Link summaries to run records without treating solver completion as physical validation."""
    notes = []
    written = set(info["written_files"])
    for summary in info["summaries"]:
        scenario = summary["scenario"]
        case_notes = list(summary["rainfall_application_notes"])
        case_notes.extend(summary.get("forcing_reception_failure_notes", []))
        negative = summary["domain_cells_below_minus_0p01_at_any_saved_time"]
        if negative:
            case_notes.append(f"Saved frames contain {negative} active cells with depth below -0.01 m; study-area land cells: "
                              f"{summary['land_negative_cells_below_minus_0p01']}.")
        nonfinite = summary["native_nonfinite_active_values_at_saved_times"]
        if nonfinite:
            case_notes.append(f"Saved frames contain {nonfinite} non-finite depth values in active cells.")
        rising = summary["late_rising_near_peak_cells"]
        if rising:
            case_notes.append(f"At the end, {rising} land cells with depth >=0.30 m remain near their recorded peak and continue rising; "
                              "results represent maxima over the specified simulation window.")
        notes.extend(f"{scenario}：{note}" for note in case_notes)
        # Add status links only to newly written summaries; preserve historical Fuzhou summaries.
        case_summary = result_root / scenario / "result_summary.json"
        run_path = result_root / scenario / "run_status.json"
        if str(case_summary) in written and run_path.is_file():
            run = json.loads(run_path.read_text(encoding="utf-8-sig"))
            # Forcing reception confirms execution of that input, not scientific validity.
            invalid = (run.get("scientific_use_valid") is False
                       or str(run.get("status", "")).startswith("invalid")
                       or run.get("status") == "failed_forcing_reception")
            if not invalid:
                run["status"] = "completed_with_quality_notes" if case_notes else "completed"
            run.update(result_summary=str(case_summary), quality_notes=case_notes,
                       forcing_reception_validation_status=summary.get("forcing_reception_validation_status"),
                       forcing_reception=summary.get("forcing_reception"))
            _write_status(run_path, run)
    path = result_root / "batch_run_status.json"
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    data["latest_result_summary"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "return_periods": info["return_periods"], "summary_csv": info["summary_csv"],
        "summary_json": info["summary_json"],
        "written_files": info["written_files"], "reused_files": info["reused_files"],
        "forcing_reception_validation": {
            summary["scenario"]: summary.get("forcing_reception_validation_status", "not_checked")
            for summary in info["summaries"]
        },
    }
    if notes and data.get("status") == "completed":
        data["status"] = "completed_with_quality_notes"
    data["latest_summary_quality_notes"] = notes
    _write_status(path, data)
    for note in notes:
        print(f"[{info['city']}] Result note: {note}", flush=True)


def execute_city(args: argparse.Namespace, city: str, *, skip_run: bool = False) -> None:
    config = city_config(city)
    periods = sorted(set(args.return_periods or config["return_periods"]))
    if not periods or any(period not in config["return_periods"] for period in periods):
        raise ValueError()
    model_root = (args.input_root / city).resolve()
    result_root = (args.output_root / city).resolve()
    if (model_root == result_root or model_root in result_root.parents
            or result_root in model_root.parents):
        raise ValueError()
    print(f"[{city}] {args.action}; return periods={periods}; results={result_root}", flush=True)
    # Import on demand so command-line help does not load raster or plotting libraries.
    if args.action in {"run", "all"} and not skip_run:
        from src.runtime import run_city
        run_city(city, model_root, result_root, periods,
                 threads=args.threads, max_concurrent=args.max_concurrent, image=args.image)
    if args.action in {"export", "all"}:
        from src.export_depth import export_depth
        records = export_depth(city, model_root, result_root, periods, overwrite=args.overwrite)
        record_export(result_root, periods, records)
        from src.summary import summarize_city
        info = summarize_city(city, model_root, result_root, periods, config=config)
        record_summary(result_root, info)
        print(f"[{city}] Summary saved: {info['summary_csv']}", flush=True)
    if args.action in {"plot", "all"}:
        # Do not plot results already invalidated by forcing checks.
        status_path = result_root / "batch_run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
        if (status.get("scientific_use_valid") is False
                or str(status.get("status", "")).startswith("invalid")
                or status.get("status") == "failed_forcing_reception"):
            raise RuntimeError()
        from src.plotting import plot_city
        info = plot_city(city, model_root, result_root, periods, config=config, dpi=args.dpi,
                         roads=args.roads, boundary=args.boundary, without_roads=args.without_roads)
        for note in info.get("notes", []):
            print(f"[{city}] Plot note: {note}", flush=True)
        status_path = result_root / "batch_run_status.json"
        status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
        status["latest_figures"] = {"updated_utc": datetime.now(timezone.utc).isoformat(), **info}
        _write_status(status_path, status)
        print(f"[{city}] Created {len(info['maximum_output_png']) + len(info['final_output_png'])} PNG files.",
              flush=True)
    if args.action == "status":
        from src.runtime import collect_city
        collect_city(city, result_root, periods)


def main() -> int:
    args = arguments()
    if args.action in {"run", "all"}:
        # Use one shared queue across cities rather than separate per-city schedulers.
        from src.runtime import run_cities
        roots, periods_by_city = {}, {}
        try:
            for city in args.cities:
                config = city_config(city)
                periods = sorted(set(args.return_periods or config["return_periods"]))
                if not periods or any(period not in config["return_periods"] for period in periods):
                    raise ValueError()
                roots[city] = ((args.input_root / city).resolve(), (args.output_root / city).resolve())
                periods_by_city[city] = periods
            callback = None
            if args.action == "all":
                # Post-process a completed city serially while other containers continue running.
                callback = lambda city, model, result, periods: execute_city(args, city, skip_run=True)
            run_cities(roots, periods_by_city, threads=args.threads, max_concurrent=args.max_concurrent,
                       image=args.image, on_city_complete=callback)
            return 0
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            None
            return 1
    failures = []
    # Guard standalone post-processing against concurrent callbacks.
    from src.runtime import city_result_guard
    for city in args.cities:
        try:
            with city_result_guard(city, (args.output_root / city).resolve()):
                execute_city(args, city)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
            None
            failures.append(city)
    if failures:
        None
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
