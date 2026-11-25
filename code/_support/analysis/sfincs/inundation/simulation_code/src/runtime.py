"""Shared Docker execution, resume and status collection across cities."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Mapping
from uuid import uuid4

from _release import ROOT

DEFAULT_IMAGE = "deltares/sfincs-cpu:sfincs-v2.4.0-Galibier-Release"
_SOURCE = Path(__file__).resolve().parent
_SUCCESS = {"solver_finished", "completed", "completed_with_quality_notes"}
_ACTIVE = {"submitted", "running", "paused"}
_GLOBAL_ROOT = Path(os.environ.get("FLOOD_SCHEDULER_ROOT") or ROOT / "results/flood_simulation/recomputed")
_GLOBAL_LOCK = _GLOBAL_ROOT / "scenario_scheduler_lock.lock"
_GLOBAL_STATUS = _GLOBAL_ROOT / "scenario_scheduler_state.json"
_HARD_CAPACITY = 2


def _read_json(path: Path) -> dict:
    # Read both PowerShell UTF-8 BOM files and plain UTF-8 Python files.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomically replace one specified file so readers cannot see partial JSON.
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _global_lock():
    """Hold the project Windows file lock across submission/resume and live capacity checks."""
    import msvcrt
    _GLOBAL_ROOT.mkdir(parents=True, exist_ok=True)
    with _GLOBAL_LOCK.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 30
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError()
                time.sleep(0.25)
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _pid_alive(pid: int) -> bool:
    """Query Windows process state without sending a possible termination signal."""
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return ctypes.get_last_error() == 5  # Treat inaccessible processes as still running.
    try:
        code = ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)):
            return True
        return code.value == 259
    finally:
        kernel.CloseHandle(ctypes.c_void_p(handle))


def _claim_roots(queue_id: str, roots: Mapping[str, Path]) -> None:
    """Allow one writer per city to prevent concurrent updates to batch status."""
    with _global_lock():
        data = _read_json(_GLOBAL_STATUS) if _GLOBAL_STATUS.is_file() else {}
        owners = data.setdefault("city_owners", {})
        for city, root in roots.items():
            key = str(root).casefold()
            owner = owners.get(key)
            if owner and owner.get("queue_id") != queue_id and _pid_alive(owner["pid"]):
                raise RuntimeError()
        for city, root in roots.items():
            owners[str(root).casefold()] = {"queue_id": queue_id, "pid": os.getpid(), "city": city}
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(_GLOBAL_STATUS, data)


def _release_roots(queue_id: str) -> None:
    with _global_lock():
        data = _read_json(_GLOBAL_STATUS) if _GLOBAL_STATUS.is_file() else {}
        data["city_owners"] = {key: owner for key, owner in data.get("city_owners", {}).items()
                               if owner.get("queue_id") != queue_id}
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(_GLOBAL_STATUS, data)


@contextmanager
def city_result_guard(city: str, result_root: Path):
    """Guard export/plot/status writes per city without occupying solver capacity."""
    queue_id = uuid4().hex
    _claim_roots(queue_id, {city: Path(result_root).resolve()})
    try:
        yield
    finally:
        _release_roots(queue_id)


def _docker(arguments: list[str]) -> str:
    completed = subprocess.run(["docker", *arguments], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError()
    return completed.stdout.strip()


def _inspect(name: str) -> dict:
    records = json.loads(_docker(["inspect", name]))
    if len(records) != 1:
        raise RuntimeError()
    return records[0]


def _global_containers() -> tuple[list[dict], list[dict]]:
    """Count all local SFINCS containers, excluding paused containers from active slots."""
    listing = _docker(["ps", "--format", "{{json .}}"])
    candidates = []
    for line in listing.splitlines():
        row = json.loads(line)
        if ("sfincs" in str(row.get("Image", "")).lower()
                or str(row.get("Names", "")).lower().startswith("sfincs-")
                or re.search(r"\bsfincs\b", str(row.get("Command", "")).lower())):
            candidates.append(row["ID"])
    records = json.loads(_docker(["inspect", *candidates])) if candidates else []
    active = [row for row in records if row["State"]["Running"] and not row["State"]["Paused"]]
    paused = [row for row in records if row["State"]["Paused"]]
    if len(active) > _HARD_CAPACITY:
        raise RuntimeError()
    return active, paused


def _global_snapshot(queue_id: str, active: list[dict], paused: list[dict],
                     pending: list[str], limit: int, *, error: str | None = None) -> None:
    """Update this queue snapshot under the project lock while retaining other city ownership."""
    data = _read_json(_GLOBAL_STATUS) if _GLOBAL_STATUS.is_file() else {}
    data.update(hard_capacity=_HARD_CAPACITY, active_count=len(active),
                active_containers=[row["Name"].lstrip("/") for row in active],
                paused_containers=[row["Name"].lstrip("/") for row in paused],
                updated_utc=datetime.now(timezone.utc).isoformat())
    data.setdefault("queues", {})[queue_id] = {
        "pid": os.getpid(), "capacity_limit": limit, "pending": pending,
        "status": "failed" if error else "running", "error": error,
    }
    _write_json(_GLOBAL_STATUS, data)


def _periods(values: Iterable[int]) -> list[int]:
    periods = sorted({int(value) for value in values})
    if not periods or any(value < 1 or value > 999 for value in periods):
        raise ValueError()
    return periods


def _check_city(city: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", city):
        raise ValueError()


def _check_fews_time(input_path: Path, image: str) -> None:
    """Validate the SFINCS 2.4 minute-based forcing-time convention before submission."""
    if image != DEFAULT_IMAGE and "v2.4.0" not in image:
        return
    settings = {}
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    if not settings.get("netamprfile"):
        return
    from netCDF4 import Dataset
    file = (input_path.parent / settings["netamprfile"]).resolve()
    with Dataset(file) as dataset:
        units = str(getattr(dataset.variables["time"], "units", ""))
    if not units.startswith("minutes since "):
        raise RuntimeError()


def _find_scenario(root: Path, period: int) -> Path:
    # Identify a scenario by its return-period prefix within the city directory.
    prefix = re.compile(rf"^T{period:03d}(?:_|$)", re.IGNORECASE)
    candidates = [child for child in root.iterdir() if child.is_dir() and prefix.match(child.name)]
    if len(candidates) != 1:
        raise RuntimeError()
    return candidates[0]


def _invoke(action: str, arguments: dict[str, object]) -> str:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if not shell:
        raise RuntimeError()
    command = [
        shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-File", str(_SOURCE / "invoke_runtime.ps1"), "-Action", action,
    ]
    # Pass arguments separately to preserve spaces and non-ASCII paths without shell interpolation.
    for name, value in arguments.items():
        command.extend([f"-{name}", str(value)])
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError()
    return completed.stdout.strip()


def _stored_success(directory: Path, record: dict) -> dict:
    # Execution completion alone does not make scientifically invalid results reusable.
    status = str(record.get("status", ""))
    if (record.get("scientific_use_valid") is False or status.startswith("invalid")
            or status == "failed_forcing_reception"):
        raise RuntimeError()
    if not (directory / "sfincs_map.nc").is_file():
        raise RuntimeError()
    state_path = directory / "container_state.json"
    if state_path.is_file():
        state = _read_json(state_path)
        if state.get("Running") or state.get("ExitCode", 0) != 0 or state.get("OOMKilled", False):
            raise RuntimeError()
    # Completed results remain inspectable after their Docker containers are removed.
    return {
        "status": record["status"], "running": False, "exit_code": 0,
        "oom_killed": False, "reused": True, "result_directory": str(directory),
    }


def _collect_one(directory: Path) -> dict:
    record_path = directory / "run_status.json"
    if not record_path.is_file():
        raise RuntimeError()
    record = _read_json(record_path)
    # Check invalid inputs before interpreting completion or collecting container state.
    status = str(record.get("status", ""))
    if (record.get("scientific_use_valid") is False or status.startswith("invalid")
            or status == "failed_forcing_reception"):
        raise RuntimeError()
    if record.get("status") in _SUCCESS:
        return _stored_success(directory, record)
    if record.get("status") not in _ACTIVE:
        raise RuntimeError()
    actual = _inspect(record["container_name"])
    if actual["State"]["Paused"]:
        # Paused containers also report Running=true; do not classify them as active.
        record.update(status="paused", paused=True,
                      last_collected_utc=datetime.now(timezone.utc).isoformat())
        _write_json(directory / "container_state.json", actual["State"])
        _write_json(record_path, record)
        return {"status": "paused", "running": False, "paused": True,
                "exit_code": actual["State"]["ExitCode"], "result_directory": str(directory)}
    state = json.loads(_invoke("Collect", {"ResultDirectory": directory}))
    if not state.get("running") and state.get("status") in _SUCCESS:
        _stored_success(directory, _read_json(record_path))
    return state


def _batch_status(city: str, result_root: Path, periods: list[int], states: dict[int, dict], **settings) -> dict:
    path = result_root / "batch_run_status.json"
    record = _read_json(path) if path.is_file() else {}
    finished = sorted(period for period, state in states.items() if state.get("status") in _SUCCESS)
    failed = sorted(period for period, state in states.items() if state.get("status") == "solver_failed")
    running = sorted(period for period, state in states.items() if state.get("running"))
    paused = sorted(period for period, state in states.items() if state.get("paused"))
    queue_failed = sorted(period for period, state in states.items()
                          if state.get("status") in {"submission_failed", "resume_failed", "collection_failed"})
    pending = sorted((set(periods) - set(states)) | set(paused))
    exported = set(record.get("exported_return_periods", []))
    export_complete = set(periods).issubset(exported) and set(periods).issubset(finished)
    previous_status = str(record.get("status", ""))
    explicit_invalid = (previous_status.startswith("invalid")
                        or previous_status == "failed_forcing_reception")
    if record.get("scientific_use_valid") is False or explicit_invalid:
        # Preserve the recorded failure cause instead of replacing it with a time-unit diagnosis.
        status = previous_status if explicit_invalid else "invalid_results"
        record["scientific_use_valid"] = False
    elif failed:
        status = "solver_failed"
    elif queue_failed:
        status = "queue_failed"
    elif record.get("postprocessing_error"):
        status = "postprocessing_failed"
    elif pending or running:
        status = "solvers_running"
    elif export_complete:
        has_notes = (previous_status == "completed_with_quality_notes"
                     or any(state.get("status") == "completed_with_quality_notes" for state in states.values()))
        status = "completed_with_quality_notes" if has_notes else "completed"
    else:
        status = "solvers_finished"
    record.update({
        "status": status, "city": city, "result_root": str(result_root),
        "requested_return_periods": periods, "pending_return_periods": pending,
        "running_return_periods": running, "solver_finished_return_periods": finished,
        "paused_return_periods": paused, "queue_failed_return_periods": queue_failed,
        "solver_failed_return_periods": failed,
        "geotiff_export_complete": export_complete,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_states": {f"T{period:03d}": state for period, state in states.items()},
        **settings,
    })
    _write_json(path, record)
    return record


def collect_city(city: str, result_root: str | Path, periods: Iterable[int]) -> dict:
    """Collect scenario states without starting containers or recomputing the model."""
    _check_city(city)
    selected = _periods(periods)
    result_root = Path(result_root).resolve()
    if not result_root.is_dir():
        raise FileNotFoundError()
    states = {}
    for period in selected:
        directory = _find_scenario(result_root, period)
        states[period] = _collect_one(directory)
        print(f"[{city}/{directory.name}] {states[period]['status']}", flush=True)
    return _batch_status(city, result_root, selected, states)


def _resume_one(directory: Path) -> dict:
    """Resume an existing container under the global lock after confirming capacity."""
    path = directory / "run_status.json"
    record = _read_json(path)
    if record.get("scientific_use_valid") is False or str(record.get("status", "")).startswith("invalid"):
        raise RuntimeError()
    container = _inspect(record["container_name"])
    if not container["State"]["Paused"]:
        if container["State"]["Running"]:
            return {"status": "running", "running": True, "paused": False,
                    "result_directory": str(directory)}
        return _collect_one(directory)
    threads = int(record["threads"])
    if not 1 <= threads <= 64:
        raise ValueError()
    change = {"updated_utc": datetime.now(timezone.utc).isoformat(),
              "operation": "update_cpu_quota_then_unpause", "original_threads": threads,
              "previous_cpu_limit": container["HostConfig"]["NanoCpus"] / 1e9,
              "target_cpu_limit": threads, "quota_updated": False, "unpaused": False}
    changes = list(record.get("resource_changes", []))
    changes.append(change)
    record["resource_changes"] = changes
    try:
        _docker(["update", "--cpus", str(threads), record["container_name"]])
        updated = _inspect(record["container_name"])
        change["inspected_nano_cpus"] = updated["HostConfig"]["NanoCpus"]
        if change["inspected_nano_cpus"] != threads * 1_000_000_000:
            raise RuntimeError()
        change["quota_updated"] = True
        record["current_cpu_limit"] = threads
        _write_json(path, record)  # Record successful quota changes even when resume subsequently fails.
        _docker(["unpause", record["container_name"]])
        actual = _inspect(record["container_name"])
        if actual["State"]["Paused"]:
            raise RuntimeError()
        change["unpaused"] = True
        record.update(status="running", paused=False)
        _write_json(path, record)
        _write_json(directory / "container_state.json", actual["State"])
        print(f"[{directory.parent.name}/{directory.name}] Resumed existing container with CPU quota={threads}; progress retained.", flush=True)
        # A later collection failure must not relabel a successfully resumed container.
        return {"status": "running", "running": True, "paused": False,
                "result_directory": str(directory)}
    except (OSError, ValueError, RuntimeError) as error:
        change["error"] = str(error)
        record["status"] = "resume_failed"
        _write_json(path, record)
        raise


def run_cities(city_roots: Mapping[str, tuple[str | Path, str | Path]],
               periods: Mapping[str, Iterable[int]] | Iterable[int], *, threads: int = 8,
               max_concurrent: int = 2, image: str = DEFAULT_IMAGE, poll_seconds: int = 15,
               on_city_complete: Callable[[str, Path, Path, list[int]], None] | None = None) -> dict:
    """Run at most two scenarios globally, rotating across periods and cities.

city_roots maps city codes to model/result directories. periods may be shared or per-city.
Resume paused containers only after checking capacity under the lock."""
    if not city_roots or not 1 <= threads <= 64 or max_concurrent not in {1, 2}:
        raise ValueError()
    if not 1 <= poll_seconds <= 60:
        raise ValueError()
    cities = list(city_roots)
    common = None if isinstance(periods, Mapping) else _periods(periods)
    selected, models, results, sources, directories = {}, {}, {}, {}, {}
    for city in cities:
        _check_city(city)
        selected[city] = _periods(periods[city]) if isinstance(periods, Mapping) else list(common)
        models[city], results[city] = (Path(p).resolve() for p in city_roots[city])
        if not models[city].is_dir():
            raise FileNotFoundError()
        if (results[city] == models[city] or models[city] in results[city].parents
                or results[city] in models[city].parents):
            raise ValueError()
        for period in selected[city]:
            key = city, period
            source = _find_scenario(models[city], period)
            if not (source / "sfincs.inp").is_file():
                raise FileNotFoundError()
            _check_fews_time(source / "sfincs.inp", image)
            sources[key], directories[key] = source, results[city] / source.name
    # Reject duplicate result roots before two cities can write to one directory.
    if len(set(results.values())) != len(cities):
        raise ValueError()
    queue_id = uuid4().hex
    _claim_roots(queue_id, results)
    states = {city: {} for city in cities}
    order = [(city, period) for period in sorted({p for ps in selected.values() for p in ps})
             for city in cities if period in selected[city]]
    pending, active, callback_done, errors, records = set(), set(), set(), [], {}
    collection_retries = {}
    try:
        # Preflight inputs and histories before submission; preserve partial and invalid records.
        for city in cities:
            batch_path = results[city] / "batch_run_status.json"
            if batch_path.is_file():
                batch = _read_json(batch_path)
                if batch.get("scientific_use_valid") is False or str(batch.get("status", "")).startswith("invalid"):
                    raise RuntimeError()
            results[city].mkdir(parents=True, exist_ok=True)
        for key in order:
            city, period = key
            directory = directories[key]
            path = directory / "run_status.json"
            if path.is_file():
                state = _collect_one(directory)
                states[city][period] = state
                if state.get("paused"):
                    pending.add(key)
                elif state.get("running"):
                    active.add(key)
                elif state.get("status") not in _SUCCESS:
                    raise RuntimeError()
            elif (directory / "sfincs_map.nc").is_file():
                raise RuntimeError()
            else:
                pending.add(key)
        while True:
            # Collect only owned active scenarios, excluding paused or unselected cities.
            for key in [key for key in order if key in active]:
                city, period = key
                try:
                    state = _collect_one(directories[key])
                    collection_retries[key] = 0
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    # Confirm actual container state before treating collection failure as solver completion.
                    record = _read_json(directories[key] / "run_status.json")
                    actual = _inspect(record["container_name"])["State"]
                    collection_retries[key] = collection_retries.get(key, 0) + 1
                    scientific_invalid = (record.get("scientific_use_valid") is False
                                          or str(record.get("status", "")).startswith("invalid")
                                          or record.get("status") == "failed_forcing_reception")
                    if scientific_invalid or collection_retries[key] >= 3:
                        data = _read_json(results[city] / "batch_run_status.json") if (results[city] / "batch_run_status.json").is_file() else {}
                        data["collection_error"] = str(error)
                        data["status"] = record["status"] if scientific_invalid else "collection_failed"
                        if scientific_invalid:
                            data["scientific_use_valid"] = False
                        _write_json(results[city] / "batch_run_status.json", data)
                        raise RuntimeError() from error
                    state = {"status": "paused" if actual["Paused"] else "running",
                             "running": not actual["Paused"], "paused": actual["Paused"],
                             "collection_error": str(error), "retry_count": collection_retries[key]}
                states[city][period] = state
                print(f"[{city}/{sources[key].name}] {state['status']} {state.get('last_progress') or ''}", flush=True)
                if not state.get("running"):
                    active.discard(key)
                    if state.get("paused"):
                        pending.add(key)
                    elif state.get("status") == "solver_failed":
                        errors.append(f"{city}/T{period:03d}求解失败")
            # Capacity checks, rereads, submit/resume and persistence form one critical section.
            with _global_lock():
                global_active, global_paused = _global_containers()
                for key in [key for key in order if key in pending]:
                    if len(global_active) >= max_concurrent:
                        break
                    city, period = key
                    path = directories[key] / "run_status.json"
                    had_record = path.is_file()
                    try:
                        if had_record:
                            state = _resume_one(directories[key])
                        else:
                            _check_fews_time(sources[key] / "sfincs.inp", image)
                            output = _invoke("Run", {"ModelRoot": models[city], "ResultRoot": results[city],
                                                     "ScenarioName": sources[key].name, "Threads": threads,
                                                     "Image": image})
                            if output:
                                print(output, flush=True)
                            state = {"status": "submitted", "running": True}
                    except (OSError, ValueError, RuntimeError, KeyError) as error:
                        state = {"status": "resume_failed" if had_record else "submission_failed",
                                 "running": False, "error": str(error)}
                        errors.append(f"{city}/T{period:03d}提交/恢复失败：{error}")
                    states[city][period] = state
                    pending.discard(key)
                    if state.get("running"):
                        active.add(key)
                    elif state.get("status") == "solver_failed":
                        errors.append(f"{city}/T{period:03d}求解失败")
                    # Recheck live containers after a failed action; do not assume capacity was freed.
                    global_active, global_paused = _global_containers()
                _global_snapshot(queue_id, global_active, global_paused,
                                 [f"{city}/T{period:03d}" for city, period in order if (city, period) in pending],
                                 max_concurrent)
            for city in cities:
                records[city] = _batch_status(city, results[city], selected[city], states[city],
                                             input_root=str(models[city]), threads=threads,
                                             max_concurrent=max_concurrent, global_capacity=_HARD_CAPACITY,
                                             queue_id=queue_id, image=image)
            # Fill solver slots before serial city post-processing and batch-status writes.
            for city in cities:
                if city in callback_done or not all(states[city].get(p, {}).get("status") in _SUCCESS
                                                     for p in selected[city]):
                    continue
                callback_done.add(city)
                if on_city_complete:
                    try:
                        on_city_complete(city, models[city], results[city], selected[city])
                        path = results[city] / "batch_run_status.json"
                        data = _read_json(path) if path.is_file() else {}
                        data.pop("postprocessing_error", None)
                        _write_json(path, data)
                        for period in selected[city]:
                            states[city][period] = _collect_one(directories[city, period])
                        records[city] = _batch_status(city, results[city], selected[city], states[city])
                    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
                        message = f"{city}导出/质量汇总/绘图失败：{error}"
                        errors.append(message)
                        path = results[city] / "batch_run_status.json"
                        data = _read_json(path) if path.is_file() else {}
                        if (data.get("scientific_use_valid") is not False
                                and not str(data.get("status", "")).startswith("invalid")
                                and data.get("status") != "failed_forcing_reception"):
                            data["status"] = "postprocessing_failed"
                        data["postprocessing_error"] = message
                        _write_json(path, data)
                        records[city] = data
                        print(message, flush=True)
            if not pending and not active:
                break
            # Wait when external containers occupy all slots, even if this queue has no active scenario.
            time.sleep(poll_seconds)
        # Require every native scenario status to pass before declaring completion.
        for city in cities:
            for period in selected[city]:
                state = states[city].get(period, {})
                if state.get("status") not in _SUCCESS:
                    errors.append(f"{city}/T{period:03d}未成功完成：{state.get('status', 'missing')}")
        if errors:
            raise RuntimeError()
        with _global_lock():
            global_active, global_paused = _global_containers()
            _global_snapshot(queue_id, global_active, global_paused, [], max_concurrent)
            data = _read_json(_GLOBAL_STATUS)
            data["queues"][queue_id]["status"] = "completed"
            _write_json(_GLOBAL_STATUS, data)
        return {"queue_id": queue_id, "hard_capacity": _HARD_CAPACITY,
                "global_limit": max_concurrent, "city_records": records}
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        with _global_lock():
            data = _read_json(_GLOBAL_STATUS) if _GLOBAL_STATUS.is_file() else {}
            entry = data.setdefault("queues", {}).setdefault(queue_id, {"pid": os.getpid()})
            entry.update(status="failed", error=str(error))
            _write_json(_GLOBAL_STATUS, data)
        raise
    finally:
        _release_roots(queue_id)


def run_city(city: str, model_root: str | Path, result_root: str | Path, periods: Iterable[int],
             threads: int = 8, max_concurrent: int = 2, image: str = DEFAULT_IMAGE,
             poll_seconds: int = 15) -> dict:
    """Retain the single-city API while respecting the global two-scenario limit."""
    info = run_cities({city: (model_root, result_root)}, periods, threads=threads,
                      max_concurrent=max_concurrent, image=image, poll_seconds=poll_seconds)
    return info["city_records"][city]
