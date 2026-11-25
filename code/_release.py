"""Shared path and command helpers for the standalone archive."""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
SUPPORT = CODE / "_support" / "analysis"
ANALYSIS = ROOT / "results" / "analysis"
CITIES = ("zhengzhou", "hangzhou", "fuzhou", "chengdu", "wuhan")
PAPER_CITIES = CITIES


def load_config(name, explicit=None):
    path = Path(explicit).resolve() if explicit else CODE / f"{name}.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("")
    return path, config


def package_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def fresh_output(path, stage, parent=ANALYSIS):
    output = Path(path).resolve() if path else parent / "recomputed" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + stage
    )
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_script(script, *arguments, settings=None):
    script = Path(script)
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    module_paths = [CODE]
    restoration = SUPPORT / "restoration"
    if script.resolve().is_relative_to(restoration):
        module_paths.append(restoration)
    path = os.pathsep.join(map(str, module_paths))
    if env.get("PYTHONPATH"):
        path += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = path
    if settings:
        env.update({key: str(value) for key, value in settings.items()})
    subprocess.run(
        [sys.executable, "-B", str(script), *(str(a) for a in arguments)],
        cwd=ROOT, env=env, check=True,
    )


def record_config(path, dest):
    with Path(path).open("rb") as src, (dest / "parameters_used.yaml").open("xb") as out:
        out.write(src.read())

