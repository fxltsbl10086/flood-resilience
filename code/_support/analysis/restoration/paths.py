"""Resolve repository-relative restoration inputs and explicit user paths."""
import json
from pathlib import Path

from _release import ROOT
from _support.analysis_paths import config

DEFAULT_BATCH = ROOT / config("road_restoration", "RESTORATION_CONFIG")["risk_reference"]


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute() and path.parts and path.parts[0] in ("code", "data", "results"):
        return (ROOT / path).resolve()
    return path.resolve()


def read_json(path):
    return json.loads(resolve_path(path).read_text(encoding="utf-8-sig"))
