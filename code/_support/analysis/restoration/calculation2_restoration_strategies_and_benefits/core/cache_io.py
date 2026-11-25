"""Cache costly local computations by input content and code version; do not cache the fast ranking step."""
from pathlib import Path
import sys as _path_sys
_RESULT4_ROOT = Path(__file__).resolve().parents[2]
if str(_RESULT4_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_RESULT4_ROOT))
from _support.analysis.restoration.paths import resolve_path
import hashlib
import json
import pickle
import time


def signature(path):
    path = resolve_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def fingerprint(paths, parameters):
    records = [signature(path) for path in sorted(set(map(Path, paths)), key=str)]
    payload = {"inputs": records, "parameters": parameters}
    key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return key, payload


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def get_or_compute(path, producer, model_key, kind):
    """Load only local cache files within this module's directory whose content digests match."""
    path = Path(path)
    meta_path = path.with_suffix(".json")
    start = time.perf_counter()
    if path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model_key"] != model_key or meta["kind"] != kind:
            raise ValueError("")
        if signature(path)["sha256"] != meta["sha256"]:
            raise ValueError("")
        with path.open("rb") as stream:
            value = pickle.load(stream)
        return value, {"cache_used": True, "elapsed_seconds": time.perf_counter() - start,
                       "original_compute_seconds": meta["original_compute_seconds"], "cache_file": str(path)}
    value = producer()
    compute_seconds = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    meta = {"model_key": model_key, "kind": kind, "sha256": signature(path)["sha256"],
            "original_compute_seconds": compute_seconds}
    dump_json(meta_path, meta)
    return value, {"cache_used": False, "elapsed_seconds": time.perf_counter() - start,
                   "original_compute_seconds": compute_seconds, "cache_file": str(path)}
