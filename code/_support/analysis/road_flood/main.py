"Identify road inundation across scenarios using one road-grid match."

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_CONFIG, load_city


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run"], help="Export binary road states.")
    parser.add_argument("--city", "--cities", dest="cities", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--road-root", type=Path)
    parser.add_argument("--depth-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--matching-root", type=Path, help="Parent directory for reusable complete road-grid mapping caches.")
    parser.add_argument("--roads", type=Path, help="Explicit facility-segment file for one city; the configured layer name is retained.")
    parser.add_argument("--boundary", type=Path, help="Explicit study-area boundary for one city.")
    parser.add_argument("--threshold", type=float, help="Override the depth threshold in metres.")
    parser.add_argument("--no-edges", action="store_true", help="Export facility segments without requiring directed routing edges.")
    parser.add_argument("--batch-name", help="Explicit batch name; an existing directory is rejected.")
    args = parser.parse_args()
    args.cities = list(dict.fromkeys(args.cities))
    if len(args.cities) > 1 and (args.roads or args.boundary):
        parser.error('')
    return args


def main() -> int:
    args = arguments()
    overrides = {key: getattr(args, key) for key in
                 ("road_root", "depth_root", "output_root", "matching_root", "roads", "boundary", "threshold", "no_edges")}
    configs = [load_city(city, args.config, **overrides) for city in args.cities]
    from src.pipeline import run_city

    for config in configs:
        run_city(config, args.batch_name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        raise SystemExit(1)
