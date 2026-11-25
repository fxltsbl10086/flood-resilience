"Combine shared TOML settings with explicit city parameters."

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPECTED_KEEP_TYPES = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    # "unclassified",
]

COMMON_SECTIONS = {"spatial", "filter", "topology", "merge"}
CITY_KEYS = {"city_code", "road", "boundary", "target_crs"}


@dataclass(frozen=True)
class RoadConfig:
    "City-specific settings and shared road-network parameters."

    common_path: Path
    common_data: dict[str, Any]
    city_data: dict[str, str]

    @property
    def city_code(self) -> str:
        return self.city_data["city_code"]

    @property
    def input_paths(self) -> dict[str, Path]:
        return {
            key: Path(self.city_data[key]).expanduser().resolve()
            for key in ("road", "boundary")
        }

    @property
    def target_crs(self) -> str:
        return self.city_data["target_crs"]

    @property
    def params(self) -> dict[str, Any]:
        return {
            **self.common_data["spatial"],
            **self.common_data["filter"],
            **self.common_data["topology"],
            **self.common_data["merge"],
        }


def load_config(
    common_config_path: str | Path,
    city_settings: Mapping[str, str],
) -> RoadConfig:
    "Read shared settings and inject the selected city parameters."

    path = Path(common_config_path).resolve()
    if not path.exists():
        raise FileNotFoundError()
    with path.open(encoding="utf-8") as file:
        common_data = yaml.safe_load(file)["topology"]

    config = RoadConfig(
        common_path=path,
        common_data=common_data,
        city_data={key: str(value) for key, value in city_settings.items()},
    )
    validate_config(config)
    return config


def validate_config(config: RoadConfig) -> None:
    missing_sections = COMMON_SECTIONS.difference(config.common_data)
    if missing_sections:
        raise ValueError()

    missing_city_keys = CITY_KEYS.difference(config.city_data)
    if missing_city_keys:
        raise ValueError()

    missing_inputs = [
        key for key, path in config.input_paths.items() if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError()

    params = config.params
    keep_types = [str(value).strip().lower() for value in params["keep_types"]]
    if keep_types != EXPECTED_KEEP_TYPES:
        raise ValueError()
    if "service" in keep_types or "living_street" in keep_types:
        raise ValueError()
    if abs(float(params["clip_buffer_m"]) - 500.0) > 1e-9:
        raise ValueError()
    if float(params["min_length_m"]) <= 0:
        raise ValueError()
    if int(params["node_precision"]) < 0:
        raise ValueError()
    if str(params["unknown_oneway_as"]).upper() != "B":
        raise ValueError()
    if not str(config.target_crs).upper().startswith("EPSG:"):
        raise ValueError()

    # Require the rules used by this experiment so unsupported TOML changes are rejected.
    if not bool(params["merge_degree_two"]):
        raise ValueError()
    if not bool(params["allow_bridge_ground_endpoint_merge"]):
        raise ValueError()
    if not bool(params["require_compatible_oneway"]):
        raise ValueError()
    if str(params["dominant_attribute_rule"]) != "longest_non_bridge_source_osm":
        raise ValueError()
