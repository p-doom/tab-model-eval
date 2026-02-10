from typing import Any, Dict

import yaml


def load_generation_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load generation results from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)
