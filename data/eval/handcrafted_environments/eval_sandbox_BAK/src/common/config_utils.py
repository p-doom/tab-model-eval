import yaml
import os


def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f)
