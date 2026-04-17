from pathlib import Path

import yaml


def load_config(config_path: Path):
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except Exception as exc:
        raise RuntimeError(f"Cannot load configuration {config_path}") from exc

    return config
