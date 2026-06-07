from pathlib import Path

import yaml

def load_feature_config():
    config_path = Path(__file__).resolve().parent / "feature_seperation.yaml"

    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)