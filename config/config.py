from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_config():
    config_path = Path(__file__).resolve().parent / "config.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, rel_path in config["paths"].items():
        config["paths"][key] = str(PROJECT_ROOT / rel_path)
    config["model_path"]["path"] = str(PROJECT_ROOT / config["model_path"]["path"])

    return config    
