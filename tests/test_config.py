"""Tests for config loading and path anchoring."""
from pathlib import Path

from config.config import PROJECT_ROOT, load_config
from loan_check.feature_engineering.feature_config import load_feature_config


def test_paths_are_absolute_and_anchored():
    config = load_config()
    for key in ("raw_data", "bronze_data", "silver_data"):
        p = Path(config["paths"][key])
        assert p.is_absolute()
        assert str(p).startswith(str(PROJECT_ROOT))
    assert Path(config["model_path"]["path"]).is_absolute()


def test_config_scalar_values():
    config = load_config()
    assert config["test_size"]["size"] == 0.2
    assert config["random_state"]["state"] == 42


def test_feature_config_structure():
    fc = load_feature_config()["features"]
    # the groups the preprocessor and feature pipeline depend on
    for key in (
        "drop_columns", "encoded_raw_columns", "nominal_columns",
        "numeric_columns", "target_labels", "emp_length_map",
    ):
        assert key in fc
    assert "Fully Paid" in fc["target_labels"]["paid_statuses"]
    assert "Charged Off" in fc["target_labels"]["default_statuses"]
    assert fc["emp_length_map"]["10+ years"] == 10