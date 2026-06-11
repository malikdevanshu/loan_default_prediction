import pytest

from pyspark.ml.feature import Imputer, VectorAssembler

from loan_check.classifiers.base_classifier import BaseClassifier
from loan_check.utils.utils import (
    add_class_weights,
    build_feature_stages,
    build_param_grid,
    get_model_paths,
    get_models,
)


def test_get_models_registry():
    models = get_models()
    assert set(models) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosted",
    }
    for cls in models.values():
        assert issubclass(cls, BaseClassifier)


@pytest.mark.parametrize(
    "name,expected_combinations",
    [
        ("logistic_regression", 6),  # 3 regParam x 2 elasticNet
        ("decision_tree", 3),        # 3 maxDepth
        ("random_forest", 4),        # 2 numTrees x 2 maxDepth
        ("gradient_boosted", 4),     # 2 maxIter x 2 maxDepth
    ],
)
def test_build_param_grid_sizes(spark, name, expected_combinations):
    estimator = get_models()[name]().estimator
    grid = build_param_grid(name, estimator)
    assert len(grid) == expected_combinations


def test_build_param_grid_unknown_model_raises():
    with pytest.raises(ValueError):
        build_param_grid("xgboost", estimator=None)


def test_get_model_paths():
    path = get_model_paths(model_name="random_forest", model_type="tuned")
    assert path.name == "random_forest_tuned"
    assert path.parent.name == "models"


def test_add_class_weights_balanced(spark):
    # 8 negatives, 2 positives
    rows = [(0,)] * 8 + [(1,)] * 2
    df = spark.createDataFrame(rows, "target int")
    weighted = add_class_weights(df)

    assert "weight" in weighted.columns
    sums = {
        r["target"]: r["s"]
        for r in weighted.groupBy("target")
        .agg({"weight": "sum"})
        .withColumnRenamed("sum(weight)", "s")
        .collect()
    }
    # balanced => each class contributes total / n_classes == 10 / 2 == 5
    assert sums[0] == pytest.approx(5.0)
    assert sums[1] == pytest.approx(5.0)


def test_build_feature_stages_structure(preprocessed_df):
    weighted = add_class_weights(preprocessed_df)
    stages = build_feature_stages(weighted)

    assert isinstance(stages[0], Imputer)
    assert isinstance(stages[-1], VectorAssembler)

    assembler_inputs = stages[-1].getInputCols()
    # target and weight must never become features
    assert "target" not in assembler_inputs
    assert "weight" not in assembler_inputs
    # a real engineered numeric feature is included
    assert "fico_avg" in assembler_inputs
    # the imputer never touches the label/weight either
    assert "target" not in stages[0].getInputCols()
    assert "weight" not in stages[0].getInputCols()