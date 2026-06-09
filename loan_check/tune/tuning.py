import pandas as pd

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

from loan_check.utils.utils import (
    add_class_weights,
    build_feature_stages,
    build_param_grid,
    get_config_values,
    get_model_paths,
    get_models,
    load_and_prepare_data,
)


def tune_models(cv=3):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_and_prepare_data()  # noqa: RUF059
    train = add_class_weights(train)

    # Fit the shared feature stages once, then tune only the estimators on
    # the cached feature matrix. This avoids re-fitting the indexers /
    # imputer / vectoriser for every fold and every model.
    feature_model = Pipeline(stages=build_feature_stages(train)).fit(train)
    train_feat = (
        feature_model.transform(train)
        .select("features", "target", "weight")
        .cache()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    models = get_models()
    results = []

    for model_name, classifier_class in models.items():
        print(f"\nTuning model: {model_name}")

        classifier = classifier_class()
        param_grid = build_param_grid(model_name, classifier.estimator)

        cross_validator = CrossValidator(
            estimator=classifier.estimator,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=cv,
            seed=config_values["random_state"],
            parallelism=2,
        )

        cv_model = cross_validator.fit(train_feat)

        best_index = cv_model.avgMetrics.index(max(cv_model.avgMetrics))
        best_score = cv_model.avgMetrics[best_index]
        best_params = {
            param.name: value
            for param, value in param_grid[best_index].items()
        }

        # Stitch the already-fitted feature stages onto the best estimator so
        # the tuned model is saved in the same end-to-end format as baselines.
        classifier.model = PipelineModel(
            [*feature_model.stages, cv_model.bestModel]
        )

        model_path = get_model_paths(
            model_name=model_name,
            model_type="tuned",
        )
        classifier.save_model(model_path)

        print(f"Best cv auc: {best_score:.4f}")
        print(f"Best params: {best_params}")
        print(f"Saved tuned model to: {model_path}")

        results.append(
            {
                "model_name": model_name,
                "stage": "cross_validation",
                "best_cv_auc": best_score,
                "best_params": best_params,
                "model_path": str(model_path),
            }
        )

    results_df = pd.DataFrame(results)

    results_path = model_dir / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nGrid search results saved to: {results_path}")

    return results_df


if __name__ == "__main__":
    tune_models()