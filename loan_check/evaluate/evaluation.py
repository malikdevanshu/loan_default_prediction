import argparse

import pandas as pd

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)

from loan_check.utils.utils import (
    get_config_values,
    get_model_paths,
    get_models,
    load_and_prepare_data,
)


def evaluate_models(model_type, classifier_name="all"):
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_and_prepare_data()  # noqa: RUF059
    test = test.cache()

    auc_eval = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="accuracy",
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="target",
        predictionCol="prediction",
        metricName="f1",
    )

    models = get_models()
    if classifier_name != "all":
        models = {classifier_name: models[classifier_name]}
    results = []

    for model_name, classifier_class in models.items():
        print(f"\nEvaluating {model_type} model: {model_name}")

        model_path = get_model_paths(
            model_name=model_name,
            model_type=model_type,
        )

        if not model_path.exists():
            print(f"Skipping {model_name}. Saved files not found.")
            continue

        model = classifier_class()
        model.load_model(model_path)

        predictions = model.predict(test)

        auc = auc_eval.evaluate(predictions)
        accuracy = accuracy_eval.evaluate(predictions)
        f1 = f1_eval.evaluate(predictions)

        print(f"{model_name} auc: {auc:.4f}")
        print(f"{model_name} accuracy: {accuracy:.4f}")
        print(f"{model_name} f1: {f1:.4f}")

        results.append(
            {
                "model_name": model_name,
                "stage": f"{model_type}_evaluation",
                "auc": auc,
                "accuracy": accuracy,
                "f1": f1,
                "model_path": str(model_path),
            }
        )

    results_df = pd.DataFrame(results)

    results_path = model_dir / f"{model_type}_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nEvaluation complete.")
    print(f"Results saved to: {results_path}")

    return results_df


def main():
    models= get_models()
    parser = argparse.ArgumentParser(description="Evaluate baseline or tuned models on the test split.")
    parser.add_argument(
        "--model",
        choices=["tuned", "baseline"],
        default="baseline",
        help="Which set of saved models to evaluate.",
    )
    parser.add_argument(
        "--classifier",
        choices=[*models.keys(), "all"],
        default="all",
        help="Which classifier to evaluate (default: all).",
     )

    args = parser.parse_args()
    evaluate_models(model_type=args.model, classifier_name=args.classifier)


if __name__ == "__main__":
    main()