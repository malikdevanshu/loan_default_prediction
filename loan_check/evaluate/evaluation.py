import argparse

import numpy as np
import pandas as pd
from typing import Any

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pandas.api.extensions import ExtensionArray

from loan_check.utils.utils import (
    get_config_values,
    get_model_paths,
    get_models,
    load_and_prepare_data,
)


def _scores_to_pandas(
    pred_df: DataFrame, label_col: str = "target", prob_col: str = "probability"
) -> pd.DataFrame:
    return pred_df.select(
        vector_to_array(f.col(prob_col))[1].alias("p1"),
        f.col(label_col).cast("int").alias("y"),
    ).toPandas()


def _binary_metrics(
    y: np.ndarray | pd.Series[Any] | ExtensionArray,
    p: np.ndarray | ExtensionArray,
    threshold: float,
) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _best_threshold(
    y: np.ndarray[Any, np.dtype[np.int_]] | pd.Series[Any], p: np.ndarray
) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = _binary_metrics(y, p, t)["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def evaluate_models(
    model_type: str, classifier_name: str = "all"
) -> pd.DataFrame:
    config_values = get_config_values()
    model_dir = config_values["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_and_prepare_data()  # noqa: RUF059

    # Split the held-out data into a validation slice (used only to pick the
    # decision threshold) and a final test slice (used only to report). The
    # model trained on neither, so threshold selection stays honest.
    val, test_eval = test.randomSplit(
        [0.5, 0.5], seed=config_values["random_state"]
    )
    val = val.cache()
    test_eval = test_eval.cache()

    auc_eval = BinaryClassificationEvaluator(
        labelCol="target",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
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

        # 1) pick the threshold on validation
        val_scores = _scores_to_pandas(model.predict(val))
        best_t, _ = _best_threshold(
            val_scores["y"].values, val_scores["p1"].values
        )

        # 2) lock it in and score the final test slice
        test_preds = model.predict(test_eval)
        auc = auc_eval.evaluate(test_preds)

        test_scores = _scores_to_pandas(test_preds)
        y = test_scores["y"].array
        p = test_scores["p1"].array

        default_m = _binary_metrics(y, p, 0.5)
        tuned_m = _binary_metrics(y, p, best_t)

        print(f"  auc:                  {auc:.4f}")
        print(f"  threshold:            {best_t:.2f}  (default 0.50)")
        print(
            f"  default-class F1:     "
            f"{default_m['f1']:.4f} -> {tuned_m['f1']:.4f}"
        )
        print(
            f"  default-class recall: "
            f"{default_m['recall']:.4f} -> {tuned_m['recall']:.4f}"
        )
        print(
            f"  default-class prec.:  "
            f"{default_m['precision']:.4f} -> {tuned_m['precision']:.4f}"
        )
        print(
            f"  accuracy:             "
            f"{default_m['accuracy']:.4f} -> {tuned_m['accuracy']:.4f}"
        )

        results.append(
            {
                "model_name": model_name,
                "stage": f"{model_type}_evaluation",
                "auc": auc,
                "threshold": best_t,
                "f1_thr0.5": default_m["f1"],
                "f1_tuned": tuned_m["f1"],
                "recall_thr0.5": default_m["recall"],
                "recall_tuned": tuned_m["recall"],
                "precision_tuned": tuned_m["precision"],
                "accuracy_tuned": tuned_m["accuracy"],
                "model_path": str(model_path),
            }
        )

    results_df = pd.DataFrame(results)

    results_path = model_dir / f"{model_type}_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nEvaluation complete.")
    print(f"Results saved to: {results_path}")

    return results_df


def main() -> None:
    models = get_models()
    parser = argparse.ArgumentParser(
        description="Evaluate baseline or tuned models on the test split."
    )
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
