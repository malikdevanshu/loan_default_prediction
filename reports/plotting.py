import argparse

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless: render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np

# Reuse the exact metric + threshold logic the evaluator uses, so the figures
# stay consistent with <model>_evaluation_results.csv.
from loan_check.evaluate.evaluation import _best_threshold  # noqa: PLC2701
from loan_check.evaluate.evaluation import _binary_metrics  # noqa: PLC2701
from loan_check.evaluate.evaluation import _scores_to_pandas  # noqa: PLC2701
from loan_check.utils.utils import (
    get_config_values,
    get_model_paths,
    get_models,
    load_and_prepare_data,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


def pr_curve(y, scores):
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    positives = max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    ap = float(np.sum(np.diff(recall) * precision[1:]))  # average precision
    return recall, precision, ap


def feature_names(transformed, features_col="features"):
    # Map assembled-vector indices back to readable feature names using the
    # ML attribute metadata that VectorAssembler/OneHotEncoder attach.
    meta = transformed.schema[features_col].metadata.get("ml_attr", {})
    size = meta.get("num_attrs")
    attrs = meta.get("attrs", {})
    names = {}
    for group in attrs.values():
        for a in group:
            names[a["idx"]] = a["name"]
    if size is None:
        size = (max(names) + 1) if names else 0
    return [names.get(i, f"f{i}") for i in range(size)]


def collect_scores(models, model_type, val, test_eval):
    """Load each saved model and return per-model (val, test) score frames."""
    data = {}
    for name, cls in models.items():
        path = get_model_paths(model_name=name, model_type=model_type)
        if not path.exists():
            print(f"  skipping {name}: {path} not found")
            continue
        model = cls()
        model.load_model(path)
        val_scores = _scores_to_pandas(model.predict(val))
        test_preds = model.predict(test_eval)
        test_scores = _scores_to_pandas(test_preds)
        best_t, _ = _best_threshold(
            val_scores["y"].values, val_scores["p1"].values
        )
        data[name] = {
            "val": val_scores,
            "test": test_scores,
            "threshold": best_t,
            "model": model,
            "test_preds": test_preds,
        }
    return data


def plot_pr_curves(data):
    plt.figure(figsize=(7, 6))
    for name, d in data.items():
        y, p = d["test"]["y"].values, d["test"]["p1"].values
        recall, precision, ap = pr_curve(y, p)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    baseline = np.mean(next(iter(data.values()))["test"]["y"].values)
    plt.axhline(
        baseline, ls="--", c="grey", lw=1, label=f"no-skill ({baseline:.2f})"
    )
    plt.xlabel("Recall (default class)")
    plt.ylabel("Precision (default class)")
    plt.title("Precision-Recall curves (test set)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    _save("pr_curves.png")


def plot_threshold_sweep(data):
    grid = np.linspace(0.05, 0.95, 91)
    n = len(data)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False
    )
    for ax, (name, d) in zip(axes.flat, data.items(), strict=True):
        y, p = d["val"]["y"].values, d["val"]["p1"].values
        prec = [_binary_metrics(y, p, t)["precision"] for t in grid]
        rec = [_binary_metrics(y, p, t)["recall"] for t in grid]
        f1 = [_binary_metrics(y, p, t)["f1"] for t in grid]
        ax.plot(grid, prec, label="precision")
        ax.plot(grid, rec, label="recall")
        ax.plot(grid, f1, label="F1", lw=2)
        ax.axvline(
            d["threshold"], ls="--", c="k", label=f"chosen={d['threshold']:.2f}"
        )
        ax.set_title(name)
        ax.set_xlabel("threshold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Threshold sweep (validation) - default class")
    fig.tight_layout()
    _save("threshold_sweep.png")


def plot_confusion_matrices(data):
    n = len(data)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False
    )
    for ax, (name, d) in zip(axes.flat, data.items(), strict=True):
        y, p = d["test"]["y"].values, d["test"]["p1"].values
        t = d["threshold"]
        pred = (p >= t).astype(int)
        cm = np.array(
            [
                [
                    int(((pred == 0) & (y == 0)).sum()),
                    int(((pred == 1) & (y == 0)).sum()),
                ],
                [
                    int(((pred == 0) & (y == 1)).sum()),
                    int(((pred == 1) & (y == 1)).sum()),
                ],
            ]
        )
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        ax.set_xticks([0, 1], ["pred paid", "pred default"])
        ax.set_yticks([0, 1], ["true paid", "true default"])
        ax.set_title(f"{name} (thr={t:.2f})")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices at tuned threshold (test set)")
    fig.tight_layout()
    _save("confusion_matrix.png")


def plot_feature_importance(data, top_n=20):
    # Prefer a tree-based model (these expose featureImportances).
    order = ["gradient_boosted", "random_forest", "decision_tree"]
    name = next((m for m in order if m in data), None)
    if name is None:
        print("  no tree-based model available; skipping feature importance")
        return
    d = data[name]
    stage = d["model"].model.stages[-1]
    if not hasattr(stage, "featureImportances"):
        print(f"  {name} has no featureImportances; skipping")
        return
    importances = stage.featureImportances.toArray()
    names = feature_names(d["test_preds"])
    if len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]
    idx = np.argsort(importances)[::-1][:top_n][::-1]
    plt.figure(figsize=(8, max(4, 0.35 * len(idx))))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [names[i] for i in idx], fontsize=8)
    plt.xlabel("importance")
    plt.title(f"Top {len(idx)} features - {name}")
    plt.tight_layout()
    _save("feature_importance.png")


def _save(fname):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate README evaluation figures."
    )
    parser.add_argument(
        "--model", choices=["tuned", "baseline"], default="tuned"
    )
    args = parser.parse_args()

    cfg = get_config_values()
    train, test = load_and_prepare_data()  # noqa: RUF059
    val, test_eval = test.randomSplit([0.5, 0.5], seed=cfg["random_state"])
    val = val.cache()
    test_eval = test_eval.cache()

    models = get_models()
    print(f"Loading {args.model} models...")
    data = collect_scores(models, args.model, val, test_eval)
    if not data:
        print("No models found. Run training/tuning first.")
        return

    print("Generating figures...")
    plot_pr_curves(data)
    plot_threshold_sweep(data)
    plot_confusion_matrices(data)
    plot_feature_importance(data)
    print(f"\nDone. Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
