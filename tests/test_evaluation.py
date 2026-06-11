import numpy as np
import pytest

from loan_check.evaluate.evaluation import _best_threshold, _binary_metrics


def test_binary_metrics_known_confusion():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.4, 0.8, 0.1])
    # at 0.5 -> preds [1, 0, 1, 0]: tp=1, fp=1, fn=1, tn=1
    m = _binary_metrics(y, p, 0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["accuracy"] == pytest.approx(0.5)


def test_binary_metrics_no_positives_no_zero_division():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.2, 0.1, 0.3, 0.05])
    # threshold above every score -> nothing predicted positive
    m = _binary_metrics(y, p, 0.99)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_best_threshold_separable():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2])
    best_t, best_f1 = _best_threshold(y, p)
    assert best_f1 == pytest.approx(1.0)
    # a perfect split exists for any cutoff in (0.4, 0.6]
    assert 0.4 < best_t <= 0.6


def test_best_threshold_returns_float():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.6, 0.5, 0.55, 0.45])
    best_t, _ = _best_threshold(y, p)
    assert isinstance(best_t, float)