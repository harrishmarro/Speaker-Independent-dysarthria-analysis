import pytest

from dysarthria.metrics import binary_classification_metrics, word_error_rate
from dysarthria.severity import experimental_band, fused_score


def test_classification_metrics_are_internally_consistent():
    metrics = binary_classification_metrics([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_word_error_rate_counts_substitution():
    assert word_error_rate("please open the window", "please close the window") == 0.25


def test_experimental_fusion_is_bounded_and_labeled():
    score = fused_score(0.8, 0.5, alpha=0.7)
    assert score == pytest.approx(0.71)
    assert experimental_band(score) == "high"
