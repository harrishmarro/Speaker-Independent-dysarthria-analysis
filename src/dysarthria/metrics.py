"""Small, auditable evaluation metrics for binary classification and ASR."""

from __future__ import annotations


def binary_classification_metrics(
    targets: list[int], predictions: list[int]
) -> dict[str, float | int]:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must have the same non-zero length")
    if set(targets + predictions) - {0, 1}:
        raise ValueError("binary metrics accept only 0 and 1")

    true_positive = sum(t == 1 and p == 1 for t, p in zip(targets, predictions, strict=True))
    true_negative = sum(t == 0 and p == 0 for t, p in zip(targets, predictions, strict=True))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(targets, predictions, strict=True))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(targets, predictions, strict=True))

    def divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    specificity = divide(true_negative, true_negative + false_positive)
    f1 = divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": divide(true_positive + true_negative, len(targets)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word-level Levenshtein distance divided by reference word count."""
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()
    if not reference_words:
        raise ValueError("reference transcript must contain at least one word")

    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution = previous[column - 1] + (reference_word != hypothesis_word)
            insertion = current[column - 1] + 1
            deletion = previous[column] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(reference_words)
