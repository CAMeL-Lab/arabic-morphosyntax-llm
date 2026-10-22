"""PATB morphology metrics for the paper benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .constants import MORPH_FEATURE_ORDER


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _macro_f1(gold_values: list[str], predicted_values: list[str]) -> float | None:
    labels = sorted(set(gold_values))
    if not labels:
        return None
    scores = []
    for label in labels:
        true_positive = sum(
            gold == label and predicted == label
            for gold, predicted in zip(gold_values, predicted_values)
        )
        false_positive = sum(
            gold != label and predicted == label
            for gold, predicted in zip(gold_values, predicted_values)
        )
        false_negative = sum(
            gold == label and predicted != label
            for gold, predicted in zip(gold_values, predicted_values)
        )
        precision = _safe_ratio(true_positive, true_positive + false_positive) or 0.0
        recall = _safe_ratio(true_positive, true_positive + false_negative) or 0.0
        scores.append(_f1(precision, recall) or 0.0)
    return sum(scores) / len(scores)


def evaluate_morphology_pairs(
    pairs: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]
) -> dict[str, Any]:
    feature_correct = Counter()
    gold_values: dict[str, list[str]] = defaultdict(list)
    predicted_values: dict[str, list[str]] = defaultdict(list)
    token_count = 0
    exact_bundles = 0
    word_matches = 0
    skipped_sentences = 0

    for gold_items, predicted_items in pairs:
        if len(gold_items) != len(predicted_items):
            skipped_sentences += 1
            continue
        if any(
            gold.get("word") != predicted.get("word")
            for gold, predicted in zip(gold_items, predicted_items)
        ):
            skipped_sentences += 1
            continue
        for gold, predicted in zip(gold_items, predicted_items):
            token_count += 1
            word_matches += gold.get("word") == predicted.get("word")
            gold_analysis = gold.get("analysis") or {}
            predicted_analysis = predicted.get("analysis") or {}
            exact_bundles += all(
                gold_analysis.get(feature) == predicted_analysis.get(feature)
                for feature in MORPH_FEATURE_ORDER
            )
            for feature in MORPH_FEATURE_ORDER:
                gold_value = str(gold_analysis.get(feature))
                predicted_value = str(predicted_analysis.get(feature))
                feature_correct[feature] += gold_value == predicted_value
                gold_values[feature].append(gold_value)
                predicted_values[feature].append(predicted_value)

    feature_accuracy = {
        feature: _safe_ratio(feature_correct[feature], token_count)
        for feature in MORPH_FEATURE_ORDER
    }
    feature_macro_f1 = {
        feature: _macro_f1(gold_values[feature], predicted_values[feature])
        for feature in MORPH_FEATURE_ORDER
    }
    total_cells = token_count * len(MORPH_FEATURE_ORDER)
    return {
        "sentences_supplied": len(pairs),
        "sentences_skipped_for_token_mismatch": skipped_sentences,
        "tokens_scored": token_count,
        "word_accuracy": _safe_ratio(word_matches, token_count),
        "exact_bundle_accuracy": _safe_ratio(exact_bundles, token_count),
        "feature_cell_accuracy": _safe_ratio(sum(feature_correct.values()), total_cells),
        "mean_feature_macro_f1": (
            sum(value for value in feature_macro_f1.values() if value is not None)
            / sum(value is not None for value in feature_macro_f1.values())
            if any(value is not None for value in feature_macro_f1.values())
            else None
        ),
        "feature_accuracy": feature_accuracy,
        "feature_macro_f1": feature_macro_f1,
    }


def evaluate_outputs(
    *,
    task: str,
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    if task not in {"morphology", "cascade"}:
        raise ValueError("Use upstream CAMeL CoNLL for parsing scores")
    prediction_by_id = {
        str(row["id"]): row
        for row in predictions
        if row.get("status") == "ok" and isinstance(row.get("prediction"), dict)
    }
    pairs = []
    for gold_row in gold:
        prediction = prediction_by_id.get(str(gold_row["id"]))
        if prediction is None:
            continue
        value = prediction["prediction"]
        predicted_tokens = value["tokens"] if task == "morphology" else value["morphology"]
        pairs.append((gold_row["morphology"], predicted_tokens))
    return {
        "task": task,
        "gold_sentences": len(gold),
        "prediction_sentences_scored": len(pairs),
        "prediction_coverage": _safe_ratio(len(pairs), len(gold)),
        "morphology": evaluate_morphology_pairs(pairs),
    }
