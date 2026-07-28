# pyright: standard
"""Nurse target의 순위·보정·고정 임계값 운용 지표를 계산한다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
OPERATING_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class PredictionSet:
    name: str
    labels: IntArray
    scores: FloatArray
    is_probability: bool


@dataclass(frozen=True, slots=True)
class UtilityMetrics:
    model: str
    auroc: float
    average_precision: float
    brier_score: float | None
    expected_calibration_error: float | None
    balanced_accuracy_at_0_5: float | None
    sensitivity_at_0_5: float | None
    false_positive_rate_at_0_5: float | None


def expected_calibration_error(labels: IntArray, probabilities: FloatArray) -> float:
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.minimum(np.digitize(probabilities, bin_edges[1:-1]), 9)
    error = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probabilities[mask].mean())
            )
    return error


def measure_utility(prediction: PredictionSet) -> UtilityMetrics:
    auroc = float(roc_auc_score(prediction.labels, prediction.scores))
    average_precision = float(average_precision_score(prediction.labels, prediction.scores))
    if not prediction.is_probability:
        return UtilityMetrics(
            model=prediction.name,
            auroc=auroc,
            average_precision=average_precision,
            brier_score=None,
            expected_calibration_error=None,
            balanced_accuracy_at_0_5=None,
            sensitivity_at_0_5=None,
            false_positive_rate_at_0_5=None,
        )

    # [METRIC][Risk:Major] 0.5 운용 임계값은 Nurse 라벨을 보기 전에 고정하며 조정하지 않는다.
    predicted_labels = prediction.scores >= OPERATING_THRESHOLD
    positive_mask = prediction.labels == 1
    negative_mask = prediction.labels == 0
    sensitivity = float(predicted_labels[positive_mask].mean())
    false_positive_rate = float(predicted_labels[negative_mask].mean())
    return UtilityMetrics(
        model=prediction.name,
        auroc=auroc,
        average_precision=average_precision,
        brier_score=float(brier_score_loss(prediction.labels, prediction.scores)),
        expected_calibration_error=expected_calibration_error(
            prediction.labels,
            prediction.scores,
        ),
        balanced_accuracy_at_0_5=float(
            balanced_accuracy_score(prediction.labels, predicted_labels)
        ),
        sensitivity_at_0_5=sensitivity,
        false_positive_rate_at_0_5=false_positive_rate,
    )
