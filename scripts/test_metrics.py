import math

import numpy as np

from scripts.metrics import (
    PredictionSet,
    expected_calibration_error,
    measure_utility,
)


def test_expected_calibration_error_uses_weighted_bin_gaps() -> None:
    labels = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([0.1, 0.9], dtype=np.float64)

    result = expected_calibration_error(labels, probabilities)

    assert math.isclose(result, 0.1, abs_tol=1e-12)


def test_probability_utility_uses_fixed_half_threshold() -> None:
    prediction = PredictionSet(
        name="known",
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        scores=np.array([0.1, 0.6, 0.4, 0.9], dtype=np.float64),
        is_probability=True,
    )

    result = measure_utility(prediction)

    assert result.brier_score is not None
    assert result.balanced_accuracy_at_0_5 is not None
    assert result.sensitivity_at_0_5 is not None
    assert result.false_positive_rate_at_0_5 is not None
    assert math.isclose(result.brier_score, 0.185, abs_tol=1e-12)
    assert math.isclose(result.balanced_accuracy_at_0_5, 0.5, abs_tol=1e-12)
    assert math.isclose(result.sensitivity_at_0_5, 0.5, abs_tol=1e-12)
    assert math.isclose(result.false_positive_rate_at_0_5, 0.5, abs_tol=1e-12)
