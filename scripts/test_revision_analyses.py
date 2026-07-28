# pyright: standard
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import analysis_adaptation as adaptation
from scripts import analysis_error_strata as error_strata
from scripts import utils_analysis as isohr
from scripts.utils_paths import RESULTS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_FILENAMES = (
    "revision_adaptation.json",
    "revision_context_aware.json",
    "revision_error_stratification.json",
    "revision_target_supervised.json",
)


def _synthetic_pair(shift: float, seed: int = 0) -> tuple[adaptation.DomainPair, np.ndarray]:
    # Given: 같은 결정 규칙을 공유하지만 2차 통계만 다른 두 도메인.
    generator = np.random.default_rng(seed)
    weights = np.array([1.2, -0.8, 0.5, 0.3])
    source = generator.normal(size=(400, 4))
    target = generator.normal(size=(300, 4)) * (1.0 + shift)
    source_labels = (source @ weights + 0.2 * generator.normal(size=400) > 0).astype(int)
    target_labels = (target @ weights + 0.2 * generator.normal(size=300) > 0).astype(int)
    pair = adaptation.DomainPair(
        source_features=source,
        source_labels=source_labels,
        target_features=target,
    )
    return pair, target_labels


def test_activity_features_match_their_documented_definitions() -> None:
    # Given: g 단위 가속도 크기 창 하나.
    magnitudes = np.array([1.0, 1.5, 0.5, 2.0, 1.0])

    # When: 활동 맥락 특징을 계산한다.
    values = isohr.activity_features(magnitudes, fs=32.0)

    # Then: 네 값이 문서화된 정의와 정확히 일치해야 한다.
    def close(actual: float, expected: float) -> bool:
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)

    assert close(values[isohr.ACC_IDX["acc_mag_mean"]], float(np.mean(magnitudes)))
    assert close(values[isohr.ACC_IDX["acc_mag_std"]], float(np.std(magnitudes)))
    assert close(values[isohr.ACC_IDX["acc_enmo"]], float(np.mean([0, 0.5, 0, 1.0, 0])))
    expected_jerk = float(np.mean(np.abs(np.diff(magnitudes))) * 32.0)
    assert close(values[isohr.ACC_IDX["acc_jerk"]], expected_jerk)


def test_activity_std_is_the_published_movement_definition() -> None:
    # Given: 기존 Nurse movement tertile이 쓰던 정의와 같은 입력.
    generator = np.random.default_rng(7)
    magnitudes = generator.normal(1.0, 0.2, size=2048)

    # When: 새 활동 특징 블록에서 표준편차 성분을 꺼낸다.
    value = isohr.activity_features(magnitudes)[isohr.ACC_IDX["acc_mag_std"]]

    # Then: 발표된 movement 값과 정확히 같아야 층화 결과가 유지된다.
    assert value == float(np.std(magnitudes))


def test_activity_features_are_nan_for_unusable_windows() -> None:
    # Given: 표준편차를 정의할 수 없는 길이 1 창.
    # When/Then: 창을 버리지 않고 NaN으로 표시해 코호트 수를 유지해야 한다.
    assert np.all(np.isnan(isohr.activity_features(np.array([1.0]))))


@pytest.mark.parametrize(
    "scorer",
    ["plain", "coral", "subspace", "tca", "importance"],
)
def test_adaptation_methods_return_probabilities(scorer: str) -> None:
    # Given: 분포 이동이 있는 합성 도메인 쌍.
    pair, target_labels = _synthetic_pair(shift=0.8)

    # When: 각 적응 기법으로 target 점수를 만든다.
    if scorer == "plain":
        scores = adaptation.plain_logistic_scores(pair)
    elif scorer == "coral":
        scores = adaptation.correlation_alignment_scores(pair)
    elif scorer == "subspace":
        scores = adaptation.subspace_alignment_scores(pair, 3)
    elif scorer == "tca":
        scores = adaptation.transfer_component_scores(pair, 3)
    else:
        scores = adaptation.importance_weighted_scores(pair)

    # Then: 확률 범위와 길이가 유효하고 유한해야 한다.
    assert scores.shape == (len(target_labels),)
    assert np.all(np.isfinite(scores))
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


@pytest.mark.parametrize("scorer", ["coral", "subspace", "tca", "importance"])
def test_adaptation_is_near_identity_without_domain_shift(scorer: str) -> None:
    # Given: 두 도메인의 분포가 같아 적응이 필요 없는 경우.
    pair, target_labels = _synthetic_pair(shift=0.0, seed=3)
    baseline = float(isohr.auroc(target_labels, adaptation.plain_logistic_scores(pair)))

    # When: 적응 기법을 적용한다.
    if scorer == "coral":
        scores = adaptation.correlation_alignment_scores(pair)
    elif scorer == "subspace":
        scores = adaptation.subspace_alignment_scores(pair, 4)
    elif scorer == "tca":
        scores = adaptation.transfer_component_scores(pair, 4)
    else:
        scores = adaptation.importance_weighted_scores(pair)

    # Then: 이동이 없으면 적응기가 기준선을 무너뜨리지 않아야 한다.
    # 구현이 사실상 무작위 사영이라면 이 불변식이 먼저 깨진다.
    assert float(isohr.auroc(target_labels, scores)) > baseline - 0.10


def test_adaptation_scores_are_deterministic_under_the_fixed_seed() -> None:
    # Given: 같은 입력을 두 번 처리한다.
    pair, _ = _synthetic_pair(shift=0.5, seed=11)

    # When: 무작위 부분표본을 쓰는 TCA를 반복 실행한다.
    first = adaptation.transfer_component_scores(pair, 3)
    second = adaptation.transfer_component_scores(pair, 3)

    # Then: 고정 seed 아래 결과가 정확히 재현되어야 한다.
    assert np.array_equal(first, second)


def test_importance_weights_stay_inside_the_declared_clip() -> None:
    # Given: 도메인 분류기가 극단 확률을 낼 수 있는 큰 이동.
    pair, _ = _synthetic_pair(shift=3.0, seed=5)

    # When: 중요도 가중을 계산하는 경로를 그대로 실행한다.
    scores = adaptation.importance_weighted_scores(pair)

    # Then: 가중치 절단 덕분에 결과가 발산하지 않아야 한다.
    assert np.all(np.isfinite(scores))


def test_stratified_auroc_ignores_unusable_strata() -> None:
    # Given: 사용 가능한 층 하나와 사용 불가 층 두 개.
    rows: list[error_strata.Stratum] = [
        {
            "stratum": "a", "status": "ok", "n_windows": 100, "n_stress": 50,
            "n_non_stress": 50, "n_subjects": 5, "single_auroc": 0.8,
            "logistic_auroc": 0.7,
        },
        {
            "stratum": "b", "status": "single_class", "n_windows": 100, "n_stress": 100,
            "n_non_stress": 0, "n_subjects": 5, "single_auroc": None,
            "logistic_auroc": None,
        },
        {
            "stratum": "c", "status": "suppressed_small_cell", "n_windows": 5,
            "n_stress": 3, "n_non_stress": 2, "n_subjects": 1, "single_auroc": None,
            "logistic_auroc": None,
        },
    ]

    # When: 층화 AUROC를 합친다.
    value = error_strata.stratified_auroc(rows, "single_auroc")

    # Then: 가중치 0인 층은 결과를 오염시키지 않아야 한다.
    assert math.isclose(value, 0.8, rel_tol=1e-9, abs_tol=1e-12)


def test_quantile_labels_separate_finite_and_missing_values() -> None:
    # Given: 결측을 포함한 연속값.
    values = np.array([0.1, 0.5, 0.9, np.nan])

    # When: 두 경계로 세 층에 배정한다.
    assignment = error_strata.quantile_labels(values, (0.3, 0.7), ("low", "mid", "high"))

    # Then: 결측은 별도 층으로 남아 조용히 사라지지 않아야 한다.
    assert list(assignment) == ["low", "mid", "high", "unavailable"]


@pytest.mark.integration
def test_new_result_artifacts_carry_no_participant_identifier() -> None:
    # Given: 실제 데이터로 생성한 리비전 결과 artifact.
    present = [name for name in RESULT_FILENAMES if (RESULTS_DIR / name).is_file()]
    if not present:
        pytest.skip("리비전 분석 artifact가 아직 생성되지 않았습니다.")

    # When: 각 파일 전체 텍스트를 검사한다.
    for name in present:
        text = (RESULTS_DIR / name).read_text(encoding="utf-8")
        payload: object = json.loads(text)
        assert payload

        # Then: corpus 이름 외에 피험자 접두사 식별자가 남아서는 안 된다.
        for prefix in ('"W_', '"P_', '"N_'):
            assert prefix not in text, f"{name} leaks a subject identifier ({prefix})"
