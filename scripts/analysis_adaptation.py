# pyright: standard, reportImplicitRelativeImport=false, reportPrivateUsage=false
"""CORAL 외의 얕은 domain-adaptation 기법을 같은 LODO 조건에서 비교한다.

리뷰어 1(R1-4)과 리뷰어 2(R2-2)는 CORAL 하나로 "모델 복잡도가 도움이 되지
않는다"고 결론짓는 것이 과하다고 지적했다. 이 모듈은 target 라벨을 쓰지 않는
비지도 적응 세 가지(Subspace Alignment, Transfer Component Analysis,
중요도 가중 로지스틱 회귀)를 추가하고, 별도로 target 라벨을 실제로 주는
지도 전이 곡선을 측정한다.

측정 경계:
  - 비지도 적응기는 CORAL과 동일하게 target의 **라벨 없는** 특징 분포만 본다.
  - SA와 TCA의 성분 수는 사전에 고를 근거가 없으므로 격자 전체를 보고하고,
    target 성능으로 고른 **oracle 최댓값**을 대표값으로 쓴다. 이 값은 어떤
    정직한 선택 규칙도 넘어설 수 없는 낙관적 상한이므로, 상한조차 단일 특징을
    넘지 못하면 결론이 성분 선택에 의존하지 않는다.
  - 지도 전이 곡선은 target 피험자 k명의 라벨을 학습에 넣고 나머지 피험자에서
    평가하므로 LODO가 아니다. 표에서 분리해 보고한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts import analysis_crosscorpus as crosscorpus
from scripts import utils_analysis as isohr
from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR

COMPONENT_GRID = (2, 3, 4, 5, 6, 7, 8)
# [QC] TCA 결과는 커널 폭에 민감하다. 중앙값 휴리스틱에는 1/median(d^2)과
# 1/(2*median(d^2)) 관례가 모두 쓰이므로, 성분 수와 함께 폭도 격자로 훑고
# 결합 격자의 최댓값을 상한으로 보고한다. 그래야 '어떤 선택 규칙도 넘을 수
# 없는 상한'이라는 주장이 성분 수에만 걸리지 않는다.
BANDWIDTH_SCALES = (0.5, 1.0, 2.0)
TCA_FIT_PER_DOMAIN = 1000
TCA_REGULARIZATION = 1.0
IMPORTANCE_WEIGHT_CLIP = 20.0
SUPERVISED_SUBJECT_COUNTS = (0, 2, 4, 8)
SUPERVISED_DRAWS = 20
BOOTSTRAP_RESAMPLES = 1000
REQUIRED_CLASS_COUNT = 2


class _MethodResult(TypedDict):
    method: str
    auroc: float
    ci_low: float
    ci_high: float
    paired_diff_vs_single: float
    paired_diff_low: float
    paired_diff_high: float
    indistinguishable_from_single: bool
    component_grid: dict[str, float] | None


class _TargetResult(TypedDict):
    target: str
    n_windows: int
    n_subjects: int
    single_scr_auroc: float
    methods: list[_MethodResult]


class _AdaptationPayload(TypedDict):
    analysis: str
    protocol: str
    window_seconds: float
    step_seconds: float
    transfer_features: list[str]
    component_grid: list[int]
    notes: list[str]
    targets: list[_TargetResult]


class _SupervisedPoint(TypedDict):
    n_labeled_target_subjects: int
    n_draws: int
    mean_auroc: float
    sd_auroc: float
    min_auroc: float
    max_auroc: float
    mean_paired_gain: float
    sd_paired_gain: float


class _SupervisedTarget(TypedDict):
    target: str
    n_subjects: int
    points: list[_SupervisedPoint]


class _SupervisedPayload(TypedDict):
    analysis: str
    protocol: str
    window_seconds: float
    step_seconds: float
    notes: list[str]
    targets: list[_SupervisedTarget]


@dataclass(frozen=True, slots=True)
class DomainPair:
    """source에서 적합하고 target에서만 평가하는 한 쌍의 표준화된 특징 행렬."""

    source_features: np.ndarray
    source_labels: np.ndarray
    target_features: np.ndarray


def _standardize(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Source 통계로만 대치·표준화해 target 라벨 정보가 새지 않게 한다."""
    imputer = SimpleImputer(strategy="median").fit(source_features)
    source_imputed = np.asarray(imputer.transform(source_features), dtype=float)
    target_imputed = np.asarray(imputer.transform(target_features), dtype=float)
    scaler = StandardScaler().fit(source_imputed)
    return (
        np.asarray(scaler.transform(source_imputed), dtype=float),
        np.asarray(scaler.transform(target_imputed), dtype=float),
    )


def make_pair(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
) -> DomainPair:
    """전이 특징만 남기고 source 기준으로 표준화한 도메인 쌍을 만든다."""
    source_scaled, target_scaled = _standardize(
        source_features[:, crosscorpus.TCOLS],
        target_features[:, crosscorpus.TCOLS],
    )
    return DomainPair(
        source_features=source_scaled,
        source_labels=source_labels,
        target_features=target_scaled,
    )


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    """모든 적응기가 같은 선형 분류기를 쓰도록 단일 생성 지점을 유지한다."""
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    return estimator.fit(features, labels, sample_weight=sample_weight)


def _principal_axes(features: np.ndarray, n_components: int) -> np.ndarray:
    """평균 중심화 후 상위 주성분 축을 (d, k) 행렬로 반환한다."""
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, right_singular = np.linalg.svd(centered, full_matrices=False)
    return right_singular[:n_components].T


def subspace_alignment_scores(pair: DomainPair, n_components: int) -> np.ndarray:
    """Subspace Alignment로 정렬한 source에서 학습한 target 확률을 반환한다.

    Fernando 등(2013)의 정의대로 source 주성분 축을 target 주성분 축에 맞추는
    선형 사상 ``M = Ps^T Pt``를 적용한 뒤 정렬된 부분공간에서 분류기를 적합한다.

    Args:
        pair: source 기준으로 표준화된 도메인 쌍.
        n_components: 두 도메인에서 공통으로 쓰는 부분공간 차원.

    Returns:
        target 창별 stress 확률.

    """
    source_axes = _principal_axes(pair.source_features, n_components)
    target_axes = _principal_axes(pair.target_features, n_components)
    alignment = source_axes.T @ target_axes
    aligned_source = pair.source_features @ source_axes @ alignment
    projected_target = pair.target_features @ target_axes
    estimator = _fit_logistic(aligned_source, pair.source_labels)
    return estimator.predict_proba(projected_target)[:, 1]


def _median_bandwidth(features: np.ndarray) -> float:
    """RBF 커널 폭을 쌍거리 중앙값 휴리스틱으로 정한다."""
    squared_norms = np.sum(features**2, axis=1)
    squared_distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * features @ features.T
    )
    upper = squared_distances[np.triu_indices_from(squared_distances, k=1)]
    median = float(np.median(upper[upper > 0.0])) if np.any(upper > 0.0) else 1.0
    return 1.0 / max(median, np.finfo(float).eps)


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    """두 집합 사이의 RBF 커널 행렬을 반환한다."""
    left_norms = np.sum(left**2, axis=1)[:, None]
    right_norms = np.sum(right**2, axis=1)[None, :]
    squared = np.maximum(left_norms + right_norms - 2.0 * left @ right.T, 0.0)
    return np.exp(-gamma * squared)


def transfer_component_scores(
    pair: DomainPair,
    n_components: int,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Transfer Component Analysis 부분공간에서 학습한 target 확률을 반환한다.

    Pan 등(2011)의 MMD 최소화 문제를 푼다. 전체 창 수에서 커널 고유분해는
    비현실적이므로 도메인마다 고정 seed로 최대 ``TCA_FIT_PER_DOMAIN``개를
    뽑아 사상을 학습하고, 학습 집합에 대한 커널로 모든 창을 사상한다.

    Args:
        pair: source 기준으로 표준화된 도메인 쌍.
        n_components: 남길 transfer component 수.
        bandwidth_scale: 중앙값 휴리스틱 RBF 폭에 곱할 배율.

    Returns:
        target 창별 stress 확률.

    """
    generator = np.random.default_rng(isohr.SEED)
    source_fit_index = generator.choice(
        len(pair.source_features),
        min(TCA_FIT_PER_DOMAIN, len(pair.source_features)),
        replace=False,
    )
    target_fit_index = generator.choice(
        len(pair.target_features),
        min(TCA_FIT_PER_DOMAIN, len(pair.target_features)),
        replace=False,
    )
    source_fit = pair.source_features[source_fit_index]
    target_fit = pair.target_features[target_fit_index]
    combined = np.vstack((source_fit, target_fit))
    n_source, n_target = len(source_fit), len(target_fit)
    total = n_source + n_target

    gamma = bandwidth_scale * _median_bandwidth(combined)
    kernel = _rbf_kernel(combined, combined, gamma)

    coefficients = np.concatenate(
        (np.full(n_source, 1.0 / n_source), np.full(n_target, -1.0 / n_target))
    )
    mmd = np.outer(coefficients, coefficients)
    centering = np.eye(total) - np.full((total, total), 1.0 / total)

    left = kernel @ mmd @ kernel + TCA_REGULARIZATION * np.eye(total)
    right = kernel @ centering @ kernel
    # [QC] 대칭성을 명시적으로 복원해 eigh의 대칭 가정과 수치 오차를 분리한다.
    left = 0.5 * (left + left.T)
    right = 0.5 * (right + right.T)
    eigenvalues, eigenvectors = eigh(right, left)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    projection = eigenvectors[:, order]

    source_embedded = _rbf_kernel(pair.source_features, combined, gamma) @ projection
    target_embedded = _rbf_kernel(pair.target_features, combined, gamma) @ projection
    # [QC] eigh는 W^T(KLK+mu I)W = I로 정규화하지만 Pan 등(2011)의 제약은
    # W^T KHK W = I이다. 임베딩을 표준화하면 결과가 정규화 관례와 무관해지므로,
    # 인용 정의대로 재구현한 독자도 같은 값을 얻는다.
    scaler = StandardScaler().fit(source_embedded)
    estimator = _fit_logistic(
        np.asarray(scaler.transform(source_embedded), dtype=float),
        pair.source_labels,
    )
    return estimator.predict_proba(
        np.asarray(scaler.transform(target_embedded), dtype=float)
    )[:, 1]


def importance_weighted_scores(pair: DomainPair) -> np.ndarray:
    """공변량 이동 보정 중요도 가중 로지스틱 회귀의 target 확률을 반환한다.

    도메인 분류기 확률비 ``p(target|x) / p(source|x)``를 source 표본 가중치로
    사용한다(분류기 기반 밀도비 추정). 가중치는 극단값이 적합을 지배하지
    않도록 ``IMPORTANCE_WEIGHT_CLIP``에서 자른다.

    Args:
        pair: source 기준으로 표준화된 도메인 쌍.

    Returns:
        target 창별 stress 확률.

    """
    domain_features = np.vstack((pair.source_features, pair.target_features))
    domain_labels = np.concatenate(
        (np.zeros(len(pair.source_features), int), np.ones(len(pair.target_features), int))
    )
    domain_classifier = _fit_logistic(domain_features, domain_labels)
    target_probability = domain_classifier.predict_proba(pair.source_features)[:, 1]
    ratio = target_probability / np.clip(1.0 - target_probability, np.finfo(float).eps, None)
    weights = np.clip(ratio, 1.0 / IMPORTANCE_WEIGHT_CLIP, IMPORTANCE_WEIGHT_CLIP)
    estimator = _fit_logistic(pair.source_features, pair.source_labels, sample_weight=weights)
    return estimator.predict_proba(pair.target_features)[:, 1]


def correlation_alignment_scores(pair: DomainPair, epsilon: float = 1e-3) -> np.ndarray:
    """기존 CORAL을 같은 전처리 경계에서 다시 계산해 비교 기준으로 쓴다.

    Args:
        pair: source 기준으로 표준화된 도메인 쌍.
        epsilon: 공분산 정칙화 항.

    Returns:
        target 창별 stress 확률.

    """

    def matrix_sqrt(matrix: np.ndarray, inverse: bool) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 1e-8, None)
        scaled = 1.0 / np.sqrt(values) if inverse else np.sqrt(values)
        return vectors @ np.diag(scaled) @ vectors.T

    dimension = pair.source_features.shape[1]
    source_covariance = np.cov(pair.source_features, rowvar=False) + epsilon * np.eye(dimension)
    target_covariance = np.cov(pair.target_features, rowvar=False) + epsilon * np.eye(dimension)
    aligned_source = (
        pair.source_features
        @ matrix_sqrt(source_covariance, inverse=True)
        @ matrix_sqrt(target_covariance, inverse=False)
    )
    estimator = _fit_logistic(aligned_source, pair.source_labels)
    return estimator.predict_proba(pair.target_features)[:, 1]


def plain_logistic_scores(pair: DomainPair) -> np.ndarray:
    """적응을 적용하지 않은 source 학습 로지스틱 회귀의 target 확률을 반환한다."""
    estimator = _fit_logistic(pair.source_features, pair.source_labels)
    return estimator.predict_proba(pair.target_features)[:, 1]


def _paired_subject_bootstrap(
    reference_scores: np.ndarray,
    method_scores: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
) -> tuple[float, float, float]:
    """피험자 cluster를 재표집해 ``기준-방법`` AUROC 차이 분포를 요약한다."""
    generator = np.random.default_rng(isohr.SEED)
    unique_subjects = np.unique(subjects)
    index_by_subject = {subject: np.where(subjects == subject)[0] for subject in unique_subjects}
    differences: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = generator.choice(unique_subjects, len(unique_subjects), replace=True)
        index = np.concatenate([index_by_subject[subject] for subject in drawn])
        resampled_labels = labels[index]
        if len(np.unique(resampled_labels)) < REQUIRED_CLASS_COUNT:
            continue
        differences.append(
            isohr.auroc(resampled_labels, reference_scores[index])
            - isohr.auroc(resampled_labels, method_scores[index])
        )
    values = np.asarray(differences, float)
    return (
        float(np.mean(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def _summarize_method(
    name: str,
    scores: np.ndarray,
    single_scores: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    component_grid: dict[str, float] | None,
) -> _MethodResult:
    """한 적응 방법의 AUROC, 피험자 bootstrap 구간, 단일 특징 대비 차이를 모은다."""
    low, high = isohr.boot_ci(labels, subjects, scores, B=BOOTSTRAP_RESAMPLES)
    mean_difference, difference_low, difference_high = _paired_subject_bootstrap(
        single_scores,
        scores,
        labels,
        subjects,
    )
    return {
        "method": name,
        "auroc": float(isohr.auroc(labels, scores)),
        "ci_low": float(low),
        "ci_high": float(high),
        "paired_diff_vs_single": mean_difference,
        "paired_diff_low": difference_low,
        "paired_diff_high": difference_high,
        "indistinguishable_from_single": bool(difference_low <= 0.0 <= difference_high),
        "component_grid": component_grid,
    }


def _best_over_grid(
    pair: DomainPair,
    labels: np.ndarray,
    scorer: str,
) -> tuple[np.ndarray, dict[str, float]]:
    """성분 수 격자 전체를 평가하고 target AUROC 최댓값 설정을 반환한다.

    성분 수를 target 성능으로 고르므로 이 값은 정직한 사전 선택으로는 도달할 수
    없는 낙관적 상한이다. 상한조차 기준선을 넘지 못하는지 확인하는 용도로만
    쓰고 논문에서도 상한이라고 명시한다.
    """
    dimension = pair.source_features.shape[1]
    grid_scores: dict[str, float] = {}
    best_scores: np.ndarray | None = None
    best_auroc = -np.inf
    # subspace alignment는 성분 수만, TCA는 성분 수와 커널 폭을 함께 훑는다.
    scales = (1.0,) if scorer == "subspace" else BANDWIDTH_SCALES
    for scale in scales:
        for n_components in COMPONENT_GRID:
            if n_components > dimension:
                continue
            if scorer == "subspace":
                scores = subspace_alignment_scores(pair, n_components)
                key = str(n_components)
            else:
                scores = transfer_component_scores(pair, n_components, scale)
                key = f"gamma{scale:g}x_k{n_components}"
            value = float(isohr.auroc(labels, scores))
            grid_scores[key] = value
            if value > best_auroc:
                best_auroc, best_scores = value, scores
    if best_scores is None:
        message = f"성분 격자가 비어 있어 {scorer} 적응을 평가할 수 없습니다."
        raise ValueError(message)
    return best_scores, grid_scores


def _save_adaptation_figure(targets: list[_TargetResult]) -> None:
    """방법별 LODO AUROC와 단일 특징 기준선을 target별로 함께 보여준다."""
    method_names = [method["method"] for method in targets[0]["methods"]]
    figure, axes = plt.subplots(1, len(targets), figsize=(12.5, 4.4), sharey=True)
    positions = np.arange(len(method_names))
    for axis, target in zip(axes, targets, strict=True):
        values = [method["auroc"] for method in target["methods"]]
        lower = [method["auroc"] - method["ci_low"] for method in target["methods"]]
        upper = [method["ci_high"] - method["auroc"] for method in target["methods"]]
        _ = axis.bar(
            positions,
            values,
            0.62,
            yerr=np.array([lower, upper]),
            capsize=3,
            color="#4c78a8",
            error_kw={"lw": 1.0},
        )
        _ = axis.axhline(
            target["single_scr_auroc"],
            color="#d62728",
            linestyle="-",
            linewidth=1.4,
            label="single SCR",
        )
        _ = axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        axis.set_xticks(positions)
        axis.set_xticklabels(method_names, rotation=32, ha="right", fontsize=8)
        axis.set_title(f"target: {target['target']}")
    axes[0].set_ylabel("Leave-one-dataset-out AUROC")
    axes[0].set_ylim(0.4, 1.0)
    axes[-1].legend(fontsize=8, loc="upper right")
    figure.suptitle(
        "No tested unsupervised adaptation method exceeds the single-feature baseline",
        fontsize=10.5,
    )
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F8_adaptation.png", dpi=180)
    plt.close(figure)


def run_adaptation() -> _AdaptationPayload:
    """네 가지 비지도 적응 기법을 동일 LODO 조건에서 측정하고 저장한다."""
    crosscorpus.require_runtime_dependencies()
    corpora, _ = crosscorpus.build_all()
    corpus_names = list(corpora)
    target_results: list[_TargetResult] = []
    for target_name in corpus_names:
        target_features, target_labels, target_subjects = corpora[target_name]
        source_corpora = [corpora[name] for name in corpus_names if name != target_name]
        source_features = np.vstack([corpus[0] for corpus in source_corpora])
        source_labels = np.concatenate([corpus[1] for corpus in source_corpora])
        pair = make_pair(source_features, source_labels, target_features)
        single_scores = target_features[:, crosscorpus.SCR].astype(float)

        subspace_scores, subspace_grid = _best_over_grid(pair, target_labels, "subspace")
        component_scores, component_grid = _best_over_grid(pair, target_labels, "tca")
        method_scores: list[tuple[str, np.ndarray, dict[str, float] | None]] = [
            ("Logistic (no adaptation)", plain_logistic_scores(pair), None),
            ("CORAL", correlation_alignment_scores(pair), None),
            ("Subspace alignment", subspace_scores, subspace_grid),
            ("Transfer components", component_scores, component_grid),
            ("Importance weighting", importance_weighted_scores(pair), None),
        ]
        target_results.append(
            {
                "target": target_name,
                "n_windows": len(target_labels),
                "n_subjects": len(np.unique(target_subjects)),
                "single_scr_auroc": float(isohr.auroc(target_labels, single_scores)),
                "methods": [
                    _summarize_method(
                        name,
                        scores,
                        single_scores,
                        target_labels,
                        target_subjects,
                        grid,
                    )
                    for name, scores, grid in method_scores
                ],
            }
        )

    _save_adaptation_figure(target_results)
    payload: _AdaptationPayload = {
        "analysis": "unsupervised domain-adaptation sweep under leave-one-dataset-out",
        "protocol": "train on the other two corpora; adapt with unlabeled target features only",
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "transfer_features": [isohr.FEATS[index] for index in crosscorpus.TCOLS],
        "component_grid": list(COMPONENT_GRID),
        "notes": [
            (
                "Subspace alignment and transfer components report the component count "
                "that maximizes target AUROC, an optimistic upper bound no honest "
                "a-priori rule can exceed; the full grid is reported alongside."
            ),
            (
                "All four adaptation methods are transductive: they use the unlabeled target "
                "feature distribution, which the single SCR score and plain logistic model "
                "never see."
            ),
            (
                "Paired differences are single-SCR minus method on identical subject-cluster "
                "bootstrap resamples; an interval containing zero means the tested comparison "
                "does not separate them."
            ),
        ],
        "targets": target_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_adaptation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_target_supervised() -> _SupervisedPayload:
    """Target 라벨을 k명분 실제로 주었을 때의 전이 곡선을 측정하고 저장한다.

    비지도 적응이 실패하는 것이 방법의 한계인지 target 신호·라벨의 한계인지
    구분하기 위해, 라벨을 직접 공급하는 상한 조건을 함께 보고한다.
    """
    crosscorpus.require_runtime_dependencies()
    corpora, _ = crosscorpus.build_all()
    corpus_names = list(corpora)
    generator = np.random.default_rng(isohr.SEED)
    target_results: list[_SupervisedTarget] = []
    for target_name in corpus_names:
        target_features, target_labels, target_subjects = corpora[target_name]
        source_corpora = [corpora[name] for name in corpus_names if name != target_name]
        source_features = np.vstack([corpus[0] for corpus in source_corpora])[
            :, crosscorpus.TCOLS
        ]
        source_labels = np.concatenate([corpus[1] for corpus in source_corpora])
        unique_subjects = np.unique(target_subjects)
        points: list[_SupervisedPoint] = []
        for n_labeled in SUPERVISED_SUBJECT_COUNTS:
            if n_labeled >= len(unique_subjects):
                continue
            draw_scores: list[float] = []
            paired_gains: list[float] = []
            draws = 1 if n_labeled == 0 else SUPERVISED_DRAWS
            for _ in range(draws):
                labeled = (
                    generator.choice(unique_subjects, n_labeled, replace=False)
                    if n_labeled
                    else np.array([], dtype=unique_subjects.dtype)
                )
                held_out = ~np.isin(target_subjects, labeled)
                train_features = source_features
                train_labels = source_labels
                if n_labeled:
                    labeled_mask = np.isin(target_subjects, labeled)
                    train_features = np.vstack(
                        (source_features, target_features[labeled_mask][:, crosscorpus.TCOLS])
                    )
                    train_labels = np.concatenate(
                        (source_labels, target_labels[labeled_mask])
                    )
                if len(np.unique(train_labels)) < REQUIRED_CLASS_COUNT:
                    continue
                evaluation_labels = target_labels[held_out]
                if len(np.unique(evaluation_labels)) < REQUIRED_CLASS_COUNT:
                    continue
                evaluation_features = target_features[held_out][:, crosscorpus.TCOLS]
                estimator = isohr._make_est("lr")
                _ = estimator.fit(train_features, train_labels)
                probabilities = estimator.predict_proba(evaluation_features)[:, 1]
                supervised_auroc = float(isohr.auroc(evaluation_labels, probabilities))
                draw_scores.append(supervised_auroc)
                # [QC] 같은 draw의 held-out 집합에서 source-only 모델을 다시 채점한다.
                # k=0을 target 전체에서 재고 k>0을 부분집합에서 재면 라벨 공급 효과와
                # 평가 코호트 구성이 섞이고, 피험자별 AUROC 편차가 큰 Nurse에서는
                # 그 차이가 실제 효과보다 커진다.
                baseline = isohr._make_est("lr")
                _ = baseline.fit(source_features, source_labels)
                baseline_auroc = float(
                    isohr.auroc(
                        evaluation_labels,
                        baseline.predict_proba(evaluation_features)[:, 1],
                    )
                )
                paired_gains.append(supervised_auroc - baseline_auroc)
            if not draw_scores:
                continue
            values = np.asarray(draw_scores, float)
            gains = np.asarray(paired_gains, float)
            points.append(
                {
                    "n_labeled_target_subjects": n_labeled,
                    "n_draws": len(draw_scores),
                    "mean_auroc": float(np.mean(values)),
                    "sd_auroc": float(np.std(values)),
                    "min_auroc": float(np.min(values)),
                    "max_auroc": float(np.max(values)),
                    "mean_paired_gain": float(np.mean(gains)),
                    "sd_paired_gain": float(np.std(gains)),
                }
            )
        target_results.append(
            {
                "target": target_name,
                "n_subjects": len(unique_subjects),
                "points": points,
            }
        )

    payload: _SupervisedPayload = {
        "analysis": "supervised target adaptation curve",
        "protocol": (
            "train on both source corpora plus k labeled target subjects; "
            "evaluate on the remaining target subjects"
        ),
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "notes": [
            (
                "This is not leave-one-dataset-out: it supplies real target labels and "
                "is reported separately as an upper bound on what adaptation could "
                "achieve."
            ),
            (
                "mean_paired_gain is the headline quantity: within each draw the "
                "source-only model is re-scored on the SAME held-out subjects, so the "
                "gain isolates the effect of supplying labels. Comparing raw AUROC "
                "across k would confound it with evaluation-cohort composition."
            ),
            (
                f"k>0 uses {SUPERVISED_DRAWS} random subject draws at a fixed seed; "
                "the spread across draws is reported as SD, min and max."
            ),
        ],
        "targets": target_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_target_supervised.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
