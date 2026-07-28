# pyright: standard, reportImplicitRelativeImport=false, reportPrivateUsage=false
"""corpus 간 특징 이동, 라벨 시각 민감도, 누수 음성 대조군을 진단한다.

특징 이동은 pairwise pooled IQR로 표준화한 1차원 Wasserstein 거리이며 인과적
domain-shift 증명이 아니다. 라벨 시각 분석은 원 protocol 전체를 동일 초만큼
이동해 center-window 라벨 규칙의 민감도만 측정한다. 음성 대조군은 피험자
내부에서 라벨을 섞은 뒤 전체 LOSO를 다시 돌려 파이프라인이 우연 수준으로
돌아오는지 확인한다. 0.5에서 크게 벗어나면 특징·분할·평가 경로 어딘가에
라벨 정보가 새고 있다는 뜻이다.
"""
from __future__ import annotations

import json
from itertools import combinations
from typing import TypedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from scripts import analysis_crosscorpus as crosscorpus
from scripts import utils_analysis as isohr
from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR

ALIGNMENT_SHIFTS_SECONDS = (-64.0, -32.0, 0.0, 32.0, 64.0)
NEGATIVE_CONTROL_REPEATS = 5
CHANCE_AUROC = 0.5


class _FeaturePair(TypedDict):
    datasets: list[str]
    mean_robust_wasserstein: float
    feature_distances: dict[str, float]


class _FeatureShiftPayload(TypedDict):
    analysis: str
    distance: str
    window_seconds: float
    step_seconds: float
    pairs: list[_FeaturePair]


class _AlignmentResult(TypedDict):
    shift_seconds: float
    n_windows: int
    n_subjects: int
    n_stress: int
    single_scr_auroc: float
    within_logreg_auroc: float


class _AlignmentPayload(TypedDict):
    analysis: str
    shift_convention: str
    window_seconds: float
    step_seconds: float
    results: list[_AlignmentResult]


def _write_payload(
    filename: str,
    payload: _FeatureShiftPayload | _AlignmentPayload,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _robust_wasserstein(first: np.ndarray, second: np.ndarray) -> float:
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    pooled = np.concatenate((first, second))
    interquartile_range = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    if interquartile_range <= np.finfo(float).eps:
        return 0.0
    return float(wasserstein_distance(first, second) / interquartile_range)


def run_feature_shift() -> _FeatureShiftPayload:
    crosscorpus.require_runtime_dependencies()
    corpora, _ = crosscorpus.build_all()
    feature_names = [isohr.FEATS[index] for index in crosscorpus.TCOLS]
    pair_results: list[_FeaturePair] = []
    distance_rows = []
    pair_labels = []
    for first_name, second_name in combinations(corpora, 2):
        first_features = corpora[first_name][0]
        second_features = corpora[second_name][0]
        distances = [
            _robust_wasserstein(first_features[:, index], second_features[:, index])
            for index in crosscorpus.TCOLS
        ]
        pair_results.append(
            {
                "datasets": [first_name, second_name],
                "mean_robust_wasserstein": float(np.mean(distances)),
                "feature_distances": dict(zip(feature_names, distances, strict=True)),
            }
        )
        distance_rows.append(distances)
        pair_labels.append(f"{first_name} vs {second_name}")

    distance_matrix = np.asarray(distance_rows)
    figure, axis = plt.subplots(figsize=(10.5, 3.8))
    image = axis.imshow(distance_matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(feature_names)), feature_names, rotation=35, ha="right")
    axis.set_yticks(range(len(pair_labels)), pair_labels)
    axis.set_title("Pairwise feature shift (Wasserstein distance / pooled IQR)")
    figure.colorbar(image, ax=axis, label="Robust standardized distance")
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F5_feature_shift.png", dpi=180)
    plt.close(figure)

    payload: _FeatureShiftPayload = {
        "analysis": "pairwise transfer-feature distribution shift",
        "distance": "1D Wasserstein distance divided by pairwise pooled IQR",
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "pairs": pair_results,
    }
    _write_payload("revision_feature_shift.json", payload)
    return payload


def run_stress_predict_alignment() -> _AlignmentPayload:
    crosscorpus.require_runtime_dependencies()
    crosscorpus.require_runtime_paths()
    results: list[_AlignmentResult] = []
    for shift_seconds in ALIGNMENT_SHIFTS_SECONDS:
        features, labels, subjects = crosscorpus.arr(
            crosscorpus.build_sp(label_shift_seconds=shift_seconds)
        )
        evaluation_indices = np.arange(len(labels))
        evaluated_labels, _, probabilities, _ = isohr.loso(
            features,
            labels,
            subjects,
            crosscorpus.TCOLS,
            evaluation_indices,
            mode="none",
            est="lr",
        )
        results.append(
            {
                "shift_seconds": shift_seconds,
                "n_windows": len(labels),
                "n_subjects": len(np.unique(subjects)),
                "n_stress": int(labels.sum()),
                "single_scr_auroc": float(crosscorpus.single_auroc(labels, features)),
                "within_logreg_auroc": float(isohr.auroc(evaluated_labels, probabilities)),
            }
        )

    shifts = [result["shift_seconds"] for result in results]
    single_scores = [result["single_scr_auroc"] for result in results]
    logreg_scores = [result["within_logreg_auroc"] for result in results]
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.plot(shifts, single_scores, marker="o", label="Single SCR")
    axis.plot(shifts, logreg_scores, marker="o", label="Within-corpus LogReg")
    axis.axvline(0.0, color="grey", linestyle="--")
    axis.set(
        xlabel="Protocol label shift relative to sensor recording (s)",
        ylabel="AUROC",
        title="Stress-Predict label-alignment sensitivity",
    )
    axis.legend()
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F6_stress_predict_alignment.png", dpi=180)
    plt.close(figure)

    payload: _AlignmentPayload = {
        "analysis": "Stress-Predict protocol-label alignment sensitivity",
        "shift_convention": "positive values move all protocol intervals later",
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "results": results,
    }
    _write_payload("revision_stress_predict_alignment.json", payload)
    return payload


class _NegativeControlCorpus(TypedDict):
    corpus: str
    n_windows: int
    n_subjects: int
    observed_auroc: float
    scrambled_mean_auroc: float
    scrambled_sd_auroc: float
    scrambled_min_auroc: float
    scrambled_max_auroc: float
    max_deviation_from_chance: float


class _NegativeControlPayload(TypedDict):
    analysis: str
    protocol: str
    window_seconds: float
    step_seconds: float
    n_repeats: int
    notes: list[str]
    corpora: list[_NegativeControlCorpus]


def run_negative_control() -> _NegativeControlPayload:
    """피험자 내부 라벨 섞기로 파이프라인의 누수 음성 대조군을 측정한다.

    관측 라벨로 얻은 within-corpus LOSO AUROC와, 같은 파이프라인을 라벨만 섞어
    다시 돌린 AUROC를 비교한다. 섞은 AUROC가 0.5 근처가 아니면 특징 계산·분할·
    평가 중 어딘가가 라벨을 보고 있다는 뜻이므로, 논문 수치를 신뢰할 수 없다.

    Returns:
        corpus별 관측 AUROC와 라벨 섞기 반복의 요약 통계.

    """
    crosscorpus.require_runtime_dependencies()
    corpora, _ = crosscorpus.build_all()
    generator = np.random.default_rng(isohr.SEED)
    corpus_results: list[_NegativeControlCorpus] = []
    for corpus_name, (features, labels, subjects) in corpora.items():
        evaluation_indices = np.arange(len(labels))
        observed_labels, _, observed_scores, _ = isohr.loso(
            features,
            labels,
            subjects,
            crosscorpus.TCOLS,
            evaluation_indices,
            mode="none",
            est="lr",
        )
        scrambled: list[float] = []
        for _ in range(NEGATIVE_CONTROL_REPEATS):
            shuffled = labels.copy()
            for subject in np.unique(subjects):
                position = np.where(subjects == subject)[0]
                shuffled[position] = generator.permutation(labels[position])
            control_labels, _, control_scores, _ = isohr.loso(
                features,
                shuffled,
                subjects,
                crosscorpus.TCOLS,
                evaluation_indices,
                mode="none",
                est="lr",
            )
            scrambled.append(float(isohr.auroc(control_labels, control_scores)))
        values = np.asarray(scrambled, float)
        corpus_results.append(
            {
                "corpus": corpus_name,
                "n_windows": len(labels),
                "n_subjects": len(np.unique(subjects)),
                "observed_auroc": float(isohr.auroc(observed_labels, observed_scores)),
                "scrambled_mean_auroc": float(np.mean(values)),
                "scrambled_sd_auroc": float(np.std(values)),
                "scrambled_min_auroc": float(np.min(values)),
                "scrambled_max_auroc": float(np.max(values)),
                "max_deviation_from_chance": float(
                    np.max(np.abs(values - CHANCE_AUROC))
                ),
            }
        )

    payload: _NegativeControlPayload = {
        "analysis": "within-subject label-scramble negative control",
        "protocol": (
            "labels permuted within each subject, then the full within-corpus "
            "LOSO logistic pipeline is re-run unchanged"
        ),
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "n_repeats": NEGATIVE_CONTROL_REPEATS,
        "notes": [
            (
                "Shuffling within subject preserves each subject's class balance, so "
                "the null is the pipeline itself rather than a different label "
                "distribution."
            ),
            (
                "A scrambled AUROC far from 0.5 would indicate that label information "
                "reaches the model through feature construction, the split, or the "
                "evaluation path."
            ),
        ],
        "corpora": corpus_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_negative_control.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
