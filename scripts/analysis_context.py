# pyright: standard, reportImplicitRelativeImport=false, reportPrivateUsage=false
"""손목 가속도 활동 맥락을 더한 context-aware 전이 모델을 평가한다.

리뷰어 2(R2-2)는 활동·업무량·환경 맥락을 넣은 multimodal 모델을 평가하라고
요청했다. 손목 E4에서 실제로 얻을 수 있는 맥락 채널은 가속도이므로, 세 corpus
모두에서 동일하게 계산한 활동 특징 4개를 전이 특징 8개 옆에 이어 붙여
corpus 내부 LOSO와 leave-one-dataset-out을 다시 측정한다.

해석 경계:
  - 활동은 corpus마다 라벨과의 연관 방향이 다르다(WESAD의 TSST는 기립·발화를
    포함해 stress에서 움직임이 늘고, Stress-Predict의 인지 과제는 오히려 준다).
    따라서 활동 특징을 넣으면 corpus 정체성 신호가 함께 들어올 수 있다.
  - 이 분석은 "활동 맥락을 더하면 전이가 회복되는가"라는 물음에만 답하며,
    활동이 stress를 유발하거나 EDA를 훼손한다는 인과 주장으로 쓰지 않는다.
"""
from __future__ import annotations

import json
from typing import TypedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts import analysis_crosscorpus as crosscorpus
from scripts import utils_analysis as isohr
from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR

BOOTSTRAP_RESAMPLES = 1000
REQUIRED_CLASS_COUNT = 2


class _WithinCorpusResult(TypedDict):
    corpus: str
    n_windows: int
    n_subjects: int
    activity_label_auroc: dict[str, float]
    activity_median_g: float
    base_loso_lr: float
    context_loso_lr: float
    base_loso_gbm: float
    context_loso_gbm: float


class _TransferResult(TypedDict):
    target: str
    single_scr_auroc: float
    base_lr_auroc: float
    context_lr_auroc: float
    context_lr_ci_low: float
    context_lr_ci_high: float
    base_gbm_auroc: float
    context_gbm_auroc: float
    activity_only_lr_auroc: float
    context_minus_base_lr: float


class _ContextPayload(TypedDict):
    analysis: str
    window_seconds: float
    step_seconds: float
    transfer_features: list[str]
    activity_features: list[str]
    notes: list[str]
    within_corpus: list[_WithinCorpusResult]
    transfer: list[_TransferResult]


def _signed_univariate_auroc(values: np.ndarray, labels: np.ndarray) -> float:
    """방향을 유지한 단변량 AUROC로 corpus 간 연관 방향 차이를 드러낸다."""
    finite = np.isfinite(values)
    if len(np.unique(labels[finite])) < REQUIRED_CLASS_COUNT:
        return float("nan")
    return float(isohr.auroc(labels[finite], values[finite]))


def _loso_auroc(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    estimator_name: str,
) -> float:
    """주어진 특징 행렬 전체를 입력으로 corpus 내부 LOSO AUROC를 구한다."""
    columns = list(range(features.shape[1]))
    evaluated_labels, _, probabilities, _ = isohr.loso(
        features,
        labels,
        subjects,
        columns,
        np.arange(len(labels)),
        mode="none",
        est=estimator_name,
    )
    return float(isohr.auroc(evaluated_labels, probabilities))


def _transfer_auroc(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    estimator_name: str,
) -> tuple[float, np.ndarray]:
    """Source 전체로 적합하고 target에서 평가한 AUROC와 확률을 반환한다."""
    estimator = isohr._make_est(estimator_name)
    _ = estimator.fit(source_features, source_labels)
    probabilities = estimator.predict_proba(target_features)[:, 1]
    return float(isohr.auroc(target_labels, probabilities)), probabilities


def _save_context_figure(transfer: list[_TransferResult]) -> None:
    """전이 target별로 기준 모델과 활동 맥락 모델을 나란히 보여준다."""
    targets = [result["target"] for result in transfer]
    positions = np.arange(len(targets))
    width = 0.26
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    _ = axis.bar(
        positions - width,
        [result["single_scr_auroc"] for result in transfer],
        width,
        label="single SCR",
        color="#1f77b4",
    )
    _ = axis.bar(
        positions,
        [result["base_lr_auroc"] for result in transfer],
        width,
        label="logistic (8 transfer features)",
        color="#ff7f0e",
    )
    _ = axis.bar(
        positions + width,
        [result["context_lr_auroc"] for result in transfer],
        width,
        label="logistic + activity context",
        color="#2ca02c",
    )
    _ = axis.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    axis.set_xticks(positions)
    axis.set_xticklabels(targets)
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("Leave-one-dataset-out AUROC")
    axis.set_title(
        "Adding wrist-accelerometer activity context does not recover transfer",
        fontsize=10.5,
    )
    axis.legend(fontsize=8.5, framealpha=0.9)
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F9_context_aware.png", dpi=180)
    plt.close(figure)


def run_context_aware() -> _ContextPayload:
    """활동 맥락 특징을 더한 모델을 corpus 내부와 전이 조건에서 측정한다."""
    crosscorpus.require_runtime_dependencies()
    corpora, raw_rows = crosscorpus.build_all()
    corpus_names = list(corpora)
    activity = {name: crosscorpus.acc_arr(raw_rows[name]) for name in corpus_names}
    base = {name: corpora[name][0][:, crosscorpus.TCOLS] for name in corpus_names}
    context = {
        name: np.hstack((base[name], activity[name])) for name in corpus_names
    }

    within: list[_WithinCorpusResult] = []
    for name in corpus_names:
        _, labels, subjects = corpora[name]
        movement = activity[name][:, isohr.ACC_IDX["acc_mag_std"]]
        within.append(
            {
                "corpus": name,
                "n_windows": len(labels),
                "n_subjects": len(np.unique(subjects)),
                "activity_label_auroc": {
                    feature_name: _signed_univariate_auroc(
                        activity[name][:, index],
                        labels,
                    )
                    for index, feature_name in enumerate(isohr.ACC_FEATS)
                },
                "activity_median_g": float(np.nanmedian(movement)),
                "base_loso_lr": _loso_auroc(base[name], labels, subjects, "lr"),
                "context_loso_lr": _loso_auroc(context[name], labels, subjects, "lr"),
                "base_loso_gbm": _loso_auroc(base[name], labels, subjects, "gbm"),
                "context_loso_gbm": _loso_auroc(context[name], labels, subjects, "gbm"),
            }
        )

    transfer: list[_TransferResult] = []
    for target_name in corpus_names:
        _, target_labels, target_subjects = corpora[target_name]
        source_names = [name for name in corpus_names if name != target_name]
        source_labels = np.concatenate([corpora[name][1] for name in source_names])
        base_auroc, _ = _transfer_auroc(
            np.vstack([base[name] for name in source_names]),
            source_labels,
            base[target_name],
            target_labels,
            "lr",
        )
        context_auroc, context_probabilities = _transfer_auroc(
            np.vstack([context[name] for name in source_names]),
            source_labels,
            context[target_name],
            target_labels,
            "lr",
        )
        base_gbm, _ = _transfer_auroc(
            np.vstack([base[name] for name in source_names]),
            source_labels,
            base[target_name],
            target_labels,
            "gbm",
        )
        context_gbm, _ = _transfer_auroc(
            np.vstack([context[name] for name in source_names]),
            source_labels,
            context[target_name],
            target_labels,
            "gbm",
        )
        activity_only, _ = _transfer_auroc(
            np.vstack([activity[name] for name in source_names]),
            source_labels,
            activity[target_name],
            target_labels,
            "lr",
        )
        low, high = isohr.boot_ci(
            target_labels,
            target_subjects,
            context_probabilities,
            B=BOOTSTRAP_RESAMPLES,
        )
        transfer.append(
            {
                "target": target_name,
                "single_scr_auroc": float(
                    crosscorpus.single_auroc(target_labels, corpora[target_name][0])
                ),
                "base_lr_auroc": base_auroc,
                "context_lr_auroc": context_auroc,
                "context_lr_ci_low": float(low),
                "context_lr_ci_high": float(high),
                "base_gbm_auroc": base_gbm,
                "context_gbm_auroc": context_gbm,
                "activity_only_lr_auroc": activity_only,
                "context_minus_base_lr": context_auroc - base_auroc,
            }
        )

    _save_context_figure(transfer)
    payload: _ContextPayload = {
        "analysis": "wrist-accelerometer context-aware transfer model",
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "transfer_features": [isohr.FEATS[index] for index in crosscorpus.TCOLS],
        "activity_features": list(isohr.ACC_FEATS),
        "notes": [
            (
                "Activity features are computed identically in all three corpora from "
                "the 32 Hz wrist accelerometer magnitude in g."
            ),
            (
                "activity_label_auroc keeps its direction: values above 0.5 mean more "
                "movement in stress windows, below 0.5 mean less. Direction is not "
                "consistent across corpora, so activity carries corpus-specific protocol "
                "information rather than a transferable stress channel."
            ),
            (
                "activity_only_lr_auroc trains on activity alone and is reported to show how "
                "much of any context-model change is protocol identity rather than "
                "stress physiology."
            ),
        ],
        "within_corpus": within,
        "transfer": transfer,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_context_aware.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
