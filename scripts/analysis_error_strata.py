# pyright: standard, reportImplicitRelativeImport=false, reportPrivateUsage=false
"""전이 오류가 어떤 창 조건에 몰려 있는지 층화해 기술한다.

리뷰어 2(R2-1)는 전이 실패의 주 원인을 좁히기 위한 오류 층화를 요청했다. 이
모듈은 leave-one-dataset-out에서 고정된 점수 벡터를 다시 적합하지 않고
움직임, HR 수준, 박동 신뢰도, source 분포까지의 Mahalanobis 거리로 층화한다.

해석 경계:
  - 층별 AUROC 차이는 연관이지 인과가 아니다. 세 관찰 corpus에서 stressor,
    라벨링, 집단, 착용 상태, 맥락을 실험적으로 분리할 수 없다.
  - 층화 AUROC(Mantel--Haenszel)와 pooled AUROC의 차이는 판별력 중 층 사이
    점수 이동에서 오는 몫을 나타낸다.
  - 결과 artifact에는 참여자 식별자를 남기지 않는다. 피험자별 값은 서로
    정렬을 맞추지 않고 각각 정렬한 벡터와 요약 통계로만 보고한다.
"""
from __future__ import annotations

import json
from typing import TypedDict

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer

from scripts import analysis_crosscorpus as crosscorpus
from scripts import utils_analysis as isohr
from scripts.utils_config import SETTINGS
from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR

MIN_STRATUM_WINDOWS = 20
MIN_STRATUM_SUBJECTS = 3
# [CONFIG] 절대 움직임 기준은 프로토콜 설정에서 읽어 한 곳에서만 정의한다.
MOVEMENT_GATE_G = SETTINGS.movement_gate_g
HR_BAND_EDGES = (70.0, 90.0)
REQUIRED_CLASS_COUNT = 2
MIN_VALID_BEATS = 3


class Stratum(TypedDict):
    """한 층의 표본 수, 사용 가능 여부, 두 모델의 층내 AUROC."""

    stratum: str
    status: str
    n_windows: int
    n_stress: int
    n_non_stress: int
    n_subjects: int
    single_auroc: float | None
    logistic_auroc: float | None


class _Stratifier(TypedDict):
    stratifier: str
    definition: str
    pooled_single_auroc: float
    pooled_logistic_auroc: float
    stratified_single_auroc: float
    stratified_logistic_auroc: float
    between_stratum_share_single: float
    between_stratum_share_logistic: float
    strata: list[Stratum]


class _SubjectSummary(TypedDict):
    n_evaluable_subjects: int
    n_single_class_subjects: int
    sorted_logistic_auroc: list[float]
    minimum: float
    first_quartile: float
    median: float
    third_quartile: float
    maximum: float


class _TargetStrata(TypedDict):
    target: str
    n_windows: int
    n_subjects: int
    stratifiers: list[_Stratifier]
    per_subject: _SubjectSummary


class _ErrorStrataPayload(TypedDict):
    analysis: str
    protocol: str
    window_seconds: float
    step_seconds: float
    suppression_rule: str
    notes: list[str]
    targets: list[_TargetStrata]


def _mahalanobis_to_source(
    source_features: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    """Source 전이 특징 분포까지의 Mahalanobis 거리를 target 창마다 계산한다."""
    imputer = SimpleImputer(strategy="median").fit(source_features)
    source = np.asarray(imputer.transform(source_features), dtype=float)
    target = np.asarray(imputer.transform(target_features), dtype=float)
    centre = source.mean(axis=0)
    covariance = np.cov(source, rowvar=False) + 1e-6 * np.eye(source.shape[1])
    precision = np.linalg.pinv(covariance)
    centred = target - centre
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centred, precision, centred), 0.0))


def quantile_labels(
    values: np.ndarray,
    edges: tuple[float, ...],
    names: tuple[str, ...],
) -> np.ndarray:
    """연속값을 고정 경계로 나눈 층 이름 배열로 바꾼다."""
    assignment = np.full(len(values), names[-1], dtype=object)
    finite = np.isfinite(values)
    assignment[~finite] = "unavailable"
    lower = -np.inf
    for edge, name in zip(edges, names[:-1], strict=True):
        assignment[finite & (values > lower) & (values <= edge)] = name
        lower = edge
    assignment[finite & (values > lower)] = names[-1]
    return assignment


def _stratum_rows(
    assignment: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    single_scores: np.ndarray,
    logistic_scores: np.ndarray,
) -> list[Stratum]:
    """층별 표본 수, 상태, 두 모델의 AUROC를 규칙에 따라 기록한다."""
    rows: list[Stratum] = []
    for name in sorted({str(value) for value in assignment}):
        mask = assignment == name
        stratum_labels = labels[mask]
        n_subjects = len(np.unique(subjects[mask]))
        n_windows = int(mask.sum())
        status = "ok"
        if n_windows < MIN_STRATUM_WINDOWS or n_subjects < MIN_STRATUM_SUBJECTS:
            status = "suppressed_small_cell"
        elif len(np.unique(stratum_labels)) < REQUIRED_CLASS_COUNT:
            status = "single_class"
        usable = status == "ok"
        rows.append(
            {
                "stratum": name,
                "status": status,
                "n_windows": n_windows,
                "n_stress": int(stratum_labels.sum()),
                "n_non_stress": int((1 - stratum_labels).sum()),
                "n_subjects": n_subjects,
                "single_auroc": (
                    float(isohr.auroc(stratum_labels, single_scores[mask])) if usable else None
                ),
                "logistic_auroc": (
                    float(isohr.auroc(stratum_labels, logistic_scores[mask])) if usable else None
                ),
            }
        )
    return rows


def stratified_auroc(rows: list[Stratum], key: str) -> float:
    """층별 AUROC를 ``n_stress * n_non_stress`` 가중으로 합친 값을 반환한다."""
    weights, values = [], []
    for row in rows:
        value = row[key]
        if row["status"] != "ok" or value is None:
            continue
        weights.append(float(row["n_stress"] * row["n_non_stress"]))
        values.append(float(value))
    if not weights or float(np.sum(weights)) <= 0.0:
        return float("nan")
    return float(np.average(values, weights=weights))


def _build_stratifier(
    name: str,
    definition: str,
    assignment: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    single_scores: np.ndarray,
    logistic_scores: np.ndarray,
) -> _Stratifier:
    """한 층화 변수에 대한 층별 결과와 pooled 대비 차이를 모은다."""
    rows = _stratum_rows(assignment, labels, subjects, single_scores, logistic_scores)
    pooled_single = float(isohr.auroc(labels, single_scores))
    pooled_logistic = float(isohr.auroc(labels, logistic_scores))
    stratified_single = stratified_auroc(rows, "single_auroc")
    stratified_logistic = stratified_auroc(rows, "logistic_auroc")
    return {
        "stratifier": name,
        "definition": definition,
        "pooled_single_auroc": pooled_single,
        "pooled_logistic_auroc": pooled_logistic,
        "stratified_single_auroc": stratified_single,
        "stratified_logistic_auroc": stratified_logistic,
        "between_stratum_share_single": pooled_single - stratified_single,
        "between_stratum_share_logistic": pooled_logistic - stratified_logistic,
        "strata": rows,
    }


def _per_subject_summary(
    labels: np.ndarray,
    subjects: np.ndarray,
    logistic_scores: np.ndarray,
) -> _SubjectSummary:
    """피험자별 AUROC를 식별자 없이 정렬 벡터와 요약값으로만 보고한다."""
    values: list[float] = []
    single_class = 0
    for subject in np.unique(subjects):
        mask = subjects == subject
        if len(np.unique(labels[mask])) < REQUIRED_CLASS_COUNT:
            single_class += 1
            continue
        values.append(float(isohr.auroc(labels[mask], logistic_scores[mask])))
    ordered = sorted(values)
    array = np.asarray(ordered, float)
    return {
        "n_evaluable_subjects": len(ordered),
        "n_single_class_subjects": single_class,
        "sorted_logistic_auroc": [round(value, 4) for value in ordered],
        "minimum": float(np.min(array)) if ordered else float("nan"),
        "first_quartile": float(np.percentile(array, 25)) if ordered else float("nan"),
        "median": float(np.median(array)) if ordered else float("nan"),
        "third_quartile": float(np.percentile(array, 75)) if ordered else float("nan"),
        "maximum": float(np.max(array)) if ordered else float("nan"),
    }


def _save_error_strata_figure(targets: list[_TargetStrata]) -> None:
    """target별로 층화가 판별력을 얼마나 설명하는지 막대로 비교한다."""
    figure, axes = plt.subplots(1, len(targets), figsize=(12.5, 4.4), sharey=True)
    for axis, target in zip(axes, targets, strict=True):
        names = [item["stratifier"] for item in target["stratifiers"]]
        positions = np.arange(len(names))
        pooled = [item["pooled_logistic_auroc"] for item in target["stratifiers"]]
        stratified = [item["stratified_logistic_auroc"] for item in target["stratifiers"]]
        _ = axis.bar(positions - 0.19, pooled, 0.36, label="pooled", color="#ff7f0e")
        _ = axis.bar(
            positions + 0.19,
            stratified,
            0.36,
            label="within-stratum (MH)",
            color="#4c78a8",
        )
        _ = axis.axhline(0.5, linestyle="--", color="grey", linewidth=1)
        axis.set_xticks(positions)
        axis.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
        axis.set_title(f"target: {target['target']}")
    axes[0].set_ylabel("Leave-one-dataset-out AUROC (logistic)")
    axes[0].set_ylim(0.4, 1.0)
    axes[-1].legend(fontsize=8.5)
    figure.suptitle(
        "Where cross-corpus discrimination survives conditioning on window context",
        fontsize=10.5,
    )
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F10_error_strata.png", dpi=180)
    plt.close(figure)


def run_error_stratification() -> _ErrorStrataPayload:
    """전이 점수를 네 가지 창 조건으로 층화하고 결과를 저장한다."""
    crosscorpus.require_runtime_dependencies()
    corpora, raw_rows = crosscorpus.build_all()
    corpus_names = list(corpora)
    targets: list[_TargetStrata] = []
    for target_name in corpus_names:
        target_features, target_labels, target_subjects = corpora[target_name]
        source_names = [name for name in corpus_names if name != target_name]
        source_features = np.vstack([corpora[name][0] for name in source_names])
        source_labels = np.concatenate([corpora[name][1] for name in source_names])

        single_scores = target_features[:, crosscorpus.SCR].astype(float)
        estimator = isohr._make_est("lr")
        _ = estimator.fit(source_features[:, crosscorpus.TCOLS], source_labels)
        logistic_scores = estimator.predict_proba(
            target_features[:, crosscorpus.TCOLS]
        )[:, 1]

        movement = crosscorpus.acc_arr(raw_rows[target_name])[
            :, isohr.ACC_IDX["acc_mag_std"]
        ]
        heart_rate = target_features[:, isohr.IDX["mean_hr"]]
        artifact = target_features[:, isohr.IDX["artifact_pct"]]
        valid_beats = target_features[:, isohr.IDX["n_nn"]]
        distance = _mahalanobis_to_source(
            source_features[:, crosscorpus.TCOLS],
            target_features[:, crosscorpus.TCOLS],
        )

        movement_tertiles = np.nanquantile(movement, [1 / 3, 2 / 3])
        quality = np.where(
            valid_beats < MIN_VALID_BEATS,
            "q0_too_few_beats",
            np.where(
                artifact <= np.nanmedian(artifact),
                "q1_low_artifact",
                "q2_high_artifact",
            ),
        ).astype(object)
        distance_quartiles = np.nanquantile(distance, [0.25, 0.5, 0.75])

        stratifiers = [
            _build_stratifier(
                "movement (within-corpus tertile)",
                f"ACC magnitude SD in g; tertile cuts {movement_tertiles[0]:.4f}, "
                f"{movement_tertiles[1]:.4f}",
                quantile_labels(
                    movement,
                    (float(movement_tertiles[0]), float(movement_tertiles[1])),
                    ("m1_low", "m2_mid", "m3_high"),
                ),
                target_labels,
                target_subjects,
                single_scores,
                logistic_scores,
            ),
            _build_stratifier(
                "movement (absolute gate)",
                f"ACC magnitude SD at or above {MOVEMENT_GATE_G} g",
                quantile_labels(
                    movement,
                    (MOVEMENT_GATE_G,),
                    ("a1_below_gate", "a2_at_or_above_gate"),
                ),
                target_labels,
                target_subjects,
                single_scores,
                logistic_scores,
            ),
            _build_stratifier(
                "heart-rate band",
                f"mean HR bands at {HR_BAND_EDGES[0]} and {HR_BAND_EDGES[1]} bpm",
                quantile_labels(
                    heart_rate,
                    HR_BAND_EDGES,
                    ("h1_below_70", "h2_70_to_90", "h3_at_or_above_90"),
                ),
                target_labels,
                target_subjects,
                single_scores,
                logistic_scores,
            ),
            _build_stratifier(
                "beat-quality band",
                (
                    "fewer than 3 valid NN intervals, or artifact fraction split at the "
                    "within-target median"
                ),
                quality,
                target_labels,
                target_subjects,
                single_scores,
                logistic_scores,
            ),
            _build_stratifier(
                "distance to source (Mahalanobis quartile)",
                f"quartile cuts {distance_quartiles[0]:.2f}, {distance_quartiles[1]:.2f}, "
                f"{distance_quartiles[2]:.2f}",
                quantile_labels(
                    distance,
                    tuple(float(value) for value in distance_quartiles),
                    ("d1_nearest", "d2", "d3", "d4_farthest"),
                ),
                target_labels,
                target_subjects,
                single_scores,
                logistic_scores,
            ),
        ]
        targets.append(
            {
                "target": target_name,
                "n_windows": len(target_labels),
                "n_subjects": len(np.unique(target_subjects)),
                "stratifiers": stratifiers,
                "per_subject": _per_subject_summary(
                    target_labels,
                    target_subjects,
                    logistic_scores,
                ),
            }
        )

    _save_error_strata_figure(targets)
    payload: _ErrorStrataPayload = {
        "analysis": "leave-one-dataset-out error stratification",
        "protocol": (
            "fixed source-trained scores, stratified without refitting; "
            "within-stratum AUROC combined with n_stress * n_non_stress weights"
        ),
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "suppression_rule": (
            f"strata with fewer than {MIN_STRATUM_WINDOWS} windows or "
            f"{MIN_STRATUM_SUBJECTS} subjects are reported with counts but no AUROC"
        ),
        "notes": [
            (
                "Between-stratum share is pooled minus within-stratum AUROC: the part "
                "of apparent discrimination that comes from score offsets between "
                "strata rather than ranking inside them."
            ),
            (
                "Strata are descriptive conditions, not manipulated factors; a difference "
                "between strata is an association and does not identify a cause."
            ),
            (
                "Per-subject values are sorted independently and carry no identifier, so "
                "rows cannot be matched back to a participant."
            ),
        ],
        "targets": targets,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_error_stratification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
