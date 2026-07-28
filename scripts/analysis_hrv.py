# pyright: standard, reportImplicitRelativeImport=false
"""64초와 300초 창에서 HRV 특징 포함 여부의 LOSO 민감도를 비교한다."""
from __future__ import annotations

import json
from typing import TypedDict

import numpy as np

from scripts import analysis_crosscorpus as crosscorpus
from scripts import utils_analysis as isohr
from scripts.utils_paths import RESULTS_DIR

HRV_FEATURE_NAMES = {
    "rmssd",
    "sdnn",
    "pnn50",
    "n_nn",
    "ibi_cov",
    "artifact_pct",
}
WINDOW_PROTOCOLS = (
    ("64 s", 64.0, 32.0),
    ("300 s", 300.0, 150.0),
)


class _CorpusHrvResult(TypedDict):
    n_windows: int
    with_hrv_auroc: float
    without_hrv_auroc: float
    difference: float


class _ProtocolHrvResult(TypedDict):
    protocol: str
    window_seconds: float
    step_seconds: float
    corpora: dict[str, _CorpusHrvResult]


class _HrvWindowPayload(TypedDict):
    analysis: str
    model: str
    hrv_features: list[str]
    protocols: list[_ProtocolHrvResult]


def _loso_auroc(
    features: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    columns: list[int],
) -> float:
    evaluated_labels, _, probabilities, _ = isohr.loso(
        features,
        labels,
        subjects,
        columns,
        np.arange(len(labels)),
        mode="none",
        est="gbm",
    )
    return float(isohr.auroc(evaluated_labels, probabilities))


def run_hrv_window_sensitivity() -> _HrvWindowPayload:
    crosscorpus.require_runtime_dependencies()
    all_columns = list(range(len(isohr.FEATS)))
    without_hrv_columns = [
        index
        for index, feature_name in enumerate(isohr.FEATS)
        if feature_name not in HRV_FEATURE_NAMES
    ]
    protocol_results: list[_ProtocolHrvResult] = []
    original_window, original_step = isohr.WIN, isohr.STEP
    try:
        for protocol_name, window_seconds, step_seconds in WINDOW_PROTOCOLS:
            isohr.WIN = window_seconds
            isohr.STEP = step_seconds
            corpora, _ = crosscorpus.build_all()
            corpus_results: dict[str, _CorpusHrvResult] = {}
            for corpus_name, (features, labels, subjects) in corpora.items():
                with_hrv = _loso_auroc(
                    features,
                    labels,
                    subjects,
                    all_columns,
                )
                without_hrv = _loso_auroc(
                    features,
                    labels,
                    subjects,
                    without_hrv_columns,
                )
                corpus_results[corpus_name] = {
                    "n_windows": len(labels),
                    "with_hrv_auroc": with_hrv,
                    "without_hrv_auroc": without_hrv,
                    "difference": with_hrv - without_hrv,
                }
            protocol_results.append(
                {
                    "protocol": protocol_name,
                    "window_seconds": window_seconds,
                    "step_seconds": step_seconds,
                    "corpora": corpus_results,
                }
            )
    finally:
        # [REPRO] 전역 창 설정을 반드시 복구해 같은 프로세스의 다음 분석 오염을 막는다.
        isohr.WIN = original_window
        isohr.STEP = original_step

    # [DOC] 이 결과는 원고의 tab:hrv_sens 표가 유일한 표현이다. 예전에는 같은
    # 열두 값을 F7 그림으로도 그렸지만 리뷰어 1이 중복을 지적해 표만 남겼으므로,
    # 여기서 그림을 다시 만들면 figures/가 원고와 어긋난다.
    payload: _HrvWindowPayload = {
        "analysis": "HRV feature contribution by window length",
        "model": "within-corpus LOSO histogram gradient boosting",
        "hrv_features": sorted(HRV_FEATURE_NAMES),
        "protocols": protocol_results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "revision_hrv_window_sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
