# pyright: standard, reportImplicitRelativeImport=false, reportPrivateUsage=false
"""리뷰어 요청 Nurse utility·민감도·특징 이동 분석의 단일 CLI 진입점."""
from __future__ import annotations

import json
from dataclasses import asdict
from enum import StrEnum
from typing import Annotated, assert_never

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer
from sklearn.metrics import precision_recall_curve

from scripts import analysis_adaptation as adaptation
from scripts import analysis_context as context
from scripts import analysis_crosscorpus as crosscorpus
from scripts import analysis_diagnostics as diagnostics
from scripts import analysis_error_strata as error_strata
from scripts import analysis_hrv as hrv_sensitivity
from scripts import utils_analysis as isohr
from scripts.metrics import PredictionSet, measure_utility
from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR


class SensitivityProtocol(StrEnum):
    NONOVERLAP = "nonoverlap"
    WINDOW300 = "window300"


app = typer.Typer(add_completion=False, no_args_is_help=True)


def save_nurse_utility_figure(
    predictions: tuple[PredictionSet, ...],
    prevalence: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for prediction in predictions:
        precision, recall, _ = precision_recall_curve(
            prediction.labels,
            prediction.scores,
        )
        axes[0].plot(recall, precision, label=prediction.name)
        if prediction.is_probability:
            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_ids = np.minimum(np.digitize(prediction.scores, bin_edges[1:-1]), 9)
            observed = []
            predicted = []
            for bin_id in range(10):
                mask = bin_ids == bin_id
                if np.any(mask):
                    observed.append(float(prediction.labels[mask].mean()))
                    predicted.append(float(prediction.scores[mask].mean()))
            axes[1].plot(predicted, observed, marker="o", label=prediction.name)

    axes[0].axhline(prevalence, color="grey", linestyle="--", label="prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Nurse precision-recall")
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set(
        xlabel="Mean predicted probability",
        ylabel="Observed stress fraction",
        title="Nurse calibration",
    )
    axes[1].legend()
    figure.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / "F4_nurse_utility.png", dpi=180)
    plt.close(figure)


@app.command("nurse-utility")
def nurse_utility() -> None:
    crosscorpus.require_runtime_dependencies()
    corpora, _ = crosscorpus.build_all()
    nurse_features, nurse_labels, _ = corpora["Nurse"]
    source_corpora = [corpora[name] for name in corpora if name != "Nurse"]
    source_features = np.vstack([corpus[0] for corpus in source_corpora])
    source_labels = np.concatenate([corpus[1] for corpus in source_corpora])

    scr_scores = nurse_features[:, crosscorpus.SCR]
    scr_mask = np.isfinite(scr_scores)
    predictions = [
        PredictionSet(
            name="Single SCR",
            labels=nurse_labels[scr_mask],
            scores=scr_scores[scr_mask],
            is_probability=False,
        )
    ]
    for name, estimator_name in (("LogReg", "lr"), ("GBM", "gbm")):
        estimator = isohr._make_est(estimator_name)
        estimator.fit(source_features[:, crosscorpus.TCOLS], source_labels)
        scores = estimator.predict_proba(nurse_features[:, crosscorpus.TCOLS])[:, 1]
        predictions.append(
            PredictionSet(
                name=name,
                labels=nurse_labels,
                scores=scores,
                is_probability=True,
            )
        )

    metrics = tuple(measure_utility(prediction) for prediction in predictions)
    payload = {
        "analysis": "Nurse leave-one-dataset-out operational utility",
        "window_seconds": isohr.WIN,
        "step_seconds": isohr.STEP,
        "threshold_policy": "fixed probability threshold 0.5; no Nurse-label tuning",
        "stress_prevalence": float(nurse_labels.mean()),
        "n_windows": len(nurse_labels),
        "n_subjects": len(np.unique(corpora["Nurse"][2])),
        "models": [asdict(metric) for metric in metrics],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "revision_nurse_utility.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_nurse_utility_figure(tuple(predictions), float(nurse_labels.mean()))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("sensitivity")
def sensitivity(
    protocol: Annotated[SensitivityProtocol, typer.Argument()],
) -> None:
    # [REPRO] 한 번의 CLI 실행에서 바뀌는 변수는 창 길이와 step뿐이다.
    match protocol:
        case SensitivityProtocol.NONOVERLAP:
            isohr.WIN = 64.0
            isohr.STEP = 64.0
        case SensitivityProtocol.WINDOW300:
            isohr.WIN = 300.0
            isohr.STEP = 150.0
        case unreachable:
            assert_never(unreachable)
    typer.echo(
        f"window_seconds={isohr.WIN:.0f} step_seconds={isohr.STEP:.0f}",
    )
    crosscorpus.run_analysis()


@app.command("feature-shift")
def feature_shift() -> None:
    payload = diagnostics.run_feature_shift()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("negative-control")
def negative_control() -> None:
    payload = diagnostics.run_negative_control()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("stress-predict-alignment")
def stress_predict_alignment() -> None:
    payload = diagnostics.run_stress_predict_alignment()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("hrv-window-sensitivity")
def hrv_window_sensitivity() -> None:
    payload = hrv_sensitivity.run_hrv_window_sensitivity()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("adaptation")
def adaptation_sweep() -> None:
    payload = adaptation.run_adaptation()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("target-supervised")
def target_supervised() -> None:
    payload = adaptation.run_target_supervised()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("context-aware")
def context_aware() -> None:
    payload = context.run_context_aware()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("error-stratification")
def error_stratification() -> None:
    payload = error_strata.run_error_stratification()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
