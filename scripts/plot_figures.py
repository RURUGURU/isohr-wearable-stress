# pyright: standard
"""권위 결과 로그에서 논문 F1--F3을 다시 생성한다.

그림 숫자를 코드에 복사하지 않고 ``results/run_crosscorpus*.txt``를
파싱한다. 따라서 수치가 바뀌면 로그와 그림이 서로 다른 상태로 남지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")
import matplotlib.pyplot as plt

from scripts.utils_paths import FIGURES_DIR, RESULTS_DIR

CORPORA = ("WESAD", "StrPred", "Nurse")
CORPUS_LABELS = ("WESAD\n(TSST)", "Stress-Predict\n(cognitive)", "Nurse\n(real-world)")
MODEL_LABELS = ("single SCR", "LogReg", "GBM", "CORAL (DG)")
COLORS = {
    "single SCR": "#1f77b4",
    "LogReg": "#ff7f0e",
    "GBM": "#2ca02c",
    "CORAL (DG)": "#d62728",
}


@dataclass(frozen=True, slots=True)
class IsoHrSeries:
    """iso-HR 분해 그림에 함께 전달되어야 하는 여섯 개의 대응 계열."""

    unmatched: list[float]
    matched: list[float]
    matched_sd: list[float]
    noncardiac: list[float]
    noncardiac_sd: list[float]
    leakage: list[float]


def _read_required(path: Path) -> str:
    """결과 원장이 없으면 빈 그림을 만들지 않고 원인을 명시해 중단한다."""
    if not path.is_file():
        message = f"권위 결과 로그가 없습니다: {path}"
        raise FileNotFoundError(message)
    return path.read_text(encoding="utf-8")


def _require_rows(
    pattern: str,
    text: str,
    expected: int,
    description: str,
) -> list[tuple[str, ...]]:
    """로그 형식 변화가 조용히 잘못된 배열로 이어지지 않도록 행 수를 고정한다."""
    rows = re.findall(pattern, text, flags=re.MULTILINE)
    if len(rows) != expected:
        message = (
            f"{description} 파싱 행 수가 다릅니다: "
            f"expected={expected}, actual={len(rows)}"
        )
        raise ValueError(message)
    return rows


def _parse_lodo(text: str) -> dict[str, list[float]]:
    rows = _require_rows(
        r"^\s+(WESAD|StrPred|Nurse)\s+"
        r"(0\.\d{3})\s+(0\.\d{3})\s+(0\.\d{3})\s+(0\.\d{3})$",
        text,
        3,
        "LODO",
    )
    by_corpus = {row[0]: [float(value) for value in row[1:]] for row in rows}
    return {
        model: [by_corpus[corpus][model_index] for corpus in CORPORA]
        for model_index, model in enumerate(MODEL_LABELS)
    }


def _parse_confidence_intervals(
    text: str,
) -> dict[str, list[tuple[float, float]]]:
    rows = _require_rows(
        r"^\s+test=(WESAD|StrPred|Nurse)\s+: single=0\.\d{3} "
        r"\[(0\.\d{3})-(0\.\d{3})\]\s+LR=0\.\d{3} "
        r"\[(0\.\d{3})-(0\.\d{3})\]$",
        text,
        3,
        "subject-bootstrap CI",
    )
    by_corpus = {
        row[0]: ((float(row[1]), float(row[2])), (float(row[3]), float(row[4])))
        for row in rows
    }
    return {
        "single SCR": [by_corpus[corpus][0] for corpus in CORPORA],
        "LogReg": [by_corpus[corpus][1] for corpus in CORPORA],
    }


def _parse_movement(text: str) -> tuple[list[float], list[float]]:
    rows = _require_rows(
        r"^\s+(low|mid|high)-move\s*:.*single=(0\.\d{3})\s+LR=(0\.\d{3})$",
        text,
        3,
        "Nurse movement tertile",
    )
    return ([float(row[1]) for row in rows], [float(row[2]) for row in rows])


def _parse_isohr(
    main_text: str,
    stats_text: str,
) -> IsoHrSeries:
    unmatched_rows = _require_rows(
        r"^\s+(WESAD|StrPred|Nurse)\s+(0\.\d{3})\s+0\.\d{3}\s+"
        r"0\.\d{3}\s+0\.\d{3}\s+[+-]0\.\d{3}\s+\d+$",
        main_text,
        3,
        "unmatched iso-HR",
    )
    matched_rows = _require_rows(
        r"^\s+(WESAD|StrPred|Nurse)\s+"
        r"(0\.\d{3})\+/-([0-9.]+)\s+"
        r"(0\.\d{3})\+/-([0-9.]+)\s+"
        r"([+-]0\.\d{3})\+/-[0-9.]+\s+\d+$",
        stats_text,
        3,
        "seven-seed iso-HR",
    )
    unmatched = {row[0]: float(row[1]) for row in unmatched_rows}
    matched = {
        row[0]: (float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
        for row in matched_rows
    }
    return IsoHrSeries(
        unmatched=[unmatched[corpus] for corpus in CORPORA],
        matched=[matched[corpus][0] for corpus in CORPORA],
        matched_sd=[matched[corpus][1] for corpus in CORPORA],
        noncardiac=[matched[corpus][2] for corpus in CORPORA],
        noncardiac_sd=[matched[corpus][3] for corpus in CORPORA],
        leakage=[matched[corpus][4] for corpus in CORPORA],
    )


def _plot_crosscorpus(
    values: dict[str, list[float]],
    intervals: dict[str, list[tuple[float, float]]],
) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    x_positions = np.arange(3)
    width = 0.20
    for model_index, model in enumerate(MODEL_LABELS):
        model_values = values[model]
        error = None
        if model in intervals:
            paired_values = zip(model_values, intervals[model], strict=True)
            lower = [value - bounds[0] for value, bounds in paired_values]
            paired_values = zip(model_values, intervals[model], strict=True)
            upper = [bounds[1] - value for value, bounds in paired_values]
            error = np.array([lower, upper])
        axis.bar(
            x_positions + (model_index - 1.5) * width,
            model_values,
            width,
            label=model,
            color=COLORS[model],
            yerr=error,
            capsize=3,
            error_kw={"lw": 1.1},
        )
    axis.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("AUROC")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(CORPUS_LABELS)
    axis.set_title(
        "Leave-one-dataset-out transfer "
        "(95% subject-bootstrap CI on single/LR)\n"
        "GBM and CORAL do not beat the single-feature / linear baseline"
    )
    axis.legend(ncol=4, loc="upper center", fontsize=9, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "F1_crosscorpus.png")
    plt.close(figure)


def _plot_movement(single: list[float], logistic: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 4.6))
    x_positions = np.arange(3)
    width = 0.36
    axis.bar(x_positions - width / 2, single, width, label="single SCR", color="#1f77b4")
    axis.bar(x_positions + width / 2, logistic, width, label="LogReg", color="#ff7f0e")
    axis.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    axis.set_ylim(0.35, 0.72)
    axis.set_ylabel("AUROC")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(["low\nmovement", "mid\nmovement", "high\nmovement"])
    axis.set_title(
        "Real-world (Nurse) discriminability by movement tertile\n"
        "near chance at every activity level (lab-to-field gap)"
    )
    axis.legend(fontsize=9, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "F2_nurse_movement.png")
    plt.close(figure)


def _plot_isohr(series: IsoHrSeries) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    x_positions = np.arange(3)
    width = 0.26
    axis.bar(
        x_positions - width,
        series.unmatched,
        width,
        label="unmatched (all features)",
        color="#aec7e8",
    )
    axis.bar(
        x_positions,
        series.matched,
        width,
        yerr=series.matched_sd,
        capsize=3,
        error_kw={"lw": 1.1},
        label="iso-HR matched (all features)",
        color="#4c78a8",
    )
    axis.bar(
        x_positions + width,
        series.noncardiac,
        width,
        yerr=series.noncardiac_sd,
        capsize=3,
        error_kw={"lw": 1.1},
        label="iso-HR matched (non-cardiac: EDA+TEMP)",
        color="#2f4b7c",
    )
    values = zip(
        x_positions,
        series.leakage,
        series.unmatched,
        strict=True,
    )
    for x_position, leakage_value, top in values:
        axis.annotate(
            f"HR-leak {leakage_value:+.3f}",
            (x_position, top + 0.014),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8.5,
            color="#333",
        )
    axis.axhline(0.5, linestyle="--", color="grey", linewidth=1)
    axis.set_ylim(0.4, 1.04)
    axis.set_ylabel("AUROC")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(CORPUS_LABELS)
    axis.set_title(
        "iso-HR decomposition (mean ± SD over 7 matching seeds)\n"
        "HR matching barely changes AUROC; matched non-cardiac signal is strong only for TSST"
    )
    axis.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "F3_isohr_decomp.png")
    plt.close(figure)


def main() -> None:
    """로그를 한 번 파싱하고 권위 리비전의 세 그림을 원자적으로 갱신한다."""
    main_text = _read_required(RESULTS_DIR / "run_crosscorpus.txt")
    stats_text = _read_required(RESULTS_DIR / "run_crosscorpus_stats.txt")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "figure.dpi": 140,
            "axes.titlesize": 10.5,
        }
    )
    _plot_crosscorpus(_parse_lodo(main_text), _parse_confidence_intervals(main_text))
    _plot_movement(*_parse_movement(main_text))
    _plot_isohr(_parse_isohr(main_text, stats_text))
    print(f"F1--F3 생성 완료: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
