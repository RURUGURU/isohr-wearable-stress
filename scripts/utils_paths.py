"""저장소·원자료·권위 리비전 경로를 환경 독립적으로 해석한다."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.utils_config import SETTINGS

_OVERRIDE_ROOT = os.environ.get("EASD_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_OVERRIDE_ROOT).expanduser().resolve()
    if _OVERRIDE_ROOT
    else Path(__file__).resolve().parents[1]
)
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
REPO_ROOT = PROJECT_ROOT
DATA_RAW_DIR = DATA_ROOT
MANUSCRIPT_DIR = PROJECT_ROOT / SETTINGS.revision_directory
MANUSCRIPT_TEX = MANUSCRIPT_DIR / SETTINGS.manuscript_filename
RESULTS_DIR = MANUSCRIPT_DIR / "results"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"

WESAD_DIR = DATA_RAW_DIR / "wesad" / "WESAD"
NURSE_DIR = DATA_RAW_DIR / "nurse"
STRESS_PREDICT_RAW_DIR = DATA_RAW_DIR / "stress_predict" / "Raw_data"
STRESS_PREDICT_PROCESSED_DIR = DATA_RAW_DIR / "stress_predict" / "Processed_data"


def require_path(path: Path, description: str) -> Path:
    """필수 입력 경로가 없으면 무엇을 어디에 두어야 하는지까지 알려주고 중단한다.

    이 저장소는 세 corpus를 재배포하지 않으므로, 처음 받은 사람이 가장 먼저
    마주치는 오류가 바로 이것이다. 경로만 찍으면 무엇을 내려받아야 하는지
    알 수 없으므로 안내 문서를 함께 가리킨다.
    """
    if path.exists():
        return path
    raise FileNotFoundError(
        "\n".join(
            (
                f"Missing {description}: {path}",
                "",
                (
                    "The three corpora are licensed third-party data and are"
                    " not redistributed with this repository."
                ),
                (
                    "See data/README.md for where to obtain each one, the"
                    " expected directory layout, and the licence terms."
                ),
                "",
                f"Resolved project root: {PROJECT_ROOT}",
                "Set EASD_PROJECT_ROOT if the data lives under a different root.",
            )
        )
    )
