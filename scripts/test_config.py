import math
from pathlib import Path

import pytest

from scripts.utils_config import AnalysisSettings


def test_analysis_settings_rejects_zero_step() -> None:
    # Given: 창 반복문을 영원히 전진시키지 못하는 0초 step 설정이 주어진다.
    # When/Then: 분석 시작 전 설정 경계에서 해당 키를 명시한 오류로 중단해야 한다.
    with pytest.raises(ValueError, match="primary_step_seconds"):
        _ = AnalysisSettings(
            revision_directory=Path("revision/example"),
            manuscript_filename="manuscript.tex",
            primary_window_seconds=64.0,
            primary_step_seconds=0.0,
            random_seed=42,
            iso_hr_bin_width_bpm=5.0,
            movement_gate_g=0.0796,
        )


def test_analysis_settings_rejects_nonfinite_window() -> None:
    # Given: 비교와 range 계산을 오염시키는 무한대 창 길이가 주어진다.
    # When/Then: 설정 객체가 만들어지기 전에 non-finite 값을 거부해야 한다.
    with pytest.raises(ValueError, match="primary_window_seconds"):
        _ = AnalysisSettings(
            revision_directory=Path("revision/example"),
            manuscript_filename="manuscript.tex",
            primary_window_seconds=math.inf,
            primary_step_seconds=32.0,
            random_seed=42,
            iso_hr_bin_width_bpm=5.0,
            movement_gate_g=0.0796,
        )
