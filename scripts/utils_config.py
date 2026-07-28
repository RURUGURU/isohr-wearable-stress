"""연구 프로토콜 설정을 한 곳에서 읽고 형식을 검증한다.

분석 파라미터를 Python 파일마다 반복해서 적으면 일부 실행만 다른 값으로
돌아갈 수 있다. 이 모듈은 ``analysis_config.ini``를 유일한 설정 원본으로
사용하며, 누락되거나 잘못된 값은 분석 시작 전에 명시적으로 실패시킨다.
"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import final


@final
class InvalidAnalysisSettingError(ValueError):
    """분석을 시작할 수 없는 설정 값과 요구 조건을 구조적으로 보존한다."""

    __slots__ = ("requirement", "setting_name", "value")

    def __init__(self, setting_name: str, value: str, requirement: str) -> None:
        """오류 필드와 사용자 표시 메시지를 같은 입력에서 구성한다."""
        self.setting_name = setting_name
        self.value = value
        self.requirement = requirement
        invalid_value = f"설정 {self.setting_name!r} 값 {self.value}"
        message = f"{invalid_value}은(는) {self.requirement} 조건을 만족해야 합니다."
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """논문 결과를 결정하는 고정 프로토콜과 권위 리비전 경로."""

    revision_directory: Path
    manuscript_filename: str
    primary_window_seconds: float
    primary_step_seconds: float
    random_seed: int
    iso_hr_bin_width_bpm: float
    movement_gate_g: float

    def __post_init__(self) -> None:
        """경로와 수치 프로토콜 불변조건을 인스턴스 생성 시 확정한다."""
        # [QC][Invariant:path] 권위 산출물 경로가 저장소 밖을 가리키면 실험 결과를
        # 예상하지 않은 위치에 덮어쓸 수 있으므로 상대 경로와 단일 TeX 파일명만 허용한다.
        if self.revision_directory.is_absolute() or ".." in self.revision_directory.parts:
            raise InvalidAnalysisSettingError(
                setting_name="revision_directory",
                value=str(self.revision_directory),
                requirement="저장소 내부의 상대 경로",
            )
        manuscript_path = Path(self.manuscript_filename)
        if (
            manuscript_path.is_absolute()
            or manuscript_path.name != self.manuscript_filename
            or manuscript_path.suffix != ".tex"
        ):
            raise InvalidAnalysisSettingError(
                setting_name="manuscript_filename",
                value=self.manuscript_filename,
                requirement="디렉터리를 포함하지 않은 .tex 파일명",
            )

        # [QC][Risk:Blocker] 0초 step은 창 반복문을 무한 루프로 만들고 NaN/Inf
        # 설정은 비교·인덱스 계산을 오염시키므로 데이터 로드 전에 함께 거부한다.
        positive_finite_settings = (
            ("primary_window_seconds", self.primary_window_seconds),
            ("primary_step_seconds", self.primary_step_seconds),
            ("iso_hr_bin_width_bpm", self.iso_hr_bin_width_bpm),
            (
                "movement_gate_g",
                self.movement_gate_g,
            ),
        )
        for setting_name, value in positive_finite_settings:
            if not isfinite(value) or value <= 0:
                raise InvalidAnalysisSettingError(
                    setting_name=setting_name,
                    value=repr(value),
                    requirement="0보다 큰 유한 실수",
                )


def _load_settings() -> AnalysisSettings:
    """INI 경계에서 값을 파싱해 내부에서는 타입이 확정된 설정만 사용한다."""
    config_path = Path(__file__).with_name("analysis_config.ini")
    parser = ConfigParser()
    loaded_files = parser.read(config_path, encoding="utf-8")
    if loaded_files != [str(config_path)]:
        message = f"분석 설정 파일을 읽을 수 없습니다: {config_path}"
        raise FileNotFoundError(message)
    if not parser.has_section("analysis"):
        message = f"[analysis] 설정 절이 없습니다: {config_path}"
        raise KeyError(message)

    # [QC] 설정 불변조건: 경로나 수치를 조용히 기본값으로 대체하면 논문 수치가
    # 다른 프로토콜에서 생성될 수 있으므로 모든 필수 값을 명시적으로 요구한다.
    revision_directory = parser.get("analysis", "revision_directory")
    if not revision_directory:
        message = "설정 'revision_directory'는 비어 있지 않은 문자열이어야 합니다."
        raise TypeError(message)

    return AnalysisSettings(
        revision_directory=Path(revision_directory),
        manuscript_filename=parser.get("analysis", "manuscript_filename"),
        primary_window_seconds=parser.getfloat(
            "analysis",
            "primary_window_seconds",
        ),
        primary_step_seconds=parser.getfloat(
            "analysis",
            "primary_step_seconds",
        ),
        random_seed=parser.getint("analysis", "random_seed"),
        iso_hr_bin_width_bpm=parser.getfloat(
            "analysis",
            "iso_hr_bin_width_bpm",
        ),
        movement_gate_g=parser.getfloat(
            "analysis",
            "movement_gate_g",
        ),
    )


SETTINGS = _load_settings()
