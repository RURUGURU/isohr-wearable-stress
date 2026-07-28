"""Nurse 설문 벽시각을 자료 수집지의 고정 UTC-5 epoch로 변환한다."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

NURSE_SURVEY_TIMEZONE = timezone(-timedelta(hours=5))


def nurse_survey_epoch(day: date, clock_time: time) -> float:
    """호스트 timezone과 무관한 Nurse 설문 epoch를 반환한다."""
    return datetime.combine(
        day,
        clock_time,
        tzinfo=NURSE_SURVEY_TIMEZONE,
    ).timestamp()
