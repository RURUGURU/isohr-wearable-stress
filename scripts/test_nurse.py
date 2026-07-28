import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_nurse_builder_reports_deidentified_input_exclusions(tmp_path: Path) -> None:
    # Given: EDA.csv가 파일이 아니라 디렉터리인 손상 세션과 최소 설문 파일이 있다.
    program = (
        "import os\n"
        "from pathlib import Path\n"
        "from openpyxl import Workbook\n"
        "root = Path(os.environ['EASD_PROJECT_ROOT'])\n"
        "nurse = root / 'data' / 'raw' / 'nurse'\n"
        "session = nurse / 'N001' / 'session_0'\n"
        "(session / 'EDA.csv').mkdir(parents=True)\n"
        "(session.parent / 'session_0.zip').touch()\n"
        "book = Workbook()\n"
        "sheet = book.active\n"
        "sheet.append(['ID', 'Stress level', 'date', 'Start time', 'End time'])\n"
        "sheet.append(['N001', 2, '1970-01-01', '00:00:00', '00:01:00'])\n"
        "book.save(nurse / 'SurveyResults.xlsx')\n"
        "from scripts import analysis_crosscorpus\n"
        "analysis_crosscorpus.build_nurse()\n"
    )
    environment = os.environ.copy()
    environment["EASD_PROJECT_ROOT"] = str(tmp_path)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    # When: canonical Nurse builder가 손상된 duration 입력을 처리한다.
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: 참여자 ID나 경로 없이 제외 단계와 건수를 stderr에 남겨야 한다.
    assert result.returncode == 0, result.stderr
    assert "Nurse input exclusions: eda_duration=1" in result.stderr
    assert "N001" not in result.stderr


@pytest.mark.integration
def test_nurse_builder_reconstructs_canonical_cohort() -> None:
    # Given: 보존된 Nurse 설문 시각과 Empatica 세션이 준비되어 있다.
    environment = os.environ.copy()
    environment["TZ"] = "Asia/Seoul"
    program = (
        "import numpy as np\n"
        "import sys\n"
        "from scripts import analysis_crosscorpus as C\n"
        "rows = C.build_nurse()\n"
        "labels = np.asarray([row[1] for row in rows])\n"
        "subjects = np.unique([row[0] for row in rows])\n"
        'print(f"{len(rows)}|{int(labels.sum())}|'
        '{int((1-labels).sum())}|{len(subjects)}")\n'
    )

    # When: UTC가 아닌 호스트 시간대에서 실제 코호트를 재구성한다.
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: 호스트 시간대와 무관하게 논문 코호트 불변식이 정확히 유지되어야 한다.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "9226|7347|1879|15"
