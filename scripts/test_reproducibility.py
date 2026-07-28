import json
import os
import shlex
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import cast

from scripts.utils_config import SETTINGS

REQUIRED_MODULES = ("sklearn", "neurokit2", "cvxopt")
EXPECTED_FIGURES = {
    "F1_crosscorpus.png",
    "F2_nurse_movement.png",
    "F3_isohr_decomp.png",
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_scientific_modules_are_importable() -> None:
    # Given: 검증 대상 연구 환경의 Python 인터프리터가 선택되어 있다.
    # When: 원시 데이터를 읽지 않고 필수 과학 패키지를 탐색한다.
    missing_modules = [
        module_name for module_name in REQUIRED_MODULES if find_spec(module_name) is None
    ]

    # Then: 실험 도중이 아니라 환경 검사 단계에서 누락 패키지가 없어야 한다.
    assert missing_modules == []


def test_eda_backend_reports_observed_fallback() -> None:
    # Given: NeuroKit2 cvxEDA가 값 오류로 실패하지만 SciPy fallback은 사용 가능한 환경이다.
    program = (
        "import json\n"
        "import numpy as np\n"
        "from scripts import utils_analysis as analysis\n"
        "def fail_cvxeda(*args, **kwargs):\n"
        "    raise ValueError('forced cvxEDA failure')\n"
        "analysis.nk.eda_phasic = fail_cvxeda\n"
        "analysis.cvx_decompose(np.linspace(0.0, 1.0, 64), 4.0)\n"
        "print(json.dumps(analysis.analysis_audit_snapshot(), sort_keys=True))\n"
    )

    # When: 실제 분해 함수를 별도 프로세스에서 한 번 실행한다.
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: 실제 선택 backend와 실패 backend가 같은 감사 경계에서 함께 보고되어야 한다.
    assert result.returncode == 0, result.stderr
    audit = cast("dict[str, object]", json.loads(result.stdout))
    assert audit["eda_backend_counts"] == {"butter-highpass": 1}
    assert audit["eda_backend_failure_counts"] == {"cvxEDA(neurokit2)": 1}


def test_crosscorpus_runner_atomically_persists_audit_ledger(tmp_path: Path) -> None:
    # Given: Conda 경계가 식별 정보 없는 machine-readable 감사 행을 출력한다.
    audit_line = "ANALYSIS_AUDIT_JSON=" + json.dumps(
        {
            "corpus_input_exclusion_counts": {},
            "eda_backend_counts": {"cvxEDA(neurokit2)": 2},
            "eda_backend_failure_counts": {},
            "input_parse_exclusion_counts": {},
            "schema_version": 1,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_conda = fake_bin / "conda"
    _ = fake_conda.write_text(
        "\n".join(
            (
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'while [[ "${1:-}" != "python" ]]; do shift; done',
                "shift",
                'if [[ "${1:-}" == "-c" ]]; then',
                '  exec python -c "${2}"',
                "fi",
                "printf '%s\\n' 'synthetic numerical result'",
                f"printf '%s\\n' {shlex.quote(audit_line)}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _ = fake_conda.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    environment["EASD_PROJECT_ROOT"] = str(tmp_path)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    bash_path = shutil.which("bash")
    assert bash_path is not None

    # When: 사용자가 실제 shell runner를 실행한다.
    result = subprocess.run(
        [bash_path, str(PROJECT_ROOT / "scripts" / "run_crosscorpus.sh")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: 결과와 감사 행이 권위 로그 하나에 원자적으로 남고 임시 파일은 사라져야 한다.
    result_log = (
        tmp_path
        / SETTINGS.revision_directory
        / "results"
        / "run_crosscorpus.txt"
    )
    assert result.returncode == 0, result.stderr
    assert result_log.read_text(encoding="utf-8") == result.stdout
    assert "ANALYSIS_AUDIT_JSON=" in result.stdout
    assert list(result_log.parent.glob(".run_crosscorpus.*")) == []


def test_plot_figures_honors_project_root_override(tmp_path: Path) -> None:
    # Given: 현재 작업 디렉터리와 무관한 임시 프로젝트 루트와 권위 로그가 있다.
    figure_dir = (
        tmp_path
        / SETTINGS.revision_directory
        / "figures"
    )
    figure_dir.mkdir(parents=True)
    results_dir = figure_dir.parent / "results"
    results_dir.mkdir()
    canonical_results = (
        PROJECT_ROOT
        / SETTINGS.revision_directory
        / "results"
    )
    for filename in ("run_crosscorpus.txt", "run_crosscorpus_stats.txt"):
        _ = shutil.copy2(canonical_results / filename, results_dir / filename)
    environment = os.environ.copy()
    environment["EASD_PROJECT_ROOT"] = str(tmp_path)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    # When: 실제 그림 엔트리포인트를 저장소 밖에서 실행한다.
    result = subprocess.run(
        [sys.executable, "-m", "scripts.plot_figures"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    # Then: 문서화된 F1--F3만 임시 권위 리비전 폴더에 생성되어야 한다.
    assert result.returncode == 0, result.stderr
    assert {path.name for path in figure_dir.iterdir()} == EXPECTED_FIGURES


def test_revision_analysis_cli_is_available() -> None:
    # Given: 리뷰어 요청 분석을 재현하는 환경이 준비되어 있다.
    command = [
        sys.executable,
        "-m",
        "scripts.main_revision",
        "--help",
    ]

    # When: 연구자가 실제 리비전 분석 CLI 도움말을 조회한다.
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    # Then: 원시 데이터를 읽지 않고 명령 표면을 확인할 수 있어야 한다.
    assert result.returncode == 0, result.stderr
