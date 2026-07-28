#!/usr/bin/env bash
set -euo pipefail

# 새로 유지보수하는 표면만 strict type/lint로 검사하고 legacy 수치 코드는 회귀 테스트로 고정한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
conda run --no-capture-output -n "${ENV_NAME}" basedpyright
conda run --no-capture-output -n "${ENV_NAME}" ruff check \
    scripts/utils_config.py scripts/utils_paths.py scripts/utils_time.py \
    scripts/metrics.py scripts/analysis_diagnostics.py scripts/analysis_hrv.py \
    scripts/analysis_adaptation.py scripts/analysis_context.py scripts/analysis_error_strata.py \
    scripts/main_revision.py scripts/plot_figures.py scripts/test_*.py
