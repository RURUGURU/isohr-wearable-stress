#!/usr/bin/env bash
set -euo pipefail

# 전체 WESAD/Stress-Predict/Nurse 분석은 데이터 검증을 통과한 뒤에만 실행한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
RESULTS_DIR="$(
    conda run --no-capture-output -n "${ENV_NAME}" \
        python -c 'from scripts.utils_paths import RESULTS_DIR; print(RESULTS_DIR)'
)"
mkdir -p "${RESULTS_DIR}"
TEMP_LOG="$(mktemp "${RESULTS_DIR}/.run_crosscorpus.XXXXXX")"

cleanup() {
    if [[ -f "${TEMP_LOG}" ]]; then
        rm -f -- "${TEMP_LOG}"
    fi
}
trap cleanup EXIT

conda run --no-capture-output -n "${ENV_NAME}" \
    python -m scripts.analysis_crosscorpus | tee "${TEMP_LOG}"
mv -- "${TEMP_LOG}" "${RESULTS_DIR}/run_crosscorpus.txt"
trap - EXIT
