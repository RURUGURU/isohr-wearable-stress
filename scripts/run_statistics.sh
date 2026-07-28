#!/usr/bin/env bash
set -euo pipefail

# 교차 코퍼스 원장을 기반으로 permutation, Holm, multi-seed, paired bootstrap을 실행한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
    python -m scripts.analysis_statistics
