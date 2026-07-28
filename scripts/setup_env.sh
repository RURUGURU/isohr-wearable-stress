#!/usr/bin/env bash
set -euo pipefail

# 재현성 계약: env.yaml의 정확한 버전으로 이름 있는 Conda 환경을 생성하거나 갱신한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

if conda run --no-capture-output -n "${ENV_NAME}" python -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))" >/dev/null 2>&1; then
    conda env update --name "${ENV_NAME}" --file "${PROJECT_ROOT}/env.yaml" --prune
else
    conda env create --name "${ENV_NAME}" --file "${PROJECT_ROOT}/env.yaml"
fi

conda run --no-capture-output -n "${ENV_NAME}" python -c \
    "import cvxopt, neurokit2, numpy, pandas, scipy, sklearn; print('Python environment: PASS')"
