#!/usr/bin/env bash
set -euo pipefail

# 권위 결과 로그를 파싱해 F1--F3을 새 리비전 폴더에만 생성한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
    python -m scripts.plot_figures
