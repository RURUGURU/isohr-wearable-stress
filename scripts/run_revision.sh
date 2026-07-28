#!/usr/bin/env bash
set -euo pipefail

# 리뷰어 요청 분석의 유일한 CLI 경계이며 모든 인자는 Typer 명령으로 그대로 전달한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
exec conda run --no-capture-output -n "${ENV_NAME}" \
    python -m scripts.main_revision "$@"
