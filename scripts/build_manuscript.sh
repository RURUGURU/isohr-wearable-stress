#!/usr/bin/env bash
set -euo pipefail

# 중앙 설정의 권위 TeX를 빌드하고 로컬 절대경로가 담긴 보조 파일은 정리한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"

cd "${PROJECT_ROOT}"
MANUSCRIPT_PATH="$(
    conda run --no-capture-output -n "${ENV_NAME}" python -c \
        'from scripts.utils_paths import MANUSCRIPT_TEX; print(MANUSCRIPT_TEX)'
)"
MANUSCRIPT_DIR="$(dirname -- "${MANUSCRIPT_PATH}")"
MANUSCRIPT_FILENAME="$(basename -- "${MANUSCRIPT_PATH}")"

cd "${MANUSCRIPT_DIR}"
latexmk -pdf -interaction=nonstopmode -halt-on-error "${MANUSCRIPT_FILENAME}"
latexmk -c -silent "${MANUSCRIPT_FILENAME}"
