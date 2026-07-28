#!/usr/bin/env bash
set -euo pipefail

# fast는 원시 데이터 없는 계약을, full은 실제 Nurse 재구성까지 함께 검사한다.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${EASD_ENV_NAME:-easd}"
MODE="${1:-full}"

cd "${PROJECT_ROOT}"
case "${MODE}" in
    fast)
        exec conda run --no-capture-output -n "${ENV_NAME}" \
            pytest -q -m "not integration"
        ;;
    full)
        exec conda run --no-capture-output -n "${ENV_NAME}" pytest -q
        ;;
    *)
        echo "사용법: scripts/run_tests.sh [fast|full]" >&2
        exit 2
        ;;
esac
