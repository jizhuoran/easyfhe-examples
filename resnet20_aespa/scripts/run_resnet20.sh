#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

TOTAL="${EASYFHE_TOTAL:-1}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  TOTAL="${1}"
  shift
fi

python -m resnet20_aespa.main --total "${TOTAL}" "$@"
