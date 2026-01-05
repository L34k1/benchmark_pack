#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/scripts/desktop/requirements-desktop.txt"

python3 -m venv "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

echo "Virtualenv created at ${VENV_DIR}"
