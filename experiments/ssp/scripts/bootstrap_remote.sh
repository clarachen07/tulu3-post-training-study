#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${SSP_PYTHON:-python3}" -m venv "${root}/.venv"
"${root}/.venv/bin/python" -m pip install -e "${root}"
echo "Environment prepared. Configure configs/study.json before starting a new run."
