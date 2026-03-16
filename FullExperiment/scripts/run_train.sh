#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -f ".env" ]]; then
  echo "Missing .env file. Copy .env.example to .env and set keys."
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
uv run python -m full_experiment.train --config configs/train.yaml

