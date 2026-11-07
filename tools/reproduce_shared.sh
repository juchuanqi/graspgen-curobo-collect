#!/usr/bin/env bash
# Bind the shared Beijing D environments, verify weights and run the two-episode smoke.
set -euo pipefail
REPRO_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPRO_GPUS="${1:-0}"
REPRO_MODE="${2:-smoke}"
REPRO_SHARED_ROOT="${SHARED_ROOT:-/mnt/public/juchuanqi}"
if [[ "$#" -gt 2 || ( "$REPRO_MODE" != smoke && "$REPRO_MODE" != --check-only ) ]]; then
  echo 'Usage: bash tools/reproduce_shared.sh [GPU_SELECTION] [--check-only]' >&2
  exit 2
fi
cd -- "$REPRO_REPO"
python3 -B tools/configure_shared.py --shared-root "$REPRO_SHARED_ROOT"
bash run.sh download-models --verify-only --output "$REPRO_SHARED_ROOT/models/graspgen"
bash run.sh check --gpus "$REPRO_GPUS"
if [[ "$REPRO_MODE" == --check-only ]]; then
  exit 0
fi
bash run.sh prepare --objects block55 --gpu "${REPRO_GPUS%%,*}"
exec bash run.sh collect --smoke --gpus "$REPRO_GPUS"
