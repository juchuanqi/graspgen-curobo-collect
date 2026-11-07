#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU_SELECTION="${1:-0,1,2,3}"
EPISODES_PER_OBJECT="${2:-30}"
WORKER_COUNT=1
if [[ "$GPU_SELECTION" == *,*,*,* ]]; then WORKER_COUNT=3; fi
bash "$REPO_DIR/run.sh" collect --smoke --gpus "$GPU_SELECTION"
exec bash "$REPO_DIR/run.sh" collect --objects all --episodes-per-object "$EPISODES_PER_OBJECT" --workers "$WORKER_COUNT" --gpus "$GPU_SELECTION"
