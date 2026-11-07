#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${LAUNCHER_PYTHON:-python3}" -B "$REPO_DIR/cli.py" "$@"
