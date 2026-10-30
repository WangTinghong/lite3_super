#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"
TASK="${TASK:-Lite3-Chase-Flat}"
exec "${PYTHON:-python3}" scripts/train_supermimic.py --task "$TASK" "$@"
