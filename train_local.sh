#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-Lite3-SuperMimic-Rough}"
case "$TASK" in
  Lite3-SuperMimic-*)
    cd "$ROOT/quadruped_lab"
    exec "${PYTHON:-python3}" scripts/train.py "$TASK" "$@"
    ;;
  *)
    cd "$ROOT"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    exec "${PYTHON:-python3}" scripts/train_supermimic.py --task "$TASK" "$@"
    ;;
esac
