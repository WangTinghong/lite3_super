#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/rsl_rl:${PYTHONPATH:-}"
cd "$ROOT"

TASK="${TASK:-lite3_stairs}"
HEADLESS="${HEADLESS:---headless}"

echo "Project: $ROOT"
python - <<'PY_CHECK'
import legged_gym
import rsl_rl
print("legged_gym:", legged_gym.__file__)
print("rsl_rl:", rsl_rl.__file__)
PY_CHECK

exec python legged_gym/scripts/train.py --task "$TASK" $HEADLESS "$@"
