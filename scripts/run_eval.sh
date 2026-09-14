#!/usr/bin/env bash
# Trigger a full evaluation batch: validate submissions, run the round-robin
# tournament, then rebuild the leaderboard.
#
# Usage:
#   bash scripts/run_eval.sh [extra args passed through to eval/tournament.py]
#
# Examples:
#   bash scripts/run_eval.sh --phase 1
#   bash scripts/run_eval.sh --phase 2 --episodes 20 --timeout 180
set -euo pipefail

# Resolve repo root as the parent of this script's directory, so this works
# regardless of the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

SUBMISSIONS_DIR="$ROOT/submissions"
RESULTS_DIR="$ROOT/results"

mkdir -p "$RESULTS_DIR"

if [ -d "$SUBMISSIONS_DIR" ]; then
    echo "==> Validating submissions in $SUBMISSIONS_DIR"
    for dir in "$SUBMISSIONS_DIR"/*/; do
        [ -d "$dir" ] || continue
        if [ -f "${dir}agent.py" ]; then
            name="$(basename "$dir")"
            echo "  - validating $name"
            if ! "$PYTHON_BIN" "$ROOT/scripts/validate_submission.py" "$dir"; then
                echo "  WARNING: submission '$name' failed validation (see above); it may still be included by eval/tournament.py" >&2
            fi
        fi
    done
else
    echo "==> No submissions/ directory found at $SUBMISSIONS_DIR, skipping validation."
fi

echo "==> Running tournament"
if [ -f "$ROOT/eval/tournament.py" ]; then
    "$PYTHON_BIN" "$ROOT/eval/tournament.py" \
        --submissions-dir "$SUBMISSIONS_DIR" \
        --results-dir "$RESULTS_DIR" \
        "$@"
else
    # Fall back to module invocation in case eval/tournament.py is only
    # runnable as a package module (e.g. it uses relative imports).
    "$PYTHON_BIN" -m eval.tournament \
        --submissions-dir "$SUBMISSIONS_DIR" \
        --results-dir "$RESULTS_DIR" \
        "$@"
fi

echo "==> Building leaderboard"
"$PYTHON_BIN" "$ROOT/eval/leaderboard.py" --results-dir "$RESULTS_DIR"

echo "==> Leaderboard written to $RESULTS_DIR/leaderboard.csv"
