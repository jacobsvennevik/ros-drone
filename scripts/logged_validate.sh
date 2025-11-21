#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF' >&2
Usage: scripts/logged_validate.sh <label> [validate_args...]

<label> is any short identifier (e.g., ring50_thr6). Remaining arguments are
passed directly to experiments/validate_hoffman_2016.py.
EOF
  exit 1
fi

LABEL="$1"
shift
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
RUN_DIR="results/sweeps/${TIMESTAMP}_${LABEL}"
mkdir -p "$RUN_DIR"

OUTPUT_PATH=""
DEFAULT_WINDOW_PATH=""
FORWARDED_ARGS=()

while (($#)); do
  case "$1" in
    --output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output expects a path." >&2
        exit 1
      fi
      OUTPUT_PATH="$2"
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    --default-window-output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --default-window-output expects a path." >&2
        exit 1
      fi
      DEFAULT_WINDOW_PATH="$2"
      FORWARDED_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      FORWARDED_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$RUN_DIR/figure.png"
  FORWARDED_ARGS+=("--output" "$OUTPUT_PATH")
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR_ABS="$PROJECT_ROOT/$RUN_DIR"
OUTPUT_ABS="$PROJECT_ROOT/$OUTPUT_PATH"
LOG_PATH="$RUN_DIR_ABS/run.log"

if [[ -n "$DEFAULT_WINDOW_PATH" ]]; then
  if [[ "$DEFAULT_WINDOW_PATH" = /* ]]; then
    DEFAULT_WINDOW_ABS="$DEFAULT_WINDOW_PATH"
  else
    DEFAULT_WINDOW_ABS="$PROJECT_ROOT/$DEFAULT_WINDOW_PATH"
  fi
else
  DEFAULT_WINDOW_ABS=""
fi

mkdir -p "$RUN_DIR_ABS"

cd "$PROJECT_ROOT"
if [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

python experiments/validate_hoffman_2016.py "${FORWARDED_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}

if [[ $STATUS -ne 0 ]]; then
  echo "Experiment failed (exit code $STATUS). See $LOG_PATH" >&2
  exit $STATUS
fi

# Ensure output artifacts live in the sweep folder
if [[ -f "$OUTPUT_ABS" ]] && [[ "$OUTPUT_ABS" != "$RUN_DIR_ABS/figure.png" ]]; then
  cp "$OUTPUT_ABS" "$RUN_DIR_ABS/figure.png"
fi

if [[ -n "$DEFAULT_WINDOW_ABS" ]] && [[ -f "$DEFAULT_WINDOW_ABS" ]]; then
  TARGET="$RUN_DIR_ABS/$(basename "$DEFAULT_WINDOW_ABS")"
  if [[ "$DEFAULT_WINDOW_ABS" != "$TARGET" ]]; then
    cp "$DEFAULT_WINDOW_ABS" "$TARGET"
  fi
fi

echo "Saved sweep artifacts to $RUN_DIR_ABS"


