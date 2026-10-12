#!/usr/bin/env bash

set -euo pipefail

: "${CKPT_PATH:?Set CKPT_PATH to an RLinf global_step directory or converted model directory}"

GPU_ID=${GPU_ID:-0}
TASK=${TASK:-insert_hole}
USE_TACTILE=${USE_TACTILE:-auto}
PORT=${PORT:-8000}
POLICY_PYTHON=${POLICY_PYTHON:-$(command -v python)}

if [[ "$USE_TACTILE" != "auto" && "$USE_TACTILE" != "true" && "$USE_TACTILE" != "false" ]]; then
  echo "USE_TACTILE must be auto, true, or false, got: $USE_TACTILE" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
NORM_STATS=${NORM_STATS:-auto}

if [[ ! -x "$POLICY_PYTHON" ]]; then
  echo "SimpleTac policy Python is not executable: $POLICY_PYTHON" >&2
  exit 2
fi

export PYTHONPATH="$REPO_DIR:$REPO_DIR/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --checkpoint "$CKPT_PATH"
  --task "$TASK"
  --port "$PORT"
  --tactile-mode "$USE_TACTILE"
  --tokenizer "${SIMPLETOUCH_TOKENIZER_PATH:-$REPO_DIR/assets/paligemma_tokenizer.model}"
  --tactile-encoder "${SIMPLETOUCH_T3_PATH:-$REPO_DIR/assets/t3_large}"
)
if [[ "$NORM_STATS" != "auto" ]]; then
  ARGS+=(--norm-stats "$NORM_STATS")
fi

exec env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$POLICY_PYTHON" "$SCRIPT_DIR/serve_policy.py" "${ARGS[@]}"
