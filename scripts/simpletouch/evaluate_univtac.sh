#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
: "${TASK:?Set TASK to one of the six released UniVTAC tasks}"

if [[ -z "${CKPT_PATH:-}" ]]; then
  CKPT_PATH=$(
    "${SIMPLETOUCH_PYTHON:-python}" "$script_dir/download_checkpoint.py" --task "$TASK"
  )
  export CKPT_PATH
fi

if [[ -n "${SIMPLETOUCH_EVAL_GPUS:-}" ]]; then
  exec "${SIMPLETOUCH_PYTHON:-python}" \
    "$repo_root/scripts/univtac/eval_simpletac_managed.py" \
    --checkpoint "$CKPT_PATH" --task "$TASK" --gpus "$SIMPLETOUCH_EVAL_GPUS" "$@"
fi

exec bash "$repo_root/scripts/univtac/run_eval_simpletac.sh" "$@"
