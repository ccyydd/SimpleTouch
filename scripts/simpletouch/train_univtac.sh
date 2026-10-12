#!/usr/bin/env bash
# Reproduce the SimpleTouch UniVTAC recipe in Table 5 of the paper.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

task=${SIMPLETOUCH_TASK:-insert_hole}
case "$task" in
  lift_bottle|pull_out_key|put_bottle_in_shelf|insert_hole) camera_type=head ;;
  lift_can|insert_tube) camera_type=all ;;
  *) echo "Unsupported paper task: $task" >&2; exit 2 ;;
esac

num_gpus=${SIMPLETOUCH_NUM_GPUS:-8}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || (( num_gpus > 64 || 64 % num_gpus != 0 )); then
  echo "SIMPLETOUCH_NUM_GPUS must divide the global batch of 64" >&2
  exit 2
fi
per_gpu=$((64 / num_gpus))
export SIMPLETAC_GPU_PLACEMENT=${SIMPLETOUCH_GPU_PLACEMENT:-0-$((num_gpus - 1))}
export EMBODIED_PATH="$repo_root/examples/sft"
export PYTHONPATH="$repo_root:$repo_root/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SIMPLETOUCH_DATA_ROOT:?Set SIMPLETOUCH_DATA_ROOT to the UniVTAC dataset root}"
: "${SIMPLETOUCH_NORM_STATS_PATH:?Set SIMPLETOUCH_NORM_STATS_PATH for this task}"
: "${SIMPLETOUCH_BASE_MODEL_PATH:?Set SIMPLETOUCH_BASE_MODEL_PATH to the converted pi0.5 model directory}"
: "${SIMPLETOUCH_T3_PATH:?Set SIMPLETOUCH_T3_PATH to the T3-large checkpoint directory}"
: "${SIMPLETOUCH_TOKENIZER_PATH:?Set SIMPLETOUCH_TOKENIZER_PATH to paligemma_tokenizer.model}"
: "${SIMPLETOUCH_FUTURE_CACHE_ROOT:?Set SIMPLETOUCH_FUTURE_CACHE_ROOT to the completed grid-6 T3 cache}"
export SIMPLETOUCH_OUTPUT_ROOT=${SIMPLETOUCH_OUTPUT_ROOT:-"$repo_root/outputs"}

if [[ "${SIMPLETOUCH_DRY_RUN:-0}" != 1 ]]; then
  for required_file in \
    "$SIMPLETOUCH_DATA_ROOT/$task/clean/0.hdf5" \
    "$SIMPLETOUCH_NORM_STATS_PATH" \
    "$SIMPLETOUCH_BASE_MODEL_PATH/model.safetensors" \
    "$SIMPLETOUCH_T3_PATH/encoders/mini.pth" \
    "$SIMPLETOUCH_T3_PATH/trunk.pth" \
    "$SIMPLETOUCH_TOKENIZER_PATH" \
    "$SIMPLETOUCH_FUTURE_CACHE_ROOT/$task/manifest.json"; do
    if [[ ! -f "$required_file" ]]; then
      echo "Required input is missing: $required_file" >&2
      exit 2
    fi
  done
fi

command=(
  "${SIMPLETOUCH_PYTHON:-python}"
  "$repo_root/examples/sft/train_vla_sft.py"
  --config-path "$repo_root/examples/sft/config"
  --config-name simpletac_univtac
  "data.task=$task"
  "data.camera_type=$camera_type"
  "data.tactile_expert_future_cache_path=$SIMPLETOUCH_FUTURE_CACHE_ROOT"
  "actor.micro_batch_size=$per_gpu"
  "actor.global_batch_size=64"
)
command+=("$@")

if [[ "${SIMPLETOUCH_DRY_RUN:-0}" == 1 ]]; then
  printf 'GPU placement: %s; per-GPU batch: %s\n' "$SIMPLETAC_GPU_PLACEMENT" "$per_gpu"
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
