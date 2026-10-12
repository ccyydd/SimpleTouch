#!/usr/bin/env bash

set -euo pipefail

if [[ "${TEMPORAL_ENSEMBLE:-false}" != "false" && "${TEMPORAL_ENSEMBLE:-false}" != "0" ]]; then
  echo "Temporal ensemble is no longer supported." >&2
  exit 2
fi

GPU_ID=${GPU_ID:-0}
TASK=${TASK:-insert_hole}
TASK_CONFIG=${TASK_CONFIG:-demo}
USE_TACTILE=${USE_TACTILE:-auto}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
PORT=${PORT:-8000}
# Resolve task defaults in the client after reading the checkpoint horizon.
EXECUTION_HORIZON=${EXECUTION_HORIZON:-task}
TOTAL_NUM=${TOTAL_NUM:-100}
ISAAC_HTTP_PORT=${ISAAC_HTTP_PORT:-$((18000 + GPU_ID))}
ISAAC_CPU_THREADS=${ISAAC_CPU_THREADS:-16}

validate_tcp_port() {
  local name=$1
  local value=$2
  if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    echo "$name must be an integer in [1, 65535], got: $value" >&2
    exit 2
  fi
}

validate_positive_integer() {
  local name=$1
  local value=$2
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

validate_tcp_port PORT "$PORT"
validate_tcp_port ISAAC_HTTP_PORT "$ISAAC_HTTP_PORT"
validate_positive_integer ISAAC_CPU_THREADS "$ISAAC_CPU_THREADS"
if [[ "$PORT" == "$ISAAC_HTTP_PORT" ]]; then
  echo "PORT and ISAAC_HTTP_PORT must be different, got: $PORT" >&2
  exit 2
fi

if [[ "$USE_TACTILE" != "auto" && "$USE_TACTILE" != "true" && "$USE_TACTILE" != "false" ]]; then
  echo "USE_TACTILE must be auto, true, or false, got: $USE_TACTILE" >&2
  exit 2
fi

if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be one physical GPU index, got: $GPU_ID" >&2
  echo "Do not pass a reordered CUDA_VISIBLE_DEVICES list here." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to construct a stable CUDA/Vulkan mapping." >&2
  exit 2
fi

mapfile -t AVAILABLE_GPU_IDS < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
)
GPU_FOUND=false
for available_gpu_id in "${AVAILABLE_GPU_IDS[@]}"; do
  if [[ "$available_gpu_id" == "$GPU_ID" ]]; then
    GPU_FOUND=true
    break
  fi
done
if [[ "$GPU_FOUND" != "true" ]]; then
  echo "GPU_ID=$GPU_ID is not an available physical GPU index." >&2
  exit 2
fi
EXPECTED_GPU_UUID=$(
  nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' '
)
EXPECTED_VULKAN_UUID=${EXPECTED_GPU_UUID#GPU-}

SCRIPT_DIR=$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -P "$SCRIPT_DIR/../.." && pwd)
: "${UNIVTAC_DIR:?Set UNIVTAC_DIR to your UniVTAC checkout}"
: "${CUROBO_ROOT:?Set CUROBO_ROOT to your cuRobo checkout}"
export SIMPLETOUCH_EVAL_ROOT=${SIMPLETOUCH_EVAL_ROOT:-$REPO_DIR/results/UniVTAC}
if [[ "$SIMPLETOUCH_EVAL_ROOT" != /* || "$SIMPLETOUCH_EVAL_ROOT" == / ]]; then
  echo "SIMPLETOUCH_EVAL_ROOT must be an absolute, non-root directory" >&2
  exit 2
fi
UNIVTAC_UPSTREAM_ROOT=${UNIVTAC_UPSTREAM_ROOT:-$UNIVTAC_DIR}
UNIVTAC_PYTHON=${UNIVTAC_PYTHON:-python}
UNIVTAC_ENVIRONMENT=${UNIVTAC_ENVIRONMENT:-reference}
case "$UNIVTAC_ENVIRONMENT" in
  reference)
    export UNIVTAC_LIVESTREAM=${UNIVTAC_LIVESTREAM:-2}
    if [[ "$UNIVTAC_LIVESTREAM" != "2" ]]; then
      echo "The reference evaluation requires UNIVTAC_LIVESTREAM=2." >&2
      exit 2
    fi
    if [[ "${UNIVTAC_REFERENCE_READY:-false}" != "true" ]]; then
      RUNTIME_ROOT=${SIMPLETOUCH_RUNTIME_ROOT:-$REPO_DIR/.runtime}
      REFERENCE_TARGET=${SIMPLETOUCH_REFERENCE_RUNTIME:-$RUNTIME_ROOT/univtac_reference/gpu$GPU_ID/UniVTAC}
      UNIVTAC_DIR=$(python3 "$SCRIPT_DIR/reference_runtime.py" \
        --upstream "$UNIVTAC_UPSTREAM_ROOT" --target "$REFERENCE_TARGET" \
        --kit-root "$(dirname "$REFERENCE_TARGET")/kit")
      export UNIVTAC_DIR
    elif [[ ! -f "$UNIVTAC_DIR/reference-runtime.json" ]]; then
      echo "UNIVTAC_REFERENCE_READY requires a verified reference runtime." >&2
      exit 2
    fi
    DEFAULT_EVAL_DRIVER=native-routed
    ;;
  public)
    DEFAULT_EVAL_DRIVER=simpletouch
    ;;
  *)
    echo "UNIVTAC_ENVIRONMENT must be reference or public." >&2
    exit 2
    ;;
esac
if [[ -n "${SIMPLETAC_MANAGED_PLAN:-}" ]]; then
  # Preserve the existing managed launcher's bounded shard adapter.
  DEFAULT_EVAL_DRIVER=simpletouch
fi
PLUGIN_SOURCE="$REPO_DIR/examples/UniVTAC/plugin/policy/SimpleTac"
PLUGIN_TARGET="$UNIVTAC_DIR/policy/SimpleTac"
TACEX_ASSETS_SOURCE="$UNIVTAC_UPSTREAM_ROOT/third_party/TacEx/source/tacex_assets"

if [[ -n "${CONDA_SH:-}" ]]; then
  # Conda activation scripts may refer to unset variables.
  set +u
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-uni}"
  set -u
fi

mkdir -p "$PLUGIN_TARGET"
cp "$PLUGIN_SOURCE/__init__.py" "$PLUGIN_TARGET/__init__.py"
cp "$PLUGIN_SOURCE/deploy_policy.py" "$PLUGIN_TARGET/deploy_policy.py"

# UniVTAC/UsdRT requires CUDA logical device 0, while Vulkan keeps the physical
# GPU enumeration. Expose only the requested CUDA device, use cuda:0 for
# physics, and override renderer activeGpu with the original physical index.
# Disabling renderer multi-GPU keeps all heavy work on that same physical GPU.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
export WANDB_MODE=offline
export OMNI_KIT_ACCEPT_EULA=YES
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0+PTX}"
# These variables affect only this Isaac Sim process. They cap CPU operators
# used by simulation-side PyTorch/NumPy without changing the policy server.
export OMP_NUM_THREADS="$ISAAC_CPU_THREADS"
export MKL_NUM_THREADS="$ISAAC_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$ISAAC_CPU_THREADS"
export NUMEXPR_NUM_THREADS="$ISAAC_CPU_THREADS"
# Do not import policy-environment packages into the separate simulator Python.
export PYTHONPATH="$REPO_DIR/packages/openpi-client/src:$TACEX_ASSETS_SOURCE:$UNIVTAC_DIR:$CUROBO_ROOT/src"
# Simulator-only optional dependencies installed outside the shared environment.
if [[ -n "${UNIVTAC_EXTRA_PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$PYTHONPATH:$UNIVTAC_EXTRA_PYTHONPATH"
fi
export SIMPLETAC_HOST="$SERVER_HOST"
export SIMPLETAC_PORT="$PORT"
export SIMPLETAC_USE_TACTILE="$USE_TACTILE"
export SIMPLETAC_EXECUTION_HORIZON="$EXECUTION_HORIZON"
export SIMPLETAC_TASK="$TASK"

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/nvcc" ]]; then
  export CUDA_HOME="$CONDA_PREFIX"
elif [[ -z "${CUDA_HOME:-}" || ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Standalone vulkaninfo and Isaac Kit can enumerate the same devices in
# different orders. Kit's /renderer/activeGpu follows its own physical/PCI
# ordering on the UniVTAC hosts, which matches nvidia-smi's physical GPU ID.
# Use vulkaninfo only to verify that the requested device is visible; feeding
# its list index to Kit can select the wrong GPU or make Kit reject the device.
if ! timeout 30 vulkaninfo --summary 2>/dev/null | awk \
  -v target="$(tr '[:upper:]' '[:lower:]' <<<"$EXPECTED_VULKAN_UUID")" '
    $1 == "deviceUUID" && tolower($3) == target { found = 1 }
    END { exit(found ? 0 : 1) }
  '; then
  echo "Could not find physical GPU $GPU_ID UUID $EXPECTED_GPU_UUID in Vulkan." >&2
  exit 2
fi
RENDERER_ACTIVE_GPU_ID=$GPU_ID

cd "$UNIVTAC_DIR"
echo "UniVTAC GPU mapping: physical_gpu=$GPU_ID uuid=$EXPECTED_GPU_UUID"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  AppLauncher physics=cuda:0 renderer_active_gpu=$RENDERER_ACTIVE_GPU_ID (Kit physical/PCI index, Vulkan UUID verified, renderer multi-GPU disabled, internal_http_port=$ISAAC_HTTP_PORT)"
echo "  Isaac Sim CPU threads=$ISAAC_CPU_THREADS (Carbonite, TBB, physics, OpenMP/MKL)"
echo "  SimpleTac execution_horizon=$EXECUTION_HORIZON"
EVAL_DRIVER="$SCRIPT_DIR/eval_policy_simpletouch.py"
case "${UNIVTAC_EVAL_DRIVER:-$DEFAULT_EVAL_DRIVER}" in
  native-routed)
    EVAL_DRIVER="$SCRIPT_DIR/eval_policy_native.py"
    ;;
  native)
    EVAL_DRIVER="$UNIVTAC_DIR/scripts/eval_policy.py"
    ;;
  simpletouch)
    if [[ -n "${SIMPLETAC_MANAGED_PLAN:-}" ]]; then
      EVAL_DRIVER="$SCRIPT_DIR/eval_policy_managed.py"
    fi
    ;;
  *)
    echo "UNIVTAC_EVAL_DRIVER must be native-routed, native or simpletouch." >&2
    exit 2
    ;;
esac
SEED_ARGS=()
for seed_name in UNIVTAC_START_SEED UNIVTAC_MAX_SEED; do
  seed_value=${!seed_name:-}
  if [[ -n "$seed_value" ]]; then
    if ! [[ "$seed_value" =~ ^[0-9]+$ ]]; then
      echo "$seed_name must be a nonnegative integer." >&2
      exit 2
    fi
    if [[ "$seed_name" == UNIVTAC_START_SEED ]]; then
      SEED_ARGS+=(--start_seed "$seed_value")
    else
      SEED_ARGS+=(--max_seed "$seed_value")
    fi
  fi
done
echo "UniVTAC evaluator: $EVAL_DRIVER (seed arguments: ${SEED_ARGS[*]})"
exec "$UNIVTAC_PYTHON" "$EVAL_DRIVER" \
  "$TASK" \
  "$TASK_CONFIG" \
  "$PLUGIN_SOURCE/deploy.yml" \
  --total_num "$TOTAL_NUM" \
  "${SEED_ARGS[@]}" \
  --device cuda:0 \
  --kit_args="--/renderer/activeGpu=$RENDERER_ACTIVE_GPU_ID --/renderer/multiGpu/enabled=false --/app/livestream/port=$((49100 + 10 * GPU_ID)) --/app/livestream/fixedHostPort=$((49000 + 10 * GPU_ID)) --/exts/omni.services.transport.server.http/port=$ISAAC_HTTP_PORT --/exts/omni.services.transport.server.http/allow_port_range=false --/plugins/carb.tasking.plugin/threadCount=$ISAAC_CPU_THREADS --/plugins/omni.tbb.globalcontrol/maxThreadCount=$ISAAC_CPU_THREADS --/persistent/physics/numThreads=$ISAAC_CPU_THREADS"
