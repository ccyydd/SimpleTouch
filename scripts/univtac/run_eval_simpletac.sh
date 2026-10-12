#!/usr/bin/env bash

set -euo pipefail

if [[ "${TEMPORAL_ENSEMBLE:-false}" != "false" && "${TEMPORAL_ENSEMBLE:-false}" != "0" ]]; then
  echo "Temporal ensemble is no longer supported." >&2
  exit 2
fi

: "${CKPT_PATH:?Set CKPT_PATH to an RLinf global_step directory or converted model directory}"

# GPU_ID is the preferred interface. Legacy POLICY_GPU_ID/SIM_GPU_ID overrides
# remain supported, but both default to the same physical GPU.
GPU_ID=${GPU_ID:-${POLICY_GPU_ID:-${SIM_GPU_ID:-0}}}
POLICY_GPU_ID=${POLICY_GPU_ID:-$GPU_ID}
SIM_GPU_ID=${SIM_GPU_ID:-$GPU_ID}
TASK=${TASK:-insert_hole}
TASK_CONFIG=${TASK_CONFIG:-demo}
USE_TACTILE=${USE_TACTILE:-auto}
# Derive both local service ports from their physical GPU after validating the
# GPU IDs, so malformed legacy GPU lists still receive a clear error.
PORT=${PORT:-}
ISAAC_HTTP_PORT=${ISAAC_HTTP_PORT:-}
EVAL_GPU_LOCK_FILE=${EVAL_GPU_LOCK_FILE:-/tmp/simpletac_eval_gpu_${SIM_GPU_ID}.lock}
# task: IH/Insert Tube execute 25 steps; other tasks use the checkpoint horizon.
# Explicit numeric values and legacy auto (full checkpoint chunk) are preserved.
EXECUTION_HORIZON=${EXECUTION_HORIZON:-task}
TOTAL_NUM=${TOTAL_NUM:-100}
# Limit only the Isaac Sim evaluation process. The policy server is launched
# separately and intentionally keeps its existing CPU-thread configuration.
ISAAC_CPU_THREADS=${ISAAC_CPU_THREADS:-16}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-900}
VULKAN_PREFLIGHT=${VULKAN_PREFLIGHT:-true}
FIX_CHECKPOINT_PERMISSIONS=${FIX_CHECKPOINT_PERMISSIONS:-true}
# Optional host-specific Vulkan runtime repair. A release checkout must not
# depend on another research repository or a particular user's NAS mount.
VULKAN_RUNTIME_ENV=${VULKAN_RUNTIME_ENV:-}
VULKAN_SETUP_SCRIPT=${VULKAN_SETUP_SCRIPT:-}

validate_gpu_id() {
  local name=$1
  local value=$2
  local available_gpu_ids
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$name must be one physical GPU index, got: $value" >&2
    echo "Do not pass a reordered CUDA_VISIBLE_DEVICES list here." >&2
    exit 2
  fi
  available_gpu_ids=$(
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr -d ' '
  )
  if ! grep -Fxq "$value" <<<"$available_gpu_ids"; then
    echo "$name=$value is not an available physical GPU index." >&2
    exit 2
  fi
}

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

port_is_listening() {
  local value=$1
  ss -H -ltn "sport = :$value" 2>/dev/null | grep -q .
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required to validate the physical GPU mapping." >&2
  exit 2
fi

validate_gpu_id POLICY_GPU_ID "$POLICY_GPU_ID"
validate_gpu_id SIM_GPU_ID "$SIM_GPU_ID"
validate_positive_integer ISAAC_CPU_THREADS "$ISAAC_CPU_THREADS"
PORT=${PORT:-$((28000 + POLICY_GPU_ID))}
ISAAC_HTTP_PORT=${ISAAC_HTTP_PORT:-$((18000 + SIM_GPU_ID))}
validate_tcp_port PORT "$PORT"
validate_tcp_port ISAAC_HTTP_PORT "$ISAAC_HTTP_PORT"
if [[ "$PORT" == "$ISAAC_HTTP_PORT" ]]; then
  echo "PORT and ISAAC_HTTP_PORT must be different, got: $PORT" >&2
  exit 2
fi

# Hold this lock until the wrapper exits. This prevents two Isaac Sim
# processes from being launched on the same physical simulation GPU.
if command -v flock >/dev/null 2>&1; then
  if [[ -n "${SIMPLETAC_RESERVED_GPU_FD:-}" ]]; then
    # The managed launcher passes an already-locked open file description.
    # Keep that reservation across preflight and launch, without a race window.
    if ! [[ "$SIMPLETAC_RESERVED_GPU_FD" =~ ^[0-9]+$ ]] || \
       [[ ! "/proc/$$/fd/$SIMPLETAC_RESERVED_GPU_FD" -ef "$EVAL_GPU_LOCK_FILE" ]]; then
      echo "Invalid inherited evaluation GPU reservation." >&2
      exit 2
    fi
    EVAL_GPU_LOCK_FD=$SIMPLETAC_RESERVED_GPU_FD
  else
    exec {EVAL_GPU_LOCK_FD}>"$EVAL_GPU_LOCK_FILE"
  fi
  if ! flock -n "$EVAL_GPU_LOCK_FD"; then
    echo "Physical simulation GPU $SIM_GPU_ID is already reserved by another SimpleTac evaluation." >&2
    echo "Lock: $EVAL_GPU_LOCK_FILE" >&2
    exit 2
  fi
else
  echo "Warning: flock is unavailable; same-GPU evaluation exclusion is disabled." >&2
fi

# Do this before checkpoint permission repair, Vulkan setup, and the large
# policy load. A conflict therefore fails in seconds rather than minutes.
if command -v ss >/dev/null 2>&1; then
  if port_is_listening "$PORT"; then
    echo "Policy server port $PORT is already in use. Set PORT to another value." >&2
    exit 2
  fi
  if port_is_listening "$ISAAC_HTTP_PORT"; then
    echo "Isaac Sim HTTP port $ISAAC_HTTP_PORT is already in use. Set ISAAC_HTTP_PORT to another value." >&2
    exit 2
  fi
else
  echo "Warning: ss is unavailable; TCP port preflight is disabled." >&2
fi

if [[ "$VULKAN_PREFLIGHT" != "true" && "$VULKAN_PREFLIGHT" != "false" ]]; then
  echo "VULKAN_PREFLIGHT must be true or false, got: $VULKAN_PREFLIGHT" >&2
  exit 2
fi
if [[ "$FIX_CHECKPOINT_PERMISSIONS" != "true" && "$FIX_CHECKPOINT_PERMISSIONS" != "false" ]]; then
  echo "FIX_CHECKPOINT_PERMISSIONS must be true or false, got: $FIX_CHECKPOINT_PERMISSIONS" >&2
  exit 2
fi

if [[ "$USE_TACTILE" != "auto" && "$USE_TACTILE" != "true" && "$USE_TACTILE" != "false" ]]; then
  echo "USE_TACTILE must be auto, true, or false, got: $USE_TACTILE" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

ensure_checkpoint_readable() {
  local checkpoint_path=$1
  local current_user private_group current_uid current_gid
  local -a unreadable_files=()
  local file

  if [[ ! -d "$checkpoint_path" ]]; then
    echo "CKPT_PATH is not a checkpoint directory: $checkpoint_path" >&2
    exit 2
  fi

  while IFS= read -r -d '' file; do
    unreadable_files+=("$file")
  done < <(find "$checkpoint_path" -type f ! -readable -print0)

  if (( ${#unreadable_files[@]} == 0 )); then
    return
  fi

  echo "Found ${#unreadable_files[@]} unreadable checkpoint file(s)." >&2
  printf '  %s\n' "${unreadable_files[@]}" >&2
  if [[ "$FIX_CHECKPOINT_PERMISSIONS" != "true" ]]; then
    echo "Set FIX_CHECKPOINT_PERMISSIONS=true to repair private group-read access." >&2
    exit 2
  fi

  current_user=$(id -un)
  private_group=$(id -gn)
  current_uid=$(id -u)
  current_gid=$(id -g)
  if [[ "$private_group" != "$current_user" || "$current_gid" != "$current_uid" ]]; then
    echo "Refusing automatic permission repair because the current primary group is not private:" >&2
    echo "  user=$current_user uid=$current_uid group=$private_group gid=$current_gid" >&2
    exit 2
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to repair root-owned checkpoint permissions." >&2
    exit 2
  fi
  if ! sudo -n true 2>/dev/null; then
    if [[ -t 0 ]]; then
      echo "Checkpoint files are root-owned; authenticate once to grant private group-read access." >&2
      sudo -v
    else
      echo "Checkpoint permission repair needs sudo authentication, but this launch is non-interactive." >&2
      echo "Run 'sudo -v' in an interactive shell, then launch the evaluation again." >&2
      exit 2
    fi
  fi

  # Keep root ownership and deny access to other users. Only the current
  # account's private primary group receives read permission.
  sudo chgrp -- "$private_group" "${unreadable_files[@]}"
  sudo chmod -- u+rw,g+r,o-rwx "${unreadable_files[@]}"

  for file in "${unreadable_files[@]}"; do
    if [[ ! -r "$file" ]]; then
      echo "Checkpoint file is still unreadable after permission repair: $file" >&2
      exit 2
    fi
  done
  echo "Checkpoint permissions repaired as root:${private_group} with private group-read access."
}

ensure_checkpoint_readable "$CKPT_PATH"

MODALITY=auto
if [[ "$USE_TACTILE" == "true" ]]; then
  MODALITY=tactile
elif [[ "$USE_TACTILE" == "false" ]]; then
  MODALITY=vision
fi

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs/SimpleTouch/${TASK}_${MODALITY}_${RUN_STAMP}}
SERVER_LOG="$LOG_DIR/policy_server.log"
EVAL_LOG="$LOG_DIR/univtac_eval.log"
mkdir -p "$LOG_DIR"

# A previously prepared user-space runtime takes precedence over mismatched
# GLX/EGL packages in the container. If the host driver changes, the preflight
# below will fail and regenerate this file for the new version.
if [[ -f "$VULKAN_RUNTIME_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$VULKAN_RUNTIME_ENV"
fi
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

if [[ "$VULKAN_PREFLIGHT" == "true" ]]; then
  VULKAN_LOG="$LOG_DIR/vulkan_preflight.log"
  if command -v vulkaninfo >/dev/null 2>&1; then
    if ! timeout 30 vulkaninfo --summary >"$VULKAN_LOG" 2>&1; then
      cp "$VULKAN_LOG" "$LOG_DIR/vulkan_preflight_initial.log"
      echo "Initial Vulkan preflight failed; trying the user-space NVIDIA runtime setup." >&2
      if [[ -f "$VULKAN_SETUP_SCRIPT" ]] && \
          bash "$VULKAN_SETUP_SCRIPT" >"$LOG_DIR/vulkan_setup.log" 2>&1 && \
          [[ -f "$VULKAN_RUNTIME_ENV" ]]; then
        # shellcheck disable=SC1090
        source "$VULKAN_RUNTIME_ENV"
      fi
      if ! timeout 30 vulkaninfo --summary >"$VULKAN_LOG" 2>&1; then
        echo "Vulkan preflight failed after automatic setup; refusing to load the checkpoint." >&2
        echo "Initial log: $LOG_DIR/vulkan_preflight_initial.log" >&2
        echo "Setup log: $LOG_DIR/vulkan_setup.log" >&2
        echo "Final log: $VULKAN_LOG" >&2
        tail -n 40 "$VULKAN_LOG" >&2 || true
        exit 2
      fi
    fi
    EXPECTED_RENDER_UUID=$(
      nvidia-smi --id="$SIM_GPU_ID" --query-gpu=uuid --format=csv,noheader,nounits \
        | tr -d ' ' | sed 's/^GPU-//'
    )
    VULKAN_GPU_UUIDS=$(awk '$1 == "deviceUUID" { print tolower($3) }' "$VULKAN_LOG")
    if ! grep -Fxiq "$EXPECTED_RENDER_UUID" <<<"$VULKAN_GPU_UUIDS"; then
      echo "Vulkan did not enumerate physical GPU $SIM_GPU_ID UUID $EXPECTED_RENDER_UUID." >&2
      echo "See: $VULKAN_LOG" >&2
      exit 2
    fi
  else
    echo "Warning: vulkaninfo is unavailable; skipping the Vulkan preflight." >&2
  fi
fi

if [[ "$POLICY_GPU_ID" == "$SIM_GPU_ID" ]]; then
  echo "Using shared physical GPU $POLICY_GPU_ID for policy, CUDA simulation, and Vulkan rendering."
else
  echo "Using physical GPU $POLICY_GPU_ID for policy and GPU $SIM_GPU_ID for simulation/rendering."
fi
echo "Evaluation ports: policy=$PORT isaac_http=$ISAAC_HTTP_PORT"
echo "Action execution: horizon=$EXECUTION_HORIZON"

SERVER_PID=""
EVAL_PID=""

terminate_eval_group() {
  local eval_pid=$1
  local elapsed

  if ! kill -0 "$eval_pid" 2>/dev/null; then
    wait "$eval_pid" 2>/dev/null || true
    return
  fi

  echo "Stopping UniVTAC evaluation process group (pgid=$eval_pid)..."
  kill -TERM -- "-$eval_pid" 2>/dev/null || kill -TERM "$eval_pid" 2>/dev/null || true
  for ((elapsed = 0; elapsed < 30; elapsed++)); do
    if ! kill -0 "$eval_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  if kill -0 "$eval_pid" 2>/dev/null; then
    echo "UniVTAC evaluation did not stop within 30s; killing its process group." >&2
    kill -KILL -- "-$eval_pid" 2>/dev/null || kill -KILL "$eval_pid" 2>/dev/null || true
  fi
  wait "$eval_pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$EVAL_PID" ]]; then
    terminate_eval_group "$EVAL_PID"
  fi
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping SimpleTac policy server (pid=$SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting SimpleTac policy server on physical GPU $POLICY_GPU_ID..."
GPU_ID="$POLICY_GPU_ID" \
CKPT_PATH="$CKPT_PATH" \
TASK="$TASK" \
USE_TACTILE="$USE_TACTILE" \
PORT="$PORT" \
bash "$SCRIPT_DIR/serve_simpletac.sh" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for the policy server (timeout: ${STARTUP_TIMEOUT}s)..."
SERVER_READY=false
for ((elapsed = 0; elapsed < STARTUP_TIMEOUT; elapsed++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Policy server exited before becoming ready. Last log lines:" >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi

  if grep -q "Serving SimpleTac on" "$SERVER_LOG" 2>/dev/null; then
    SERVER_READY=true
    break
  fi
  sleep 1
done

if [[ "$SERVER_READY" != "true" ]]; then
  echo "Policy server did not become ready within ${STARTUP_TIMEOUT}s. Last log lines:" >&2
  tail -n 80 "$SERVER_LOG" >&2 || true
  exit 1
fi

echo "Policy server is ready. Starting UniVTAC evaluation on physical GPU $SIM_GPU_ID..."
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required to isolate and reliably clean up the Isaac Sim process group." >&2
  exit 2
fi

set +e
GPU_ID="$SIM_GPU_ID" \
TASK="$TASK" \
TASK_CONFIG="$TASK_CONFIG" \
USE_TACTILE="$USE_TACTILE" \
SERVER_HOST=127.0.0.1 \
PORT="$PORT" \
ISAAC_HTTP_PORT="$ISAAC_HTTP_PORT" \
ISAAC_CPU_THREADS="$ISAAC_CPU_THREADS" \
EXECUTION_HORIZON="$EXECUTION_HORIZON" \
TOTAL_NUM="$TOTAL_NUM" \
setsid bash "$SCRIPT_DIR/eval_simpletac.sh" > >(tee "$EVAL_LOG") 2>&1 &
EVAL_PID=$!
wait "$EVAL_PID"
EVAL_STATUS=$?
EVAL_PID=""
set -e

if (( EVAL_STATUS != 0 )); then
  echo "UniVTAC evaluation exited with status $EVAL_STATUS. See: $EVAL_LOG" >&2
  exit "$EVAL_STATUS"
fi

echo "Evaluation finished. Logs are in: $LOG_DIR"
