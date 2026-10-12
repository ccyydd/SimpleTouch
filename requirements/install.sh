#!/usr/bin/env bash
# Install only the native PyTorch SimpleTouch policy/training environment.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv="$repo/.venv"
python_version=3.11
platform=nvidia
backend=auto
torch_version=2.7.1
vision_version=0.22.1
unsupported() { echo 'SimpleTouch supports only: embodied --model openpi --env policy-only' >&2; exit 2; }
while (( $# )); do
  case "$1" in
    embodied) shift ;;
    --model) [[ "${2:-}" == openpi ]] || unsupported; shift 2 ;;
    --env) [[ "${2:-}" == policy-only ]] || unsupported; shift 2 ;;
    --venv) venv=$2; shift 2 ;;
    --python) python_version=$2; shift 2 ;;
    --platform) platform=$2; shift 2 ;;
    --torch-backend) backend=$2; shift 2 ;;
    --torch) torch_version=$2; shift 2 ;;
    --torchvision) vision_version=$2; shift 2 ;;
    --no-root|--no-flash-attn|--install-rlinf) shift ;;
    -h|--help)
      echo 'Usage: bash requirements/install.sh embodied --model openpi --env policy-only'
      echo 'Options: --venv PATH --python 3.11 --platform nvidia|amd'
      echo '         --torch VERSION --torchvision VERSION --torch-backend BACKEND'
      echo 'AMD: explicitly select matching ROCm torch/torchvision versions and backend.'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ $(uname -s) == Linux ]] || { echo 'The policy runtime requires Linux.' >&2; exit 2; }
case "$platform" in
  nvidia) ;;
  amd)
    [[ "$backend" == rocm* ]] || { echo 'AMD requires --torch-backend rocm<VERSION> and matching torch/torchvision versions.' >&2; exit 2; } ;;
  *) echo "Unsupported platform: $platform" >&2; exit 2 ;;
esac
command -v uv >/dev/null || {
  echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 2
}
if [[ -e "$venv" && ! -f "$venv/pyvenv.cfg" ]]; then
  echo "Refusing to overwrite a non-venv directory: $venv" >&2; exit 2
fi
if [[ ! -f "$venv/pyvenv.cfg" ]]; then
  uv venv --python "$python_version" "$venv"
fi
uv pip install --python "$venv/bin/python" --torch-backend "$backend" \
  "torch==$torch_version" "torchvision==$vision_version" \
  -r "$repo/requirements/policy.txt" -e "$repo" -e "$repo/packages/openpi-client"
uv pip check --python "$venv/bin/python"
cd "$repo"
"$venv/bin/python" - <<'PY'
import torch
import scripts.univtac.serve_policy
import rlinf.workers.sft.fsdp_vla_sft_worker
print(f'SimpleTouch imports OK. torch={torch.__version__}, CUDA available={torch.cuda.is_available()}')
PY
echo "Ready. Activate with: source $venv/bin/activate"
