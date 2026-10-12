#!/usr/bin/env bash
# Pinned, isolated UniVTAC simulator installation.
set -euo pipefail
# Build tools parse English compiler/assembler version strings.
export LC_ALL=C LANG=C LANGUAGE=C
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
support="$repo/requirements/univtac"
prefix= gpu= arch= accepted=0 jobs=4
usage() {
  echo 'Install pinned simulator dependencies; run an evaluation smoke test afterwards.'
  echo 'Usage: bash requirements/install_univtac.sh --prefix ABSOLUTE_PATH --gpu INDEX --cuda-arch NN [--jobs N] --accept-isaac-eula'
  echo 'Requires installed uv, Conda, Git LFS, GCC/G++ 11, Linux x86_64 and NVIDIA driver.'
  echo 'Supported --cuda-arch values: 70, 75, 80, 86, 87, 89, 90 (CUDA 12.4).'
  echo '--jobs controls native build parallelism (default: 4); choose according to free CPU and RAM.'
  echo 'Creates an owned prefix and private temporary scratch; resumes identical inputs under an exclusive lock.'
  echo 'Does not download policy weights, run RL, evaluate episodes, or change system packages.'
}
fail() { echo "ERROR: $*" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --prefix|--gpu|--cuda-arch|--jobs)
      (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || fail "Missing value for $1"
      case "$1" in --prefix) prefix=$2;; --gpu) gpu=$2;; --cuda-arch) arch=$2;; --jobs) jobs=$2;; esac
      shift 2 ;;
    --accept-isaac-eula) accepted=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || fail '--jobs must be a positive integer.'
[[ -n "$prefix" && "$gpu" =~ ^[0-9]+$ && "$arch" =~ ^[0-9]{2,3}$ ]] || { usage >&2; exit 2; }
case "$arch" in
  70|75|80|86|87|89|90) ;;
  *) fail "Unsupported --cuda-arch $arch for this CUDA 12.4 recipe; supported: 70, 75, 80, 86, 87, 89, 90. Blackwell is not supported." ;;
esac
(( accepted )) || fail 'Read the NVIDIA Isaac Sim EULA and explicitly pass --accept-isaac-eula.'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || fail 'Only Linux x86_64 is supported.'
for tool in uv python3 git flock gcc g++ nvidia-smi; do
  command -v "$tool" >/dev/null || fail "Required tool is missing: $tool"
done
git lfs version >/dev/null || fail 'Install Git LFS before running this installer.'
uv_exe=$(type -P uv)
host_python=$(type -P python3)
nvidia_smi_exe=$(type -P nvidia-smi)
conda_exe=${CONDA_EXE:-$(type -P conda || true)}
[[ -x "$conda_exe" ]] || fail 'Conda executable not found; set CONDA_EXE.'
# Resolve tools before dropping the caller's activated policy environment.
host_path=/usr/local/bin:/usr/bin:/bin
gcc_exe=$(PATH="$host_path" command -v gcc); gxx_exe=$(PATH="$host_path" command -v g++)
[[ $("$gcc_exe" -dumpversion) == 11* && $("$gxx_exe" -dumpversion) == 11* ]] || fail 'This recipe requires host GCC/G++ 11.'
actual_arch=$("$nvidia_smi_exe" --id="$gpu" --query-gpu=compute_cap --format=csv,noheader | tr -d '. ')
[[ "$actual_arch" == "$arch" ]] || fail "--cuda-arch $arch does not match GPU $gpu ($actual_arch)."
state() { "$host_python" -I -S "$support/installer_state.py" "$@"; }
state prepare "$prefix"
[[ ! -L "$prefix/.install.lock" ]] || fail 'Installation lock must not be a symlink.'
exec 9>"$prefix/.install.lock"
flock -n 9 || fail 'Another installer is using this prefix.'
state manifest "$prefix" "$repo" "$gpu" "$arch"
# shellcheck source=univtac/versions.env
source "$support/versions.env"
mkdir -p "$prefix"/{sources,tools,tmp,logs,state,runtime,cache}
export TMPDIR="$prefix/tmp" CONDA_PKGS_DIRS="$prefix/cache/conda"
export CONDA_ENVS_PATH="$prefix/envs" CONDA_REGISTER_ENVS=false CONDARC=/dev/null
export UV_CACHE_DIR="$prefix/cache/uv" UV_PYTHON_INSTALL_DIR="$prefix/tools/uv-python"
export PIP_CACHE_DIR="$prefix/cache/pip" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$prefix/cache/runtime" XDG_CONFIG_HOME="$prefix/runtime/config" XDG_DATA_HOME="$prefix/runtime/data"
export CUDA_CACHE_PATH="$prefix/cache/cuda" TORCH_EXTENSIONS_DIR="$prefix/cache/torch-extensions"
export NUMBA_CACHE_DIR="$prefix/cache/numba" __GL_SHADER_DISK_CACHE_PATH="$prefix/cache/gl-shader"
export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=3 PIP_DEFAULT_TIMEOUT=60
export PIP_CONFIG_FILE=/dev/null UV_NO_CONFIG=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
unset ISAAC_PATH EXP_PATH CARB_APP_PATH
unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CMAKE_PREFIX_PATH CMAKE_TOOLCHAIN_FILE
unset CFLAGS CXXFLAGS LDFLAGS CMAKE_ARGS CUDACXX CUDAHOSTCXX
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL UV_INDEX UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL
export PATH="$host_path" OMNI_KIT_ACCEPT_EULA=YES
export CC="$gcc_exe" CXX="$gxx_exe" MAX_JOBS="$jobs" CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
sim_python="$prefix/env/bin/python"
univtac="$prefix/sources/UniVTAC"
lab="$univtac/third_party/IsaacLab"
tacex="$univtac/third_party/TacEx/source"
curobo="$univtac/third_party/curobo"
vcpkg="$prefix/sources/vcpkg"
pip_install() { "$uv_exe" pip install --python "$sim_python" -c "$support/constraints.txt" "$@"; }
pip_check() { "$sim_python" -m pip check; }
stage() {
  local name=$1; shift
  if [[ -f "$prefix/state/$name.done" ]]; then
    echo "Verified-input resume: $name"
  else
    echo "Installing stage: $name"
    ( "$@"; pip_check ) 2>&1 | tee "$prefix/logs/$name.log"
    date -u +%FT%TZ > "$prefix/state/$name.done.tmp"
    mv "$prefix/state/$name.done.tmp" "$prefix/state/$name.done"
  fi
  # A later interrupted stage may have left incomplete dependencies. Let the
  # first unfinished stage repair them; each new stage and the final check are
  # still strict, and no failed stage receives a completion marker.
}
clone_at() {
  local url=$1 destination=$2 commit=$3
  if [[ ! -e "$destination" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone "$url" "$destination"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" checkout --detach "$commit"
  fi
  [[ ! -L "$destination" && -d "$destination/.git" ]] || fail "Not an owned standalone checkout: $destination"
  [[ $(git -C "$destination" remote get-url origin) == "$url" ]] || fail "Unexpected remote: $destination"
  [[ $(git -C "$destination" rev-parse HEAD) == "$commit" ]] || fail "Unexpected revision: $destination; use a new prefix."
}
install_toolchain() {
  # Never run a Conda solve AFTER pip-installed Isaac/CUDA runtime packages.
  if [[ ! -f "$prefix/env/conda-meta/history" ]]; then
    [[ ! -e "$prefix/env" ]] || fail 'Incomplete environment directory; use a new prefix.'
    "$conda_exe" create --prefix "$prefix/env" --override-channels \
      -c nvidia/label/cuda-12.4.1 -c conda-forge \
      "python=$PYTHON_VERSION" pip "cuda-toolkit=$CUDA_TOOLKIT_VERSION" -y
  fi
  "$prefix/env/bin/nvcc" --version
  "$sim_python" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
  pip_install cmake==3.30.1 ninja pkgconfig toml 'setuptools<82' wheel pybind11
  if [[ ! -f "$prefix/tools/native/pyvenv.cfg" ]]; then
    [[ ! -e "$prefix/tools/native" ]] || fail 'Incomplete native tools environment; use a new prefix.'
    "$uv_exe" venv --python "$sim_python" "$prefix/tools/native"
  fi
  "$uv_exe" pip install --python "$prefix/tools/native/bin/python" cmake==3.30.1 ninja
  "$uv_exe" pip check --python "$prefix/tools/native/bin/python"
}
install_sources() {
  clone_at https://github.com/univtac/UniVTAC.git "$univtac" "$UNIVTAC_COMMIT"
  git -C "$univtac" lfs install --local
  git -C "$univtac" lfs pull --include='' --exclude=''
  git -C "$univtac" lfs fsck
  clone_at https://github.com/isaac-sim/IsaacLab.git "$lab" "$ISAACLAB_COMMIT"
  clone_at https://github.com/NVlabs/curobo.git "$curobo" "$CUROBO_COMMIT"
  # UniVTAC supplies the task assets. cuRobo's optional example LFS collection
  # is not needed for the UniVTAC sphere-planner path.
  clone_at https://github.com/microsoft/vcpkg.git "$vcpkg" "$VCPKG_COMMIT"
  bash "$vcpkg/bootstrap-vcpkg.sh" -disableMetrics
  state patch "$univtac" "$support/torch-scatter-url.json"
  state copy-client "$repo/packages/openpi-client" "$prefix/sources/openpi-client"
  git -C "$univtac" diff -- third_party/TacEx/source/tacex/setup.py > "$prefix/logs/compatibility.patch"
}
install_isaac() {
  pip_install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --default-index https://download.pytorch.org/whl/cu124
  pip_install 'isaacsim[all,extscache]==4.5.0' --index https://pypi.nvidia.com
}
install_lab() {
  pip_install flatdict==4.0.1 --no-build-isolation
  # Runtime-only installation; no RL frameworks or Mimic extras.
  for extension in isaaclab isaaclab_assets isaaclab_tasks; do
    pip_install -e "$lab/source/$extension" --no-build-isolation
  done
}
install_curobo() {
  pip_install setuptools_scm warp-lang==1.0.0
  pip_install -e "$curobo" --no-build-isolation
}
install_tacex() {
  for extension in tacex tacex_assets; do
    pip_install -e "$tacex/$extension" --no-build-isolation
  done
  # Isaac's asset-browser extension imports requests but omits it from its
  # package dependencies; keep the normal livestream-enabled app functional.
  pip_install h5py decorator transforms3d trimesh tetgen 'requests>=2.31,<3' -e "$prefix/sources/openpi-client"
}
install_uipc() {
  # Upstream's native-architecture probe still needs a visible, idle device.
  [[ -z $("$nvidia_smi_exe" --id="$gpu" --query-compute-apps=pid --format=csv,noheader) ]] || fail "Build GPU $gpu is busy."
  local build="$tacex/tacex_uipc/build" code=0
  mkdir -p "$build" "$prefix/cache/vcpkg-downloads" "$prefix/cache/vcpkg-binary"
  export CMAKE_TOOLCHAIN_FILE="$vcpkg/scripts/buildsystems/vcpkg.cmake"
  export VCPKG_ROOT="$vcpkg" VCPKG_DOWNLOADS="$prefix/cache/vcpkg-downloads"
  export VCPKG_DEFAULT_BINARY_CACHE="$prefix/cache/vcpkg-binary" VCPKG_MAX_CONCURRENCY="$jobs"
  export VCPKG_KEEP_ENV_VARS='LC_ALL;LANG;LANGUAGE'
  state tinygltf-overlay "$vcpkg" "$prefix/overlay-ports/tinygltf" "$support/tinygltf-source.json"
  export VCPKG_OVERLAY_PORTS="$prefix/overlay-ports"
  "$sim_python" "$tacex/tacex_uipc/libuipc/scripts/gen_vcpkg_json.py" "$build" --build_gui=OFF --dev_mode=1 \
    > "$prefix/logs/uipc-manifest.log" 2>&1 || code=$?
  cat "$prefix/logs/uipc-manifest.log"
  if (( code != 0 )); then
    # The reviewed generator intentionally exits 1 when creating/changing its
    # manifest. Do not confuse an unrelated Python error with that convention.
    [[ "$code" == 1 ]] && grep -Eq 'Generated vcpkg.json at:|vcpkg.json content has changed' "$prefix/logs/uipc-manifest.log" || return "$code"
  fi
  "$sim_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["builtin-baseline"] == sys.argv[2]' "$build/vcpkg.json" "$UIPC_BASELINE"
  # Use native host tools for ports; no Conda runtime libraries on their path.
  (
    unset LD_LIBRARY_PATH CONDA_PREFIX CONDA_DEFAULT_ENV CUDA_HOME CUDA_PATH
    export PATH="$prefix/tools/native/bin:$host_path" VCPKG_FORCE_SYSTEM_BINARIES=1
    "$vcpkg/vcpkg" install --triplet x64-linux --x-manifest-root="$build" \
      --x-install-root="$build/vcpkg_installed" --overlay-ports="$VCPKG_OVERLAY_PORTS" --disable-metrics
  )
  # libuipc generates Python stubs; install explicitly under our constraints
  # instead of allowing CMake to invoke an unconstrained pip download.
  pip_install mypy==2.3.1
  # Stub generation uses Unix sockets, whose paths cannot fit a deep NAS prefix.
  # Keep a short private scratch directory (including on failure for diagnosis).
  local build_tmp
  build_tmp=$(mktemp -d /tmp/simpletouch-uipc.XXXXXX)
  echo "UIPC temporary directory: $build_tmp"
  (
    export TMPDIR="$build_tmp"
    # Omit developer applications, not simulator functionality. The upstream
    # editable build reuses these options without overriding them.
    cmake -S "$tacex/tacex_uipc/libuipc" -B "$build" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" \
      -DUIPC_BUILD_PYBIND=1 -DUIPC_DEV_MODE=1 -DUIPC_BUILD_GUI=0 \
      -DUIPC_BUILD_EXAMPLES=OFF -DUIPC_BUILD_TESTS=OFF -DUIPC_BUILD_BENCHMARKS=OFF \
      -DUIPC_PYTHON_EXECUTABLE_PATH="$sim_python"
    cmake --build "$build" --parallel "$jobs"
    pip_install -e "$tacex/tacex_uipc" --no-build-isolation
  )
}
stage 01-toolchain install_toolchain
export CUDA_HOME="$prefix/env" CUDA_PATH="$prefix/env" CUDA_VISIBLE_DEVICES="$gpu"
export PATH="$prefix/env/bin:$host_path"
export LD_LIBRARY_PATH="$prefix/env/lib:$prefix/env/targets/x86_64-linux/lib"
export TORCH_CUDA_ARCH_LIST="${arch:0:${#arch}-1}.${arch: -1}" CMAKE_CUDA_ARCHITECTURES="$arch"
stage 02-sources install_sources
# Resume must not trust a stage stamp after checkouts have been edited/replaced.
clone_at https://github.com/univtac/UniVTAC.git "$univtac" "$UNIVTAC_COMMIT"
clone_at https://github.com/isaac-sim/IsaacLab.git "$lab" "$ISAACLAB_COMMIT"
clone_at https://github.com/NVlabs/curobo.git "$curobo" "$CUROBO_COMMIT"
clone_at https://github.com/microsoft/vcpkg.git "$vcpkg" "$VCPKG_COMMIT"
state patch "$univtac" "$support/torch-scatter-url.json"
state copy-client "$repo/packages/openpi-client" "$prefix/sources/openpi-client"
[[ $(git -C "$univtac" diff HEAD --name-only) == third_party/TacEx/source/tacex/setup.py ]] || fail 'Unexpected tracked UniVTAC modifications.'
for checkout in "$lab" "$curobo" "$vcpkg"; do
  git -C "$checkout" diff HEAD --quiet || fail "Unexpected tracked changes in $checkout"
done
stage 03-isaac install_isaac
stage 04-isaaclab install_lab
stage 05-curobo install_curobo
stage 06-tacex install_tacex
stage 07-uipc install_uipc
pip_check | tee "$prefix/logs/pip-check-final.log"
"$sim_python" - <<'PY'
import importlib.metadata as metadata
import torch
assert torch.__version__ == '2.5.1+cu124', torch.__version__
assert torch.version.cuda == '12.4', torch.version.cuda
assert metadata.version('torchvision') == '0.20.1+cu124'
assert metadata.version('torch-scatter') == '2.1.2+pt25cu124'
PY
"$sim_python" -m pip freeze > "$prefix/logs/pip-freeze.txt"
"$conda_exe" list --prefix "$prefix/env" --explicit > "$prefix/logs/conda-explicit.txt"
# A wrapper isolates CUDA/runtime exports from the caller's policy environment.
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'prefix=%q\n' "$prefix"
  cat <<'SH'
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME
unset ISAAC_PATH EXP_PATH CARB_APP_PATH
# Evaluation selects a private, verified reference runtime. Keep that selection
# while isolating all unrelated packages from the caller's policy environment.
runtime_univtac=${UNIVTAC_DIR:-$prefix/sources/UniVTAC}
if [[ "$runtime_univtac" != "$prefix/sources/UniVTAC" ]]; then
  [[ "$runtime_univtac" == /* && -f "$runtime_univtac/reference-runtime.json" ]] || {
    echo 'UNIVTAC_DIR override must be a prepared reference runtime.' >&2; exit 2;
  }
fi
export PYTHONPATH="$prefix/sources/openpi-client/src:$prefix/sources/UniVTAC/third_party/TacEx/source/tacex_assets:$runtime_univtac:$prefix/sources/UniVTAC/third_party/curobo/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMNI_KIT_ACCEPT_EULA=YES
export CUDA_HOME="$prefix/env" CUDA_PATH="$prefix/env"
export PATH="$prefix/env/bin:/usr/local/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$prefix/env/lib:$prefix/env/targets/x86_64-linux/lib"
export TMPDIR="$prefix/tmp" XDG_CACHE_HOME="$prefix/cache/runtime"
export XDG_CONFIG_HOME="$prefix/runtime/config" XDG_DATA_HOME="$prefix/runtime/data"
export CUDA_CACHE_PATH="$prefix/cache/cuda" TORCH_EXTENSIONS_DIR="$prefix/cache/torch-extensions"
export NUMBA_CACHE_DIR="$prefix/cache/numba" __GL_SHADER_DISK_CACHE_PATH="$prefix/cache/gl-shader"
# Restore only the caller's explicit graphics setup, never arbitrary policy libs.
if [[ -n "${VULKAN_RUNTIME_ENV:-}" ]]; then
  [[ -f "$VULKAN_RUNTIME_ENV" ]] || { echo "Missing VULKAN_RUNTIME_ENV: $VULKAN_RUNTIME_ENV" >&2; exit 2; }
  source "$VULKAN_RUNTIME_ENV"
fi
exec "$prefix/env/bin/python" "$@"
SH
} > "$prefix/run-simulator-python"
chmod u+x "$prefix/run-simulator-python"
{
  printf 'export UNIVTAC_DIR=%q\n' "$univtac"
  printf 'export UNIVTAC_UPSTREAM_ROOT=%q\n' "$univtac"
  echo 'export UNIVTAC_ENVIRONMENT=reference'
  printf 'export SIMPLETOUCH_RUNTIME_ROOT=%q\n' "$prefix/runtime/univtac-eval"
  printf 'export CUROBO_ROOT=%q\n' "$curobo"
  printf 'export UNIVTAC_PYTHON=%q\n' "$prefix/run-simulator-python"
  echo 'unset CONDA_SH CONDA_ENV'
} > "$prefix/env.sh"
echo "Installation commands completed. Runtime smoke/episode acceptance is NOT implied."
echo "Evaluation paths: source $prefix/env.sh"
