#!/usr/bin/env bash
# Prepare a process-local NVIDIA Vulkan/EGL runtime on Debian/Ubuntu containers.
# Never install packages system-wide or change the host NVIDIA driver.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime=${SIMPLETOUCH_VULKAN_ROOT:-$repo/.runtime/vulkan}
libs=${NVIDIA_LIB_DIR:-/usr/lib/x86_64-linux-gnu}
for command in nvidia-smi vulkaninfo timeout; do
  command -v "$command" >/dev/null || { echo "Required: $command" >&2; exit 2; }
done
version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'Cannot determine NVIDIA driver version' >&2; exit 2; }
mkdir -p "$runtime"
runtime=$(cd "$runtime" && pwd)
mkdir -p "$runtime/xdg"
chmod 700 "$runtime/xdg"
root="$runtime/$version"
driver="$root/driver-lib"
egl="$root/egl-root/usr/lib/x86_64-linux-gnu"
mkdir -p "$driver" "$egl" "$root/packages"
for entry in GLX:0 EGL:0 GLESv1_CM:1 GLESv2:2; do
  name=${entry%:*}
  suffix=${entry#*:}
  source_library="$libs/lib${name}_nvidia.so.$version"
  [[ -f "$source_library" ]] || { echo "Missing host-matched library: $source_library" >&2; exit 2; }
  ln -sfn "$source_library" "$driver/lib${name}_nvidia.so.$suffix"
done
if [[ ! -f "$egl/libEGL.so.1" || ! -f "$egl/libXt.so.6" ]]; then
  command -v apt-get >/dev/null && command -v dpkg-deb >/dev/null || {
    echo 'This helper requires Debian/Ubuntu apt-get and dpkg-deb.' >&2; exit 2;
  }
  # Isaac Sim's MaterialX libraries also require Xt in headless mode.
  (cd "$root/packages" && apt-get download libegl1 libegl-mesa0 libxt6)
  while IFS= read -r -d '' package; do
    dpkg-deb -x "$package" "$root/egl-root"
  done < <(find "$root/packages" -maxdepth 1 -type f -name '*.deb' -print0)
fi
[[ -f "$egl/libEGL.so.1" ]] || { echo 'Missing extracted libEGL.so.1' >&2; exit 2; }
[[ -f "$egl/libXt.so.6" ]] || { echo 'Missing extracted libXt.so.6' >&2; exit 2; }
printf '%s\n' '{"file_format_version":"1.0.1","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' > "$root/nvidia_icd.json"
printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$root/10_nvidia.json"
{
  echo '# Source this file before launching evaluation.'
  printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n' "$egl:$driver"
  printf 'export VK_ICD_FILENAMES=%q\n' "$root/nvidia_icd.json"
  printf 'export __EGL_VENDOR_LIBRARY_FILENAMES=%q\n' "$root/10_nvidia.json"
  printf 'if [[ -z ${XDG_RUNTIME_DIR:-} ]]; then export XDG_RUNTIME_DIR=%q; fi\n' "$runtime/xdg"
} > "$runtime/env.sh"
source "$runtime/env.sh"
ldd "$egl/libXt.so.6" > "$runtime/libxt-dependencies.log"
if grep -q 'not found' "$runtime/libxt-dependencies.log"; then
  cat "$runtime/libxt-dependencies.log" >&2
  echo 'Install the missing X11 runtime dependencies before starting Isaac Sim.' >&2
  exit 2
fi
if ! timeout 30 vulkaninfo --summary > "$runtime/vulkaninfo.log" 2>&1; then
  tail -60 "$runtime/vulkaninfo.log" >&2; exit 1
fi
grep -q 'deviceName[[:space:]]*=[[:space:]]*NVIDIA' "$runtime/vulkaninfo.log" || {
  echo "NVIDIA GPU not found; inspect $runtime/vulkaninfo.log" >&2; exit 1;
}
grep -q "driverInfo[[:space:]]*=[[:space:]]*$version" "$runtime/vulkaninfo.log" || {
  echo 'Loaded Vulkan driver does not match the host' >&2; exit 1;
}
echo "NVIDIA Vulkan ready. Run: source '$runtime/env.sh'"
