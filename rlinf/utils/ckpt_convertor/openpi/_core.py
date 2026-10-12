# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load native SimpleTouch state dictionaries and remove wrapper prefixes."""

from __future__ import annotations

import pathlib
import shutil
import json
from typing import Any, Mapping

import torch

# Wrapper / FSDP prefixes that may sit in front of the bare ``Pi0`` keys in a
# trained checkpoint. ``OpenPiPytorchActionModel`` adds ``model.`` (the vendored
# Pi0 lives at ``wrapper.model``); FSDP/compile may add the others. No bare Pi0
# key begins with any of these, so stripping them is safe.
_WRAPPER_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
    "model.",
)


def strip_wrapper_prefix(
    state_dict: Mapping[str, torch.Tensor],
    *,
    cast_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Drop wrapper/FSDP key prefixes, returning bare ``Pi0`` keys.

    Removes any leading combination of the known wrapper/FSDP prefixes from each
    key. When ``cast_dtype`` is given, floating-point tensors are cast to it (the
    new-format eval loader validates that every checkpoint tensor is bf16);
    integer/bool buffers are passed through. Two distinct source keys must never
    collapse to the same bare key, or a tensor would be silently dropped, so that
    raises instead.
    """
    bare: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        bare_key = key
        while True:
            for prefix in _WRAPPER_PREFIXES:
                if bare_key.startswith(prefix):
                    bare_key = bare_key[len(prefix) :]
                    break
            else:
                break
        if cast_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(cast_dtype)
        if bare_key in bare:
            raise ValueError(
                f"duplicate bare key {bare_key!r} after prefix strip "
                "(two checkpoint keys collapsed to one); refusing to drop a tensor."
            )
        bare[bare_key] = tensor.detach().cpu().contiguous()
    return bare


def as_state_dict(loaded: Any) -> Mapping[str, torch.Tensor]:
    """Unwrap common checkpoint containers down to a key->tensor mapping."""
    obj = loaded
    for _ in range(4):
        if (
            isinstance(obj, Mapping)
            and obj
            and all(isinstance(v, torch.Tensor) for v in obj.values())
        ):
            return obj
        if isinstance(obj, Mapping):
            for wrapper_key in ("model", "state_dict", "module"):
                if wrapper_key in obj and isinstance(obj[wrapper_key], Mapping):
                    obj = obj[wrapper_key]
                    break
            else:
                break
        else:
            break
    if isinstance(obj, Mapping) and obj:
        return obj
    raise TypeError(
        f"Could not extract a state dict from the checkpoint (got {type(loaded)!r})."
    )


def save_safetensors(
    state_dict: Mapping[str, torch.Tensor], path: str | pathlib.Path
) -> None:
    """Save a state dict to a safetensors file, creating parent dirs as needed."""
    import safetensors.torch

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(dict(state_dict), str(path))


def write_config_json(
    config: Mapping[str, Any], output_model: str | pathlib.Path
) -> None:
    """Write ``config.json`` into the output checkpoint directory."""
    output_model = pathlib.Path(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    (output_model / "config.json").write_text(json.dumps(dict(config), indent=2) + "\n")


def copy_norm_stats(src: str | pathlib.Path, dst: str | pathlib.Path) -> None:
    """Copy the norm-stats file from ``src`` to ``dst`` verbatim (straight copy).

    This is the single definition shared by every mode (the old per-script
    duplicates in ``old_to_new`` and ``sft_to_new_pytorch`` are folded here).
    """
    src = pathlib.Path(src)
    dst = pathlib.Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"input norm stats not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
