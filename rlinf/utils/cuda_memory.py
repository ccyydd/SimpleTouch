"""Host-side CUDA/ROCm allocator counters, without synchronization or allocation."""

import torch


def begin_memory_measurement() -> dict[str, int] | None:
    """Reset only peak counters on the worker's already initialized device."""
    if not torch.cuda.is_initialized():
        return None
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.memory_stats()


def memory_metrics(before: dict[str, int] | None) -> dict[str, float]:
    """Read local per-step peaks and retries; callers may average across ranks.

    These are allocator numbers, not total nvidia-smi device consumption.
    They include diagnostic work when diagnostics run within the measured step.
    """
    if before is None:
        return {}
    after = torch.cuda.memory_stats()
    metrics = {
        f"gpu_memory_{kind}_{stat}_gib": after.get(f"{kind}_bytes.all.{stat}", 0)
        / 2**30
        for kind in ("allocated", "reserved")
        for stat in ("current", "peak")
    }
    for key in ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free"):
        metrics[f"gpu_memory_{key}"] = float(after.get(key, 0) - before.get(key, 0))
    return metrics
