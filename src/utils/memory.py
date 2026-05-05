"""
Memory inspection and debugging utilities.
"""

import torch


def inspect_memory(step_name: str, device: torch.device = None) -> dict:
    """Inspect GPU memory usage.

    Args:
        step_name: Name/description of the current step (for logging).
        device: CUDA device to inspect. Uses current device if None.

    Returns:
        Dictionary with memory statistics in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "peak": 0.0, "reserved": 0.0}

    torch.cuda.synchronize()

    if device is None:
        device = torch.cuda.current_device()

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3

    stats = {
        "allocated": allocated,
        "peak": peak,
        "reserved": reserved,
    }

    print(
        f"[{step_name}] "
        f"Current: {allocated:.2f}GB | "
        f"Peak: {peak:.2f}GB | "
        f"Reserved: {reserved:.2f}GB"
    )

    return stats


def reset_peak_memory_stats(device: torch.device = None) -> None:
    """Reset peak memory statistics.

    Args:
        device: CUDA device. Uses current device if None.
    """
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)


def empty_cache() -> None:
    """Empty CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
