"""
Checkpoint saving and loading utilities.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator


def save_checkpoint(
    model: nn.Module,
    accelerator: Accelerator,
    save_dir: str,
    filename: str = "best_model_weights.pth",
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Save model checkpoint.

    Args:
        model: Model to save.
        accelerator: Accelerator instance.
        save_dir: Directory to save checkpoint.
        filename: Name of checkpoint file.
        optimizer: Optional optimizer to save.
        scheduler: Optional scheduler to save.
        epoch: Optional epoch number.
        metrics: Optional metrics dictionary.

    Returns:
        Path to saved checkpoint.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)

    checkpoint = {
        "model_state_dict": unwrapped_model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    checkpoint_path = save_path / filename
    accelerator.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        map_location: Device to map checkpoint to.

    Returns:
        Dictionary with checkpoint metadata (epoch, metrics, etc.).
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Handle both full checkpoint and state_dict only
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        return {}

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
    }


def load_model_weights(
    model: nn.Module, weights_path: str, map_location: str = "cpu"
) -> None:
    """Load model weights from a file.

    Simpler version of load_checkpoint that only loads model weights.

    Args:
        model: Model to load weights into.
        weights_path: Path to weights file.
        map_location: Device to map weights to.
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=map_location)

    # Handle both full checkpoint and state_dict only
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
