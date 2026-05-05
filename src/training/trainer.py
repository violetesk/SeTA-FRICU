"""
Training utilities and main training loop.

This module provides the core training functionality with distributed
training support via HuggingFace Accelerate.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

from ..data.batch_utils import get_global_max_seq_len, prepare_batch_tensors
from .evaluator import evaluate
from .losses import WeightedMSELoss

logger = get_logger(__name__, log_level="INFO")


class Trainer:
    """Trainer class for clinical risk prediction models.

    Handles the complete training loop including:
    - Distributed training with Accelerate
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - TensorBoard logging
    - K-Fold cross validation support

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test/validation data.
        config: Training configuration dictionary.
        accelerator: HuggingFace Accelerator instance.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        accelerator: Accelerator,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.accelerator = accelerator

        # Extract config values
        train_cfg = config["training"]
        lr_cfg = config["lr_scheduler"]

        self.output_dim = train_cfg["output_dim"]
        self.weight = train_cfg["weig"]
        self.test_step = train_cfg["test_step"]
        self.num_epochs = train_cfg["num_epochs"]
        self.learning_rate = train_cfg["learning_rate"]

        # Initialize optimizer with parameter groups
        self.optimizer = self._create_optimizer(train_cfg)

        # Initialize scheduler
        self.scheduler = self._create_scheduler(train_cfg, lr_cfg)

        # Loss function
        self.loss_fn = WeightedMSELoss(positive_weight=self.weight, reduction="none")

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.scheduler,
        ) = accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.test_loader,
            self.scheduler,
        )

        # Training state
        self.global_step = 0
        self.best_metric = 0.0
        self.last_best_epoch = 0
        self.early_stopping_patience = 50

        # Logging
        self.writer: Optional[SummaryWriter] = None
        self._setup_logging()

        # K-Fold support
        self.fold_idx = os.getenv("KFOLD_INDEX")

    def _create_optimizer(self, train_cfg: Dict) -> torch.optim.Optimizer:
        """Create optimizer with separate learning rates for MCE predictor."""
        mce_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "mce_predictor" in name:
                mce_params.append(param)
            else:
                base_params.append(param)

        base_lr = train_cfg["learning_rate"]
        optim_groups = [
            {"params": base_params, "lr": base_lr},
            {"params": mce_params, "lr": base_lr * 10.0},
        ]

        return torch.optim.AdamW(optim_groups, weight_decay=0.01)

    def _create_scheduler(
        self, train_cfg: Dict, lr_cfg: Dict
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        num_batches = len(self.train_loader)
        num_training_steps = train_cfg["num_epochs"] * num_batches
        warmup_ratio = 0.05
        num_warmup_steps = int(num_training_steps * warmup_ratio)

        return get_scheduler(
            name=lr_cfg["type"],
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def _setup_logging(self) -> None:
        """Setup file and TensorBoard logging."""
        if not self.accelerator.is_main_process:
            return

        log_dir = Path(self.config["checkpoint"]["log_save_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)

        # File logging
        log_file = log_dir / "train.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        formatter = logging.Formatter(
            "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger("root")
        root_logger.handlers = [
            h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)
        ]
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.INFO)

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir = Path(self.config["logging"].get("tensorboard_dir", "runs")) / timestamp
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))

        logger.info(f"TensorBoard logs: {tb_dir}")

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        train_loop = tqdm(
            self.train_loader,
            disable=not self.accelerator.is_main_process,
            desc=f"Epoch {epoch}",
        )

        for batch in train_loop:
            loss = self._train_step(batch)
            total_loss += loss
            num_steps += 1

            # Update progress bar
            if self.accelerator.is_main_process:
                self._update_progress_bar(train_loop, loss)
                self._log_training_step(loss)

            self.global_step += 1

        return total_loss / max(num_steps, 1)

    def _train_step(self, batch: list) -> float:
        """Execute a single training step.

        Args:
            batch: List of patient dictionaries.

        Returns:
            Loss value for this step.
        """
        # Get global sequence length
        final_len = get_global_max_seq_len(batch, self.accelerator)

        # Prepare tensors
        feature_dim = self.model.d_model
        (input_tensor, label_tensor, time_tensor, mask_tensor) = prepare_batch_tensors(
            batch, self.accelerator.device, feature_dim, self.output_dim, final_len
        )

        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(input_tensor, time_tensor)

        # Compute loss
        raw_loss = self.loss_fn(output, label_tensor)
        expanded_mask = mask_tensor.unsqueeze(-1)
        raw_loss = raw_loss.masked_fill(expanded_mask, 0.0)

        total_loss = raw_loss.sum()
        total_valid = (~mask_tensor).sum() * self.output_dim
        mean_loss = total_loss / total_valid

        # Synchronize loss across processes
        global_loss = self.accelerator.reduce(mean_loss, reduction="sum")
        global_loss = global_loss / self.accelerator.num_processes

        # Backward pass
        self.accelerator.backward(mean_loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return global_loss.item()

    def _update_progress_bar(self, pbar: tqdm, loss: float) -> None:
        """Update progress bar with current metrics."""
        device = self.accelerator.device
        mem_alloc = torch.cuda.memory_allocated(device) / (1024**3)
        mem_res = torch.cuda.memory_reserved(device) / (1024**3)

        pbar.set_postfix_str(
            f"loss={loss:.4f} | alloc={mem_alloc:.3f}G | res={mem_res:.3f}G"
        )

    def _log_training_step(self, loss: float) -> None:
        """Log training metrics to TensorBoard."""
        if self.writer is not None:
            self.writer.add_scalar("Loss/train", loss, self.global_step)
            self.writer.add_scalar(
                "LR/train", self.scheduler.get_last_lr()[0], self.global_step
            )

    def evaluate_and_checkpoint(self, epoch: int) -> Tuple[Dict, bool]:
        """Run evaluation and save checkpoint if best.

        Args:
            epoch: Current epoch number.

        Returns:
            Tuple of (metrics_dict, is_best).
        """
        self.accelerator.wait_for_everyone()

        metrics = evaluate(
            self.model,
            self.test_loader,
            self.accelerator,
            self.loss_fn,
            epoch,
            self.weight,
        )

        is_best = False
        if self.accelerator.is_main_process:
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/eval", metrics["loss"], self.global_step)
                self.writer.add_scalar(
                    "AUPRC/micro", metrics.get("micro_auprc", 0), self.global_step
                )
                self.writer.add_scalar(
                    "AUPRC/sample_wise",
                    metrics.get("sample_wise_auprc", 0),
                    self.global_step,
                )

            # Check if best model
            current_metric = metrics.get("micro_auprc", 0)
            if current_metric > self.best_metric:
                logger.info(
                    f"Epoch {epoch}: New best model! AUPRC: {current_metric:.4f}"
                )
                self.best_metric = current_metric
                self.last_best_epoch = epoch
                self._save_checkpoint()
                is_best = True

        self.accelerator.wait_for_everyone()
        return metrics, is_best

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        save_dir = Path(self.config["checkpoint"]["save_dir"]) / "best_model"
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.save(
            unwrapped_model.state_dict(), save_dir / "best_model_weights.pth"
        )

    def check_early_stopping(self, epoch: int) -> bool:
        """Check if early stopping criteria is met.

        Args:
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        stop_training = False

        if self.accelerator.is_main_process:
            patience = epoch - self.last_best_epoch
            logger.info(
                f"Epoch {epoch}: Patience {patience}/{self.early_stopping_patience}"
            )

            if patience >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {self.early_stopping_patience} "
                    "epochs without improvement."
                )
                stop_training = True

        # Broadcast stop signal to all processes
        if self.accelerator.num_processes > 1:
            stop_signal = torch.tensor(
                int(stop_training), device=self.accelerator.device
            )
            dist.broadcast(stop_signal, src=0)
            stop_training = bool(stop_signal.item())

        return stop_training

    def save_fold_results(self) -> None:
        """Save K-Fold cross validation results."""
        if not self.accelerator.is_main_process:
            return
        if self.fold_idx is None:
            return

        result_file = (
            Path(self.config["checkpoint"]["log_save_dir"]) / "fold_result.json"
        )
        with open(result_file, "w") as f:
            json.dump(
                {
                    "fold": int(self.fold_idx),
                    "best_auc": self.best_metric,
                    "config": str(self.config),
                },
                f,
                indent=2,
            )

    def train(self) -> Dict[str, float]:
        """Run the complete training loop.

        Returns:
            Dictionary with final training metrics.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(self.num_epochs):
            # Train one epoch
            train_loss = self.train_epoch(epoch)

            # Evaluate
            if self.global_step > 0:
                metrics, _ = self.evaluate_and_checkpoint(epoch)

            # Check early stopping
            if self.check_early_stopping(epoch):
                break

        # Save fold results if in K-Fold mode
        self.save_fold_results()

        return {"best_metric": self.best_metric, "final_epoch": epoch}


def setup_kfold_config(config: Dict, accelerator: Accelerator) -> Dict:
    """Setup configuration for K-Fold cross validation.

    Modifies config paths based on KFOLD environment variables.

    Args:
        config: Original configuration dictionary.
        accelerator: Accelerator instance.

    Returns:
        Modified configuration dictionary.
    """
    fold_idx = os.getenv("KFOLD_INDEX")

    if fold_idx is None:
        return config

    if accelerator.is_main_process:
        print(f"\n[K-Fold Mode] Detected Fold Index: {fold_idx}")

    # Override data paths
    if os.getenv("OVERRIDE_TRAIN_DIR"):
        config["data"]["train_parquet"] = os.getenv("OVERRIDE_TRAIN_DIR")
    if os.getenv("OVERRIDE_TEST_DIR"):
        config["data"]["test_parquet"] = os.getenv("OVERRIDE_TEST_DIR")

    # Create fold-specific directories
    base_log_dir = Path(config["checkpoint"]["log_save_dir"]) / f"fold_{fold_idx}"
    base_ckpt_dir = Path(config["checkpoint"]["save_dir"]) / f"fold_{fold_idx}"
    base_tb_dir = (
        Path(config["logging"].get("tensorboard_dir", "runs")) / f"fold_{fold_idx}"
    )

    config["checkpoint"]["log_save_dir"] = str(base_log_dir)
    config["checkpoint"]["save_dir"] = str(base_ckpt_dir)
    config["logging"]["tensorboard_dir"] = str(base_tb_dir)

    if accelerator.is_main_process:
        base_log_dir.mkdir(parents=True, exist_ok=True)
        base_ckpt_dir.mkdir(parents=True, exist_ok=True)

    return config
