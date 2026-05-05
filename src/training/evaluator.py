"""
Evaluation utilities for clinical risk prediction models.

This module provides the evaluation loop with distributed training support
via HuggingFace Accelerate.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from tqdm import tqdm

from ..data.batch_utils import get_global_max_seq_len, prepare_batch_tensors
from .metrics import compute_all_metrics, compute_sample_wise_auprc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    loss_fn: nn.Module,
    epoch: int,
    weight: float,
    k_values: Optional[List[int]] = None,
    save_results: bool = False,
    results_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate model on test set with distributed support.

    This function handles:
    - Variable-length sequence padding with global synchronization
    - Distributed metric gathering
    - Optional result saving for analysis

    Args:
        model: The model to evaluate.
        test_loader: DataLoader for test data.
        accelerator: HuggingFace Accelerator instance.
        loss_fn: Loss function for evaluation.
        epoch: Current epoch number (for logging).
        weight: Weight for positive samples in loss computation.
        k_values: List of k values for Precision@K computation.
        save_results: Whether to save predictions for analysis.
        results_dir: Directory to save results.

    Returns:
        Dictionary containing evaluation metrics.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    model.eval()
    threshold = 0.5

    total_epoch_loss = 0.0
    total_epoch_steps = 0
    device = accelerator.device

    # Containers for collecting predictions
    all_micro_preds = []
    all_micro_targets = []
    all_patient_scores_with_ids = []
    batch_patient_scores = []
    collected_params = []

    test_loop = tqdm(
        test_loader,
        disable=not accelerator.is_main_process,
        desc=f"Epoch {epoch} [Eval]",
    )

    for batch in test_loop:
        # Get global max sequence length across all processes
        final_len = get_global_max_seq_len(batch, accelerator)

        batch_size = len(batch)
        feature_dim = model.d_model if hasattr(model, "d_model") else 4096
        output_dim = model.output_dim if hasattr(model, "output_dim") else 1440

        # Prepare batch tensors
        (padded_input, padded_labels, padded_times, padded_mask) = (
            prepare_batch_tensors(batch, device, feature_dim, output_dim, final_len)
        )

        # Forward pass
        output = model(padded_input, padded_times)

        # Collect MCE parameters for analysis (main process only)
        if accelerator.is_main_process:
            _collect_mce_params(model, accelerator, padded_mask, collected_params)

        # Compute loss
        raw_loss = loss_fn(output, padded_labels)
        expanded_mask = padded_mask.unsqueeze(-1)
        raw_loss = raw_loss.masked_fill(expanded_mask, 0.0)

        total_loss = raw_loss.sum()
        total_valid_elements = (~padded_mask).sum() * output_dim

        if total_valid_elements > 0:
            mean_loss = total_loss / total_valid_elements
        else:
            mean_loss = torch.tensor(0.0, device=device)

        global_mean_loss = accelerator.reduce(mean_loss, reduction="sum")
        global_mean_loss = global_mean_loss / accelerator.num_processes

        total_epoch_loss += global_mean_loss.item()
        total_epoch_steps += 1

        # Gather predictions across processes
        preds = torch.clamp(output, 0.0, 1.0).float()
        labels = padded_labels.float()
        local_patient_ids = [p.get("patient_id", i) for i, p in enumerate(batch)]

        preds_gathered = accelerator.gather_for_metrics(preds)
        labels_gathered = accelerator.gather_for_metrics(labels)
        mask_gathered = accelerator.gather_for_metrics(padded_mask)
        ids_gathered = accelerator.gather_for_metrics(local_patient_ids)

        if accelerator.is_main_process:
            preds_np = preds_gathered.detach().cpu().numpy()
            labels_np = labels_gathered.detach().cpu().numpy()
            mask_np = mask_gathered.detach().cpu().numpy()
            ids_np = np.array(ids_gathered)

            # Store valid predictions for micro metrics
            valid_indices = ~mask_np
            all_micro_preds.append(preds_np[valid_indices])
            all_micro_targets.append(labels_np[valid_indices])

            # Compute per-patient scores
            _compute_patient_scores(
                preds_np,
                labels_np,
                mask_np,
                ids_np,
                threshold,
                batch_patient_scores,
                all_patient_scores_with_ids,
            )

        if accelerator.is_main_process:
            test_loop.set_postfix_str(f"loss={global_mean_loss.item():.4f}")

        # Clean up
        del preds_gathered, labels_gathered, mask_gathered

    # Compute final metrics
    avg_epoch_loss = total_epoch_loss / max(total_epoch_steps, 1)
    metrics = {"loss": avg_epoch_loss}

    if accelerator.is_main_process:
        accelerator.print("Aggregating results...")

        # Sample-wise AUPRC
        if batch_patient_scores:
            metrics["sample_wise_auprc"] = np.mean(batch_patient_scores)
        else:
            metrics["sample_wise_auprc"] = 0.0

        # Compute micro metrics
        if all_micro_preds:
            final_preds = np.concatenate(all_micro_preds, axis=0)
            final_targets = np.concatenate(all_micro_targets, axis=0)

            micro_metrics = compute_all_metrics(
                final_preds.reshape(-1, output_dim)
                if final_preds.ndim == 1
                else final_preds,
                final_targets.reshape(-1, output_dim)
                if final_targets.ndim == 1
                else final_targets,
                threshold,
                k_values,
            )
            metrics.update(micro_metrics)

            # Save results if requested
            if save_results and results_dir:
                _save_evaluation_results(
                    results_dir,
                    epoch,
                    final_preds,
                    final_targets,
                    collected_params,
                    all_patient_scores_with_ids,
                )
        else:
            metrics.update({"micro_auprc": 0.0, "auroc": 0.0, "micro_f1": 0.0})
            for k in k_values:
                metrics[f"p@{k}"] = 0.0

        # Print results
        _print_metrics(
            accelerator, epoch, metrics, k_values, all_patient_scores_with_ids
        )

    accelerator.wait_for_everyone()
    model.train()

    return metrics


def _collect_mce_params(
    model: nn.Module,
    accelerator: Accelerator,
    mask: torch.Tensor,
    collected_params: List[Dict],
) -> None:
    """Collect MCE slope and peak parameters for analysis."""
    unwrapped_model = accelerator.unwrap_model(model)
    layers = getattr(unwrapped_model, "layers", [])

    valid_mask_expanded = ~mask.unsqueeze(1).unsqueeze(-1)
    mask_np = valid_mask_expanded.cpu().numpy()

    for layer_idx, layer_module in enumerate(layers):
        attn_module = getattr(layer_module, "attn", None)
        if attn_module is None:
            continue

        cached_slope = getattr(attn_module, "cached_slope", None)
        cached_peak = getattr(attn_module, "cached_peak", None)

        if cached_slope is None or cached_peak is None:
            continue

        slope_np = cached_slope.float().cpu().numpy()
        peak_np = cached_peak.float().cpu().numpy()

        _, num_heads, _, _ = slope_np.shape

        for h in range(num_heads):
            head_mask = mask_np[:, 0, :, 0].astype(bool)
            valid_slopes = slope_np[:, h, :, 0][head_mask]
            valid_peaks = peak_np[:, h, :, 0][head_mask]

            for s, p in zip(valid_slopes, valid_peaks):
                collected_params.append(
                    {"Layer": layer_idx, "Head": f"Head {h}", "Slope": s, "Peak": p}
                )


def _compute_patient_scores(
    preds_np: np.ndarray,
    labels_np: np.ndarray,
    mask_np: np.ndarray,
    ids_np: np.ndarray,
    threshold: float,
    batch_scores: List[float],
    scores_with_ids: List[Dict],
) -> None:
    """Compute per-patient AUPRC scores."""
    from sklearn.metrics import average_precision_score

    batch_size = preds_np.shape[0]
    for i in range(batch_size):
        valid_mask = ~mask_np[i]
        if not np.any(valid_mask):
            continue

        p_preds = preds_np[i][valid_mask].ravel()
        p_labels = (labels_np[i][valid_mask] > threshold).astype(int).ravel()

        if np.sum(p_labels) > 0:
            try:
                score = average_precision_score(p_labels, p_preds)
                batch_scores.append(score)
                scores_with_ids.append({"patient_id": ids_np[i], "score": score})
            except Exception:
                pass


def _save_evaluation_results(
    results_dir: str,
    epoch: int,
    preds: np.ndarray,
    targets: np.ndarray,
    collected_params: List[Dict],
    patient_scores: List[Dict],
) -> None:
    """Save evaluation results for later analysis."""
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions
    np.savez_compressed(
        os.path.join(results_dir, f"results_epoch_{epoch}.npz"),
        preds=preds,
        targets=targets,
    )

    # Save MCE parameters
    if collected_params:
        pkl_path = os.path.join(results_dir, f"mce_params_epoch_{epoch}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "epoch": epoch,
                    "collected_params": collected_params,
                    "patient_scores": patient_scores,
                },
                f,
            )


def _print_metrics(
    accelerator: Accelerator,
    epoch: int,
    metrics: Dict[str, float],
    k_values: List[int],
    patient_scores: List[Dict],
) -> None:
    """Print evaluation metrics."""
    accelerator.print(f"\n{'=' * 50}")
    accelerator.print(f"Evaluation Results - Epoch {epoch}")
    accelerator.print(f"{'=' * 50}")
    accelerator.print(f"Loss:              {metrics['loss']:.5f}")
    accelerator.print(f"Sample-wise AUPRC: {metrics.get('sample_wise_auprc', 0):.4f}")
    accelerator.print(f"Micro AUPRC:       {metrics.get('micro_auprc', 0):.4f}")
    accelerator.print(f"AUROC:             {metrics.get('auroc', 0):.4f}")
    accelerator.print(f"Micro F1:          {metrics.get('micro_f1', 0):.4f}")

    for k in k_values:
        accelerator.print(f"P@{k}:              {metrics.get(f'p@{k}', 0):.4f}")

    if patient_scores:
        accelerator.print(f"\nTop 10 Patients by AUPRC:")
        sorted_patients = sorted(patient_scores, key=lambda x: x["score"], reverse=True)
        for rank, p in enumerate(sorted_patients[:10], 1):
            accelerator.print(f"  {rank}. Patient {p['patient_id']}: {p['score']:.4f}")

    accelerator.print(f"{'=' * 50}\n")
