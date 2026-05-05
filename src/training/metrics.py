"""
Metrics computation for clinical risk prediction evaluation.

This module provides functions for computing various evaluation metrics
including AUPRC, AUROC, F1, and Precision@K.
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_micro_auprc(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute micro-averaged AUPRC across all predictions.

    Args:
        predictions: Predicted probabilities, flattened or 2D.
        targets: Ground truth labels (will be binarized).
        threshold: Threshold for binarizing targets.

    Returns:
        Micro-averaged AUPRC score.
    """
    flat_preds = predictions.ravel().astype(np.float32)
    flat_targets = (targets.ravel() > threshold).astype(int)

    if np.sum(flat_targets) == 0:
        return 0.0

    try:
        return average_precision_score(flat_targets, flat_preds)
    except Exception:
        return 0.0


def compute_auroc(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute AUROC score.

    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Threshold for binarizing targets.

    Returns:
        AUROC score, or 0.0 if computation fails.
    """
    flat_preds = predictions.ravel().astype(np.float32)
    flat_targets = (targets.ravel() > threshold).astype(int)

    if len(np.unique(flat_targets)) < 2:
        return 0.0

    try:
        return roc_auc_score(flat_targets, flat_preds)
    except Exception:
        return 0.0


def compute_micro_f1(
    predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5
) -> float:
    """Compute micro-averaged F1 score.

    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Threshold for binarizing both predictions and targets.

    Returns:
        Micro-averaged F1 score.
    """
    flat_preds = (predictions.ravel() > threshold).astype(int)
    flat_targets = (targets.ravel() > threshold).astype(int)

    try:
        return f1_score(flat_targets, flat_preds, average="micro")
    except Exception:
        return 0.0


def compute_precision_at_k(
    predictions: np.ndarray,
    targets: np.ndarray,
    k_values: List[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Precision@K for multiple k values.

    For each sample, selects the top-k predictions and computes
    the precision (fraction of relevant items in top-k).

    Args:
        predictions: Predicted probabilities of shape [num_samples, num_classes].
        targets: Ground truth labels of same shape.
        k_values: List of k values to compute precision for.
        threshold: Threshold for determining relevance.

    Returns:
        Dictionary mapping 'p@k' to precision values.
    """
    results = {}

    try:
        for k in k_values:
            if predictions.shape[1] < k:
                results[f"p@{k}"] = 0.0
                continue

            # Get indices of top-k predictions for each sample
            top_k_indices = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]

            # Get the corresponding targets
            rows = np.arange(predictions.shape[0])[:, None]
            top_k_targets = targets[rows, top_k_indices]

            # Count relevant items in top-k
            relevant_count = (top_k_targets > threshold).sum(axis=1)
            results[f"p@{k}"] = np.mean(relevant_count / k)

    except Exception:
        for k in k_values:
            results[f"p@{k}"] = 0.0

    return results


def compute_sample_wise_auprc(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> tuple:
    """Compute per-patient AUPRC scores.

    Args:
        predictions: Predictions of shape [batch, seq, dim].
        targets: Targets of shape [batch, seq, dim].
        mask: Padding mask of shape [batch, seq] (True = padding).
        threshold: Threshold for binarizing targets.

    Returns:
        Tuple of (mean_auprc, list_of_patient_scores).
    """
    patient_scores = []
    batch_size = predictions.shape[0]

    for i in range(batch_size):
        valid_mask = ~mask[i]
        if not np.any(valid_mask):
            continue

        p_preds = predictions[i][valid_mask]
        p_labels = targets[i][valid_mask]

        flat_preds = p_preds.ravel()
        flat_labels = (p_labels > threshold).astype(int).ravel()

        if np.sum(flat_labels) > 0:
            try:
                score = average_precision_score(flat_labels, flat_preds)
                patient_scores.append(score)
            except Exception:
                pass

    mean_score = np.mean(patient_scores) if patient_scores else 0.0
    return mean_score, patient_scores


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        predictions: Predicted probabilities.
        targets: Ground truth labels.
        threshold: Threshold for binarization.
        k_values: List of k values for Precision@K.

    Returns:
        Dictionary containing all computed metrics.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    metrics = {
        "micro_auprc": compute_micro_auprc(predictions, targets, threshold),
        "auroc": compute_auroc(predictions, targets, threshold),
        "micro_f1": compute_micro_f1(predictions, targets, threshold),
    }

    # Add Precision@K if predictions are 2D
    if predictions.ndim >= 2:
        pk_metrics = compute_precision_at_k(predictions, targets, k_values, threshold)
        metrics.update(pk_metrics)

    return metrics
