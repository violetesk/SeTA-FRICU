#!/usr/bin/env python3
"""
Sample analysis script for clinical risk prediction models.

This script provides detailed analysis of model predictions on individual
patient samples, including attention visualization data extraction.

Usage:
    python sample_analysis.py --config config/test.yaml

    # With accelerate:
    accelerate launch sample_analysis.py --config config/test.yaml
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ICURiskDataset, simple_collate
from src.data.batch_utils import get_global_max_seq_len, prepare_batch_tensors
from src.models import PatientRiskTransformer
from src.utils import load_config, load_model_weights


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze model predictions on sample patients."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plot/plotData",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=0, help="Layer index for attention analysis"
    )
    return parser.parse_args()


@torch.no_grad()
def analyze_sample(
    model: nn.Module,
    test_loader: DataLoader,
    accelerator: Accelerator,
    output_dir: str,
    layer_idx: int = 0,
) -> None:
    """Analyze model on sample patients and save visualization data.

    Args:
        model: The trained model.
        test_loader: DataLoader for test data.
        accelerator: HuggingFace Accelerator instance.
        output_dir: Directory to save results.
        layer_idx: Which layer to analyze attention from.
    """
    model.eval()
    device = accelerator.device

    os.makedirs(output_dir, exist_ok=True)

    test_loop = tqdm(
        test_loader, disable=not accelerator.is_main_process, desc="Analyzing samples"
    )

    for batch_idx, batch in enumerate(test_loop):
        # Get global sequence length
        final_len = get_global_max_seq_len(batch, accelerator)

        feature_dim = model.d_model if hasattr(model, "d_model") else 4096
        output_dim = model.output_dim if hasattr(model, "output_dim") else 1440

        # Prepare batch
        (padded_input, padded_labels, padded_times, padded_mask) = (
            prepare_batch_tensors(batch, device, feature_dim, output_dim, final_len)
        )

        if accelerator.is_main_process:
            try:
                # Take first patient for detailed analysis
                patient_data = batch[0]
                event_ids_list = [
                    e.get("event_id", i) for i, e in enumerate(patient_data["event"])
                ]
                valid_len = len(event_ids_list)

                # Single sample input
                sample_input = padded_input[0:1, :valid_len, :]
                sample_time = padded_times[0:1, :valid_len]

                # Forward pass
                outputs = model(sample_input, sample_time)

                # Extract attention data
                unwrapped_model = accelerator.unwrap_model(model)
                target_layer = unwrapped_model.layers[layer_idx]
                attn_module = target_layer.attn

                attn_weights = attn_module.last_attn_weights

                if attn_weights is not None:
                    attn_weights_np = attn_weights[0].float().cpu().numpy()

                    # Get slope and peak statistics
                    head_slopes = []
                    head_peaks = []

                    if (
                        hasattr(attn_module, "cached_slope")
                        and attn_module.cached_slope is not None
                    ):
                        slope_tensor = attn_module.cached_slope.float().cpu()
                        head_slopes = slope_tensor.mean(dim=(0, 2, 3)).tolist()
                    else:
                        head_slopes = [0.0] * attn_weights_np.shape[0]

                    if (
                        hasattr(attn_module, "cached_peak")
                        and attn_module.cached_peak is not None
                    ):
                        peak_tensor = attn_module.cached_peak.float().cpu()
                        head_peaks = peak_tensor.mean(dim=(0, 2, 3)).tolist()
                    else:
                        head_peaks = [0.0] * attn_weights_np.shape[0]

                    # Save attention data
                    attn_package = {
                        "patient_id": patient_data.get("patient_id", "unknown"),
                        "event_ids": event_ids_list,
                        "layer_idx": layer_idx,
                        "attn_weights": attn_weights_np,
                        "head_slopes": head_slopes,
                        "head_peaks": head_peaks,
                    }

                    attn_path = os.path.join(output_dir, "plot_attention_hot.pkl")
                    with open(attn_path, "wb") as f:
                        pickle.dump(attn_package, f)

                    # Save predictions
                    predictions_np = outputs[0].float().cpu().numpy()

                    pred_package = {
                        "patient_id": patient_data.get("patient_id", "unknown"),
                        "event_ids": event_ids_list,
                        "predictions": predictions_np,
                    }

                    pred_path = os.path.join(output_dir, "plot_prediction_data.pkl")
                    with open(pred_path, "wb") as f:
                        pickle.dump(pred_package, f)

                    accelerator.print(f"✅ Analysis data saved to {output_dir}")
                    accelerator.print(f"   - Attention data: {attn_path}")
                    accelerator.print(f"   - Prediction data: {pred_path}")

            except Exception as e:
                accelerator.print(f"⚠️ Failed to save analysis data: {e}")
                import traceback

                traceback.print_exc()

        # Only analyze first batch
        break


def main():
    """Main analysis function."""
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load config
    config = load_config(args.config)

    train_cfg = config["training"]
    data_cfg = config["data"]

    # Create test dataset
    test_dataset = ICURiskDataset(data_dir=Path(data_cfg["test_parquet"]))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_cfg["batch_size"],
        collate_fn=simple_collate,
        num_workers=4,
        shuffle=False,
    )

    # Create model
    model = PatientRiskTransformer(
        d_model=train_cfg["d_model"],
        output_dim=train_cfg["output_dim"],
        nhead=train_cfg["nhead"],
        num_layers=train_cfg["num_layers"],
    )

    # Load checkpoint
    checkpoint_path = (
        Path(config["checkpoint"]["save_dir"]) / "best_model" / "best_model_weights.pth"
    )

    accelerator.print(f"Loading model from {checkpoint_path}...")

    try:
        load_model_weights(model, str(checkpoint_path))
        accelerator.print("Model weights loaded successfully.")
    except FileNotFoundError:
        accelerator.print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Prepare with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)

    # Run analysis
    with accelerator.autocast():
        analyze_sample(
            model=model,
            test_loader=test_loader,
            accelerator=accelerator,
            output_dir=args.output_dir,
            layer_idx=args.layer_idx,
        )

    accelerator.print("Sample analysis completed.")


if __name__ == "__main__":
    main()
