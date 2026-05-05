#!/usr/bin/env python3
"""
Evaluation script for clinical risk prediction models.

This script provides the main entry point for evaluating trained models
on test datasets with comprehensive metric computation.

Usage:
    python test.py --config config/test.yaml

    # With accelerate for distributed evaluation:
    accelerate launch test.py --config config/test.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ICURiskDataset, simple_collate
from src.models import PatientRiskTransformer
from src.training import evaluate
from src.utils import load_config, load_model_weights


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate risk prediction model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the test config YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--save-results", action="store_true", help="Save predictions for analysis"
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load config
    config = load_config(args.config)

    train_cfg = config["training"]
    data_cfg = config["data"]

    # Create test dataset and loader
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
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = (
            Path(config["checkpoint"]["save_dir"])
            / "best_model"
            / "best_model_weights.pth"
        )

    accelerator.print(f"Loading model from {checkpoint_path}...")

    try:
        load_model_weights(model, str(checkpoint_path))
        accelerator.print("Model weights loaded successfully.")
    except FileNotFoundError:
        accelerator.print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Setup loss function
    loss_fn = nn.MSELoss(reduction="none")

    # Prepare with accelerator
    model, test_loader = accelerator.prepare(model, test_loader)

    # Run evaluation
    weight = train_cfg.get("weig", 100.0)

    results_dir = None
    if args.save_results:
        results_dir = str(Path(config["checkpoint"]["save_dir"]) / "analysis_results")

    with accelerator.autocast():
        metrics = evaluate(
            model=model,
            test_loader=test_loader,
            accelerator=accelerator,
            loss_fn=loss_fn,
            epoch=0,
            weight=weight,
            save_results=args.save_results,
            results_dir=results_dir,
        )

    if accelerator.is_main_process:
        print("\n" + "=" * 50)
        print("Final Evaluation Results")
        print("=" * 50)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
