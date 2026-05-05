#!/usr/bin/env python3
"""
Training script for clinical risk prediction models.

This script provides the main entry point for training transformer-based
models for ICU patient risk prediction with distributed training support.

Usage:
    python train.py --config config/train.yaml

    # With accelerate for distributed training:
    accelerate launch train.py --config config/train.yaml
"""

import argparse
import sys
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ICURiskDataset, simple_collate
from src.models import PatientRiskTransformer
from src.training import Trainer, setup_kfold_config
from src.utils import load_config

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train risk prediction model with Accelerate."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML file",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()
    set_seed(args.seed)

    # Load and setup config
    config = load_config(args.config)
    config = setup_kfold_config(config, accelerator)

    if accelerator.is_main_process:
        logger.info(f"Loaded config from: {args.config}")
        logger.info(f"Accelerator type: {accelerator.distributed_type}")

    # Extract config values
    train_cfg = config["training"]
    data_cfg = config["data"]

    # Create datasets
    train_dataset = ICURiskDataset(data_dir=Path(data_cfg["train_parquet"]))
    test_dataset = ICURiskDataset(data_dir=Path(data_cfg["test_parquet"]))

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_cfg["batch_size"],
        collate_fn=simple_collate,
        num_workers=4,
        shuffle=True,
    )
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

    if accelerator.is_main_process:
        logger.info(f"Model parameters: {model.num_parameters:,}")

    # Create trainer and run training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        accelerator=accelerator,
    )

    results = trainer.train()

    if accelerator.is_main_process:
        logger.info(f"Training completed. Best metric: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()
