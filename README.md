# SeTA: Semantic-Temporal Alignment for ICU Risk Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

SeTA (**Se**mantic-**T**emporal **A**lignment) is a transformer-based framework for predicting clinical risks in ICU patients from temporal electronic health record (EHR) data. It introduces **MCE-Aware Time-ALiBi** — a multi-head attention mechanism that learns to modulate attention weights based on the temporal distance between clinical events, allowing the model to discover which time horizons matter most for each prediction task.

The framework ingests 4096-dimensional semantic embeddings of clinical events (produced by a Qwen3-Embedding-8B encoder), respects the irregular time intervals inherent to EHR data, and outputs 1440-dimensional risk predictions covering a 24-hour horizon at minute-level resolution.

---

## Model Architecture

The core model is `PatientRiskTransformer`, a pre-norm encoder stack built from `TransformerBlock` layers:

```
Input: [B, S, 4096] event embeddings + [B, S] timestamps
  └─ TransformerBlock ×6
       ├─ RMSNorm → MCE-Aware Time-ALiBi Attention → Residual
       └─ RMSNorm → SwiGLU FFN → Residual
  └─ RMSNorm → Linear(4096 → 1440)
Output: [B, S, 1440] risk predictions
```

### MCE-Aware Time-ALiBi Attention

Standard positional encodings treat sequence positions uniformly. Clinical time series don't work that way — the relevance of a past lab result or vital sign depends on *when* it occurred relative to the prediction point, and that relationship varies across attention heads and clinical contexts.

MCE-Aware Time-ALiBi addresses this by learning per-head, per-query temporal attention profiles:

1. **Time distance matrix** — Compute pairwise absolute time differences and map to log-space: `log(|Δt|/60 + 1)`
2. **MCE predictor** — A small MLP (`head_dim → 64 → 2`) predicts `[slope_scale, peak_offset]` from each query vector
3. **Dynamic parameters** — Combine predicted deltas with learned static priors:
   - `slope = static_slope × exp(Δslope)`, clamped to `[1e-4, 2.5]`
   - `peak = sigmoid(static_peak_logit + Δpeak × 4) × 10`
4. **Tent bias** — Apply a tent function centered at `peak` with decay rate `slope`:
   `bias = −slope × |log_time_diff − peak|`
5. **Attention** — Add bias to standard scaled dot-product scores before softmax

The result: each head learns a different temporal receptive field — some attend to recent events (low peak), others to events at specific time lags (high peak) — and the MCE predictor lets these profiles adapt per-query rather than remaining fixed.

### Other Components

| Component | Description |
|---|---|
| **RMSNorm** | Root-mean-square normalization (faster than LayerNorm, no mean centering) |
| **SwiGLU FFN** | `SiLU(W₁x) ⊙ W₂x` gated activation with dimension 11008 |
| **Gradient checkpointing** | Activated during training to reduce peak GPU memory |
| **Causal + padding mask** | Prevents attending to future positions and padding tokens |

---

## Installation

```bash
git clone https://github.com/seta-project/SeTA.git
cd SeTA

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Dependencies

| Package | Version |
|---|---|
| `torch` | ≥ 2.0 |
| `accelerate` | ≥ 1.0 |
| `transformers` | ≥ 4.40 |
| `deepspeed` | ≥ 0.14 |
| `scikit-learn` | ≥ 1.3 |
| `pandas` | ≥ 2.0 |
| `pyarrow` | ≥ 14.0 |
| `numpy` | ≥ 1.24 |
| `pyyaml` | ≥ 6.0 |
| `tqdm` | ≥ 4.60 |
| `tensorboard` | ≥ 2.14 |
| `einops` | ≥ 0.7 |
| `flash-attn` | ≥ 2.5 |

> **Note:** `flash-attn` requires CUDA-compatible hardware. For CPU-only environments, omit it from the installation.

---

## Data Preparation

SeTA includes a 5-step preprocessing pipeline for MIMIC-IV data. Each step is a standalone script under `preprocessing/`:

| Step | Script | Description |
|---|---|---|
| 1 | `csv_to_json.py` | Convert raw CSV events → per-patient JSON sequences (narrative format) |
| 2 | `filter_before_anchor.py` | Truncate each patient's timeline to events before the prediction anchor |
| 3 | `text_cut.py` | Split long event texts; filter patients below minimum event count |
| 4 | `data_embed_parallel.py` | Multi-GPU embedding generation (Qwen3-Embedding-8B) |
| 5 | `split_test_train.py` | Stratified 8:1:1 train/val/test split |

### Step-by-step

```bash
# Step 1: CSV → JSON
python -m preprocessing.csv_to_json \
    --input-csv data/mimic_events.csv \
    --output-dir data/json/ \
    --n-jobs 100

# Step 2: Filter before anchor
python -m preprocessing.filter_before_anchor \
    --input-dir data/json/ \
    --output-dir data/json_filtered/ \
    --max-workers 8

# Step 3: Truncate and filter
python -m preprocessing.text_cut \
    --target-folder data/json_filtered/ \
    --max-text-length 5000 \
    --min-events 50 \
    --max-events 512

# Step 4: Generate embeddings (multi-GPU)
python -m preprocessing.data_embed_parallel \
    --input-dir data/json_filtered/ \
    --output-dir data/embedded/

# Step 5: Train/val/test split
python -m preprocessing.split_test_train \
    --src-pos data/embedded/positive/ \
    --src-neg data/embedded/negative/ \
    --target-train data/train/ \
    --target-val data/val/ \
    --target-test data/test/
```

After preprocessing, each patient is stored as a Parquet file containing embedded events, labels, and timestamps — ready for the `ICURiskDataset` loader.

---

## Training

### Single GPU

```bash
python train.py --config config/train.yaml
```

### Multi-GPU (Accelerate + DeepSpeed)

```bash
bash scripts/train.sh config/train.yaml
```

The launch script auto-detects your virtual environment and Accelerate configuration. It looks for an Accelerate config in this order:

1. `{config_name}_dp.yaml` — co-located with your training config
2. `config/accelerate_config.yaml` — project-level default
3. `~/.cache/huggingface/accelerate/default_config.yaml` — system default

The included `config/accelerate_config.yaml` configures DeepSpeed ZeRO Stage 2 with BF16 mixed precision.

### Training Features

- **Distributed training** via HuggingFace Accelerate + DeepSpeed
- **Differential learning rates** — MCE predictor parameters use 10× the base LR
- **Cosine LR schedule** with 5% warmup
- **Weighted MSE loss** to handle class imbalance in clinical outcomes
- **Early stopping** with patience of 50 epochs (monitors micro-AUPRC)
- **TensorBoard logging** of training loss, learning rate, and evaluation metrics
- **Gradient checkpointing** for memory-efficient training on long sequences

---

## Evaluation

### Full Evaluation

```bash
bash scripts/test.sh config/test.yaml
```

### Per-Patient Analysis with Attention Visualization

```bash
bash scripts/sample_analysis.sh config/test.yaml

# Specify attention layer and output directory
bash scripts/sample_analysis.sh config/test.yaml --layer-idx 3 --output-dir analysis/patient_001/
```

The sample analysis script extracts:

- **Attention weights** from the specified layer (saved as `.pkl`)
- **Per-head slope and peak parameters** from the MCE predictor
- **Risk prediction curves** for the analyzed patient

### Evaluation Metrics

| Metric | Scope | Description |
|---|---|---|
| AUPRC (micro) | Global | Area under the precision-recall curve, micro-averaged |
| AUPRC (sample-wise) | Per-patient | Mean of per-patient AUPRC scores |
| AUROC | Global | Area under the ROC curve |
| F1 (micro) | Global | Micro-averaged F1 at threshold 0.5 |
| Precision@K | Per-patient | Fraction of relevant items in top-K predictions |

---

## K-Fold Cross Validation

```bash
bash scripts/run_kfold.sh config/train.yaml
```

Configure the fold count and data paths by editing the variables at the top of `scripts/run_kfold.sh`:

```bash
ENABLE_KFOLD=true
KFOLD_COUNT=4
KFOLD_DATA_ROOT="data/icu/kfold_event_balanced_cv"
```

The script iterates over folds, setting `KFOLD_INDEX`, `OVERRIDE_TRAIN_DIR`, and `OVERRIDE_TEST_DIR` environment variables for each run. Results are saved to `SeTA/logs/fold_{i}/fold_result.json`.

---

## Configuration

All hyperparameters are specified in YAML config files. Key parameters from `config/train.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `training.d_model` | 4096 | Transformer hidden dimension |
| `training.nhead` | 32 | Number of attention heads |
| `training.num_layers` | 6 | Transformer encoder layers |
| `training.output_dim` | 1440 | Output prediction dimensions (24h × 60min) |
| `training.batch_size` | 2 | Training batch size per GPU |
| `training.learning_rate` | 5e-5 | Peak learning rate (cosine schedule) |
| `training.T_scaling` | 1209600.0 | Time scaling factor (14 days in seconds) |
| `training.weig` | 1.0 | Positive sample weight for weighted MSE |
| `training.num_epochs` | 750 | Maximum training epochs |
| `training.test_step` | 5 | Evaluate every N epochs |
| `lr_scheduler.type` | cosine | Learning rate schedule |
| `lr_scheduler.step` | 8000 | Total scheduler steps |

> The MCE predictor parameters are automatically configured with a 10× higher learning rate inside the `Trainer` class — no manual config needed.

### Environment Variables

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_SNAPSHOT_TRIGGER=oom
export TOKENIZERS_PARALLELISM=false
export LOG_LEVEL=INFO
export ENABLE_FILE_LOGGING=true
```

These are set automatically by the shell scripts. Set them manually if invoking `train.py` directly.

---

## Project Structure

```
SeTA/
├── train.py                          # Training entry point
├── test.py                           # Evaluation entry point
├── sample_analysis.py                # Per-patient analysis with attention visualization
├── setup.py                          # Package installation
├── requirements.txt                  # Python dependencies
├── LICENSE                           # Apache 2.0
│
├── config/
│   ├── train.yaml                    # Training configuration
│   ├── test.yaml                     # Evaluation configuration
│   └── accelerate_config.yaml        # Accelerate/DeepSpeed ZeRO-2 + BF16
│
├── scripts/
│   ├── train.sh                      # Distributed training launcher
│   ├── test.sh                       # Evaluation launcher
│   ├── run_kfold.sh                  # K-Fold cross-validation runner
│   └── sample_analysis.sh            # Sample analysis launcher
│
├── src/
│   ├── data/
│   │   ├── dataset.py                # ICURiskDataset + simple_collate
│   │   └── batch_utils.py            # Dynamic padding, global max-seq-len sync
│   ├── models/
│   │   ├── attention.py              # MCE-Aware Time-ALiBi attention
│   │   ├── normalization.py          # RMSNorm
│   │   └── transformer.py            # PatientRiskTransformer + TransformerBlock
│   ├── training/
│   │   ├── trainer.py                # Trainer class with early stopping
│   │   ├── evaluator.py              # Distributed evaluation loop
│   │   ├── losses.py                 # WeightedMSELoss
│   │   └── metrics.py                # AUPRC, AUROC, F1, Precision@K
│   └── utils/
│       ├── config.py                 # YAML config loading + validation
│       ├── checkpoint.py             # Save/load model checkpoints
│       └── memory.py                 # GPU memory monitoring
│
└── preprocessing/
    ├── csv_to_json.py                # Step 1: CSV → JSON narrative format
    ├── filter_before_anchor.py       # Step 2: Filter events before prediction anchor
    ├── text_cut.py                   # Step 3: Split long texts, filter by event count
    ├── data_embed_parallel.py        # Step 4: Multi-GPU embedding generation
    ├── split_test_train.py           # Step 5: Train/val/test split (8:1:1)
    └── embedder.py                   # Qwen3-Embedding-8B wrapper
```

---

## Citation

If you use SeTA in your research, please cite:

```bibtex
@inproceedings{seta2026,
  title     = {SeTA: Semantic-Temporal Alignment for ICU Patient Risk Prediction},
  author    = {SeTA Authors},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM)},
  year      = {2026}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
