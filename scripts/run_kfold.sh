#!/bin/bash
# ==============================================================================
# K-Fold Cross Validation Script
#
# Usage:
#   bash scripts/run_kfold.sh config/train.yaml
#
# Configuration:
#   Modify ENABLE_KFOLD, KFOLD_COUNT, and KFOLD_DATA_ROOT below.
# ==============================================================================

set -e  # Exit on error

# ==============================================================================
# K-Fold Configuration
# ==============================================================================
ENABLE_KFOLD=true
KFOLD_COUNT=4
KFOLD_DATA_ROOT="data/icu/kfold_event_balanced_cv"
TRAIN_FILE_NAME="train"
TEST_FILE_NAME="test"

# ==============================================================================
# Environment Variables
# ==============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_SNAPSHOT_TRIGGER=oom
export TOKENIZERS_PARALLELISM=false
export LOG_LEVEL=INFO
export ENABLE_FILE_LOGGING=true

echo "=========================================="
echo "Environment Configuration"
echo "=========================================="
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo ""

# ==============================================================================
# Virtual Environment Setup
# ==============================================================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/.venv"

if [ -f "$VENV_PATH/bin/python" ] && "$VENV_PATH/bin/python" --version &>/dev/null 2>&1; then
    PYTHON="$VENV_PATH/bin/python"
    echo "Using Python from virtual environment: $VENV_PATH"
else
    PYTHON="python"
    echo "Using system Python"
fi
echo ""

# ==============================================================================
# Argument Parsing
# ==============================================================================
CONFIG_FILE="$1"
EXTRA_ARGS=("${@:2}")

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file specified"
    echo "Usage: bash scripts/run_kfold.sh <config.yaml>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ==============================================================================
# Locate Training Script
# ==============================================================================
TRAIN_SCRIPT="${PROJECT_ROOT}/train.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

# ==============================================================================
# Accelerate Configuration
# ==============================================================================
CONFIG_DIR=$(dirname "$CONFIG_FILE")
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
FSDP_CONFIG="${CONFIG_DIR}/${CONFIG_BASENAME}_dp.yaml"
DEFAULT_ACCELERATE_CONFIG="$HOME/.cache/huggingface/accelerate/default_config.yaml"

ACCELERATE_CONFIG_TO_USE=""
# 1. First priority: Naming convention [config_name]_dp.yaml
if [ -f "$FSDP_CONFIG" ]; then
    ACCELERATE_CONFIG_TO_USE="$FSDP_CONFIG"
    echo "Using derived config: $FSDP_CONFIG"
# 2. Second priority: Project default config
elif [ -f "${CONFIG_DIR}/accelerate_config.yaml" ]; then
    ACCELERATE_CONFIG_TO_USE="${CONFIG_DIR}/accelerate_config.yaml"
    echo "Using project config: ${CONFIG_DIR}/accelerate_config.yaml"
# 3. Third priority: User's system default
elif [ -f "$DEFAULT_ACCELERATE_CONFIG" ]; then
    ACCELERATE_CONFIG_TO_USE="$DEFAULT_ACCELERATE_CONFIG"
    echo "Using default Accelerate config"
else
    echo "Error: No Accelerate config found."
    exit 1
fi

# ==============================================================================
# Training Function
# ==============================================================================
run_training() {
    local fold_idx=$1
    local extra_msg=""
    
    if [ -n "$fold_idx" ]; then
        extra_msg="(Fold $fold_idx/$((KFOLD_COUNT-1)))"
    fi

    echo ""
    echo "========================================================"
    echo "🚀 Starting Training Task $extra_msg"
    echo "   Config: $CONFIG_FILE"
    if [ -n "$fold_idx" ]; then
        echo "   Fold Index: $fold_idx"
        echo "   Train Data: $OVERRIDE_TRAIN_DIR"
        echo "   Test Data:  $OVERRIDE_TEST_DIR"
    fi
    echo "========================================================"
    echo ""

    "$PYTHON" -m accelerate.commands.launch \
        --config_file "$ACCELERATE_CONFIG_TO_USE" \
        "$TRAIN_SCRIPT" \
        --config "$CONFIG_FILE" \
        "${EXTRA_ARGS[@]}"
}

# ==============================================================================
# Main Execution
# ==============================================================================
if [ "$ENABLE_KFOLD" = true ]; then
    echo ">>> K-Fold Cross Validation Enabled (Total Folds: $KFOLD_COUNT)"
    echo ">>> Data Root: $KFOLD_DATA_ROOT"
    
    if [ ! -d "$KFOLD_DATA_ROOT" ]; then
        echo "Error: K-Fold data root not found: $KFOLD_DATA_ROOT"
        exit 1
    fi

    for (( i=0; i<KFOLD_COUNT; i++ )); do
        export KFOLD_INDEX=$i
        
        CURRENT_FOLD_DIR="${KFOLD_DATA_ROOT}/fold_${i}"
        export OVERRIDE_TRAIN_DIR="${CURRENT_FOLD_DIR}/${TRAIN_FILE_NAME}"
        export OVERRIDE_TEST_DIR="${CURRENT_FOLD_DIR}/${TEST_FILE_NAME}"

        if [ ! -d "$OVERRIDE_TRAIN_DIR" ]; then
            echo "Error: Train data missing for fold $i: $OVERRIDE_TRAIN_DIR"
            exit 1
        fi

        run_training "$i"
        
        if [ $? -ne 0 ]; then
            echo "❌ Fold $i Failed!"
            exit 1
        else
            echo "✅ Fold $i Completed successfully."
        fi

        sleep 5
    done
    
    echo ">>> All $KFOLD_COUNT folds completed successfully."

else
    echo ">>> Standard Single Run Mode"
    
    unset KFOLD_INDEX
    unset OVERRIDE_TRAIN_DIR
    unset OVERRIDE_TEST_DIR
    
    run_training ""
fi

echo ""
echo "=========================================="
echo "All Tasks Finished"
echo "=========================================="
