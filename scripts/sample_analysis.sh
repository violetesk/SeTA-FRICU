#!/bin/bash
# ==============================================================================
# Sample Analysis Script
#
# Usage:
#   bash scripts/sample_analysis.sh config/test.yaml
#
# Options:
#   --layer-idx N    Layer index for attention analysis (default: 0)
#   --output-dir DIR Output directory for analysis results
# ==============================================================================

set -e  # Exit on error

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
    echo "Usage: bash scripts/sample_analysis.sh <config.yaml>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ==============================================================================
# Locate Analysis Script
# ==============================================================================
ANALYSIS_SCRIPT="${PROJECT_ROOT}/sample_analysis.py"

if [ ! -f "$ANALYSIS_SCRIPT" ]; then
    echo "Error: Analysis script not found: $ANALYSIS_SCRIPT"
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
if [ -f "$FSDP_CONFIG" ]; then
    ACCELERATE_CONFIG_TO_USE="$FSDP_CONFIG"
    echo "Using FSDP config: $FSDP_CONFIG"
elif [ -f "$DEFAULT_ACCELERATE_CONFIG" ]; then
    ACCELERATE_CONFIG_TO_USE="$DEFAULT_ACCELERATE_CONFIG"
    echo "Using default Accelerate config"
else
    echo "Error: No Accelerate config found."
    echo "Run 'accelerate config' to create one."
    exit 1
fi

# ==============================================================================
# Launch Analysis
# ==============================================================================
echo "=========================================="
echo "Sample Analysis Configuration"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Accelerate config: $ACCELERATE_CONFIG_TO_USE"
echo "Python: $PYTHON"
echo "=========================================="
echo ""

"$PYTHON" -m accelerate.commands.launch \
    --config_file "$ACCELERATE_CONFIG_TO_USE" \
    "$ANALYSIS_SCRIPT" \
    --config "$CONFIG_FILE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=========================================="
echo "Sample Analysis Completed"
echo "=========================================="
