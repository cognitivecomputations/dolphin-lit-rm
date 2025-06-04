#!/bin/bash

# Example script to run the full pipeline WITHOUT Poetry
# Ensure you are in the project root directory (dolphin-lit-rm)
# and have the required dependencies installed in your Python environment

# Set PYTHONPATH so Python can find the dolphin_lit_rm module
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Default config files (can be overridden by CLI options if needed)
RUN_CONFIG="dolphin_lit_rm/config/run.yaml"
DATASETS_CONFIG="dolphin_lit_rm/config/datasets.yaml"
RUBRIC_CONFIG="dolphin_lit_rm/config/rubric.yaml"

# Note: If you get import errors, install the package first:
#   pip install -r requirements.txt
#   pip install -e .

# --- Option 1: Start a new run ---
# The CLI will create a new run directory under `runs_parent_dir` specified in run.yaml
echo "Starting a new pipeline run..."
python3 cli.py  \
    --run-config "$RUN_CONFIG" \
    --datasets-config "$DATASETS_CONFIG" \
    --rubric-config "$RUBRIC_CONFIG" \
    process-all

# --- Option 2: Resume a specific run ---
# Replace 'path/to/your/existing_run_YYYYMMDD_HHMMSS_xxxxxx' with the actual directory
# EXISTING_RUN_DIR="output/runs/run_20231121_100000_abcdef" 
# if [ -d "$EXISTING_RUN_DIR" ]; then
#   echo "Resuming pipeline run from: $EXISTING_RUN_DIR"
#   python3 cli.py process-all --resume "$EXISTING_RUN_DIR"
# else
#   echo "Resume directory $EXISTING_RUN_DIR not found. Cannot resume."
# fi

# --- Option 3: Run specific stages ---
# echo "Running only ingestion and filter stages for a new run..."
# python3 cli.py process-all \
#     --run-config "$RUN_CONFIG" \
#     --datasets-config "$DATASETS_CONFIG" \
#     --rubric-config "$RUBRIC_CONFIG" \
#     --run-only-stages "ingest,filter"

# --- Option 4: Skip specific stages ---
# echo "Running all stages EXCEPT normalization and scoring for a new run..."
# python3 cli.py process-all \
#     --run-config "$RUN_CONFIG" \
#     --datasets-config "$DATASETS_CONFIG" \
#     --rubric-config "$RUBRIC_CONFIG" \
#     --skip-stages "normalize,score"

echo "Pipeline execution script finished."