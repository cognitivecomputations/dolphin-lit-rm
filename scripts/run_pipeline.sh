#!/bin/bash

# Example script to run the full pipeline
# Ensure you are in the project root directory (dolphin-lit-rm)
# and have poetry environment activated or use `poetry run`

# Default config files (can be overridden by CLI options if needed)
RUN_CONFIG="dolphin-lit-rm/config/run.yml"
DATASETS_CONFIG="dolphin-lit-rm/config/datasets.yml"
RUBRIC_CONFIG="dolphin-lit-rm/config/rubric.yml"

# --- Option 1: Start a new run ---
# The CLI will create a new run directory under `runs_parent_dir` specified in run.yml
echo "Starting a new pipeline run..."
poetry run dolphin-lit-rm process-all \
    --run-config "$RUN_CONFIG" \
    --datasets-config "$DATASETS_CONFIG" \
    --rubric-config "$RUBRIC_CONFIG"

# --- Option 2: Resume a specific run ---
# Replace 'path/to/your/existing_run_YYYYMMDD_HHMMSS_xxxxxx' with the actual directory
# EXISTING_RUN_DIR="output/runs/run_20231121_100000_abcdef" 
# if [ -d "$EXISTING_RUN_DIR" ]; then
#   echo "Resuming pipeline run from: $EXISTING_RUN_DIR"
#   poetry run dolphin-lit-rm process-all --resume "$EXISTING_RUN_DIR"
# else
#   echo "Resume directory $EXISTING_RUN_DIR not found. Cannot resume."
# fi

# --- Option 3: Run specific stages ---
# echo "Running only ingestion and filter stages for a new run..."
# poetry run dolphin-lit-rm process-all \
#     --run-config "$RUN_CONFIG" \
#     --datasets-config "$DATASETS_CONFIG" \
#     --rubric-config "$RUBRIC_CONFIG" \
#     --run-only-stages "ingest,filter"

# --- Option 4: Skip specific stages ---
# echo "Running all stages EXCEPT normalization and scoring for a new run..."
# poetry run dolphin-lit-rm process-all \
#     --run-config "$RUN_CONFIG" \
#     --datasets-config "$DATASETS_CONFIG" \
#     --rubric-config "$RUBRIC_CONFIG" \
#     --skip-stages "normalize,score"


echo "Pipeline execution script finished."