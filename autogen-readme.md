# Dolphin Literary Reward Model (LRM) Data Pipeline

This project implements a data pipeline to process various text corpora into a format suitable for training a Literary Reward Model. The pipeline ingests raw data, filters, segments, classifies, reconstructs prompts, scores texts based on a rubric, and prepares final datasets.

## Project Structure

(Refer to the initial problem description for the detailed directory structure)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dolphin-lit-rm
    ```

2.  **Install dependencies:**
    - Using pip:
    ```bash
    pip install -r requirements.txt
    ```
    OR using poetry:
    ```bash
    poetry install
    ```

3.  **Download necessary NLP models:**
    ```bash
    poetry run python -m spacy download en_core_web_sm
    # Download fastText lid.176.bin model if not automatically handled by the library
    # (Refer to fastText documentation for model download)
    ```

## Configuration

Configuration is managed through YAML files in `dolphin-lit-rm/config/`:

*   `run.yml`: Main run parameters, API keys, model endpoints, paths, stage-specific settings.
*   `datasets.yml`: Definitions of input datasets, their types, paths, and ingestion limits.
*   `rubric.yml`: Metrics for the reward model.
*   `prompts/`: Jinja templates for LLM interactions.

Default configurations are provided. You can override them by:
1.  Editing the YAML files directly.
2.  Providing alternative config file paths via CLI arguments.
3.  Overriding specific parameters via CLI arguments (where supported).

## Usage

The pipeline is controlled via a CLI tool: `dolphin-lit-rm`.

**General Command Structure:**

```bash
dolphin-lit-rm [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

**Example: Running the full pipeline for a new run:**

```bash
poetry run dolphin-lit-rm process-all \
    --run-config-file dolphin-lit-rm/config/run.yml \
    --datasets-config-file dolphin-lit-rm/config/datasets.yml \
    --rubric-config-file dolphin-lit-rm/config/rubric.yml
```

This will create a new run directory (e.g., `runs/run_YYYYMMDD_HHMMSS_xxxx`) containing logs, artifacts, and a copy of the configs used.

**Example: Resuming a previous run:**

```bash
poetry run dolphin-lit-rm process-all --resume runs/run_YYYYMMDD_HHMMSS_xxxx
```

**Available Commands (Stages):**

*   `ingest`: Ingest raw datasets.
*   `preprocess`: Run pre-filtering, segmentation.
*   `classify`: Classify texts by genre/sub-genre.
*   `normalize`: Apply quota-based sampling.
*   `reconstruct-prompts`: Generate prompts for prompt-less texts.
*   `score`: Score texts using the judge model.
*   `postprocess`: Calibrate scores and create final splits.
*   `process-all`: Run all stages sequentially.

Use `poetry run dolphin-lit-rm --help` and `poetry run dolphin-lit-rm <COMMAND> --help` for detailed options.

## Run Directory

Each execution of the pipeline operates within a "run directory".
*   If `--resume [path_to_run_dir]` is specified, the pipeline attempts to resume from that directory.
*   Otherwise, a new run directory is created (e.g., `runs/run_YYYYMMDD_HHMMSS_xxxxxx`).

The run directory contains:
*   `config/`: A copy of the configuration files used for the run.
*   `logs/`: Log files for each stage.
*   `artifacts/`: All intermediate and final data products.
    *   `state/`: Files tracking processed items for resumability.
    *   `raw/`, `filtered/`, `segmented/`, `classified/`, `normalized/`, `reconstructed/`, `scored/`, `final/`

## Development

(Add notes on running tests, linters, etc.)