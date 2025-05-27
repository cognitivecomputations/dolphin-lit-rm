import shutil
import sys
from pathlib import Path
import typer
from typing import Optional, List
from loguru import logger
import datetime
import random
import string
import yaml # For loading/dumping configs

from dolphin_lit_rm.core_configs import RunConfig, DatasetsConfig, RubricConfig, AppConfig
from dolphin_lit_rm.utils.state_manager import StateManager

# Import stage runners
from dolphin_lit_rm.ingestion.main import run_ingestion_stage
from dolphin_lit_rm.preprocessing.filter import run_filter_stage
from dolphin_lit_rm.preprocessing.segment import run_segmentation_stage
from dolphin_lit_rm.classification.classifier import run_classification_stage # Needs global app_config fix
from dolphin_lit_rm.normalization.sampler import run_normalization_stage
from dolphin_lit_rm.preprocessing.prompt_reconstruct import run_prompt_reconstruction_stage
from dolphin_lit_rm.scoring.judge import run_scoring_stage
from dolphin_lit_rm.postprocessing.pipeline import run_postprocessing_stage


app = typer.Typer(name="dolphin-lit-rm", help="Literary Reward Model Data Pipeline CLI")

# --- Helper to load configurations ---
def load_configuration(
    run_config_path: Optional[Path],
    datasets_config_path: Optional[Path],
    rubric_config_path: Optional[Path],
    existing_run_dir: Optional[Path] = None # For resuming
) -> AppConfig:
    
    if existing_run_dir:
        logger.info(f"Resuming run from: {existing_run_dir}")
        # Load configs from the existing run directory
        run_config_path = existing_run_dir / "config" / "run.yml"
        datasets_config_path = existing_run_dir / "config" / "datasets.yml"
        rubric_config_path = existing_run_dir / "config" / "rubric.yml"
        if not all([p.exists() for p in [run_config_path, datasets_config_path, rubric_config_path]]):
            logger.error(f"One or more config files not found in resumed run directory {existing_run_dir}/config. Exiting.")
            raise typer.Exit(code=1)
    else:
        # Ensure default paths are provided if specific ones aren't
        if not run_config_path: run_config_path = Path("dolphin-lit-rm/config/run.yml")
        if not datasets_config_path: datasets_config_path = Path("dolphin-lit-rm/config/datasets.yml")
        if not rubric_config_path: rubric_config_path = Path("dolphin-lit-rm/config/rubric.yml")

    try:
        with open(run_config_path, 'r') as f:
            run_config_data = yaml.safe_load(f)
        run_cfg = RunConfig(**run_config_data)

        with open(datasets_config_path, 'r') as f:
            datasets_config_data = yaml.safe_load(f)
        datasets_cfg = DatasetsConfig(**datasets_config_data)

        with open(rubric_config_path, 'r') as f:
            rubric_config_data = yaml.safe_load(f)
        rubric_cfg = RubricConfig(**rubric_config_data)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e.filename}. Exiting.")
        raise typer.Exit(code=1)
    except Exception as e: # Catch Pydantic validation errors or other YAML issues
        logger.error(f"Error loading or validating configuration: {e}. Exiting.")
        raise typer.Exit(code=1)

    return AppConfig(run=run_cfg, datasets=datasets_cfg, rubric=rubric_cfg)


# --- Helper to setup run directory and logging ---
def setup_run_environment(app_cfg: AppConfig, resume_dir: Optional[Path]) -> AppConfig:
    if resume_dir:
        current_run_dir = resume_dir
        # Configs are already loaded from resume_dir by load_configuration
    else:
        # Create new run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name = f"run_{timestamp}_{random_suffix}"
        current_run_dir = Path(app_cfg.run.runs_parent_dir) / run_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"New run directory created: {current_run_dir}")

        # Copy current configs to the new run directory for reproducibility
        run_config_copy_dir = current_run_dir / "config"
        run_config_copy_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine original config paths (this is a bit of a hack, assumes they were default or passed)
        # This part needs to be robust if CLI overrides were used for config paths.
        # For simplicity, assume they are the default paths if not resuming.
        orig_run_cfg_path = Path(getattr(app_cfg, "_cli_run_config_path", "dolphin-lit-rm/config/run.yml"))
        orig_ds_cfg_path = Path(getattr(app_cfg, "_cli_datasets_config_path", "dolphin-lit-rm/config/datasets.yml"))
        orig_rb_cfg_path = Path(getattr(app_cfg, "_cli_rubric_config_path", "dolphin-lit-rm/config/rubric.yml"))

        try:
            shutil.copy(orig_run_cfg_path, run_config_copy_dir / "run.yml")
            shutil.copy(orig_ds_cfg_path, run_config_copy_dir / "datasets.yml")
            shutil.copy(orig_rb_cfg_path, run_config_copy_dir / "rubric.yml")
            app_cfg.run.run_config_copy_path = run_config_copy_dir / "run.yml"
        except Exception as e:
            logger.warning(f"Could not copy config files to run directory: {e}")


    # Update AppConfig with dynamic paths
    app_cfg.run.current_run_dir = current_run_dir
    app_cfg.run.artifacts_dir = current_run_dir / "artifacts"
    app_cfg.run.logs_dir = current_run_dir / "logs"
    app_cfg.run.state_dir = current_run_dir / "artifacts" / "state" # State inside artifacts

    app_cfg.run.artifacts_dir.mkdir(parents=True, exist_ok=True)
    app_cfg.run.logs_dir.mkdir(parents=True, exist_ok=True)
    app_cfg.run.state_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to file within the run directory
    log_file_path = app_cfg.run.logs_dir / "pipeline.log"
    logger.remove() # Remove default handler
    logger.add(sys.stderr, level=app_cfg.run.default_log_level.upper())
    logger.add(log_file_path, level=app_cfg.run.default_log_level.upper(), rotation="10 MB")
    
    logger.info(f"Logging to console and to: {log_file_path}")
    logger.info(f"Artifacts will be stored in: {app_cfg.run.artifacts_dir}")
    logger.info(f"State information in: {app_cfg.run.state_dir}")

    # Initialize StateManager
    app_cfg.state_manager = StateManager(state_dir=app_cfg.run.state_dir)
    
    return app_cfg

# --- Global options ---
@app.callback()
def global_options(
    ctx: typer.Context,
    run_config_file: Optional[Path] = typer.Option(None, "--run-config", "-rc", help="Path to the run configuration YAML file.", exists=False, dir_okay=False, resolve_path=True),
    datasets_config_file: Optional[Path] = typer.Option(None, "--datasets-config", "-dc", help="Path to the datasets configuration YAML file.", exists=False, dir_okay=False, resolve_path=True),
    rubric_config_file: Optional[Path] = typer.Option(None, "--rubric-config", "-rbc", help="Path to the rubric configuration YAML file.", exists=False, dir_okay=False, resolve_path=True),
    resume_run_dir: Optional[Path] = typer.Option(None, "--resume", "-r", help="Path to an existing run directory to resume.", exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    # Add force flags if needed, e.g. --force-ingest, --force-all-stages
    # force_all: bool = typer.Option(False, "--force-all", help="Force re-running all stages, ignoring existing artifacts/state."),
    # force_stages: Optional[List[str]] = typer.Option(None, "--force-stages", help="Comma-separated list of stages to force re-run.")
):
    """Dolphin LRM Data Pipeline: Process texts for reward modeling."""
    
    # Store original config paths if provided, for copying later
    # This is a bit of a workaround to pass these to setup_run_environment
    # Pydantic models in AppConfig are the source of truth after loading.
    _cli_options = {
        "_cli_run_config_path": run_config_file,
        "_cli_datasets_config_path": datasets_config_file,
        "_cli_rubric_config_path": rubric_config_file,
    }

    app_config = load_configuration(run_config_file, datasets_config_file, rubric_config_file, resume_run_dir)
    
    # Update app_config with any CLI-provided paths for setup_run_environment to use if not resuming
    for k, v in _cli_options.items():
        if v is not None:
            setattr(app_config, k, v)

    app_config = setup_run_environment(app_config, resume_run_dir)
    
    # Make AppConfig available to subcommands
    ctx.obj = app_config


# --- Pipeline Stages Commands ---

PIPELINE_STAGES = {
    "ingest": run_ingestion_stage,
    "filter": run_filter_stage,
    "segment": run_segmentation_stage,
    "classify": run_classification_stage,
    "normalize": run_normalization_stage,
    "reconstruct-prompts": run_prompt_reconstruction_stage,
    "score": run_scoring_stage,
    "postprocess": run_postprocessing_stage,
}

@app.command()
def ingest(ctx: typer.Context):
    """1. Ingest raw datasets."""
    app_config: AppConfig = ctx.obj
    run_ingestion_stage(app_config)

@app.command()
def filter(ctx: typer.Context):
    """2. Pre-filter raw data (length, lang, dedupe)."""
    app_config: AppConfig = ctx.obj
    run_filter_stage(app_config)

@app.command()
def segment(ctx: typer.Context):
    """3. Segment long-form documents."""
    app_config: AppConfig = ctx.obj
    run_segmentation_stage(app_config)

@app.command()
def classify(ctx: typer.Context):
    """4. Classify texts by genre/sub-genre."""
    app_config: AppConfig = ctx.obj
    run_classification_stage(app_config)

@app.command()
def normalize(ctx: typer.Context):
    """5. Apply quota-based sampling to classified data."""
    app_config: AppConfig = ctx.obj
    run_normalization_stage(app_config)

@app.command(name="reconstruct-prompts") # Name in CLI
def reconstruct_prompts_command(ctx: typer.Context):
    """6. Reconstruct prompts for prompt-less texts."""
    app_config: AppConfig = ctx.obj
    run_prompt_reconstruction_stage(app_config)

@app.command()
def score(ctx: typer.Context):
    """7. Score texts using the judge model based on the rubric."""
    app_config: AppConfig = ctx.obj
    run_scoring_stage(app_config)

@app.command()
def postprocess(ctx: typer.Context):
    """8. Calibrate scores and create final train/val/test splits."""
    app_config: AppConfig = ctx.obj
    run_postprocessing_stage(app_config)


@app.command()
def process_all(
    ctx: typer.Context,
    skip_stages: Optional[str] = typer.Option(None, help="Comma-separated list of stages to skip."),
    run_only_stages: Optional[str] = typer.Option(None, help="Comma-separated list of stages to run exclusively.")
):
    """Run all pipeline stages sequentially."""
    app_config: AppConfig = ctx.obj
    logger.info("--- Starting Full Pipeline Process ---")

    stages_to_skip = set(skip_stages.split(',')) if skip_stages else set()
    stages_to_run_only = set(run_only_stages.split(',')) if run_only_stages else set()

    for stage_name, stage_func in PIPELINE_STAGES.items():
        if stage_name in stages_to_skip:
            logger.info(f"Skipping stage: {stage_name} as per --skip-stages.")
            continue
        if stages_to_run_only and stage_name not in stages_to_run_only:
            logger.info(f"Skipping stage: {stage_name} as it's not in --run-only-stages.")
            continue
        
        try:
            stage_func(app_config)
        except Exception as e:
            logger.error(f"Error during stage '{stage_name}': {e}", exc_info=True)
            logger.error(f"Pipeline halted at stage '{stage_name}'. Fix the error and resume if possible.")
            raise typer.Exit(code=1)
            
    logger.info("--- Full Pipeline Process Completed Successfully ---")


if __name__ == "__main__":
    app()