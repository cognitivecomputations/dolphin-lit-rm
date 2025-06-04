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

from dolphin_lit_rm.core_configs import RunConfig, DatasetsConfig, RubricConfig, AppConfig, RubricsConfig
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

def _read_yaml(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh)

# --- Helper to load configurations ---
def load_configuration(
    run_config_path: Optional[Path],
    datasets_config_path: Optional[Path],
    rubric_paths: Optional[List[Path]],        # ← allow *several* rubric files
    existing_run_dir: Optional[Path] = None,
) -> AppConfig:
    
    if existing_run_dir:
        logger.info(f"Resuming run from: {existing_run_dir}")

        run_config_path      = existing_run_dir / "config" / "run.yaml"
        datasets_config_path = existing_run_dir / "config" / "datasets.yaml"

        # ── locate rubric YAMLs (supports both old and new layouts) ────────────
        rubric_dir   = existing_run_dir / "config" / "rubrics"
        if rubric_dir.is_dir():
            rubric_paths = sorted(rubric_dir.glob("*.yaml"))
        else:                                  # legacy single-file case
            rubric_paths = [existing_run_dir / "config" / "rubric.yaml"]

        if not (run_config_path.exists() and datasets_config_path.exists() and rubric_paths):
            logger.error(
                f"One or more config files not found in resumed run directory "
                f"{existing_run_dir}/config. Exiting."
            )
            raise typer.Exit(code=1)


    try:
        with open(run_config_path, 'r') as f:
            run_config_data = yaml.safe_load(f)
        
        # Create runs_parent_dir if it doesn't exist before Pydantic validation
        if 'runs_parent_dir' in run_config_data:
            runs_parent_path = Path(run_config_data['runs_parent_dir'])
            runs_parent_path.mkdir(parents=True, exist_ok=True)
        
        # ----------------------- load run + datasets -----------------------------
        run_cfg     = RunConfig(**_read_yaml(run_config_path))
        datasets_cfg = DatasetsConfig(**_read_yaml(datasets_config_path))

        # ----------------------- load all rubric YAMLs ---------------------------
        rubric_files: List[Path] = []
        if rubric_paths:
            for p in rubric_paths:
                rubric_files.extend(sorted(
                    p.glob("*.yaml") if p.is_dir() else [p]
                ))
        else:                                         # default single-file case
            rubric_files = [Path("dolphin_lit_rm/config/rubric.yaml")]

        rubrics_dict = {}
        for f in rubric_files:
            loaded = _read_yaml(f)
            cfg     = RubricConfig(**loaded)
            if cfg.name in rubrics_dict:
                logger.error(f"Duplicate rubric name '{cfg.name}' in {f}.")
                raise typer.Exit(code=1)
            rubrics_dict[cfg.name] = cfg

        rubrics_cfg = RubricsConfig(root=rubrics_dict)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e.filename}. Exiting.")
        raise typer.Exit(code=1)
    except Exception as e: # Catch Pydantic validation errors or other YAML issues
        logger.error(f"Error loading or validating configuration: {e}. Exiting.")
        raise typer.Exit(code=1)

    return AppConfig(run=run_cfg, datasets=datasets_cfg, rubrics=rubrics_cfg)


# --- Helper to setup run directory and logging ---
def setup_run_environment(app_cfg: AppConfig, resume_dir: Optional[Path]) -> AppConfig:
    """
    Creates / restores the directory structure for a pipeline run and
    copies the exact configuration files that were used to launch it.
    """
    # ── 1. decide run directory ────────────────────────────────────────────
    if resume_dir:
        current_run_dir = resume_dir                   # nothing to copy
    else:
        ts            = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix        = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name      = f"run_{ts}_{suffix}"
        current_run_dir = Path(app_cfg.run.runs_parent_dir) / run_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"New run directory created: {current_run_dir}")

        # ── 2. copy configuration files for reproducibility ───────────────
        cfg_dst = current_run_dir / "config"
        cfg_dst.mkdir(parents=True, exist_ok=True)

        # original paths as remembered by the CLI callback
        orig_run_cfg = Path(getattr(app_cfg, "_cli_run_config_path",
                                    "dolphin_lit_rm/config/run.yaml"))
        orig_ds_cfg  = Path(getattr(app_cfg, "_cli_datasets_config_path",
                                    "dolphin_lit_rm/config/datasets.yaml"))
        rubrics_srcs = getattr(app_cfg, "_cli_rubric_config_path", None)  # Path | List[Path] | None

        try:
            shutil.copy(orig_run_cfg, cfg_dst / "run.yaml")
            shutil.copy(orig_ds_cfg,  cfg_dst / "datasets.yaml")
        except Exception as e:
            logger.warning(f"Could not copy run/datasets YAMLs to run directory: {e}")

        # ── 2a. copy rubric YAMLs (supports file or directory, many or one)
        rubrics_dst = cfg_dst / "rubrics"
        rubrics_dst.mkdir(parents=True, exist_ok=True)

        def _copy_rubric(p: Path):
            try:
                shutil.copy(p, rubrics_dst / p.name)
            except Exception as e:
                logger.warning(f"Failed copying rubric {p}: {e}")

        if rubrics_srcs is None:
            _copy_rubric(Path("dolphin_lit_rm/config/rubric.yaml"))
        else:
            if not isinstance(rubrics_srcs, (list, tuple)):
                rubrics_srcs = [rubrics_srcs]
            for src in rubrics_srcs:
                src = Path(src)
                if src.is_dir():
                    for yml in src.glob("*.yaml"):
                        _copy_rubric(yml)
                else:
                    _copy_rubric(src)

        app_cfg.run.run_config_copy_path = cfg_dst / "run.yaml"

    # ── 3. create artifacts / logs / state sub-dirs ────────────────────────
    app_cfg.run.current_run_dir = current_run_dir
    app_cfg.run.artifacts_dir   = current_run_dir / "artifacts"
    app_cfg.run.logs_dir        = current_run_dir / "logs"
    app_cfg.run.state_dir       = app_cfg.run.artifacts_dir / "state"

    app_cfg.run.artifacts_dir.mkdir(parents=True, exist_ok=True)
    app_cfg.run.logs_dir.mkdir(parents=True, exist_ok=True)
    app_cfg.run.state_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. configure logging ───────────────────────────────────────────────
    log_file = app_cfg.run.logs_dir / "pipeline.log"
    logger.remove()
    logger.add(sys.stderr, level=app_cfg.run.default_log_level.upper())
    logger.add(log_file,  level=app_cfg.run.default_log_level.upper(), rotation="10 MB")

    logger.info(f"Logging to console and to: {log_file}")
    logger.info(f"Artifacts will be stored in: {app_cfg.run.artifacts_dir}")
    logger.info(f"State information in: {app_cfg.run.state_dir}")

    # ── 5. initialise state manager ────────────────────────────────────────
    app_cfg.state_manager = StateManager(state_dir=app_cfg.run.state_dir)

    return app_cfg


# --- Global options ---
@app.callback()
def global_options(
    ctx: typer.Context,
    run_config_file: Optional[Path] = typer.Option(None, "--run-config", "-rc", help="Path to the run configuration YAML file.", exists=False, dir_okay=False, resolve_path=True),
    datasets_config_file: Optional[Path] = typer.Option(None, "--datasets-config", "-dc", help="Path to the datasets configuration YAML file.", exists=False, dir_okay=False, resolve_path=True),
    rubric_config_files: List[Path] = typer.Option(
        None, "--rubric-config", "-rbc", help="Path(s) or directory with rubric YAML files.",
        exists=False, dir_okay=True, file_okay=True, resolve_path=True,
    ),
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
        "_cli_rubric_config_path": rubric_config_files,
    }

    app_config = load_configuration(run_config_file, datasets_config_file, rubric_config_files, resume_run_dir)
    
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