import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm

from dolphin_lit_rm.core_configs import AppConfig, PostprocessingConfig, RubricConfig
from dolphin_lit_rm.utils import file_io, state_manager
from dolphin_lit_rm.utils.schema_def import Record


def calibrate_scores(
    records: List[Dict[str, Any]], 
    rubric_metrics: List[Dict[str, Any]], # from rubric_config.metrics
    calibration_config: Dict[str, Any] # from postprocessing_config.calibration
) -> List[Dict[str, Any]]:
    """
    Rescales each metric so 5th percentile -> 0, 95th percentile -> 1.
    Operates in-place on the list of record dictionaries.
    """
    if not calibration_config.get("enabled", False) or not records:
        logger.info("Score calibration is disabled or no records to calibrate.")
        return records

    logger.info("Starting score calibration...")
    lower_p = calibration_config["lower_percentile"]
    upper_p = calibration_config["upper_percentile"]

    for metric_info in rubric_metrics:
        metric_name = metric_info["name"]
        
        # Extract all valid scores for this metric
        metric_scores = [
            rec["scores"][metric_name] 
            for rec in records 
            if rec.get("scores") and isinstance(rec["scores"].get(metric_name), (int, float))
        ]

        if not metric_scores or len(metric_scores) < 2: # Need at least 2 points to find percentiles
            logger.warning(f"Not enough valid scores for metric '{metric_name}' to calibrate. Skipping.")
            continue

        try:
            p_lower_val = np.percentile(metric_scores, lower_p)
            p_upper_val = np.percentile(metric_scores, upper_p)
        except Exception as e:
            logger.error(f"Error calculating percentiles for metric '{metric_name}': {e}. Skipping calibration for this metric.")
            continue

        logger.info(f"Metric '{metric_name}': Original range [{min(metric_scores):.2f}, {max(metric_scores):.2f}]. "
                    f"{lower_p}th percentile = {p_lower_val:.2f}, {upper_p}th percentile = {p_upper_val:.2f}")

        if p_upper_val == p_lower_val: # Avoid division by zero if all scores in percentile range are same
            logger.warning(f"Percentiles {lower_p}th and {upper_p}th are identical for metric '{metric_name}'. "
                           f"Assigning 0.5 to all or keeping original if range is 0.")
            for rec in records:
                if rec.get("scores") and isinstance(rec["scores"].get(metric_name), (int, float)):
                    # If all values are same, map to 0.5 if range is zero, else map to 0 or 1 based on value
                    # A simpler approach: if range is zero, all calibrated scores are 0.5
                    rec["scores"][f"{metric_name}_calibrated"] = 0.5 if p_upper_val == p_lower_val else \
                        (0.0 if rec["scores"][metric_name] <= p_lower_val else 1.0)

            continue


        # Apply calibration
        for rec in records:
            if rec.get("scores") and isinstance(rec["scores"].get(metric_name), (int, float)):
                original_score = rec["scores"][metric_name]
                calibrated_score = (original_score - p_lower_val) / (p_upper_val - p_lower_val)
                calibrated_score = max(0.0, min(1.0, calibrated_score)) # Clamp to [0, 1]
                rec["scores"][f"{metric_name}_calibrated"] = calibrated_score
            # else: keep original None or missing score as is for calibrated field too

    logger.info("Score calibration completed.")
    return records


def filter_by_missing_metrics(
    records: List[Dict[str, Any]],
    rubric_metrics: List[Dict[str, Any]],
    min_metrics_present_percent: float
) -> List[Dict[str, Any]]:
    """Filters records that have too many missing scores."""
    if min_metrics_present_percent <= 0:
        return records

    num_total_metrics = len(rubric_metrics)
    min_metrics_required = int(num_total_metrics * min_metrics_present_percent)
    
    retained_records = []
    dropped_count = 0
    for rec in records:
        if not rec.get("scores"): # No scores dict at all
            if min_metrics_required > 0: # If any metrics are required, drop
                dropped_count +=1
                continue
            else: # No metrics required, keep
                retained_records.append(rec)
                continue

        present_metrics_count = sum(
            1 for metric in rubric_metrics 
            if isinstance(rec["scores"].get(metric["name"]), (int, float)) # Check original scores
        )
        if present_metrics_count >= min_metrics_required:
            retained_records.append(rec)
        else:
            dropped_count += 1
            
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} records due to missing metrics (required {min_metrics_required}/{num_total_metrics}).")
    return retained_records


def split_dataset(
    records: List[Dict[str, Any]], 
    split_ratios: Dict[str, float]
) -> Dict[str, List[Dict[str, Any]]]:
    """Splits records into train/validation/test sets based on hash of record ID."""
    # Ensure splits sum to 1 (approximately)
    if not np.isclose(sum(split_ratios.values()), 1.0):
        logger.warning(f"Split ratios {split_ratios} do not sum to 1. Normalizing.")
        total_ratio = sum(split_ratios.values())
        split_ratios = {k: v / total_ratio for k, v in split_ratios.items()}

    # Sort splits to process in a defined order (e.g., train, val, test)
    # This ensures that if ratios are e.g. train:0.8, val:0.1, test:0.1,
    # an ID gets assigned to 'train' if its hash falls in the first 0.8 range.
    sorted_splits = sorted(split_ratios.items(), key=lambda item: item[0]) # Sort by name for consistency

    split_boundaries = {}
    current_boundary = 0.0
    for name, ratio in sorted_splits:
        split_boundaries[name] = (current_boundary, current_boundary + ratio)
        current_boundary += ratio
    
    # Ensure the last boundary reaches 1.0
    # last_split_name = sorted_splits[-1][0]
    # split_boundaries[last_split_name] = (split_boundaries[last_split_name][0], 1.0)


    split_records: Dict[str, List[Dict[str, Any]]] = {name: [] for name in split_ratios.keys()}

    for rec in records:
        record_id = rec["id"]
        # Use a part of sha256 hash for stable splitting
        # Convert hex hash to an integer, then scale to [0,1)
        h = hashlib.sha256(record_id.encode('utf-8')).hexdigest()
        # Take first 8 hex chars (32 bits), convert to int
        hash_val = int(h[:8], 16) 
        # Normalize to [0,1) range
        normalized_hash = hash_val / (2**32) 

        assigned = False
        for name, (lower_bound, upper_bound) in split_boundaries.items():
            if lower_bound <= normalized_hash < upper_bound:
                split_records[name].append(rec)
                assigned = True
                break
        if not assigned: # Should not happen if boundaries are correct, but as a fallback
            logger.warning(f"Record {record_id} (hash {normalized_hash}) not assigned to any split. Assigning to first split: {sorted_splits[0][0]}")
            split_records[sorted_splits[0][0]].append(rec)


    for name, data in split_records.items():
        logger.info(f"Split '{name}': {len(data)} records ({len(data)/len(records)*100 if records else 0:.2f}%).")
        
    return split_records


def run_postprocessing_stage(app_config: AppConfig):
    """
    Final stage:

    1.  Optionally calibrate every metric (5–95 pct → 0–1).
    2.  Drop records that have fewer than the configured fraction of
        *any* score values.
    3.  Deterministically split into train/val/test.
    4.  Save each split to artifacts/final/.
    """
    logger.info("--- Starting Post-processing Stage ---")
    post_cfg   = app_config.run.postprocessing
    rubrics    = app_config.rubrics              # mapping name → RubricConfig

    # ------------------------------------------------------------------ I/O
    input_dir  = Path(app_config.run.artifacts_dir) / "scored"
    output_dir = Path(app_config.run.artifacts_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_files = list(input_dir.glob(f"*_scored.{app_config.run.artifact_ext}"))
    if not scored_files:
        logger.warning(f"No scored files found in {input_dir}. Skipping post-processing.")
        return

    if len(scored_files) == 1:
        ds_combined = file_io.load_records(scored_files[0])
    else:
        ds_parts = [file_io.load_records(fp) for fp in scored_files if fp.stat().st_size > 0]
        ds_combined = concatenate_datasets(ds_parts) if ds_parts else None

    if not ds_combined or len(ds_combined) == 0:
        logger.warning("No data to post-process. Exiting.")
        return

    records = ds_combined.to_list()
    logger.info(f"Loaded {len(records)} records for post-processing.")

    # ------------------------------------------------------------------ build global metric list (namespaced)
    metric_dicts = []
    for rubric_name, rubric_cfg in rubrics.root.items():
        for m in rubric_cfg.metrics:
            metric_dicts.append(
                {
                    "name": f"{rubric_name}.{m.name}",
                    "description": m.description,
                    "prompt_hint": m.prompt_hint,
                }
            )

    # ------------------------------------------------------------------ 1. calibration
    if post_cfg.calibration.enabled:
        records = calibrate_scores(
            records,
            metric_dicts,
            vars(post_cfg.calibration),
        )

    # ------------------------------------------------------------------ 2. drop by missing-metric ratio
    records = filter_by_missing_metrics(
        records,
        metric_dicts,
        post_cfg.min_metrics_present_percent,
    )
    if not records:
        logger.warning("All records were filtered out after missing-metric check.")
        return

    # ------------------------------------------------------------------ 3. split
    split_ratios = vars(post_cfg.splits)
    splits = split_dataset(records, split_ratios)

    # ------------------------------------------------------------------ 4. write
    prefix = post_cfg.final_dataset_name_prefix
    for split_name, data in splits.items():
        if not data:
            logger.warning(f"No records for split '{split_name}'. Skipping file write.")
            continue
        outfile = output_dir / f"{prefix}.{split_name}.{app_config.run.artifact_ext}"
        file_io.save_records(data, outfile)
        logger.info(f"Saved {len(data)} records to {outfile}")

    logger.info("--- Post-processing Stage Completed ---")
