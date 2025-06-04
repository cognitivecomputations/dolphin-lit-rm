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
    logger.info("--- Starting Post-processing Stage ---")
    post_config = app_config.run.postprocessing
    rubric_config = app_config.rubric
    
    input_dir = Path(app_config.run.artifacts_dir) / "scored"
    output_dir = Path(app_config.run.artifacts_dir) / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # This stage expects one primary scored file (e.g., all_sources_normalized_scored.parquet)
    scored_files = list(input_dir.glob("*_scored.parquet"))
    if not scored_files:
        logger.warning(f"No scored files found in {input_dir} (expected '*_scored.parquet'). Skipping post-processing.")
        return
    
    if len(scored_files) > 1:
        logger.warning(f"Multiple scored files found: {scored_files}. Concatenating them for post-processing.")
        all_scored_datasets = [file_io.load_records_from_arrow(f) for f in scored_files if f.stat().st_size > 0]
        if not all_scored_datasets:
            logger.error("No data in scored files after loading. Skipping post-processing.")
            return
        try:
            combined_scored_dataset = concatenate_datasets(all_scored_datasets)
        except Exception as e:
            logger.error(f"Failed to concatenate scored datasets: {e}. Processing first file only: {scored_files[0]}")
            combined_scored_dataset = file_io.load_records_from_arrow(scored_files[0])
    elif scored_files:
        combined_scored_dataset = file_io.load_records_from_arrow(scored_files[0])
    else: # Should be caught by the first check
        return

    if not combined_scored_dataset or len(combined_scored_dataset) == 0:
        logger.warning("No records in combined scored dataset. Skipping post-processing.")
        return

    logger.info(f"Loaded {len(combined_scored_dataset)} records for post-processing.")
    
    # Convert to list of dicts for easier manipulation
    records_list = combined_scored_dataset.to_list()

    # 1. Calibrate scores (optional)
    if post_config.calibration.enabled:
        records_list = calibrate_scores(records_list, rubric_config.metrics, vars(post_config.calibration))

    # 2. Drop records with > X% missing metrics
    records_list = filter_by_missing_metrics(records_list, rubric_config.metrics, post_config.min_metrics_present_percent)
    if not records_list:
        logger.warning("No records remaining after filtering by missing metrics. Halting post-processing.")
        return

    # 3. Partition into train/val/test
    split_ratios_dict = vars(post_config.splits)
    final_splits = split_dataset(records_list, split_ratios_dict)

    # 4. Write final dataset releases
    dataset_name_prefix = post_config.final_dataset_name_prefix
    for split_name, split_data in final_splits.items():
        if split_data: # Only save if there's data for the split
            output_file = output_dir / f"{dataset_name_prefix}.{split_name}.parquet"
            file_io.save_records_to_arrow(split_data, output_file)
            logger.info(f"Saved final split '{split_name}' with {len(split_data)} records to {output_file}")
        else:
            logger.warning(f"No data for split '{split_name}'. Not saving file.")
            
    logger.info("--- Post-processing Stage Completed ---")