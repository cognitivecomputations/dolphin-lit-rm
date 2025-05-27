from pathlib import Path
from typing import List, Dict, Any, Counter as TypingCounter
from collections import Counter
import random
from datasets import Dataset, concatenate_datasets
from loguru import logger
from tqdm import tqdm

from dolphin_lit_rm.core_configs import AppConfig, NormalizationQuotaConfig
from dolphin_lit_rm.utils import file_io, state_manager
from dolphin_lit_rm.utils.schema_def import Record

def run_normalization_stage(app_config: AppConfig):
    logger.info("--- Starting Normalization (Quota Sampling) Stage ---")
    norm_config = app_config.run.normalization
    
    input_dir = Path(app_config.run.artifacts_dir) / "classified"
    output_dir = Path(app_config.run.artifacts_dir) / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output is a single combined and normalized dataset
    output_file = output_dir / "all_sources_normalized.parquet"
    
    # Resumability for normalization is tricky if quotas change.
    # If output_file exists, we assume it's correctly normalized from a previous run.
    if output_file.exists() and not getattr(app_config, "force_normalization", False):
        logger.info(f"Normalized artifact {output_file} already exists. Skipping normalization.")
        return

    # 1. Load and concatenate all classified datasets
    all_classified_datasets = []
    for dataset_file in input_dir.glob("*.parquet"):
        logger.info(f"Loading classified data from {dataset_file.name} for normalization.")
        ds = file_io.load_records_from_arrow(dataset_file)
        if ds and len(ds) > 0:
            all_classified_datasets.append(ds)
    
    if not all_classified_datasets:
        logger.warning("No classified datasets found to normalize. Skipping.")
        file_io.save_records_to_arrow([], output_file) # Save empty
        return

    # Concatenate into a single Hugging Face Dataset
    # This might be memory intensive for very large total datasets.
    # Consider iterating and sampling if memory becomes an issue.
    try:
        combined_dataset = concatenate_datasets(all_classified_datasets)
        logger.info(f"Combined {len(all_classified_datasets)} classified datasets into one with {len(combined_dataset)} records.")
    except Exception as e:
        logger.error(f"Error concatenating datasets for normalization: {e}. Trying to process record by record (slower).")
        # Fallback: process as list of dicts (more memory but might avoid some HF concat issues)
        temp_records = []
        for ds in all_classified_datasets:
            temp_records.extend(ds.to_list())
        if not temp_records:
            logger.warning("No records after attempting to combine. Skipping normalization.")
            file_io.save_records_to_arrow([], output_file)
            return
        combined_dataset = Dataset.from_list(temp_records)


    # 2. Group records by class (e.g., 'class.top')
    # Convert to list of dicts for easier manipulation here
    records_list = combined_dataset.to_list()
    
    grouped_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for record_dict in tqdm(records_list, desc="Grouping records by class", unit="records"):
        # Ensure 'classification' and 'top' exist
        class_info = record_dict.get("classification", {})
        if not isinstance(class_info, dict): # Handle if it's not a dict (e.g. None)
            class_info = {}
            
        top_class = class_info.get("top", "unknown") # Default to 'unknown' if not classified
        if top_class is None: # Handle if 'top' is explicitly None
            top_class = "unknown"

        if top_class not in grouped_by_class:
            grouped_by_class[top_class] = []
        grouped_by_class[top_class].append(record_dict)

    logger.info(f"Records grouped into {len(grouped_by_class)} classes.")
    for class_name, items in grouped_by_class.items():
        logger.info(f"  Class '{class_name}': {len(items)} records")

    # 3. Apply quotas
    final_normalized_records: List[Dict[str, Any]] = []
    quotas_top_class = norm_config.quotas.get("class.top", {})
    default_quota = norm_config.default_quota_per_class

    for class_name, items_in_class in grouped_by_class.items():
        quota = quotas_top_class.get(class_name, default_quota)
        
        if len(items_in_class) > quota:
            logger.info(f"Sampling {quota} records for class '{class_name}' from {len(items_in_class)} available.")
            # Simple random sampling. For reproducibility, consider seeding random.
            # random.seed(42) # Or get seed from config
            final_normalized_records.extend(random.sample(items_in_class, quota))
        else:
            logger.info(f"Taking all {len(items_in_class)} records for class '{class_name}' (quota: {quota}).")
            final_normalized_records.extend(items_in_class)
            
    logger.info(f"Total records after normalization: {len(final_normalized_records)}")

    if final_normalized_records:
        # Shuffle the final combined list to mix datasets and classes
        random.shuffle(final_normalized_records) 
        file_io.save_records_to_arrow(final_normalized_records, output_file)
        logger.info(f"Saved {len(final_normalized_records)} normalized records to {output_file}")
    else:
        logger.warning("No records retained after normalization. Saving empty file.")
        file_io.save_records_to_arrow([], output_file)

    logger.info("--- Normalization (Quota Sampling) Stage Completed ---")