from pathlib import Path
from typing import List, Iterator
from loguru import logger
from tqdm import tqdm

from dolphin_lit_rm.core_configs import AppConfig, DatasetEntryConfig
from dolphin_lit_rm.utils.schema_def import Record
from dolphin_lit_rm.utils import file_io, state_manager
from dolphin_lit_rm.ingestion import sharegpt, standalone, longform

def ingest_dataset(
    dataset_config: DatasetEntryConfig,
    app_config: AppConfig,
    # state_manager_instance: state_manager.StateManager # Passed via app_config
) -> List[Record]:
    """Ingests a single dataset based on its configuration."""
    
    # Resumability: For ingestion, it's tricky if source files change.
    # A simple check could be if the output raw file already exists and skip.
    # However, if max_items changes, we might want to re-ingest.
    # For now, let's assume if raw output exists, we skip, unless forced.
    # More robust resumability is handled by downstream stages using record IDs.

    output_dir = Path(app_config.run.artifacts_dir) / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_config.name}.parquet" # Using parquet for HF compatibility

    # Basic skip if output exists (can be made more sophisticated)
    # if output_file.exists() and not app_config.force_stage_rerun.get("ingest"): # Assuming a force flag
    #     logger.info(f"Raw file {output_file} already exists for dataset {dataset_config.name}. Skipping ingestion.")
    #     # To use this, we'd need to load it and return, or signal to skip.
    #     # For simplicity, we'll re-process and overwrite for now, or rely on downstream stages for ID-based skipping.
    #     # A true "skip" would mean this function returns an empty list and the main loop loads existing.

    records_buffer = []
    
    try:
        if dataset_config.format == "text_dir": # Special handling for directory of text files
            if dataset_config.type != "C":
                logger.warning(f"Dataset {dataset_config.name} is format 'text_dir' but not Type C. Treating as Type C.")
            
            record_iterator = longform.process_longform_dataset_from_files(
                dataset_config, app_config.run.tokenizer_name
            )
        else: # All other formats loadable by datasets.load_dataset
            raw_hf_dataset = file_io.get_hf_dataset(
                dataset_config.path_or_hf_id,
                dataset_config.format,
                vars(dataset_config), # Pass the whole dataset_config entry
                dataset_config.max_items
            )
            if not raw_hf_dataset or len(raw_hf_dataset) == 0:
                logger.warning(f"No data loaded for dataset {dataset_config.name}. Skipping.")
                return []

            if dataset_config.type == "A": # Prompt+Assistant
                record_iterator = sharegpt.process_sharegpt_dataset(
                    dataset_config, raw_hf_dataset, app_config.run.tokenizer_name
                )
            elif dataset_config.type == "B": # Standalone
                record_iterator = standalone.process_standalone_dataset(
                    dataset_config, raw_hf_dataset, app_config.run.tokenizer_name
                )
            elif dataset_config.type == "C": # Long-form (from HF dataset)
                 record_iterator = longform.process_longform_dataset_from_hf(
                    dataset_config, raw_hf_dataset, app_config.run.tokenizer_name
                )
            else:
                logger.error(f"Unknown dataset type '{dataset_config.type}' for {dataset_config.name}")
                return []
        
        for record in tqdm(record_iterator, desc=f"Ingesting {dataset_config.name}", unit="records"):
            records_buffer.append(record.model_dump()) # Convert Pydantic model to dict for saving

    except Exception as e:
        logger.error(f"Failed to ingest dataset {dataset_config.name}: {e}", exc_info=True)
        return [] # Return empty list on failure for this dataset

    if records_buffer:
        file_io.save_records_to_arrow(records_buffer, output_file)
        logger.info(f"Successfully ingested {len(records_buffer)} records from {dataset_config.name} to {output_file}")
    else:
        logger.warning(f"No records were produced for dataset {dataset_config.name}.")
        # Save an empty file to signify completion if necessary
        file_io.save_records_to_arrow([], output_file)


    # This function now writes its own output. The CLI will call this per dataset.
    # It doesn't return records anymore to save memory for large ingestions.
    # Downstream stages will load from the saved files.

def run_ingestion_stage(app_config: AppConfig):
    logger.info("--- Starting Ingestion Stage ---")
    raw_artifacts_dir = Path(app_config.run.artifacts_dir) / "raw"
    raw_artifacts_dir.mkdir(parents=True, exist_ok=True)

    for ds_config in app_config.datasets.datasets:
        output_file = raw_artifacts_dir / f"{ds_config.name}.parquet"
        # Simple resumability: if output file exists, skip this dataset's ingestion.
        # Add a force flag in AppConfig if needed to override this.
        if output_file.exists() and not getattr(app_config, "force_ingest", False):
             logger.info(f"Raw artifact for {ds_config.name} already exists at {output_file}. Skipping ingestion.")
             continue
        
        logger.info(f"Starting ingestion for dataset: {ds_config.name}")
        ingest_dataset(ds_config, app_config) # This function now saves its own output

    logger.info("--- Ingestion Stage Completed ---")