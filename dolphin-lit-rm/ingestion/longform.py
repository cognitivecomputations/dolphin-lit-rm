from typing import List, Dict, Any, Iterator
from pathlib import Path
from datasets import Dataset
from loguru import logger

from dolphin_lit_rm.utils.schema_def import Record, Metadata, generate_record_id
from dolphin_lit_rm.core_configs import DatasetEntryConfig
from dolphin_lit_rm.utils.text_utils import clean_text

def process_longform_dataset_from_files(
    dataset_config: DatasetEntryConfig,
    # raw_dataset: Dataset, # For 'text_dir', raw_dataset is not pre-loaded by get_hf_dataset
    tokenizer_name: str
) -> Iterator[Record]:
    """
    Processes long-form documents (Type C) from a directory of text files.
    Each file is treated as a single document to be segmented later.
    Simplified: Assumes plain text files.
    """
    logger.info(f"Processing long-form dataset (text_dir): {dataset_config.name}")
    
    source_dir = Path(dataset_config.path_or_hf_id)
    if not source_dir.is_dir():
        logger.error(f"Source path for longform dataset {dataset_config.name} is not a directory: {source_dir}")
        return

    file_paths = list(source_dir.glob("*.txt")) # Simple glob for .txt files
    
    # Handle max_items for files
    if dataset_config.max_items:
        num_items = 0
        if isinstance(dataset_config.max_items, str) and "%" in dataset_config.max_items:
            percentage = int(dataset_config.max_items.replace("%", ""))
            num_items = int(len(file_paths) * (percentage / 100))
        elif isinstance(dataset_config.max_items, int):
            num_items = dataset_config.max_items
        
        if num_items > 0:
            file_paths = file_paths[:num_items]
            logger.info(f"Processing up to {len(file_paths)} files from {dataset_config.name} due to max_items={dataset_config.max_items}")


    for i, file_path in enumerate(file_paths):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            
            cleaned_content = clean_text(content)
            if not cleaned_content:
                logger.warning(f"Skipping file {file_path.name} in {dataset_config.name}: Content empty after cleaning.")
                continue

            # For Type C, the original_id is the file name (or relative path)
            source_file_id = file_path.name 

            record_id = generate_record_id(
                source_dataset_name=dataset_config.name,
                source_specific_id=source_file_id
            )

            yield Record(
                id=record_id, # This ID is for the whole document; segmentation will create new IDs for chunks
                source_dataset_name=dataset_config.name,
                prompt=None, # Long-form docs don't have prompts initially
                response=cleaned_content, # Entire document content
                meta=Metadata(
                    prompt_type=None,
                    source_path=str(file_path.resolve()),
                    original_id=source_file_id
                ),
                # Mark for segmentation
                pipeline_internal_status=f"type_C_needs_segmentation"
            )
        except Exception as e:
            logger.error(f"Error processing file {file_path.name} in {dataset_config.name}: {e}")
            continue

def process_longform_dataset_from_hf(
    dataset_config: DatasetEntryConfig,
    raw_dataset: Dataset, # Loaded by file_io.get_hf_dataset
    tokenizer_name: str
) -> Iterator[Record]:
    """
    Processes long-form documents (Type C) from a Hugging Face dataset.
    Each item in raw_dataset is treated as a document.
    """
    logger.info(f"Processing long-form HF dataset: {dataset_config.name}")
    
    text_column = dataset_config.hf_dataset_config.get("text_column", "text") if dataset_config.hf_dataset_config else "text"
    id_column = dataset_config.hf_dataset_config.get("id_column", None) if dataset_config.hf_dataset_config else None

    for i, item in enumerate(raw_dataset):
        try:
            content = item.get(text_column)
            if not content or not isinstance(content, str):
                logger.warning(f"Skipping item {i} in {dataset_config.name}: Empty or invalid text in column '{text_column}'.")
                continue
            
            cleaned_content = clean_text(content)
            if not cleaned_content:
                logger.warning(f"Skipping item {i} in {dataset_config.name}: Text became empty after cleaning.")
                continue

            source_item_id = None
            if id_column and id_column in item:
                source_item_id = str(item[id_column])
            else:
                source_item_id = f"item_{i}"
            
            record_id = generate_record_id(
                source_dataset_name=dataset_config.name,
                source_specific_id=source_item_id
            )

            yield Record(
                id=record_id,
                source_dataset_name=dataset_config.name,
                prompt=None,
                response=cleaned_content,
                meta=Metadata(
                    prompt_type=None,
                    original_id=source_item_id
                ),
                pipeline_internal_status=f"type_C_needs_segmentation"
            )
        except Exception as e:
            logger.error(f"Error processing item {i} in {dataset_config.name}: {e}. Item: {str(item)[:200]}")
            continue