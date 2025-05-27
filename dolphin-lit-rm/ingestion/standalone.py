from typing import List, Dict, Any, Iterator
from datasets import Dataset
from loguru import logger

from dolphin_lit_rm.utils.schema_def import Record, Metadata, generate_record_id
from dolphin_lit_rm.core_configs import DatasetEntryConfig
from dolphin_lit_rm.utils.text_utils import clean_text

def process_standalone_dataset(
    dataset_config: DatasetEntryConfig,
    raw_dataset: Dataset, # Loaded by file_io.get_hf_dataset
    tokenizer_name: str
) -> Iterator[Record]:
    """
    Processes datasets of standalone short pieces (Type B).
    Each item in raw_dataset is treated as a 'response'.
    """
    logger.info(f"Processing standalone dataset: {dataset_config.name}")
    
    # Determine the text column. For 'text' format, it's usually 'text'.
    # For CSV/JSONL, it might be specified in hf_dataset_config.
    text_column = dataset_config.hf_dataset_config.get("text_column", "text") if dataset_config.hf_dataset_config else "text"
    id_column = dataset_config.hf_dataset_config.get("id_column", None) if dataset_config.hf_dataset_config else None


    for i, item in enumerate(raw_dataset):
        try:
            response_text = item.get(text_column)
            if not response_text or not isinstance(response_text, str):
                logger.warning(f"Skipping item {i} in {dataset_config.name}: Empty or invalid text in column '{text_column}'.")
                continue
            
            response_text = clean_text(response_text)
            if not response_text:
                logger.warning(f"Skipping item {i} in {dataset_config.name}: Text became empty after cleaning.")
                continue

            source_item_id = None
            if id_column and id_column in item:
                source_item_id = str(item[id_column])
            else: # Fallback if no ID column
                source_item_id = f"item_{i}"

            record_id = generate_record_id(
                source_dataset_name=dataset_config.name,
                source_specific_id=source_item_id,
                content_for_hash=response_text if not source_item_id or source_item_id == f"item_{i}" else None
            )

            yield Record(
                id=record_id,
                source_dataset_name=dataset_config.name,
                prompt=None,
                response=response_text,
                meta=Metadata(
                    prompt_type=None, # Will be 'reconstructed' later if applicable
                    original_id=source_item_id
                )
            )
        except Exception as e:
            logger.error(f"Error processing item {i} in {dataset_config.name}: {e}. Item: {str(item)[:200]}")
            continue