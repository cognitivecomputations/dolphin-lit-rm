from typing import List, Dict, Any, Iterator
from datasets import Dataset
from loguru import logger

from dolphin_lit_rm.utils.schema_def import Record, Metadata, generate_record_id
from dolphin_lit_rm.core_configs import DatasetEntryConfig

def process_sharegpt_dataset(
    dataset_config: DatasetEntryConfig,
    raw_dataset: Dataset, # Loaded by file_io.get_hf_dataset
    tokenizer_name: str
) -> Iterator[Record]:
    """
    Processes a ShareGPT-like dataset.
    Extracts the last user prompt and the assistant reply.
    """
    logger.info(f"Processing ShareGPT-style dataset: {dataset_config.name}")
    
    # Determine column names - this might need to be more flexible or configurable
    # For standard ShareGPT, conversations is a list of dicts with "from" and "value"
    # For other formats, this logic would need to adapt based on dataset_config.hf_dataset_config
    conversations_col = dataset_config.hf_dataset_config.get("conversations_column", "conversations")
    # Some ShareGPT variants might have 'id' or 'dataset' fields at the top level
    id_col = dataset_config.hf_dataset_config.get("id_column", "id")


    for i, item in enumerate(raw_dataset):
        try:
            # Try to get a unique ID from the source item
            source_item_id = item.get(id_col)
            if source_item_id is None: # Fallback if 'id' column is not present or None
                source_item_id = f"item_{i}"

            conversations = item.get(conversations_col)
            if not conversations or not isinstance(conversations, list) or len(conversations) < 2:
                logger.warning(f"Skipping item {source_item_id} in {dataset_config.name}: Not enough turns or invalid format.")
                continue

            # Find the last user prompt and the immediately following assistant reply
            last_user_prompt = None
            assistant_reply = None

            # Iterate backwards to find the last assistant turn preceded by a user turn
            for turn_idx in range(len(conversations) - 1, 0, -1):
                current_turn = conversations[turn_idx]
                prev_turn = conversations[turn_idx-1]

                # Heuristic: 'from' might be 'human', 'user', 'gpt', 'assistant', etc.
                # This needs to be robust or configurable.
                is_current_assistant = 'assistant' in current_turn.get('from','').lower() or \
                                       'gpt' in current_turn.get('from','').lower()
                is_prev_user = 'human' in prev_turn.get('from','').lower() or \
                               'user' in prev_turn.get('from','').lower()

                if is_current_assistant and is_prev_user:
                    assistant_reply = current_turn.get('value')
                    last_user_prompt = prev_turn.get('value')
                    break
            
            if not last_user_prompt or not assistant_reply:
                # logger.warning(f"Skipping item {source_item_id} in {dataset_config.name}: Could not find valid user-assistant pair at the end.")
                continue

            record_id = generate_record_id(
                source_dataset_name=dataset_config.name,
                source_specific_id=str(source_item_id) # Ensure it's a string
            )
            
            yield Record(
                id=record_id,
                source_dataset_name=dataset_config.name,
                prompt=str(last_user_prompt),
                response=str(assistant_reply),
                meta=Metadata(
                    prompt_type="human", # Assuming ShareGPT user prompts are human
                    original_id=str(source_item_id)
                )
            )
        except Exception as e:
            logger.error(f"Error processing item {i} in {dataset_config.name}: {e}. Item: {str(item)[:200]}")
            continue