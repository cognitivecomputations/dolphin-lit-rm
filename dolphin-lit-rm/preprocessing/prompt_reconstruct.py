from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template

from dolphin_lit_rm.core_configs import AppConfig, PromptReconstructionConfig
from dolphin_lit_rm.utils import text_utils, file_io, state_manager, llm_api_client
from dolphin_lit_rm.utils.schema_def import Record

# Load prompt template
PROMPT_RECONSTRUCTION_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "prompts" / "prompt_reconstruction.jinja"
try:
    with PROMPT_RECONSTRUCTION_TEMPLATE_PATH.open("r") as f:
        PROMPT_TEMPLATE = Template(f.read())
except FileNotFoundError:
    logger.error(f"Prompt reconstruction template not found at {PROMPT_RECONSTRUCTION_TEMPLATE_PATH}")
    PROMPT_TEMPLATE = None


def reconstruct_prompt_for_record(
    record_dict: Dict[str, Any],
    pr_config: PromptReconstructionConfig,
    llm_client: llm_api_client.LLMAPIClient,
    tokenizer_name: str
) -> Optional[str]:
    """Generates a prompt for a single record if needed and conditions are met."""
    if not PROMPT_TEMPLATE:
        return None

    response_text = record_dict.get("response")
    if record_dict.get("prompt") or not response_text: # Skip if prompt exists or no response
        return record_dict.get("prompt") # Return existing prompt

    # Gatekeeper: len(response) <= MAX_PROMPT_REC_TOK
    response_tokens = text_utils.count_tokens(response_text, tokenizer_name)
    if response_tokens > pr_config.max_response_tokens_for_reconstruction:
        # logger.debug(f"Skipping prompt reconstruction for record {record_dict.get('id')}: response too long ({response_tokens} tokens).")
        return None

    try:
        api_prompt = PROMPT_TEMPLATE.render(response=response_text)
        # Use chat messages format for newer models
        messages = [{"role": "user", "content": api_prompt}]
        
        api_response = llm_client.make_request(
            messages=messages,
            # model_name will be picked from llm_client's default or pr_config override
            temperature=0.1, # Slightly creative but mostly deterministic
            max_tokens=pr_config.reconstructed_prompt_max_chars // 2, # Estimate tokens from chars
        )
        
        reconstructed_prompt = llm_client.get_completion(api_response, is_chat=True)

        if reconstructed_prompt:
            # Post-process: strip, truncate
            reconstructed_prompt = reconstructed_prompt.strip()
            if len(reconstructed_prompt) > pr_config.reconstructed_prompt_max_chars:
                reconstructed_prompt = reconstructed_prompt[:pr_config.reconstructed_prompt_max_chars].rsplit(' ', 1)[0] + "..."
            return reconstructed_prompt
        else:
            logger.warning(f"Prompt reconstruction failed for record {record_dict.get('id')}: LLM returned empty response.")
            return None
            
    except Exception as e:
        logger.error(f"Error during prompt reconstruction for record {record_dict.get('id')}: {e}")
        return None


def run_prompt_reconstruction_stage(app_config: AppConfig):
    logger.info("--- Starting Prompt Reconstruction Stage ---")
    if not PROMPT_TEMPLATE:
        logger.error("Prompt reconstruction template not loaded. Skipping stage.")
        # Copy classified files to reconstructed directory
        input_dir_pr = Path(app_config.run.artifacts_dir) / "normalized" # Input from normalized
        output_dir_pr = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir_pr.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_pr.glob("*.parquet"):
            target_file = output_dir_pr / dataset_file.name
            if not target_file.exists():
                 import shutil
                 shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to reconstructed dir as template is missing.")
        return

    pr_config = app_config.run.preprocessing.prompt_reconstruction
    tokenizer_name = app_config.run.tokenizer_name
    
    # Get LLM settings, merging defaults with stage-specific overrides
    llm_settings_for_stage = app_config.run.get_llm_settings_for_stage("prompt_reconstruction")
    if not llm_settings_for_stage.api_base_url or not llm_settings_for_stage.model_name:
        logger.error("API base URL or model name for prompt reconstruction is not configured. Skipping stage.")
        # Copy files like above
        input_dir_pr = Path(app_config.run.artifacts_dir) / "normalized"
        output_dir_pr = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir_pr.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_pr.glob("*.parquet"): # Ensure this matches actual input dir
            target_file = output_dir_pr / dataset_file.name
            if not target_file.exists():
                 import shutil
                 shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to reconstructed dir as LLM config is missing.")
        return

    llm_client_instance = llm_api_client.LLMAPIClient(
        api_base_url=llm_settings_for_stage.api_base_url,
        api_key=llm_settings_for_stage.api_key,
        default_model_name=llm_settings_for_stage.model_name,
        timeout_seconds=llm_settings_for_stage.timeout_seconds or 60,
        max_retries=llm_settings_for_stage.max_retries or 3
    )
    
    input_dir = Path(app_config.run.artifacts_dir) / "normalized" # Input from normalized data
    output_dir = Path(app_config.run.artifacts_dir) / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # This stage operates on the single combined normalized file
    # (or multiple if normalization outputs per original dataset, adjust glob then)
    normalized_files = list(input_dir.glob("*.parquet"))
    if not normalized_files:
        logger.warning(f"No normalized files found in {input_dir}. Skipping prompt reconstruction.")
        return

    for dataset_file in normalized_files: # Should typically be one combined file
        dataset_name = dataset_file.stem # e.g., "all_sources_normalized"
        logger.info(f"Reconstructing prompts for: {dataset_name}")
        
        output_file = output_dir / f"{dataset_name}_reconstructed.parquet" # Add suffix
        if output_file.exists() and not getattr(app_config, "force_prompt_reconstruction", False):
            logger.info(f"Reconstructed artifact for {dataset_name} already exists. Skipping.")
            continue

        current_dataset = file_io.load_records_from_arrow(dataset_file)
        if not current_dataset or len(current_dataset) == 0:
            logger.warning(f"No records in {dataset_file}. Skipping.")
            file_io.save_records_to_arrow([], output_file)
            continue
            
        # State management for this stage (on the combined dataset)
        stage_processed_ids = app_config.state_manager.get_processed_ids("prompt_reconstruction", dataset_name)
        
        updated_records = []
        ids_processed_in_this_run = []

        # Convert to list of dicts for processing
        records_to_process = current_dataset.to_list()

        for record_dict in tqdm(records_to_process, desc=f"Reconstructing prompts for {dataset_name}", unit="records"):
            record_id = record_dict["id"]
            if record_id in stage_processed_ids:
                # If already processed (e.g. in a resumed run), just add it
                # This assumes the content of the record_dict from file is what we want
                updated_records.append(record_dict)
                continue

            # Ensure 'meta' is a dict
            if 'meta' not in record_dict or record_dict['meta'] is None:
                record_dict['meta'] = {}

            reconstructed_prompt = reconstruct_prompt_for_record(
                record_dict, pr_config, llm_client_instance, tokenizer_name
            )
            
            if reconstructed_prompt:
                record_dict["prompt"] = reconstructed_prompt
                # Ensure meta is a dict before trying to update it
                if not isinstance(record_dict.get("meta"), dict):
                    record_dict["meta"] = {}
                record_dict["meta"]["prompt_type"] = "reconstructed"
            
            updated_records.append(record_dict)
            ids_processed_in_this_run.append(record_id)

            # Batch add to state manager periodically to save on I/O
            if len(ids_processed_in_this_run) >= 100: # Configurable batch size
                app_config.state_manager.add_processed_ids_batch(ids_processed_in_this_run, "prompt_reconstruction", dataset_name)
                ids_processed_in_this_run.clear()
        
        # Add any remaining processed IDs
        if ids_processed_in_this_run:
            app_config.state_manager.add_processed_ids_batch(ids_processed_in_this_run, "prompt_reconstruction", dataset_name)

        if updated_records:
            file_io.save_records_to_arrow(updated_records, output_file)
            logger.info(f"Finished prompt reconstruction for {dataset_name}: {len(updated_records)} records processed/updated.")
        else:
            logger.warning(f"No records after prompt reconstruction for {dataset_name}.")
            file_io.save_records_to_arrow([], output_file)

    logger.info("--- Prompt Reconstruction Stage Completed ---")