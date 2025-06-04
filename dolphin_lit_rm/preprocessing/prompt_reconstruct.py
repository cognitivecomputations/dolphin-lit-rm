from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """
    Parallelised drop-in replacement of the original function.
    Uses a thread-pool for LLM calls; concurrency is controlled by
    `llm_settings.max_concurrent_requests` (defaults to 4).
    """
    logger.info("--- Starting Prompt Reconstruction Stage ---")

    # ----------------------------------------------------------------- guard: template present?
    if not PROMPT_TEMPLATE:
        logger.error("Prompt reconstruction template not loaded. Skipping stage.")
        input_dir_pr  = Path(app_config.run.artifacts_dir) / "normalized"
        output_dir_pr = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir_pr.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_pr.glob("*.parquet"):
            target_file = output_dir_pr / dataset_file.name
            if not target_file.exists():
                import shutil
                shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to reconstructed dir as template is missing.")
        return

    pr_cfg        = app_config.run.preprocessing.prompt_reconstruction
    tokenizer     = app_config.run.tokenizer_name
    llm_settings  = app_config.run.get_llm_settings_for_stage("prompt_reconstruction")

    # ----------------------------------------------------------------- guard: LLM config present?
    if not llm_settings.api_base_url or not llm_settings.model_name:
        logger.error("API base URL or model name for prompt reconstruction not configured. Skipping stage.")
        input_dir_pr  = Path(app_config.run.artifacts_dir) / "normalized"
        output_dir_pr = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir_pr.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_pr.glob("*.parquet"):
            target_file = output_dir_pr / dataset_file.name
            if not target_file.exists():
                import shutil
                shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to reconstructed dir as LLM config is missing.")
        return

    max_workers = llm_settings.max_concurrent_requests or 4

    llm_client = llm_api_client.LLMAPIClient(
        api_base_url       = llm_settings.api_base_url,
        api_key            = llm_settings.api_key,
        default_model_name = llm_settings.model_name,
        timeout_seconds    = llm_settings.timeout_seconds or 60,
        max_retries        = llm_settings.max_retries  or 3,
    )

    # ----------------------------------------------------------------- IO paths
    input_dir  = Path(app_config.run.artifacts_dir) / "normalized"
    output_dir = Path(app_config.run.artifacts_dir) / "reconstructed"
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_files = list(input_dir.glob("*.parquet"))
    if not normalized_files:
        logger.warning(f"No normalized files found in {input_dir}. Skipping prompt reconstruction.")
        return

    # ----------------------------------------------------------------- dataset loop
    for dataset_file in normalized_files:
        dataset_name = dataset_file.stem          # e.g. all_sources_normalized
        logger.info(f"Reconstructing prompts for: {dataset_name}")

        output_file = output_dir / f"{dataset_name}_reconstructed.parquet"
        if output_file.exists() and not getattr(app_config, "force_prompt_reconstruction", False):
            logger.info(f"Reconstructed artifact for {dataset_name} already exists. Skipping.")
            continue

        current_dataset = file_io.load_records_from_arrow(dataset_file)
        if not current_dataset or len(current_dataset) == 0:
            logger.warning(f"No records in {dataset_file}. Skipping.")
            file_io.save_records_to_arrow([], output_file)
            continue

        processed_ids = app_config.state_manager.get_processed_ids("prompt_reconstruction", dataset_name)
        records       = current_dataset.to_list()
        id_to_index   = {rec["id"]: idx for idx, rec in enumerate(records)}
        to_process    = [rec for rec in records if rec["id"] not in processed_ids]

        logger.info(f"{len(to_process)} / {len(records)} records need prompt reconstruction.")

        # ------------------------------------------------------------- worker func
        def _reconstruct(rec_dict: dict) -> dict:
            # ensure meta dict exists
            if 'meta' not in rec_dict or rec_dict['meta'] is None:
                rec_dict['meta'] = {}

            recon_prompt = reconstruct_prompt_for_record(
                rec_dict,
                pr_cfg,
                llm_client,
                tokenizer,
            )

            if recon_prompt:
                rec_dict["prompt"] = recon_prompt
                rec_dict["meta"]["prompt_type"] = "reconstructed"
            return rec_dict

        # ---------------------------------------------------------- parallel exec
        updated_ids = []
        if to_process:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_reconstruct, rec): rec["id"] for rec in to_process}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Reconstructing prompts for {dataset_name}",
                    unit="records",
                ):
                    rec_id = futures[fut]
                    try:
                        updated_rec = fut.result()
                    except Exception as e:
                        logger.error(f"Prompt reconstruction failed for record {rec_id}: {e}")
                        continue
                    records[id_to_index[rec_id]] = updated_rec
                    updated_ids.append(rec_id)

        # ---------------------------------------------------------- state + save
        if updated_ids:
            app_config.state_manager.add_processed_ids_batch(
                updated_ids, "prompt_reconstruction", dataset_name
            )

        file_io.save_records_to_arrow(records, output_file)
        logger.info(f"Finished prompt reconstruction for {dataset_name}: wrote {len(records)} records.")

    logger.info("--- Prompt Reconstruction Stage Completed ---")