import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

from dolphin_lit_rm.core_configs import AppConfig, ClassificationConfig
from dolphin_lit_rm.utils import text_utils, file_io, state_manager, llm_api_client
from dolphin_lit_rm.utils.schema_def import Record, ClassificationLabels
from dolphin_lit_rm.utils.concurrency import parallel_map

# Load classification prompt template
CLASSIFICATION_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "prompts" / "classification.jinja"
try:
    with CLASSIFICATION_TEMPLATE_PATH.open("r") as f:
        CLASSIFICATION_TEMPLATE = Template(f.read())
except FileNotFoundError:
    logger.error(f"Classification template not found at {CLASSIFICATION_TEMPLATE_PATH}")
    CLASSIFICATION_TEMPLATE = None

# Default genres if not specified in config (example)
DEFAULT_TOP_LEVEL_GENRES = [
    "fiction", "non-fiction", "news", "essay", "poetry", "dialogue", 
    "technical writing", "marketing copy", "academic paper", "script/screenplay", 
    "review", "biography", "historical text", "legal document", "other"
]


def classify_text_zero_shot(
    text: str,
    class_config: ClassificationConfig,
    llm_client_instance: llm_api_client.LLMAPIClient,
    app_config
) -> Tuple[Optional[str], Optional[str]]: # (top_genre, sub_genre)
    """Classifies text using a zero-shot LLM approach."""
    if not CLASSIFICATION_TEMPLATE:
        return "unknown", None

    # Determine genre list for the prompt
    # This could come from class_config.genre_taxonomy_file or a default
    top_genres_for_prompt = getattr(class_config, "top_level_genres_for_prompt", DEFAULT_TOP_LEVEL_GENRES)

    try:
        # Truncate text if too long for classification prompt context
        # A few hundred tokens should be enough for genre.
        # This limit should be configurable or based on model context.
        MAX_TEXT_TOKENS_FOR_CLASSIFICATION = 512 
        if text_utils.count_tokens(text, app_config.run.tokenizer_name) > MAX_TEXT_TOKENS_FOR_CLASSIFICATION:
            # A simple head truncation for now
            # TODO: A more sophisticated summarization or representative snippet selection might be better.
            # For tiktoken, roughly 4 chars per token.
            text_for_prompt = text[:MAX_TEXT_TOKENS_FOR_CLASSIFICATION * 4] 
        else:
            text_for_prompt = text

        api_prompt = CLASSIFICATION_TEMPLATE.render(
            response=text_for_prompt, 
            top_level_genres=top_genres_for_prompt
        )
        messages = [{"role": "user", "content": api_prompt}]
        
        api_response = llm_client_instance.make_request(
            messages=messages,
            temperature=0.0, # Deterministic for classification
            max_tokens=20  # Enough for "Primary Genre Category: <Category Name>"
        )
        
        llm_output = llm_client_instance.get_completion(api_response, is_chat=True)

        if llm_output:
            # Parse the output. Example LLM output: "Primary Genre Category: Fiction"
            # Or just "Fiction"
            # Make parsing robust.
            llm_output_lower = llm_output.lower()
            
            # Try to find a match from the provided list first for robustness
            for genre in top_genres_for_prompt + ["unknown"]: # Add unknown to the check list
                if genre.lower() in llm_output_lower:
                    # TODO: Add confidence check if LLM provides it, or use regex with specific format.
                    # For now, direct match is considered confident enough.
                    # Sub-genre classification would be a second step or a more complex prompt.
                    return genre.lower(), None # Return normalized genre

            # If no direct match from list, use the LLM output if it's simple, else unknown
            # This part is heuristic and might need refinement.
            # E.g. if LLM says "This is clearly Fiction.", extract "Fiction".
            # A simple fallback: if the output is short and seems like a label.
            if len(llm_output.split()) <= 3: # Arbitrary short length
                 # Basic sanitization: remove "Primary Genre Category:" prefix if present
                cleaned_output = re.sub(r"(?i)primary genre category:\s*", "", llm_output).strip()
                if cleaned_output.lower() in [g.lower() for g in top_genres_for_prompt]: # Check again after cleaning
                    return cleaned_output.lower(), None

            logger.warning(f"Classification output '{llm_output}' not directly parsable or not in known list. Defaulting to 'unknown'.")
            return "unknown", None
        else:
            logger.warning("Classification LLM returned empty response. Defaulting to 'unknown'.")
            return "unknown", None
            
    except Exception as e:
        logger.error(f"Error during zero-shot classification: {e}")
        return "unknown", None # Default to unknown on error


def run_classification_stage(app_config: AppConfig):
    """
    Parallelised version of the original run_classification_stage.
    Concurrency level comes from
    app_config.run.get_llm_settings_for_stage("classification").max_concurrent_requests
    (defaults to 4 if unset).
    """
    # ---------------------------------------------------- boiler-plate setup
    global app_config_global                # needed by classify_text_zero_shot
    app_config_global = app_config

    logger.info("--- Starting Classification Stage ---")

    # 1. template check ------------------------------------------------------
    if not CLASSIFICATION_TEMPLATE:
        logger.error("Classification template not loaded. Skipping stage.")
        # passthrough segmented → classified (unchanged from original)
        input_dir_clf  = Path(app_config.run.artifacts_dir) / "segmented"
        output_dir_clf = Path(app_config.run.artifacts_dir) / "classified"
        output_dir_clf.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_clf.glob("*.parquet"):
            target_file = output_dir_clf / dataset_file.name
            if not target_file.exists():
                import shutil
                shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to classified dir as template is missing.")
        return

    # 2. read config + build client -----------------------------------------
    class_config          = app_config.run.classification
    llm_settings          = app_config.run.get_llm_settings_for_stage("classification")

    if not llm_settings.api_base_url or not llm_settings.model_name:
        logger.error("API base URL or model name for classification not configured. Skipping stage.")
        # passthrough segmented → classified (unchanged from original)
        input_dir_clf  = Path(app_config.run.artifacts_dir) / "segmented"
        output_dir_clf = Path(app_config.run.artifacts_dir) / "classified"
        output_dir_clf.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_clf.glob("*.parquet"):
            target_file = output_dir_clf / dataset_file.name
            if not target_file.exists():
                import shutil
                shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to classified dir as LLM config is missing.")
        return

    max_workers = llm_settings.max_concurrent_requests or 4

    llm_client = llm_api_client.LLMAPIClient(
        api_base_url       = llm_settings.api_base_url,
        api_key            = llm_settings.api_key,
        default_model_name = llm_settings.model_name,
        timeout_seconds    = llm_settings.timeout_seconds or 60,
        max_retries        = llm_settings.max_retries  or 3,
    )

    # 3. iterate datasets ----------------------------------------------------
    input_dir  = Path(app_config.run.artifacts_dir) / "segmented"
    output_dir = Path(app_config.run.artifacts_dir) / "classified"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_file in input_dir.glob("*.parquet"):
        dataset_name = dataset_file.stem
        logger.info(f"Classifying dataset: {dataset_name}")

        output_file = output_dir / f"{dataset_name}.parquet"
        if output_file.exists() and not getattr(app_config, "force_classification", False):
            logger.info(f"Classified artifact for {dataset_name} already exists. Skipping.")
            continue

        current_dataset = file_io.load_records_from_arrow(dataset_file)
        if not current_dataset or len(current_dataset) == 0:
            logger.warning(f"No records in {dataset_file} for classification. Skipping.")
            file_io.save_records_to_arrow([], output_file)
            continue

        stage_processed_ids = app_config.state_manager.get_processed_ids("classification", dataset_name)

        # convert to list for in-place updates
        records = current_dataset.to_list()
        id_to_index = {rec["id"]: idx for idx, rec in enumerate(records)}
        to_classify = [
            rec for rec in records
            if rec["id"] not in stage_processed_ids
               or not rec.get("classification", {}).get("top")
        ]

        logger.info(f"{len(to_classify)} / {len(records)} records need classification.")

        # 4. helper for worker threads ---------------------------------------
        def _classify_record(rec_dict: dict) -> dict:
            response_text = rec_dict.get("response")
            if not response_text:
                top_genre, sub_genre = "unknown", None
            else:
                top_genre, sub_genre = classify_text_zero_shot(
                    response_text,
                    class_config,
                    llm_client,
                    app_config,
                )

            rec_dict.setdefault("classification", {})
            rec_dict["classification"]["top"] = top_genre
            rec_dict["classification"]["sub"] = sub_genre
            return rec_dict  # updated

        # 5. parallel execution ---------------------------------------------
        updated_ids = []
        if to_classify:  # skip pool if everything already done
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_classify_record, rec): rec["id"] for rec in to_classify}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Classifying {dataset_name}",
                    unit="records",
                ):
                    rec_id = futures[fut]
                    try:
                        updated_rec = fut.result()
                    except Exception as e:
                        logger.error(f"Classification failed for record {rec_id}: {e}")
                        continue
                    # place updated record back into master list
                    records[id_to_index[rec_id]] = updated_rec
                    updated_ids.append(rec_id)

        # 6. state manager & save -------------------------------------------
        if updated_ids:
            app_config.state_manager.add_processed_ids_batch(
                updated_ids, "classification", dataset_name
            )

        file_io.save_records_to_arrow(records, output_file)
        logger.info(f"Finished classification for {dataset_name}: wrote {len(records)} records.")

    logger.info("--- Classification Stage Completed ---")

# Global app_config for text_utils.count_tokens, this is a temporary workaround.
# Proper way is to pass tokenizer_name through function calls.
app_config_global: Optional[AppConfig] = None