import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template

from dolphin_lit_rm.core_configs import AppConfig, ScoringConfig, RubricConfig, MetricConfig
from dolphin_lit_rm.utils import text_utils, file_io, state_manager, llm_api_client
from dolphin_lit_rm.utils.schema_def import Record

# Load scoring prompt template
SCORING_TEMPLATE_PATH = Path(__file__).parent.parent / "config" / "prompts" / "scoring.jinja"
try:
    with SCORING_TEMPLATE_PATH.open("r") as f:
        SCORING_TEMPLATE = Template(f.read())
except FileNotFoundError:
    logger.error(f"Scoring template not found at {SCORING_TEMPLATE_PATH}")
    SCORING_TEMPLATE = None


def parse_scores_from_llm_response(
    llm_response: str, 
    metrics: List[MetricConfig]
) -> Dict[str, Optional[float]]:
    """
    Parses multiple scores from a single LLM response.
    Assumes LLM output format like:
    Score for metric_name_1 (0.0-1.0):
    0.75
    Score for metric_name_2 (0.0-1.0):
    0.60
    """
    parsed_scores: Dict[str, Optional[float]] = {metric.name: None for metric in metrics}
    if not llm_response:
        return parsed_scores

    # Iterate through metrics and try to find their score in the text
    # This regex is an example and might need to be very robust
    for metric in metrics:
        # Regex to find "Score for <Metric Name> (0.0-1.0):\s*([0-9.]+)"
        # Escape metric name for regex, handle potential case differences if necessary
        pattern = re.compile(
            rf"Score for {re.escape(metric.name)} \(0\.0-1\.0\):\s*([0-1]\.\d+|[01])", 
            re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(llm_response)
        if match:
            try:
                score_str = match.group(1)
                score_float = float(score_str)
                # Clamp score to [0,1] just in case LLM hallucinates outside range
                parsed_scores[metric.name] = max(0.0, min(1.0, score_float))
            except ValueError:
                logger.warning(f"Could not parse score for metric '{metric.name}' from '{score_str}'.")
        else:
            logger.warning(f"Could not find score for metric '{metric.name}' in LLM response.")
            # Attempt a simpler parse if the above fails: find metric name then a number
            # This is a fallback and less reliable
            simple_pattern = re.compile(rf"{re.escape(metric.name)}[^0-9]*([0-1]\.\d+|[01])", re.IGNORECASE)
            simple_match = simple_pattern.search(llm_response)
            if simple_match:
                try:
                    score_str = simple_match.group(1)
                    score_float = float(score_str)
                    parsed_scores[metric.name] = max(0.0, min(1.0, score_float))
                    logger.info(f"Used fallback parsing for metric '{metric.name}', got: {score_float}")
                except ValueError:
                    pass # Already logged primary failure

    return parsed_scores


def score_record_with_judge(
    record_dict: Dict[str, Any],
    scoring_config: ScoringConfig,
    rubric_metrics: List[MetricConfig],
    llm_client_instance: llm_api_client.LLMAPIClient
) -> Dict[str, Optional[float]]:
    """Scores a single record using the judge model for all metrics in one call."""
    if not SCORING_TEMPLATE:
        return {metric.name: None for metric in rubric_metrics}

    response_text = record_dict.get("response")
    if not response_text:
        logger.warning(f"Skipping scoring for record {record_dict.get('id')}: no response text.")
        return {metric.name: None for metric in rubric_metrics}

    try:
        # Render prompt for all metrics
        api_prompt = SCORING_TEMPLATE.render(response=response_text, metrics=rubric_metrics)
        messages = [{"role": "user", "content": api_prompt}]
        
        # Calculate max_tokens needed: num_metrics * (tokens_for_score_value + tokens_for_label_and_newlines)
        # Example: 6 metrics * (approx 4 tokens for "0.xx" + ~10 for "Score for X: \n") = ~84 tokens
        # Add buffer.
        estimated_max_tokens = len(rubric_metrics) * (scoring_config.max_tokens_per_metric_response + 15)
        
        api_response = llm_client_instance.make_request(
            messages=messages,
            temperature=0.0, # Deterministic for scoring
            max_tokens=max(estimated_max_tokens, 100) # Ensure a minimum
        )
        
        llm_output_text = llm_client_instance.get_completion(api_response, is_chat=True)
        
        if llm_output_text:
            return parse_scores_from_llm_response(llm_output_text, rubric_metrics)
        else:
            logger.warning(f"Scoring failed for record {record_dict.get('id')}: Judge LLM returned empty response.")
            return {metric.name: None for metric in rubric_metrics}
            
    except Exception as e:
        logger.error(f"Error during scoring record {record_dict.get('id')}: {e}", exc_info=True)
        return {metric.name: None for metric in rubric_metrics}


def run_scoring_stage(app_config: AppConfig):
    logger.info("--- Starting Rubric Scoring Stage ---")
    if not SCORING_TEMPLATE:
        logger.error("Scoring template not loaded. Skipping stage.")
        # Copy reconstructed files to scored directory
        input_dir_score = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir_score = Path(app_config.run.artifacts_dir) / "scored"
        output_dir_score.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_score.glob("*.parquet"):
            target_file = output_dir_score / dataset_file.name # Or add _scored suffix
            if not target_file.exists():
                 import shutil
                 shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to scored dir as template is missing.")
        return

    scoring_config = app_config.run.scoring
    rubric_metrics = app_config.rubric.metrics
    if not rubric_metrics:
        logger.error("No metrics defined in rubric. Skipping scoring stage.")
        # Copy files as above
        return

    llm_settings_for_stage = app_config.run.get_llm_settings_for_stage("scoring")
    if not llm_settings_for_stage.api_base_url or not llm_settings_for_stage.model_name:
        logger.error("API base URL or model name for scoring is not configured. Skipping stage.")
        # Copy files as above
        return

    llm_client_instance = llm_api_client.LLMAPIClient(
        api_base_url=llm_settings_for_stage.api_base_url,
        api_key=llm_settings_for_stage.api_key,
        default_model_name=llm_settings_for_stage.model_name,
        timeout_seconds=llm_settings_for_stage.timeout_seconds or 120, # Longer timeout for scoring
        max_retries=llm_settings_for_stage.max_retries or 2
    )
    
    input_dir = Path(app_config.run.artifacts_dir) / "reconstructed" # Input from reconstructed data
    output_dir = Path(app_config.run.artifacts_dir) / "scored"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # This stage operates on the single combined reconstructed file (or multiple if structure differs)
    reconstructed_files = list(input_dir.glob("*_reconstructed.parquet")) # Matches output from prompt_reconstruct
    if not reconstructed_files:
        logger.warning(f"No reconstructed files found in {input_dir} (expected '*_reconstructed.parquet'). Skipping scoring.")
        return

    for dataset_file in reconstructed_files: # Should typically be one combined file
        # Output name: e.g., all_sources_normalized_reconstructed_scored.parquet
        base_name = dataset_file.stem.replace("_reconstructed", "")
        output_file = output_dir / f"{base_name}_scored.parquet"
        
        logger.info(f"Scoring dataset: {dataset_file.name}")
        
        if output_file.exists() and not getattr(app_config, "force_scoring", False):
            logger.info(f"Scored artifact for {dataset_file.name} already exists at {output_file}. Skipping.")
            continue

        current_dataset = file_io.load_records_from_arrow(dataset_file)
        if not current_dataset or len(current_dataset) == 0:
            logger.warning(f"No records in {dataset_file} for scoring. Skipping.")
            file_io.save_records_to_arrow([], output_file)
            continue
            
        # State management for this stage (on the combined dataset)
        # Use a unique name for state tracking, e.g., base_name
        stage_processed_ids = app_config.state_manager.get_processed_ids("scoring", base_name)
        
        updated_records = []
        ids_processed_in_this_run = []

        records_to_process = current_dataset.to_list()

        for record_dict in tqdm(records_to_process, desc=f"Scoring {dataset_file.name}", unit="records"):
            record_id = record_dict["id"]
            # Check if all scores are already present for this record_id
            all_metrics_present = True
            if record_dict.get("scores"): # Check if scores dict exists
                for metric_conf in rubric_metrics:
                    if metric_conf.name not in record_dict["scores"] or record_dict["scores"][metric_conf.name] is None:
                        all_metrics_present = False
                        break
            else: # No scores dict at all
                all_metrics_present = False

            if record_id in stage_processed_ids and all_metrics_present:
                updated_records.append(record_dict)
                continue

            # Ensure 'scores' field exists and is a dict
            if "scores" not in record_dict or not isinstance(record_dict.get("scores"), dict):
                 record_dict["scores"] = {}
            
            # Only score if not all metrics are present or if forced
            if not all_metrics_present or getattr(app_config, "force_scoring", False):
                scores = score_record_with_judge(
                    record_dict, scoring_config, rubric_metrics, llm_client_instance
                )
                record_dict["scores"].update(scores) # Update existing scores dict
            
            updated_records.append(record_dict)
            ids_processed_in_this_run.append(record_id)

            if len(ids_processed_in_this_run) >= 50: # Smaller batch for scoring due to cost/time
                app_config.state_manager.add_processed_ids_batch(ids_processed_in_this_run, "scoring", base_name)
                ids_processed_in_this_run.clear()
        
        if ids_processed_in_this_run:
            app_config.state_manager.add_processed_ids_batch(ids_processed_in_this_run, "scoring", base_name)

        if updated_records:
            file_io.save_records_to_arrow(updated_records, output_file)
            logger.info(f"Finished scoring for {dataset_file.name}: {len(updated_records)} records processed.")
        else:
            logger.warning(f"No records after scoring for {dataset_file.name}.")
            file_io.save_records_to_arrow([], output_file)

    logger.info("--- Rubric Scoring Stage Completed ---")