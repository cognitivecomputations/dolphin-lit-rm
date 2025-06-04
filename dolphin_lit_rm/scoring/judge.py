import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

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
            max_tokens= 2048 #max(estimated_max_tokens, 100) # Ensure a minimum
        )
        
        llm_output_text = llm_client_instance.get_completion(api_response, is_chat=True)
        print(llm_output_text)
        
        if llm_output_text:
            return parse_scores_from_llm_response(llm_output_text, rubric_metrics)
        else:
            logger.warning(f"Scoring failed for record {record_dict.get('id')}: Judge LLM returned empty response.")
            return {metric.name: None for metric in rubric_metrics}
            
    except Exception as e:
        logger.error(f"Error during scoring record {record_dict.get('id')}: {e}", exc_info=True)
        return {metric.name: None for metric in rubric_metrics}


def run_scoring_stage(app_config: AppConfig):
    """
    Parallelised, drop-in replacement for the original rubric-scoring stage.
    Concurrency is controlled by
    `app_config.run.get_llm_settings_for_stage("scoring").max_concurrent_requests`
    (defaults to 4 threads when unspecified).
    """
    logger.info("--- Starting Rubric Scoring Stage ---")

    # -------------------------------------------------- template & rubric guards
    if not SCORING_TEMPLATE:
        logger.error("Scoring template not loaded. Skipping stage.")
        input_dir = Path(app_config.run.artifacts_dir) / "reconstructed"
        output_dir = Path(app_config.run.artifacts_dir) / "scored"
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir.glob(f"*.{app_config.run.artifact_ext}"):
            target = output_dir / dataset_file.name
            if not target.exists():
                import shutil
                shutil.copy(dataset_file, target)
            logger.warning(f"Copied {dataset_file.name} to scored dir as template is missing.")
        return

    scoring_cfg   = app_config.run.scoring
    rubric_metrics = app_config.rubric.metrics
    if not rubric_metrics:
        logger.error("No metrics defined in rubric. Skipping scoring stage.")
        return

    llm_settings = app_config.run.get_llm_settings_for_stage("scoring")
    if not llm_settings.api_base_url or not llm_settings.model_name:
        logger.error("API base URL or model name for scoring not configured. Skipping stage.")
        return

    max_workers = llm_settings.max_concurrent_requests or 4

    llm_client = llm_api_client.LLMAPIClient(
        api_base_url       = llm_settings.api_base_url,
        api_key            = llm_settings.api_key,
        default_model_name = llm_settings.model_name,
        timeout_seconds    = llm_settings.timeout_seconds or 120,
        max_retries        = llm_settings.max_retries  or 2,
    )

    # -------------------------------------------------- dirs & files
    input_dir  = Path(app_config.run.artifacts_dir) / "reconstructed"
    output_dir = Path(app_config.run.artifacts_dir) / "scored"
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstructed_files = list(input_dir.glob(f"*_reconstructed.{app_config.run.artifact_ext}"))
    if not reconstructed_files:
        logger.warning(f"No reconstructed files found in {input_dir}. Skipping scoring.")
        return

    # -------------------------------------------------- dataset loop
    for dataset_file in reconstructed_files:
        base_name   = dataset_file.stem.replace("_reconstructed", "")
        output_file = output_dir / f"{base_name}_scored.{app_config.run.artifact_ext}"

        logger.info(f"Scoring dataset: {dataset_file.name}")

        if output_file.exists() and not getattr(app_config, "force_scoring", False):
            logger.info(f"{output_file} already exists. Skipping.")
            continue

        current_dataset = file_io.load_records(dataset_file)
        if not current_dataset or len(current_dataset) == 0:
            logger.warning(f"No records in {dataset_file}. Skipping.")
            file_io.save_records([], output_file)
            continue

        processed_ids = app_config.state_manager.get_processed_ids("scoring", base_name)
        records       = current_dataset.to_list()
        id_to_index   = {rec["id"]: idx for idx, rec in enumerate(records)}

        # ---------------- identify which need scoring ------------------------
        def _needs_scoring(rec_dict):
            if getattr(app_config, "force_scoring", False):
                return True
            if rec_dict["id"] in processed_ids:
                # verify all metrics present
                scores = rec_dict.get("scores") or {}
                return any(scores.get(m.name) is None for m in rubric_metrics)
            return True

        to_score = [rec for rec in records if _needs_scoring(rec)]

        logger.info(f"{len(to_score)} / {len(records)} records require scoring.")

        # ---------------- worker fn ------------------------------------------
        def _score(rec_dict: dict) -> dict:
            rec_dict.setdefault("scores", {})

            scores = score_record_with_judge(
                rec_dict,
                scoring_cfg,
                rubric_metrics,
                llm_client,
            )
            rec_dict["scores"].update(scores)
            return rec_dict

        # ---------------- parallel execution ---------------------------------
        newly_scored_ids = []
        if to_score:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_score, rec): rec["id"] for rec in to_score}
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Scoring {dataset_file.name}",
                    unit="records",
                ):
                    rec_id = futures[fut]
                    try:
                        updated_rec = fut.result()
                    except Exception as e:
                        logger.error(f"Scoring failed for record {rec_id}: {e}")
                        continue
                    records[id_to_index[rec_id]] = updated_rec
                    newly_scored_ids.append(rec_id)

        # ---------------- state manager & save -------------------------------
        if newly_scored_ids:
            app_config.state_manager.add_processed_ids_batch(newly_scored_ids, "scoring", base_name)

        file_io.save_records(records, output_file)
        logger.info(f"Finished scoring {dataset_file.name}: wrote {len(records)} records.")

    logger.info("--- Rubric Scoring Stage Completed ---")