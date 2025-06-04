import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from jinja2 import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch

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
    metrics: List[MetricConfig],
) -> Dict[str, Optional[float]]:
    """
    Parse a JSON object produced by the judge LLM and return a mapping from
    metric name → score.

    The LLM **must** return something that can be parsed by `json.loads`,
    possibly wrapped in Markdown fences.  If parsing fails, the function logs
    the error and returns a dict whose values are all `None`.

    Parameters
    ----------
    llm_response : str
        Raw text returned by the judge model.
    metrics : list[MetricConfig]
        Metric definitions loaded from `rubric.yaml`.

    Returns
    -------
    dict[str, Optional[float]]
        Keys are the metric names from the rubric; values are floats in the
        range [0.0 – 1.0] or `None` when a score could not be extracted.
    """
    import json
    import re

    metric_names = [m.name for m in metrics]
    parsed_scores: Dict[str, Optional[float]] = {k: None for k in metric_names}

    if not llm_response:
        logger.warning("LLM response is empty – all scores set to None.")
        return parsed_scores

    # -- strip possible Markdown code fences --------------------------------
    cleaned = llm_response.strip()
    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
    fence_match = fence_pattern.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1)

    # -- isolate first {...} pair if extra prose remains --------------------
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        logger.error(
            "Could not locate a JSON object in the LLM response. "
            "Returning None for all metrics.\nResponse: {}", cleaned[:300]
        )
        return parsed_scores

    json_str = cleaned[first_brace : last_brace + 1]

    # -- parse JSON ---------------------------------------------------------
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON decoding failed: {}. Response snippet: {}", exc, cleaned[:300]
        )
        return parsed_scores

    # -- extract & clamp scores --------------------------------------------
    for name in metric_names:
        val = data.get(name)
        if isinstance(val, (int, float)):
            parsed_scores[name] = max(0.0, min(1.0, float(val)))
        elif val is not None:
            logger.warning(
                "Metric '{}' present in JSON but not numeric (value: {}). "
                "Setting it to None.",
                name,
                val,
            )

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
    Parallel rubric-aware scoring stage.

    Concurrency is limited by
    app_config.run.get_llm_settings_for_stage("scoring").max_concurrent_requests
    (defaults to 4).
    """
    logger.info("--- Starting Rubric Scoring Stage ---")

    # ── guard: template present ──────────────────────────────────────────
    if not SCORING_TEMPLATE:
        logger.error("Scoring template not loaded. Skipping stage.")
        src_dir = Path(app_config.run.artifacts_dir) / "reconstructed"
        dst_dir = Path(app_config.run.artifacts_dir) / "scored"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fp in src_dir.glob(f"*.{app_config.run.artifact_ext}"):
            target = dst_dir / fp.name
            if not target.exists():
                import shutil
                shutil.copy(fp, target)
            logger.warning(f"Copied {fp.name} to scored dir because template was missing.")
        return

    # ── LLM settings ------------------------------------------------------
    llm_settings = app_config.run.get_llm_settings_for_stage("scoring")
    if not llm_settings.api_base_url or not llm_settings.model_name:
        logger.error("API base URL or model name for scoring not configured. Skipping stage.")
        return

    llm_client = llm_api_client.LLMAPIClient(
        api_base_url       = llm_settings.api_base_url,
        api_key            = llm_settings.api_key,
        default_model_name = llm_settings.model_name,
        timeout_seconds    = llm_settings.timeout_seconds or 120,
        max_retries        = llm_settings.max_retries or 2,
    )
    max_workers = llm_settings.max_concurrent_requests or 4

    # ── rubrics and routing ----------------------------------------------
    rubrics: dict[str, RubricConfig] = app_config.rubrics  # RootModel acts like dict
    default_rubric = rubrics.get("default") or next(iter(rubrics))
    if not default_rubric.metrics:
        logger.error("No metrics defined in any rubric. Skipping scoring stage.")
        return

    routing_rules = app_config.run.model_dump().get("rubric_mapping", {})

    def _select_rubric(class_label: str) -> str:
        for pattern, name in routing_rules.items():
            if pattern == "*" or fnmatch.fnmatch(class_label, pattern):
                return name
        return "default"

    # ── I/O paths ---------------------------------------------------------
    input_dir  = Path(app_config.run.artifacts_dir) / "reconstructed"
    output_dir = Path(app_config.run.artifacts_dir) / "scored"
    output_dir.mkdir(parents=True, exist_ok=True)

    recon_files = list(input_dir.glob(f"*_reconstructed.{app_config.run.artifact_ext}"))
    if not recon_files:
        logger.warning(f"No reconstructed files found in {input_dir}. Skipping scoring.")
        return

    # ── per-file processing ----------------------------------------------
    for file_path in recon_files:
        base_name   = file_path.stem.replace("_reconstructed", "")
        output_file = output_dir / f"{base_name}_scored.{app_config.run.artifact_ext}"

        logger.info(f"Scoring dataset: {file_path.name}")

        if output_file.exists() and not getattr(app_config, "force_scoring", False):
            logger.info(f"{output_file} already exists. Skipping.")
            continue

        ds = file_io.load_records(file_path)
        if not ds or len(ds) == 0:
            logger.warning(f"No records in {file_path}. Skipping.")
            file_io.save_records([], output_file)
            continue

        records       = ds.to_list()
        id_to_index   = {rec["id"]: idx for idx, rec in enumerate(records)}
        processed_ids = app_config.state_manager.get_processed_ids("scoring", base_name)

        # ---------- queue selection --------------------------------------
        def _needs_scoring(rec):
            if getattr(app_config, "force_scoring", False):
                return True
            if rec["id"] in processed_ids:
                return False
            return not bool(rec.get("scores"))  # score dict missing or empty

        to_score = [rec for rec in records if _needs_scoring(rec)]
        logger.info(f"{len(to_score)} / {len(records)} records require scoring.")

        # ---------- worker ------------------------------------------------
        def _score(rec_dict: dict) -> dict:
            rec_dict.setdefault("scores", {})
            class_label = (rec_dict.get("classification") or {}).get("top", "unknown")
            rubric_name = _select_rubric(class_label)
            rubric_cfg  = rubrics.get(rubric_name, default_rubric)

            scores = score_record_with_judge(
                rec_dict,
                app_config.run.scoring,
                rubric_cfg.metrics,
                llm_client,
            )
            for metric_name, val in scores.items():
                rec_dict["scores"][f"{rubric_name}.{metric_name}"] = val
            return rec_dict

        # ---------- parallel execution -----------------------------------
        newly_scored_ids = []
        if to_score:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_score, rec): rec["id"] for rec in to_score}
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc=f"Scoring {file_path.name}",
                                unit="records"):
                    rec_id = futures[fut]
                    try:
                        updated = fut.result()
                        records[id_to_index[rec_id]] = updated
                        newly_scored_ids.append(rec_id)
                    except Exception as exc:
                        logger.error(f"Scoring failed for record {rec_id}: {exc}")

        # ---------- persist ----------------------------------------------
        if newly_scored_ids:
            app_config.state_manager.add_processed_ids_batch(newly_scored_ids, "scoring", base_name)

        file_io.save_records(records, output_file)
        logger.info(f"Finished scoring {file_path.name}: wrote {len(records)} records.")

    logger.info("--- Rubric Scoring Stage Completed ---")
