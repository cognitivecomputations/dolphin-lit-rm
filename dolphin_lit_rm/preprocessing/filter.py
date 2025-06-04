import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset, Features, Value
from loguru import logger
from tqdm import tqdm
import fasttext # Requires model: lid.176.bin
import lmdb
import hashlib

from dolphin_lit_rm.core_configs import AppConfig, FilterConfig
from dolphin_lit_rm.utils import text_utils, file_io, state_manager
from dolphin_lit_rm.utils.schema_def import Record

# Load fastText language model (ensure model is downloaded)
# fasttext.FastText.eprint = lambda x: None # Suppress warnings if any
try:
    # You might need to specify the path to the model if it's not in a standard location
    # For example: fasttext_model_path = Path.home() / ".fasttext_models" / "lid.176.bin"
    # if not fasttext_model_path.exists():
    #    logger.error(f"FastText model not found at {fasttext_model_path}. Please download it.")
    #    raise FileNotFoundError("FastText model lid.176.bin not found.")
    # lang_model = fasttext.load_model(str(fasttext_model_path))
    lang_model = fasttext.load_model('lid.176.bin') # Assumes it's findable by fasttext
except Exception as e:
    logger.error(f"Failed to load fastText language model: {e}. Language filtering will be disabled.")
    lang_model = None


class Deduplicator:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing deduplication LMDB at {self.db_path}")
        # Increased map_size for potentially large number of hashes
        self.env = lmdb.open(str(self.db_path), map_size=1024**4) # 1 TB, adjust as needed

    def is_duplicate(self, text_hash: str) -> bool:
        with self.env.begin() as txn:
            return txn.get(text_hash.encode()) is not None

    def add_hash(self, text_hash: str) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(text_hash.encode(), b'1') # Value doesn't matter, just existence

    def close(self):
        self.env.close()
        logger.info(f"Closed deduplication LMDB at {self.db_path}")


def apply_filters_to_record(
    record_dict: Dict[str, Any], # A dict, not Pydantic model here
    filter_config: FilterConfig,
    tokenizer_name: str,
    deduplicator: Optional[Deduplicator] = None,
    lang_id_model: Optional[Any] = lang_model # Pass the loaded model
) -> Optional[Dict[str, Any]]:
    """Applies filters to a single record dictionary. Returns None if record is dropped."""
    response_text = record_dict.get("response", "")
    if not response_text:
        return None # Drop if no response

    # 1. Length filter (tokens)
    num_tokens = text_utils.count_tokens(response_text, tokenizer_name)
    if not (filter_config.min_response_tokens <= num_tokens <= filter_config.max_response_tokens):
        # logger.debug(f"Record {record_dict.get('id')} dropped: token count {num_tokens} out of range.")
        return None

    # 2. Language ID filter
    if lang_id_model and filter_config.lang_id_threshold > 0:
        # fastText expects clean text, newlines can affect prediction
        cleaned_for_langid = response_text.replace("\n", " ")
        predictions = lang_id_model.predict(cleaned_for_langid, k=1)
        lang, prob = predictions[0][0].replace("__label__", ""), predictions[1][0]
        
        # Assuming English for now, this could be configurable
        if not (lang == "en" and prob >= filter_config.lang_id_threshold):
            # logger.debug(f"Record {record_dict.get('id')} dropped: lang {lang} ({prob:.2f}) not meeting threshold.")
            record_dict["meta"]["lang"] = lang # Store detected lang even if dropped for analysis
            return None
        record_dict["meta"]["lang"] = "en" # Store detected lang

    # 3. Blacklist regexes
    if filter_config.blacklist_regex_patterns:
        for pattern in filter_config.blacklist_regex_patterns:
            if re.search(pattern, response_text, re.IGNORECASE): # Added IGNORECASE
                # logger.debug(f"Record {record_dict.get('id')} dropped: matched blacklist pattern '{pattern}'.")
                return None
    
    # 4. Deduplication (on sha1 of response)
    if deduplicator:
        # Using sha1 for deduplication hash as it's faster and sufficient for this purpose
        text_hash = hashlib.sha1(response_text.encode('utf-8')).hexdigest()
        if deduplicator.is_duplicate(text_hash):
            # logger.debug(f"Record {record_dict.get('id')} dropped: duplicate content (hash {text_hash}).")
            return None
        deduplicator.add_hash(text_hash) # Add after check, only for non-duplicates

    return record_dict


def run_filter_stage(app_config: AppConfig):
    logger.info("--- Starting Pre-filtering Stage ---")
    filter_config = app_config.run.preprocessing.filter
    tokenizer_name = app_config.run.tokenizer_name
    
    input_dir = Path(app_config.run.artifacts_dir) / "raw"
    output_dir = Path(app_config.run.artifacts_dir) / "filtered"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize deduplicator once for the entire stage
    dedup_db_path = Path(app_config.run.state_dir) / filter_config.deduplication_cache_db
    deduplicator = Deduplicator(dedup_db_path)
    
    processed_count_total = 0
    retained_count_total = 0

    for dataset_file in input_dir.glob(f"*.{app_config.run.artifact_ext}"):
        dataset_name = dataset_file.stem
        logger.info(f"Filtering dataset: {dataset_name}")
        
        output_file = output_dir / f"{dataset_name}.{app_config.run.artifact_ext}"
        if output_file.exists() and not getattr(app_config, "force_filter", False):
            logger.info(f"Filtered artifact for {dataset_name} already exists. Skipping.")
            # Need to load and count if we want accurate totals for resumed runs
            # For now, just skip processing.
            try:
                # Still load hashes into deduplicator if resuming and file exists
                # This is tricky because we don't want to re-add if already processed in a *previous* run
                # The LMDB is persistent across runs.
                # A simpler approach: if resuming, the LMDB is already populated.
                # If it's a new run, it starts empty.
                # So, no special handling needed here for deduplicator on resume.
                existing_data = file_io.load_records(output_file)
                processed_count_total += len(existing_data) # Assume all were processed to get here
                retained_count_total += len(existing_data)
            except Exception:
                pass # If loading fails, it will be reprocessed
            continue

        raw_dataset = file_io.load_records(dataset_file)
        if not raw_dataset or len(raw_dataset) == 0:
            logger.warning(f"No records found in raw file {dataset_file} for {dataset_name}. Skipping.")
            file_io.save_records([], output_file) # Save empty
            continue
        
        processed_count_total += len(raw_dataset)
        
        # Using .map for Hugging Face Datasets is efficient
        # Need to ensure the function can be pickled or use with_indices
        # For simplicity and control, iterate and build a list.
        # For very large datasets, consider HF .map() with a more complex setup or Spark/Dask.
        
        filtered_records = []
        # Convert HF Dataset to list of dicts for processing
        # This loads all into memory for this dataset. For extremely large individual files,
        # stream processing would be better.
        raw_records_list = raw_dataset.to_list()

        for record_dict in tqdm(raw_records_list, desc=f"Filtering {dataset_name}", unit="records"):
            # Ensure 'meta' exists and is a dict
            if 'meta' not in record_dict or record_dict['meta'] is None:
                record_dict['meta'] = {}
            
            # If 'lang' is already set (e.g. by ingester), respect it or re-evaluate
            # For now, filter.py will attempt to set/overwrite it.

            processed_record = apply_filters_to_record(
                record_dict, filter_config, tokenizer_name, deduplicator, lang_model
            )
            if processed_record:
                filtered_records.append(processed_record)
        
        if filtered_records:
            file_io.save_records(filtered_records, output_file)
            logger.info(f"Finished filtering {dataset_name}: {len(filtered_records)} records retained out of {len(raw_dataset)}.")
        else:
            logger.warning(f"No records retained for {dataset_name} after filtering.")
            file_io.save_records([], output_file) # Save empty
            
        retained_count_total += len(filtered_records)

    deduplicator.close()
    logger.info(f"--- Pre-filtering Stage Completed ---")
    logger.info(f"Total records processed: {processed_count_total}, Total records retained: {retained_count_total}")