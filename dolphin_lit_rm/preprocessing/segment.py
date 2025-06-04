import spacy
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datasets import Dataset
from loguru import logger
from tqdm import tqdm

from dolphin_lit_rm.core_configs import AppConfig, SegmentationConfig
from dolphin_lit_rm.utils import text_utils, file_io, state_manager
from dolphin_lit_rm.utils.schema_def import Record, Metadata, generate_record_id

# Load Spacy model
# spacy.prefer_gpu() # Uncomment if GPU is available and desired
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Only need sentencizer
    nlp.max_length = 2000000 # Increase if very long documents are expected as single strings
except OSError:
    logger.error("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None


def segment_document(
    doc_record_dict: Dict[str, Any], # A dict from the dataset
    segment_config: SegmentationConfig,
    tokenizer_name: str
) -> List[Dict[str, Any]]:
    """Segments a single document (passed as a record dict) into smaller chunks."""
    if not nlp:
        logger.error("Spacy model not loaded. Segmentation cannot proceed.")
        # Return the original document as a single chunk if spacy fails
        # This might exceed token limits downstream, but prevents data loss here.
        return [doc_record_dict] 

    text = doc_record_dict.get("response", "")
    if not text:
        return []

    doc = nlp(text)
    sentences = list(doc.sents)
    
    chunks = []
    current_chunk_sents = []
    current_chunk_tokens = 0
    start_char_offset = 0

    for i, sent in enumerate(sentences):
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        
        sent_tokens = text_utils.count_tokens(sent_text, tokenizer_name)

        if current_chunk_tokens + sent_tokens > segment_config.max_chunk_tokens and current_chunk_sents:
            # Finalize current chunk
            chunk_text = " ".join(s.text.strip() for s in current_chunk_sents)
            end_char_offset = current_chunk_sents[-1].end_char
            
            # Create new record for the chunk
            chunk_id = generate_record_id(
                source_dataset_name=doc_record_dict["source_dataset_name"],
                source_specific_id=doc_record_dict["meta"].get("original_id", doc_record_dict["id"]), # Use original doc ID as base
                segment_index=len(chunks) # Simple index for this document
            )
            new_meta = doc_record_dict["meta"].copy() # Shallow copy meta
            new_meta["char_span"] = [start_char_offset, end_char_offset]
            
            chunk_record = {
                "id": chunk_id,
                "source_dataset_name": doc_record_dict["source_dataset_name"],
                "prompt": None, # Chunks of long-form docs don't have prompts yet
                "response": chunk_text,
                "meta": new_meta,
                "classification": {}, # Will be filled later
                "scores": {} # Will be filled later
            }
            chunks.append(chunk_record)
            
            # Start new chunk, potentially with overlap
            if segment_config.sentence_overlap_count > 0 and len(current_chunk_sents) > segment_config.sentence_overlap_count:
                # Take last N sentences for overlap
                overlap_sents = current_chunk_sents[-segment_config.sentence_overlap_count:]
                current_chunk_sents = list(overlap_sents) # Make a copy
                current_chunk_tokens = sum(text_utils.count_tokens(s.text.strip(), tokenizer_name) for s in current_chunk_sents)
                start_char_offset = current_chunk_sents[0].start_char
            else:
                current_chunk_sents = []
                current_chunk_tokens = 0
                start_char_offset = sent.start_char # Start of the current sentence that didn't fit

        # Add current sentence to the current_chunk_sents (even if it started a new chunk)
        if not current_chunk_sents: # If it's the first sentence of a new chunk
            start_char_offset = sent.start_char
        current_chunk_sents.append(sent)
        current_chunk_tokens += sent_tokens


    # Add the last remaining chunk
    if current_chunk_sents:
        chunk_text = " ".join(s.text.strip() for s in current_chunk_sents)
        end_char_offset = current_chunk_sents[-1].end_char
        chunk_id = generate_record_id(
            source_dataset_name=doc_record_dict["source_dataset_name"],
            source_specific_id=doc_record_dict["meta"].get("original_id", doc_record_dict["id"]),
            segment_index=len(chunks)
        )
        new_meta = doc_record_dict["meta"].copy()
        new_meta["char_span"] = [start_char_offset, end_char_offset]
        chunk_record = {
            "id": chunk_id,
            "source_dataset_name": doc_record_dict["source_dataset_name"],
            "prompt": None,
            "response": chunk_text,
            "meta": new_meta,
            "classification": {},
            "scores": {}
        }
        chunks.append(chunk_record)
        
    return chunks


def run_segmentation_stage(app_config: AppConfig):
    logger.info("--- Starting Segmentation Stage ---")
    if not nlp:
        logger.error("Spacy model not available. Skipping segmentation stage.")
        # Copy filtered files to segmented directory to allow pipeline to continue if desired
        input_dir_seg = Path(app_config.run.artifacts_dir) / "filtered"
        output_dir_seg = Path(app_config.run.artifacts_dir) / "segmented"
        output_dir_seg.mkdir(parents=True, exist_ok=True)
        for dataset_file in input_dir_seg.glob(f"*.{app_config.run.artifact_ext}"):
            target_file = output_dir_seg / dataset_file.name
            if not target_file.exists(): # Avoid error if CLI is re-run and file exists
                 import shutil
                 shutil.copy(dataset_file, target_file)
            logger.warning(f"Copied {dataset_file.name} to segmented dir as spacy is unavailable.")
        return

    segment_config = app_config.run.preprocessing.segmentation
    tokenizer_name = app_config.run.tokenizer_name
    
    input_dir = Path(app_config.run.artifacts_dir) / "filtered"
    output_dir = Path(app_config.run.artifacts_dir) / "segmented"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_file in input_dir.glob(f"*.{app_config.run.artifact_ext}"):
        dataset_name = dataset_file.stem
        logger.info(f"Segmenting dataset: {dataset_name}")

        output_file = output_dir / f"{dataset_name}.{app_config.run.artifact_ext}"
        if output_file.exists() and not getattr(app_config, "force_segment", False):
            logger.info(f"Segmented artifact for {dataset_name} already exists. Skipping.")
            continue

        filtered_dataset = file_io.load_records(dataset_file)
        if not filtered_dataset or len(filtered_dataset) == 0:
            logger.warning(f"No records found in filtered file {dataset_file} for {dataset_name}. Skipping segmentation.")
            file_io.save_records([], output_file) # Save empty
            continue
        
        all_segmented_records = []
        # Convert to list of dicts for processing
        records_to_process = filtered_dataset.to_list()

        for record_dict in tqdm(records_to_process, desc=f"Segmenting {dataset_name}", unit="docs"):
            # Check if this record was marked for segmentation by ingestion (Type C)
            # Or if its type is 'C' based on original dataset config (more robust)
            # For now, assume all records in 'filtered' might need segmentation if long enough.
            # A more precise way: check record_dict['pipeline_internal_status'] or original type.
            
            # Simple heuristic: if it's much larger than max_chunk_tokens, it's probably Type C or long Type B
            # A dedicated 'type' field in the Record schema propagated from DatasetEntryConfig would be better.
            # For now, let's assume if 'pipeline_internal_status' has "needs_segmentation" or if it's simply too long.
            
            is_long_doc = record_dict.get("pipeline_internal_status") == "type_C_needs_segmentation"
            # Also segment if it's just too long, regardless of original type
            # (e.g. a very long "standalone" piece)
            if not is_long_doc:
                current_tokens = text_utils.count_tokens(record_dict.get("response",""), tokenizer_name)
                if current_tokens > segment_config.max_chunk_tokens * 1.1: # Add a small buffer
                    is_long_doc = True # Treat as needing segmentation

            if is_long_doc:
                # Ensure 'meta' is a dict
                if 'meta' not in record_dict or not isinstance(record_dict['meta'], dict):
                    record_dict['meta'] = {}

                chunks = segment_document(record_dict, segment_config, tokenizer_name)
                all_segmented_records.extend(chunks)
            else:
                # Pass through records that don't need segmentation
                # Ensure schema consistency (classification, scores dicts)
                if "classification" not in record_dict or record_dict["classification"] is None:
                    record_dict["classification"] = {}
                if "scores" not in record_dict or record_dict["scores"] is None:
                    record_dict["scores"] = {}
                all_segmented_records.append(record_dict)
        
        if all_segmented_records:
            file_io.save_records(all_segmented_records, output_file)
            logger.info(f"Finished segmenting {dataset_name}: {len(all_segmented_records)} total records/chunks produced.")
        else:
            logger.warning(f"No records/chunks produced for {dataset_name} after segmentation attempt.")
            file_io.save_records([], output_file) # Save empty

    logger.info("--- Segmentation Stage Completed ---")