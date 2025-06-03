import json
from pathlib import Path
from typing import List, Dict, Any, Iterator, Union, Optional
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_dataset, Features, Value
from loguru import logger

from dolphin_lit_rm.utils.schema_def import Record # Assuming schema_def.py

def save_records_to_arrow(records: List[Dict[str, Any]], output_path: Path, schema: Optional[pa.Schema] = None) -> None:
    """Saves a list of record dictionaries to an Arrow file."""
    if not records:
        logger.warning(f"No records to save to {output_path}")
        # Create an empty file with schema if provided, or handle as needed
        if schema:
            table = pa.Table.from_arrays([[] for _ in schema.names], schema=schema)
            with pa.OSFile(str(output_path), 'wb') as sink:
                with pa.ipc.new_file(sink, schema=table.schema) as writer:
                    writer.write_table(table)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Convert list of Pydantic models (if they are) or dicts to a Dataset
        # This is a common pattern if records are [record.model_dump() for record in pydantic_records]
        # For HF Datasets, it's often easier to create a Dataset object first
        
        # Dynamically create features if not provided or infer from first record
        # For simplicity, let's assume records are dicts and infer
        hf_dataset = Dataset.from_list(records)
        hf_dataset.to_parquet(str(output_path)) # .arrow often means Parquet for HF Datasets
        logger.info(f"Saved {len(records)} records to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save records to {output_path}: {e}")
        # Fallback or re-raise
        # For robustness, could try saving as JSONL if Arrow fails for some reason
        # with output_path.with_suffix(".jsonl").open("w") as f:
        #     for record in records:
        #         f.write(json.dumps(record) + "\n")
        # logger.warning(f"Saved as JSONL fallback: {output_path.with_suffix('.jsonl')}")
        raise

def load_records_from_arrow(file_path: Path) -> Dataset:
    """Loads records from an Arrow file into a Hugging Face Dataset."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}, returning empty Dataset.")
        # Define an empty schema based on Record Pydantic model for consistency
        # This is complex; for now, let's rely on HF to handle empty if it can
        # Or return an empty list and let caller handle it
        return Dataset.from_list([]) 
    try:
        dataset = Dataset.from_parquet(str(file_path))
        logger.info(f"Loaded {len(dataset)} records from {file_path}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load records from {file_path}: {e}")
        raise

def stream_jsonl_file(file_path: Path) -> Iterator[Dict[str, Any]]:
    """Streams records from a JSONL file."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line in {file_path}: {e} - Line: {line[:100]}...")

def stream_text_file(file_path: Path) -> Iterator[str]:
    """Streams lines from a text file."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()

def get_hf_dataset(
    path_or_hf_id: str,
    format_type: str,
    config: Dict[str, Any], # dataset specific config from datasets.yml
    max_items: Optional[Union[int, str]] = None
) -> Dataset:
    """
    Loads a dataset from various sources into a Hugging Face Dataset object.
    Handles max_items for slicing.
    """
    dataset_args = config.get("hf_dataset_config", {})
    split = dataset_args.get("split", "train") # Default to train

    if isinstance(max_items, str) and "%" in max_items:
        percentage = int(max_items.replace("%", ""))
        split_arg = f"{split}[:{percentage}%]"
    elif isinstance(max_items, int):
        split_arg = f"{split}[:{max_items}]"
    else:
        split_arg = split

    logger.info(f"Loading dataset: {path_or_hf_id}, format: {format_type}, split_arg: {split_arg}")

    if format_type == "hf_dataset":
        ds = load_dataset(path_or_hf_id, split=split_arg, name=dataset_args.get("name"))
    elif format_type == "jsonl":
        ds = load_dataset("json", data_files=path_or_hf_id, split=split_arg)
    elif format_type == "arrow" or format_type == "parquet":
        # load_dataset can also load local arrow/parquet files
        ds = Dataset.from_parquet(path_or_hf_id) # Assumes single file
        if max_items: # Manual slicing after load for local parquet/arrow
             if isinstance(max_items, str) and "%" in max_items:
                num_to_take = int(len(ds) * (int(max_items.replace("%", "")) / 100))
                ds = ds.select(range(min(num_to_take, len(ds))))
             elif isinstance(max_items, int):
                ds = ds.select(range(min(max_items, len(ds))))

    elif format_type == "text":
        # load_dataset can load text files, each line becomes a record
        ds = load_dataset("text", data_files=path_or_hf_id, split=split_arg)
    elif format_type == "csv":
        ds = load_dataset("csv", data_files=path_or_hf_id, split=split_arg)
    # 'text_dir' needs custom handling in the ingester, not directly by load_dataset for multiple files as one dataset easily
    else:
        raise ValueError(f"Unsupported dataset format: {format_type}")
    
    logger.info(f"Successfully loaded {len(ds)} items for {path_or_hf_id}")
    return ds
