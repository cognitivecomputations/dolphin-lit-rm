import hashlib
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

PromptType = Literal["human", "synthetic", "reconstructed"]

class Metadata(BaseModel):
    prompt_type: Optional[PromptType] = None
    source_path: Optional[str] = None # Filesystem trace for Type C
    char_span: Optional[List[int]] = None # For segmented long documents [start_char, end_char]
    lang: Optional[str] = None
    original_id: Optional[str] = None # ID from the source dataset, if available
    # Add other relevant metadata fields as needed
    # e.g. source_url, original_split_from_source

class ClassificationLabels(BaseModel):
    top: Optional[str] = None # Top-level genre, e.g., fiction, news
    sub: Optional[str] = None # Sub-genre, e.g., sci-fi, political_news

class Record(BaseModel):
    id: str # Deterministic hash: sha256(source_name + original_id_or_hash_of_content + segment_offset)
    source_dataset_name: str # Short name of the source dataset (e.g., "sharegpt")
    # orig_split: Optional[str] = None # Original split if any (can be in meta)
    prompt: Optional[str] = None
    response: str # The actual text chunk
    meta: Metadata = Field(default_factory=Metadata)
    classification: ClassificationLabels = Field(default_factory=ClassificationLabels)
    scores: Dict[str, Optional[float]] = Field(default_factory=dict) # Metric_name -> score

    # Custom fields for pipeline control, not part of the final schema for RM training
    # These might be dropped before final export or used for internal tracking
    pipeline_internal_status: Optional[str] = Field(None, exclude=True) # e.g. "type_C_needs_segmentation"

def generate_record_id(
    source_dataset_name: str,
    source_specific_id: Optional[str] = None,
    content_for_hash: Optional[str] = None,
    segment_index: Optional[int] = None
) -> str:
    """
    Generates a deterministic ID for a record.
    Uses source_specific_id if available, otherwise hashes content.
    Includes segment_index for segmented documents.
    """
    base_string = source_dataset_name
    if source_specific_id:
        base_string += f":{source_specific_id}"
    elif content_for_hash:
        # Use a hash of the initial content if no stable ID is available
        # This is important for Type B/C where items might not have inherent IDs
        content_hash = hashlib.sha256(content_for_hash.encode('utf-8')).hexdigest()[:16]
        base_string += f":contenthash:{content_hash}"
    else:
        # Fallback, though ideally one of the above should be provided
        # This might happen if we only have an index from a list without content or ID yet
        # Consider raising an error if neither ID nor content is available for hashing
        import uuid
        base_string += f":uuid:{uuid.uuid4().hex}"


    if segment_index is not None:
        base_string += f":seg:{segment_index}"
    
    return f"sha256:{hashlib.sha256(base_string.encode('utf-8')).hexdigest()}"