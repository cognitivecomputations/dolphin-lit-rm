from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, FilePath, DirectoryPath, RootModel
from pathlib import Path

# --- Pydantic models for configuration files ---

class LLMSettings(BaseModel):
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    timeout_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    max_concurrent_requests: Optional[int] = None


class FilterConfig(BaseModel):
    min_response_tokens: int = 10
    max_response_tokens: int = 4000
    lang_id_threshold: float = 0.9
    blacklist_regex_patterns: List[str] = Field(default_factory=list)
    deduplication_cache_db: str = "dedup_cache.lmdb"

class SegmentationConfig(BaseModel):
    max_chunk_tokens: int = 3800
    sentence_overlap_count: int = 1

class PromptReconstructionConfig(BaseModel):
    model_name: Optional[str] = None # Specific model for this task
    max_response_tokens_for_reconstruction: int = 1024
    reconstructed_prompt_max_chars: int = 256
    llm_settings: Optional[LLMSettings] = None # Overrides default

class PreprocessingConfig(BaseModel):
    filter: FilterConfig = Field(default_factory=FilterConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    prompt_reconstruction: PromptReconstructionConfig = Field(default_factory=PromptReconstructionConfig)

class ClassificationConfig(BaseModel):
    model_name: Optional[str] = None # Specific model for this task
    confidence_threshold: Optional[float] = None # If applicable
    llm_settings: Optional[LLMSettings] = None # Overrides default
    top_level_genres_for_prompt: List[str] = Field(default_factory=list)
    genre_taxonomy_file: Optional[FilePath] = None # Path to a more detailed taxonomy YAML

class NormalizationQuotaConfig(BaseModel):
    # class.top: {"news": 10000, "fiction": 15000}
    quotas: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    default_quota_per_class: int = 5000

class ScoringConfig(BaseModel):
    model_name: Optional[str] = None
    max_tokens_per_metric_response: int = 8
    llm_settings: Optional[LLMSettings] = None # Overrides default

class CalibrationConfig(BaseModel):
    enabled: bool = True
    lower_percentile: int = 5
    upper_percentile: int = 95

class PostprocessingSplitsConfig(BaseModel):
    train: float = 0.90
    validation: float = 0.05
    test: float = 0.05

class PostprocessingConfig(BaseModel):
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    min_metrics_present_percent: float = 0.7
    splits: PostprocessingSplitsConfig = Field(default_factory=PostprocessingSplitsConfig)
    final_dataset_name_prefix: str = "dolphin_lit_rm_v0.1"

class RunConfig(BaseModel):
    runs_parent_dir: DirectoryPath = Field(default_factory=lambda: Path("./output/runs"))
    default_log_level: str = "INFO"
    
    default_llm_settings: LLMSettings = Field(default_factory=LLMSettings)
    
    # Stage specific configs
    # ingestion: Optional[Dict[str, Any]] = None # Ingestion config is mostly in datasets.yaml
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    normalization: NormalizationQuotaConfig = Field(default_factory=NormalizationQuotaConfig)
    # prompt_reconstruction is part of preprocessing config now
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    
    tokenizer_name: str = "gpt-4" # For tiktoken, or a HF tokenizer path

    # These will be set at runtime by the CLI based on --run-dir or new creation
    current_run_dir: Optional[DirectoryPath] = None # Actual path to the current run (e.g. output/runs/run_xyz)
    artifacts_dir: Optional[DirectoryPath] = None
    logs_dir: Optional[DirectoryPath] = None
    state_dir: Optional[DirectoryPath] = None
    run_config_copy_path: Optional[FilePath] = None # Path where this config is copied for the run
    artifact_ext: str = "jsonl"

    def get_llm_settings_for_stage(self, stage_name: str) -> LLMSettings:
        """Merges default LLM settings with stage-specific overrides."""
        stage_config_map = {
            "prompt_reconstruction": self.preprocessing.prompt_reconstruction,
            "classification": self.classification,
            "scoring": self.scoring,
        }
        
        stage_specific_config = stage_config_map.get(stage_name)
        
        # Start with a copy of default settings
        merged_settings = self.default_llm_settings.model_copy(deep=True)

        if stage_specific_config and hasattr(stage_specific_config, 'llm_settings') and stage_specific_config.llm_settings:
            # Update with non-None values from stage-specific LLM settings
            stage_llm_overrides = stage_specific_config.llm_settings.model_dump(exclude_none=True)
            merged_settings = merged_settings.model_copy(update=stage_llm_overrides)
        
        # Also, some stages have a direct model_name attribute (e.g., classification.model_name)
        # This should take precedence if stage_specific_config.llm_settings.model_name is not set
        if stage_specific_config and hasattr(stage_specific_config, 'model_name') and stage_specific_config.model_name:
            if merged_settings.model_name is None : # Only if not already set by llm_settings block
                 merged_settings.model_name = stage_specific_config.model_name
        
        return merged_settings


class DatasetEntryConfig(BaseModel):
    name: str
    path_or_hf_id: str
    format: Literal["jsonl", "arrow", "parquet", "hf_dataset", "text", "csv", "text_dir"]
    type: Literal["A", "B", "C"] # A: Prompt+Assistant, B: Standalone, C: Long-form
    hf_dataset_config: Optional[Dict[str, Any]] = None # e.g., split, name, column mappings
    max_items: Optional[Union[int, str]] = None # e.g., 1000 or "10%"
    # lang_filter: Optional[str] = None # Example of dataset-specific pre-filter

class DatasetsConfig(BaseModel):
    datasets: List[DatasetEntryConfig]

class MetricConfig(BaseModel):
    name: str
    description: str
    prompt_hint: str # Short hint for the scoring prompt

class RubricConfig(BaseModel):
    name: str                                  # unique identifier (“fiction”, “non_fiction”, …)
    metrics: List[MetricConfig]

class RubricsConfig(RootModel):
    """
    Mapping rubric-name → RubricConfig stored as the root value.
    """
    root: Dict[str, RubricConfig]

    # convenience helpers so existing code keeps working
    def __iter__(self):
        return iter(self.root.values())

    def __getitem__(self, k):
        return self.root[k]

    def get(self, k, default=None):
        return self.root.get(k, default)

# --- Global AppConfig to hold all loaded configurations ---
class AppConfig(BaseModel):
    run: RunConfig
    datasets: DatasetsConfig
    rubrics: RubricsConfig          # ← new canonical field

    # populated later by CLI
    state_manager: Optional[Any] = None

    # ── backward-compat shim ──────────────────────────────────────────────
    @property
    def rubric(self) -> RubricsConfig:     # old attribute still works
        return self.rubrics

    class Config:
        arbitrary_types_allowed = True