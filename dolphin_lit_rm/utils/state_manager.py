import json
from pathlib import Path
from typing import Set, Optional, List, Dict
from loguru import logger

class StateManager:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.processed_ids_cache: Dict[str, Set[str]] = {} # Cache loaded sets of IDs

    def _get_state_file_path(self, stage_name: str, dataset_name: Optional[str] = None) -> Path:
        """Generates the path for a state file."""
        filename = f"{stage_name}"
        if dataset_name:
            filename += f"_{dataset_name}"
        filename += "_processed_ids.jsonl"
        return self.state_dir / filename

    def _load_processed_ids(self, state_file_path: Path) -> Set[str]:
        """Loads a set of processed IDs from a file."""
        if state_file_path.exists():
            with state_file_path.open("r") as f:
                return {line.strip() for line in f if line.strip()}
        return set()

    def get_processed_ids(self, stage_name: str, dataset_name: Optional[str] = None) -> Set[str]:
        """
        Gets the set of processed IDs for a given stage and optional dataset.
        Uses a cache to avoid repeated file reads.
        """
        cache_key = f"{stage_name}_{dataset_name}" if dataset_name else stage_name
        if cache_key not in self.processed_ids_cache:
            state_file = self._get_state_file_path(stage_name, dataset_name)
            self.processed_ids_cache[cache_key] = self._load_processed_ids(state_file)
            logger.debug(f"Loaded {len(self.processed_ids_cache[cache_key])} processed IDs for {cache_key} from {state_file}")
        return self.processed_ids_cache[cache_key]

    def add_processed_id(self, record_id: str, stage_name: str, dataset_name: Optional[str] = None) -> None:
        """Adds a processed ID to the state file and cache for a given stage and dataset."""
        state_file = self._get_state_file_path(stage_name, dataset_name)
        with state_file.open("a") as f:
            f.write(f"{record_id}\n")
        
        cache_key = f"{stage_name}_{dataset_name}" if dataset_name else stage_name
        if cache_key in self.processed_ids_cache:
            self.processed_ids_cache[cache_key].add(record_id)
        # else: it will be loaded on next get_processed_ids call

    def add_processed_ids_batch(self, record_ids: List[str], stage_name: str, dataset_name: Optional[str] = None) -> None:
        """Adds a batch of processed IDs to the state file and cache."""
        if not record_ids:
            return
        state_file = self._get_state_file_path(stage_name, dataset_name)
        with state_file.open("a") as f:
            for record_id in record_ids:
                f.write(f"{record_id}\n")
        
        cache_key = f"{stage_name}_{dataset_name}" if dataset_name else stage_name
        if cache_key in self.processed_ids_cache:
            self.processed_ids_cache[cache_key].update(record_ids)
        logger.debug(f"Added batch of {len(record_ids)} IDs to {state_file} for {cache_key}")

    def is_processed(self, record_id: str, stage_name: str, dataset_name: Optional[str] = None) -> bool:
        """Checks if a record ID has been processed for a given stage and dataset."""
        return record_id in self.get_processed_ids(stage_name, dataset_name)

    def clear_state(self, stage_name: str, dataset_name: Optional[str] = None) -> None:
        """Clears the state for a given stage and dataset (deletes the state file)."""
        state_file = self._get_state_file_path(stage_name, dataset_name)
        cache_key = f"{stage_name}_{dataset_name}" if dataset_name else stage_name
        if state_file.exists():
            state_file.unlink()
            logger.info(f"Cleared state file: {state_file}")
        if cache_key in self.processed_ids_cache:
            del self.processed_ids_cache[cache_key]
            logger.info(f"Cleared cache for state: {cache_key}")