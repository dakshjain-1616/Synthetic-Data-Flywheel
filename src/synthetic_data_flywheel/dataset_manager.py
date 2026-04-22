"""Dataset Manager - HuggingFace datasets integration, save/load, versioning."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import structlog
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, create_repo, upload_file

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import SyntheticPair

logger = structlog.get_logger()


class DatasetManager:
    """Manager for dataset operations including save/load and versioning."""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """Initialize dataset manager.
        
        Args:
            data_dir: Local directory for dataset storage
            hf_token: HuggingFace API token
        """
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or settings.huggingface_token
        self.hf_api = HfApi(token=self.hf_token) if self.hf_token else None
        
        logger.info(
            "dataset_manager_initialized",
            data_dir=str(self.data_dir),
            hf_authenticated=self.hf_token is not None,
        )
    
    def pairs_to_dicts(self, pairs: List[SyntheticPair]) -> List[Dict[str, Any]]:
        """Convert SyntheticPair objects to dictionaries."""
        return [
            {
                "id": str(p.id),
                "instruction": p.instruction,
                "input": p.input or "",
                "output": p.output,
                "context": p.context or "",
                "category": p.category or "",
                "difficulty": p.difficulty or "",
                "metadata": p.metadata or {},
                "source_seed": p.source_seed or "",
                "cycle_id": p.cycle_id,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in pairs
        ]
    
    def dicts_to_pairs(self, data: List[Dict[str, Any]]) -> List[SyntheticPair]:
        """Convert dictionaries to SyntheticPair objects."""
        pairs = []
        for item in data:
            try:
                pair = SyntheticPair(
                    id=UUID(item["id"]) if "id" in item else None,
                    instruction=item.get("instruction", ""),
                    input=item.get("input") or None,
                    output=item.get("output", ""),
                    context=item.get("context") or None,
                    category=item.get("category") or None,
                    difficulty=item.get("difficulty") or None,
                    metadata=item.get("metadata", {}),
                    source_seed=item.get("source_seed") or None,
                    cycle_id=item.get("cycle_id"),
                )
                pairs.append(pair)
            except Exception as e:
                logger.warning("failed_to_parse_pair", item=item, error=str(e))
        return pairs
    
    def save_local(
        self,
        pairs: List[SyntheticPair],
        filename: str = "synthetic_data.json",
        split: Optional[str] = None,
    ) -> Path:
        """Save pairs to local JSON file.
        
        Args:
            pairs: List of synthetic pairs to save
            filename: Output filename
            split: Optional dataset split (train/test/validation)
            
        Returns:
            Path to saved file
        """
        if split:
            output_dir = self.data_dir / split
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.data_dir
        
        output_path = output_dir / filename
        data = self.pairs_to_dicts(pairs)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(
            "dataset_saved_locally",
            path=str(output_path),
            count=len(pairs),
            split=split,
        )
        
        return output_path
    
    def load_local(
        self,
        filename: str = "synthetic_data.json",
        split: Optional[str] = None,
    ) -> List[SyntheticPair]:
        """Load pairs from local JSON file.
        
        Args:
            filename: Input filename
            split: Optional dataset split
            
        Returns:
            List of synthetic pairs
        """
        if split:
            input_path = self.data_dir / split / filename
        else:
            input_path = self.data_dir / filename
        
        if not input_path.exists():
            logger.warning("local_dataset_not_found", path=str(input_path))
            return []
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        pairs = self.dicts_to_pairs(data)
        
        logger.info(
            "dataset_loaded_locally",
            path=str(input_path),
            count=len(pairs),
            split=split,
        )
        
        return pairs
    
    def save_to_huggingface(
        self,
        pairs: List[SyntheticPair],
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """Save dataset to HuggingFace Hub.
        
        Args:
            pairs: List of synthetic pairs
            repo_id: HuggingFace repository ID (username/repo-name)
            private: Whether to create private repo
            commit_message: Optional commit message
            
        Returns:
            URL to uploaded dataset
        """
        if not self.hf_api:
            raise ValueError("HuggingFace token required for upload")
        
        # Create dataset
        data = self.pairs_to_dicts(pairs)
        dataset = Dataset.from_list(data)
        
        # Create or get repo
        try:
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=private,
                repo_type="dataset",
                exist_ok=True,
            )
        except Exception as e:
            logger.warning("repo_creation_warning", error=str(e))
        
        # Push to hub
        dataset.push_to_hub(
            repo_id,
            token=self.hf_token,
            commit_message=commit_message or f"Add {len(pairs)} synthetic pairs",
        )
        
        url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(
            "dataset_uploaded_to_hub",
            repo_id=repo_id,
            url=url,
            count=len(pairs),
        )
        
        return url
    
    def load_from_huggingface(
        self,
        repo_id: str,
        split: str = "train",
    ) -> List[SyntheticPair]:
        """Load dataset from HuggingFace Hub.
        
        Args:
            repo_id: HuggingFace repository ID
            split: Dataset split to load
            
        Returns:
            List of synthetic pairs
        """
        dataset = load_dataset(repo_id, split=split)
        data = [dict(row) for row in dataset]
        pairs = self.dicts_to_pairs(data)
        
        logger.info(
            "dataset_loaded_from_hub",
            repo_id=repo_id,
            split=split,
            count=len(pairs),
        )
        
        return pairs
    
    def create_train_test_split(
        self,
        pairs: List[SyntheticPair],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple[List[SyntheticPair], List[SyntheticPair], List[SyntheticPair]]:
        """Split pairs into train/test/validation sets.
        
        Args:
            pairs: List of pairs to split
            test_size: Fraction for test set
            val_size: Fraction for validation set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train, test, validation) lists
        """
        import random
        
        random.seed(random_seed)
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        
        n = len(shuffled)
        test_count = int(n * test_size)
        val_count = int(n * val_size)
        
        test_pairs = shuffled[:test_count]
        val_pairs = shuffled[test_count:test_count + val_count]
        train_pairs = shuffled[test_count + val_count:]
        
        logger.info(
            "dataset_split_created",
            total=n,
            train=len(train_pairs),
            test=len(test_pairs),
            validation=len(val_pairs),
        )
        
        return train_pairs, test_pairs, val_pairs
    
    def save_split_datasets(
        self,
        train: List[SyntheticPair],
        test: List[SyntheticPair],
        val: List[SyntheticPair],
        base_filename: str = "synthetic_data",
    ) -> Dict[str, Path]:
        """Save train/test/val splits to separate files.
        
        Args:
            train: Training pairs
            test: Test pairs
            val: Validation pairs
            base_filename: Base filename for splits
            
        Returns:
            Dictionary mapping split names to file paths
        """
        paths = {}
        
        if train:
            paths["train"] = self.save_local(train, f"{base_filename}_train.json", "train")
        if test:
            paths["test"] = self.save_local(test, f"{base_filename}_test.json", "test")
        if val:
            paths["validation"] = self.save_local(val, f"{base_filename}_val.json", "validation")
        
        return paths
    
    def get_dataset_info(self, pairs: List[SyntheticPair]) -> Dict[str, Any]:
        """Get statistics about a dataset.
        
        Args:
            pairs: List of pairs to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        if not pairs:
            return {"count": 0}
        
        categories = {}
        difficulties = {}
        total_input_len = 0
        total_output_len = 0
        
        for pair in pairs:
            cat = pair.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
            
            diff = pair.difficulty or "unknown"
            difficulties[diff] = difficulties.get(diff, 0) + 1
            
            total_input_len += len(pair.instruction) + len(pair.input or "")
            total_output_len += len(pair.output)
        
        return {
            "count": len(pairs),
            "avg_input_length": total_input_len / len(pairs),
            "avg_output_length": total_output_len / len(pairs),
            "categories": categories,
            "difficulties": difficulties,
        }
    
    def merge_datasets(
        self,
        datasets: List[List[SyntheticPair]],
        deduplicate: bool = True,
    ) -> List[SyntheticPair]:
        """Merge multiple datasets, optionally deduplicating.
        
        Args:
            datasets: List of pair lists to merge
            deduplicate: Whether to remove duplicates
            
        Returns:
            Merged list of pairs
        """
        merged = []
        seen = set() if deduplicate else None
        
        for dataset in datasets:
            for pair in dataset:
                if deduplicate:
                    key = (pair.instruction, pair.output)
                    if key in seen:
                        continue
                    seen.add(key)
                merged.append(pair)
        
        logger.info(
            "datasets_merged",
            input_count=sum(len(d) for d in datasets),
            output_count=len(merged),
            deduplicated=deduplicate,
        )
        
        return merged


def create_dataset_manager(
    data_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> DatasetManager:
    """Factory function to create a dataset manager.
    
    Args:
        data_dir: Local data directory
        hf_token: HuggingFace API token
        
    Returns:
        Configured DatasetManager
    """
    return DatasetManager(data_dir=data_dir, hf_token=hf_token)
