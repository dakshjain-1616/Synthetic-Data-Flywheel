"""Dataset Manager - HuggingFace datasets integration, save/load, versioning."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import SyntheticPair


class DatasetManager:
    """Manager for dataset operations including save/load and versioning."""
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """Initialize dataset manager."""
        settings = get_settings()
        self.data_dir = Path(data_dir or settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or settings.huggingface_token
    
    def pairs_to_dicts(self, pairs: List[SyntheticPair]) -> List[Dict[str, Any]]:
        """Convert SyntheticPair objects to dictionaries."""
        return [p.to_dict() for p in pairs]
    
    def dicts_to_pairs(self, data: List[Dict[str, Any]]) -> List[SyntheticPair]:
        """Convert dictionaries to SyntheticPair objects."""
        pairs = []
        for item in data:
            try:
                pairs.append(SyntheticPair.from_dict(item))
            except Exception:
                pass
        return pairs
    
    def save_local(
        self,
        pairs: List[SyntheticPair],
        filename: str = "synthetic_data.json",
        split: Optional[str] = None,
    ) -> Path:
        """Save pairs to local JSON file."""
        if split:
            output_dir = self.data_dir / split
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.data_dir
        
        output_path = output_dir / filename
        data = self.pairs_to_dicts(pairs)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def load_local(
        self,
        filename: str = "synthetic_data.json",
        split: Optional[str] = None,
    ) -> List[SyntheticPair]:
        """Load pairs from local JSON file."""
        if split:
            input_path = self.data_dir / split / filename
        else:
            input_path = self.data_dir / filename
        
        if not input_path.exists():
            return []
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        return self.dicts_to_pairs(data)
    
    def save_to_huggingface(
        self,
        pairs: List[SyntheticPair],
        repo_id: str,
        private: bool = False,
    ) -> str:
        """Save dataset to HuggingFace Hub."""
        if not self.hf_token:
            raise ValueError("HuggingFace token required for upload")
        
        data = self.pairs_to_dicts(pairs)
        dataset = Dataset.from_list(data)
        
        dataset.push_to_hub(
            repo_id,
            token=self.hf_token,
            private=private,
        )
        
        return f"https://huggingface.co/datasets/{repo_id}"
    
    def load_from_huggingface(
        self,
        repo_id: str,
        split: str = "train",
    ) -> List[SyntheticPair]:
        """Load dataset from HuggingFace Hub."""
        dataset = load_dataset(repo_id, split=split)
        data = [dict(row) for row in dataset]
        return self.dicts_to_pairs(data)
    
    def create_train_test_split(
        self,
        pairs: List[SyntheticPair],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_seed: int = 42,
    ) -> Tuple[List[SyntheticPair], List[SyntheticPair], List[SyntheticPair]]:
        """Split pairs into train/test/validation sets."""
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
        
        return train_pairs, test_pairs, val_pairs
    
    def get_dataset_info(self, pairs: List[SyntheticPair]) -> Dict[str, Any]:
        """Get statistics about a dataset."""
        if not pairs:
            return {"count": 0}
        
        categories = {}
        for pair in pairs:
            cat = pair.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "count": len(pairs),
            "categories": categories,
        }


def create_dataset_manager(
    data_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> DatasetManager:
    """Factory function to create a dataset manager."""
    return DatasetManager(data_dir=data_dir, hf_token=hf_token)
