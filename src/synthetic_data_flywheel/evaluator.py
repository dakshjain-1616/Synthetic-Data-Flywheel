"""Evaluator module - Held-out test evaluation, metric computation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_data_flywheel.models import JudgmentResult, SyntheticPair


class Evaluator:
    """Evaluator for assessing model performance and data quality."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute exact match accuracy."""
        if len(predictions) != len(references) or not predictions:
            return 0.0
        
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        return matches / len(predictions)
    
    def evaluate_judgments(
        self,
        judgments: List[JudgmentResult],
    ) -> Dict[str, Any]:
        """Evaluate judgment results."""
        if not judgments:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_overall": 0.0,
            }
        
        passed = sum(1 for j in judgments if j.passed)
        
        return {
            "total": len(judgments),
            "passed": passed,
            "failed": len(judgments) - passed,
            "pass_rate": passed / len(judgments),
            "avg_coherence": sum(j.scores.coherence for j in judgments) / len(judgments),
            "avg_accuracy": sum(j.scores.accuracy for j in judgments) / len(judgments),
            "avg_helpfulness": sum(j.scores.helpfulness for j in judgments) / len(judgments),
            "avg_overall": sum(j.scores.overall for j in judgments) / len(judgments),
        }
    
    def evaluate_dataset(
        self,
        pairs: List[SyntheticPair],
        judgments: Optional[List[JudgmentResult]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a dataset of synthetic pairs."""
        if not pairs:
            return {"error": "Empty dataset"}
        
        # Category distribution
        categories = {}
        for p in pairs:
            cat = p.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        
        result = {
            "total_samples": len(pairs),
            "categories": categories,
        }
        
        if judgments:
            result["judgments"] = self.evaluate_judgments(judgments)
        
        return result


def create_evaluator() -> Evaluator:
    """Factory function to create an evaluator."""
    return Evaluator()
