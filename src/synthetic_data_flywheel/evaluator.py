"""Evaluator module - Held-out test evaluation, metric computation."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from datasets import Dataset

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.models import JudgmentResult, SyntheticPair

logger = structlog.get_logger()


class Evaluator:
    """Evaluator for assessing model performance and data quality."""
    
    def __init__(self):
        """Initialize evaluator."""
        settings = get_settings()
        self.metrics = {}
        
        logger.info("evaluator_initialized")
    
    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute exact match accuracy.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            Exact match accuracy (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return 0.0
        
        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if pred.strip().lower() == ref.strip().lower()
        )
        
        return matches / len(predictions)
    
    def compute_contains_match(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """Compute if prediction contains reference or vice versa.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            Contains match rate (0-1)
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return 0.0
        
        matches = 0
        for pred, ref in zip(predictions, references):
            pred_lower = pred.strip().lower()
            ref_lower = ref.strip().lower()
            
            # Check if either contains the other
            if ref_lower in pred_lower or pred_lower in ref_lower:
                matches += 1
        
        return matches / len(predictions)
    
    def compute_token_overlap(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute token overlap metrics.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            Dictionary with precision, recall, F1
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        if not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        total_precision = 0.0
        total_recall = 0.0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if not pred_tokens:
                continue
            
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if pred_tokens else 0
            recall = overlap / len(ref_tokens) if ref_tokens else 0
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(predictions)
        avg_recall = total_recall / len(predictions)
        
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) \
            if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": f1,
        }
    
    def compute_length_stats(
        self,
        texts: List[str],
    ) -> Dict[str, float]:
        """Compute length statistics.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with length statistics
        """
        if not texts:
            return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        
        lengths = [len(t) for t in texts]
        lengths_sorted = sorted(lengths)
        
        return {
            "mean": sum(lengths) / len(lengths),
            "median": lengths_sorted[len(lengths_sorted) // 2],
            "min": min(lengths),
            "max": max(lengths),
        }
    
    def evaluate_judgments(
        self,
        judgments: List[JudgmentResult],
    ) -> Dict[str, Any]:
        """Evaluate judgment results.
        
        Args:
            judgments: List of judgment results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not judgments:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_coherence": 0.0,
                "avg_accuracy": 0.0,
                "avg_helpfulness": 0.0,
                "avg_overall": 0.0,
            }
        
        passed = sum(1 for j in judgments if j.passed)
        
        coherence_scores = [j.scores.coherence for j in judgments]
        accuracy_scores = [j.scores.accuracy for j in judgments]
        helpfulness_scores = [j.scores.helpfulness for j in judgments]
        overall_scores = [j.scores.overall for j in judgments]
        
        return {
            "total": len(judgments),
            "passed": passed,
            "failed": len(judgments) - passed,
            "pass_rate": passed / len(judgments),
            "avg_coherence": sum(coherence_scores) / len(coherence_scores),
            "avg_accuracy": sum(accuracy_scores) / len(accuracy_scores),
            "avg_helpfulness": sum(helpfulness_scores) / len(helpfulness_scores),
            "avg_overall": sum(overall_scores) / len(overall_scores),
        }
    
    def evaluate_dataset(
        self,
        pairs: List[SyntheticPair],
        judgments: Optional[List[JudgmentResult]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a dataset of synthetic pairs.
        
        Args:
            pairs: List of synthetic pairs
            judgments: Optional list of judgments
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        if not pairs:
            return {"error": "Empty dataset"}
        
        # Compute length statistics
        instruction_lengths = self.compute_length_stats([p.instruction for p in pairs])
        output_lengths = self.compute_length_stats([p.output for p in pairs])
        
        # Category distribution
        categories = {}
        for p in pairs:
            cat = p.category or "unknown"
            categories[cat] = categories.get(cat, 0) + 1
        
        # Difficulty distribution
        difficulties = {}
        for p in pairs:
            diff = p.difficulty or "unknown"
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        result = {
            "total_samples": len(pairs),
            "instruction_lengths": instruction_lengths,
            "output_lengths": output_lengths,
            "categories": categories,
            "difficulties": difficulties,
        }
        
        # Add judgment metrics if available
        if judgments:
            judgment_metrics = self.evaluate_judgments(judgments)
            result["judgments"] = judgment_metrics
        
        logger.info(
            "dataset_evaluated",
            samples=len(pairs),
            categories=len(categories),
        )
        
        return result
    
    def evaluate_against_baseline(
        self,
        test_pairs: List[SyntheticPair],
        baseline_pairs: List[SyntheticPair],
    ) -> Dict[str, Any]:
        """Compare test dataset against baseline.
        
        Args:
            test_pairs: Test dataset
            baseline_pairs: Baseline dataset
            
        Returns:
            Comparison metrics
        """
        test_eval = self.evaluate_dataset(test_pairs)
        baseline_eval = self.evaluate_dataset(baseline_pairs)
        
        return {
            "test": test_eval,
            "baseline": baseline_eval,
            "comparison": {
                "sample_ratio": test_eval["total_samples"] / baseline_eval["total_samples"]
                    if baseline_eval["total_samples"] > 0 else 0,
                "output_length_ratio": test_eval["output_lengths"]["mean"] / baseline_eval["output_lengths"]["mean"]
                    if baseline_eval["output_lengths"]["mean"] > 0 else 0,
            },
        }
    
    def save_evaluation(
        self,
        metrics: Dict[str, Any],
        output_path: Path,
    ) -> Path:
        """Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("evaluation_saved", path=str(output_path))
        
        return output_path


def create_evaluator() -> Evaluator:
    """Factory function to create an evaluator.
    
    Returns:
        Configured Evaluator
    """
    return Evaluator()
