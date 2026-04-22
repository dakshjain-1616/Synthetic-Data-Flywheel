"""Flywheel Engine - Main loop orchestration, cycle management, checkpointing."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import structlog

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.dataset_manager import DatasetManager
from synthetic_data_flywheel.evaluator import Evaluator
from synthetic_data_flywheel.generator import OpenRouterClient, PromptTemplate
from synthetic_data_flywheel.judge import OllamaClient, QualityJudge
from synthetic_data_flywheel.models import CycleState, JudgmentResult, SyntheticPair
from synthetic_data_flywheel.report_generator import ReportGenerator
from synthetic_data_flywheel.trainer import Trainer

logger = structlog.get_logger()


class FlywheelEngine:
    """Main engine for the synthetic data flywheel."""
    
    def __init__(
        self,
        generator: Optional[OpenRouterClient] = None,
        judge: Optional[QualityJudge] = None,
        dataset_manager: Optional[DatasetManager] = None,
        trainer: Optional[Trainer] = None,
        evaluator: Optional[Evaluator] = None,
        report_generator: Optional[ReportGenerator] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """Initialize flywheel engine.
        
        Args:
            generator: OpenRouter client for generation
            judge: Quality judge for filtering
            dataset_manager: Dataset manager
            trainer: Training notebook generator
            evaluator: Evaluation metrics
            report_generator: Report generator
            checkpoint_dir: Directory for checkpoints
        """
        settings = get_settings()
        
        self.generator = generator or OpenRouterClient()
        self.judge = judge or QualityJudge()
        self.dataset_manager = dataset_manager or DatasetManager()
        self.trainer = trainer or Trainer()
        self.evaluator = evaluator or Evaluator()
        self.report_generator = report_generator or ReportGenerator()
        
        self.checkpoint_dir = Path(checkpoint_dir or settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.cycles: List[CycleState] = []
        self.current_cycle: int = 0
        
        logger.info(
            "flywheel_engine_initialized",
            checkpoint_dir=str(self.checkpoint_dir),
        )
    
    async def run_cycle(
        self,
        seeds: List[str],
        cycle_id: Optional[int] = None,
        template_type: str = PromptTemplate.INSTRUCTION,
        max_concurrent: int = 5,
    ) -> CycleState:
        """Run a single flywheel cycle.
        
        Args:
            seeds: Seed prompts for generation
            cycle_id: Optional cycle ID
            template_type: Prompt template type
            max_concurrent: Max concurrent generations
            
        Returns:
            Cycle state with results
        """
        cycle_id = cycle_id or self.current_cycle + 1
        self.current_cycle = cycle_id
        
        start_time = datetime.now()
        logger.info("cycle_started", cycle_id=cycle_id, seeds=len(seeds))
        
        # Step 1: Generate synthetic data
        logger.info("generating_synthetic_data", cycle_id=cycle_id)
        generated_pairs = await self.generator.generate_batch(
            seeds=seeds,
            template_type=template_type,
            cycle_id=cycle_id,
            max_concurrent=max_concurrent,
        )
        
        logger.info(
            "generation_complete",
            cycle_id=cycle_id,
            generated=len(generated_pairs),
        )
        
        # Step 2: Judge quality
        logger.info("judging_quality", cycle_id=cycle_id)
        judgments = self.judge.judge_batch(generated_pairs)
        
        passed_count = sum(1 for j in judgments if j.passed)
        logger.info(
            "judgment_complete",
            cycle_id=cycle_id,
            total=len(judgments),
            passed=passed_count,
            failed=len(judgments) - passed_count,
        )
        
        # Step 3: Filter pairs
        passed_pairs, passed_judgments = self.judge.filter_pairs(
            generated_pairs,
            judgments,
        )
        
        # Step 4: Evaluate
        eval_metrics = self.evaluator.evaluate_dataset(passed_pairs, passed_judgments)
        
        # Step 5: Save dataset
        dataset_path = self.dataset_manager.save_local(
            passed_pairs,
            filename=f"cycle_{cycle_id:03d}_passed.json",
        )
        
        # Step 6: Generate training artifacts
        artifacts = self.trainer.prepare_training_artifacts(
            passed_pairs,
            cycle_id,
            self.dataset_manager,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create cycle state
        cycle_state = CycleState(
            cycle_id=cycle_id,
            status="completed",
            seeds=seeds,
            generated_pairs=generated_pairs,
            judgments=judgments,
            passed_pairs=passed_pairs,
            passed_judgments=passed_judgments,
            eval_metrics=eval_metrics,
            dataset_path=str(dataset_path),
            artifacts={k: str(v) for k, v in artifacts.items()},
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )
        
        self.cycles.append(cycle_state)
        
        # Save checkpoint
        self._save_checkpoint(cycle_state)
        
        logger.info(
            "cycle_complete",
            cycle_id=cycle_id,
            duration=duration,
            passed=len(passed_pairs),
            pass_rate=cycle_state.pass_rate,
        )
        
        return cycle_state
    
    def _extract_failure_seeds(
        self,
        cycle_state: CycleState,
    ) -> List[str]:
        """Extract seeds from failed pairs for next cycle.
        
        Args:
            cycle_state: Completed cycle state
            
        Returns:
            List of failure seeds
        """
        failure_seeds = []
        
        # Map judgments to pairs
        judgment_map = {j.pair_id: j for j in cycle_state.judgments}
        
        for pair in cycle_state.generated_pairs:
            judgment = judgment_map.get(pair.id)
            if judgment and not judgment.passed:
                # Use the original instruction as seed for retry
                failure_seeds.append(pair.instruction)
        
        logger.info(
            "failure_seeds_extracted",
            cycle_id=cycle_state.cycle_id,
            failures=len(failure_seeds),
        )
        
        return failure_seeds
    
    def _feedback_loop(
        self,
        cycle_state: CycleState,
        new_seeds: Optional[List[str]] = None,
    ) -> List[str]:
        """Prepare seeds for next cycle.
        
        Args:
            cycle_state: Current cycle state
            new_seeds: Optional new seeds to add
            
        Returns:
            Seeds for next cycle
        """
        # Get failure seeds
        failure_seeds = self._extract_failure_seeds(cycle_state)
        
        # Combine with new seeds
        next_seeds = failure_seeds.copy()
        if new_seeds:
            next_seeds.extend(new_seeds)
        
        # Deduplicate
        seen = set()
        unique_seeds = []
        for seed in next_seeds:
            if seed not in seen:
                seen.add(seed)
                unique_seeds.append(seed)
        
        logger.info(
            "feedback_loop_complete",
            cycle_id=cycle_state.cycle_id,
            next_cycle=cycle_state.cycle_id + 1,
            seeds_for_next=len(unique_seeds),
        )
        
        return unique_seeds
    
    async def run_full_loop(
        self,
        initial_seeds: List[str],
        max_cycles: Optional[int] = None,
        min_pass_rate: float = 0.5,
        template_type: str = PromptTemplate.INSTRUCTION,
    ) -> List[CycleState]:
        """Run full flywheel loop with feedback.
        
        Args:
            initial_seeds: Starting seeds
            max_cycles: Maximum cycles to run
            min_pass_rate: Minimum pass rate to continue
            template_type: Prompt template type
            
        Returns:
            List of cycle states
        """
        settings = get_settings()
        max_cycles = max_cycles or settings.max_cycles
        
        seeds = initial_seeds.copy()
        
        for cycle_num in range(1, max_cycles + 1):
            if not seeds:
                logger.info("no_more_seeds", cycle=cycle_num)
                break
            
            # Run cycle
            cycle_state = await self.run_cycle(
                seeds=seeds,
                cycle_id=cycle_num,
                template_type=template_type,
            )
            
            # Check if we should continue
            if cycle_state.pass_rate < min_pass_rate:
                logger.warning(
                    "pass_rate_below_threshold",
                    cycle=cycle_num,
                    pass_rate=cycle_state.pass_rate,
                    threshold=min_pass_rate,
                )
            
            # Prepare seeds for next cycle
            seeds = self._feedback_loop(cycle_state)
        
        # Generate final report
        if self.cycles:
            report_path = self.report_generator.generate_flywheel_report(self.cycles)
            logger.info("final_report_generated", path=str(report_path))
        
        return self.cycles
    
    def _save_checkpoint(self, cycle_state: CycleState) -> Path:
        """Save cycle checkpoint.
        
        Args:
            cycle_state: Cycle state to save
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"cycle_{cycle_state.cycle_id:03d}.json"
        
        with open(checkpoint_path, "w") as f:
            json.dump(cycle_state.to_dict(), f, indent=2, default=str)
        
        logger.info("checkpoint_saved", path=str(checkpoint_path))
        
        return checkpoint_path
    
    def load_checkpoint(self, cycle_id: int) -> Optional[CycleState]:
        """Load cycle checkpoint.
        
        Args:
            cycle_id: Cycle ID to load
            
        Returns:
            Cycle state or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"cycle_{cycle_id:03d}.json"
        
        if not checkpoint_path.exists():
            logger.warning("checkpoint_not_found", path=str(checkpoint_path))
            return None
        
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
        
        cycle_state = CycleState.from_dict(data)
        
        logger.info("checkpoint_loaded", cycle_id=cycle_id)
        
        return cycle_state
    
    def list_checkpoints(self) -> List[int]:
        """List available checkpoint cycle IDs.
        
        Returns:
            List of cycle IDs
        """
        checkpoints = []
        
        for file in self.checkpoint_dir.glob("cycle_*.json"):
            try:
                cycle_id = int(file.stem.split("_")[1])
                checkpoints.append(cycle_id)
            except (IndexError, ValueError):
                continue
        
        return sorted(checkpoints)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all cycles.
        
        Returns:
            Summary statistics
        """
        if not self.cycles:
            return {"cycles": 0}
        
        total_generated = sum(len(c.generated_pairs) for c in self.cycles)
        total_passed = sum(len(c.passed_pairs) for c in self.cycles)
        
        return {
            "cycles": len(self.cycles),
            "total_generated": total_generated,
            "total_passed": total_passed,
            "overall_pass_rate": total_passed / total_generated if total_generated else 0,
            "avg_quality_per_cycle": [
                {"cycle": c.cycle_id, "avg_quality": c.avg_quality_score}
                for c in self.cycles
            ],
        }


def create_engine(
    generator: Optional[OpenRouterClient] = None,
    judge: Optional[QualityJudge] = None,
    **kwargs,
) -> FlywheelEngine:
    """Factory function to create flywheel engine.
    
    Args:
        generator: Generator client
        judge: Quality judge
        **kwargs: Additional kwargs for engine
        
    Returns:
        Configured FlywheelEngine
    """
    return FlywheelEngine(
        generator=generator,
        judge=judge,
        **kwargs
    )
