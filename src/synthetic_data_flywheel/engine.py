"""Flywheel Engine - Main loop orchestration, cycle management, checkpointing."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from synthetic_data_flywheel.config import get_settings
from synthetic_data_flywheel.dataset_manager import create_dataset_manager
from synthetic_data_flywheel.evaluator import create_evaluator
from synthetic_data_flywheel.generator import create_generator
from synthetic_data_flywheel.judge import create_judge
from synthetic_data_flywheel.models import CycleState, JudgmentResult, SyntheticPair
from synthetic_data_flywheel.trainer import create_trainer


class FlywheelEngine:
    """Main engine for the synthetic data flywheel."""
    
    def __init__(
        self,
        seeds: List[str],
        checkpoint_dir: Optional[str] = None,
        max_cycles: Optional[int] = None,
    ):
        """Initialize the flywheel engine."""
        settings = get_settings()
        
        self.seeds = seeds
        self.max_cycles = max_cycles or settings.max_cycles
        self.checkpoint_dir = Path(checkpoint_dir or settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.generator = create_generator()
        self.judge = create_judge()
        self.dataset_manager = create_dataset_manager()
        self.trainer = create_trainer()
        self.evaluator = create_evaluator()
        
        # State tracking
        self.current_cycle = 0
        self.cycles: List[CycleState] = []
        self.all_passed_pairs: List[SyntheticPair] = []
    
    async def run_cycle(self, cycle_id: int) -> CycleState:
        """Run a single flywheel cycle."""
        print(f"\n{'='*60}")
        print(f"Starting Cycle {cycle_id}")
        print(f"{'='*60}")
        
        cycle_start = datetime.utcnow()
        
        # Get seeds for this cycle
        cycle_seeds = self._get_seeds_for_cycle(cycle_id)
        print(f"Using {len(cycle_seeds)} seeds")
        
        # Generate synthetic pairs
        print("Generating synthetic data...")
        generated_pairs = await self.generator.generate_batch(
            seeds=cycle_seeds,
            template_type="INSTRUCTION",
        )
        print(f"Generated {len(generated_pairs)} pairs")
        
        # Judge quality
        print("Judging quality...")
        judgments = self.judge.judge_batch(generated_pairs)
        passed_pairs, failed_pairs = self.judge.filter_pairs(generated_pairs, judgments)
        print(f"Passed: {len(passed_pairs)}, Failed: {len(failed_pairs)}")
        
        # Evaluate
        eval_metrics = self.evaluator.evaluate_judgments(judgments)
        
        # Save dataset
        dataset_path = self.dataset_manager.save_local(
            passed_pairs,
            filename=f"cycle_{cycle_id:03d}_passed.json",
        )
        
        # Generate training notebook
        artifacts = self.trainer.prepare_training_artifacts(
            passed_pairs,
            cycle_id,
            self.dataset_manager,
        )
        
        cycle_end = datetime.utcnow()
        
        # Create cycle state
        cycle_state = CycleState(
            cycle_id=cycle_id,
            status="completed",
            seeds=cycle_seeds,
            generated_pairs=[p.to_dict() for p in generated_pairs],
            judgments=[j.to_dict() for j in judgments],
            passed_pairs=[p.to_dict() for p in passed_pairs],
            passed_judgments=[j.to_dict() for j in judgments if j.passed],
            eval_metrics=eval_metrics,
            dataset_path=str(dataset_path),
            artifacts={k: str(v) for k, v in artifacts.items()},
            timing={
                "started_at": cycle_start.isoformat(),
                "completed_at": cycle_end.isoformat(),
                "duration_seconds": (cycle_end - cycle_start).total_seconds(),
            },
        )
        
        # Update state
        self.cycles.append(cycle_state)
        self.all_passed_pairs.extend(passed_pairs)
        
        # Save checkpoint
        self._save_checkpoint()
        
        print(f"Cycle {cycle_id} complete. Pass rate: {cycle_state.pass_rate:.2%}")
        
        return cycle_state
    
    async def run_full_loop(self) -> List[CycleState]:
        """Run the full flywheel loop."""
        print(f"Starting Flywheel with max_cycles={self.max_cycles}")
        
        for cycle_id in range(1, self.max_cycles + 1):
            self.current_cycle = cycle_id
            
            try:
                cycle_state = await self.run_cycle(cycle_id)
                
                # Check if we should continue
                if cycle_state.pass_rate < 0.1:
                    print(f"Low pass rate ({cycle_state.pass_rate:.2%}), stopping")
                    break
                
            except Exception as e:
                print(f"Error in cycle {cycle_id}: {e}")
                break
        
        print(f"\nFlywheel complete. Ran {len(self.cycles)} cycles.")
        return self.cycles
    
    def _get_seeds_for_cycle(self, cycle_id: int) -> List[str]:
        """Get seeds for a cycle, including failure seeds from previous cycles."""
        if cycle_id == 1:
            return self.seeds
        
        # Add failure seeds from previous cycle
        failure_seeds = self._extract_failure_seeds(cycle_id - 1)
        return self.seeds + failure_seeds
    
    def _extract_failure_seeds(self, cycle_id: int) -> List[str]:
        """Extract failure cases from a cycle to use as seeds."""
        cycle = next((c for c in self.cycles if c.cycle_id == cycle_id), None)
        if not cycle:
            return []

        # Get failed judgments
        failed_judgments = [
            j for j in cycle.judgments
            if not j.passed
        ]

        # Extract seeds from failed pairs
        failure_seeds = []
        for judgment in failed_judgments:
            pair_id = str(judgment.pair_id)
            # Find the corresponding pair
            for pair in cycle.generated_pairs:
                if str(pair.id) == pair_id:
                    # Use instruction as seed
                    seed = pair.instruction
                    if seed:
                        failure_seeds.append(seed)
                    break
        
        # Limit to top failures
        return failure_seeds[:10]
    
    def _save_checkpoint(self) -> Path:
        """Save current state to checkpoint file."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.current_cycle:03d}.json"
        
        state = {
            "current_cycle": self.current_cycle,
            "cycles": [c.to_dict() for c in self.cycles],
            "all_passed_pairs_count": len(self.all_passed_pairs),
        }
        
        with open(checkpoint_path, "w") as f:
            json.dump(state, f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load state from checkpoint file."""
        if checkpoint_path:
            path = Path(checkpoint_path)
        else:
            # Find latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoints:
                return False
            path = checkpoints[-1]
        
        if not path.exists():
            return False
        
        with open(path, "r") as f:
            state = json.load(f)
        
        self.current_cycle = state.get("current_cycle", 0)
        self.cycles = [
            CycleState.from_dict(c) for c in state.get("cycles", [])
        ]
        
        print(f"Loaded checkpoint from cycle {self.current_cycle}")
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all cycles."""
        return {
            "total_cycles": len(self.cycles),
            "total_passed_pairs": len(self.all_passed_pairs),
            "avg_pass_rate": sum(c.pass_rate for c in self.cycles) / len(self.cycles) if self.cycles else 0,
            "cycles": [
                {
                    "cycle_id": c.cycle_id,
                    "pass_rate": c.pass_rate,
                    "avg_quality": c.avg_quality_score,
                    "duration": c.timing.get("duration_seconds", 0),
                }
                for c in self.cycles
            ],
        }


def create_engine(
    seeds: List[str],
    checkpoint_dir: Optional[str] = None,
    max_cycles: Optional[int] = None,
) -> FlywheelEngine:
    """Factory function to create a flywheel engine."""
    return FlywheelEngine(
        seeds=seeds,
        checkpoint_dir=checkpoint_dir,
        max_cycles=max_cycles,
    )
