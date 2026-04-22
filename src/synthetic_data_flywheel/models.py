"""Core data models - Pydantic models for SyntheticPair, JudgmentResult, CycleState, FlywheelConfig."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SyntheticPair(BaseModel):
    """A synthetic training data pair (instruction/output)."""
    
    id: Union[str, UUID] = Field(default_factory=uuid4)
    instruction: str = Field(..., description="The instruction or question")
    input: Optional[str] = Field(None, description="Optional input context")
    output: str = Field(..., description="The response or answer")
    context: Optional[str] = Field(None, description="Additional context")
    category: Optional[str] = Field(None, description="Category of the pair")
    difficulty: Optional[str] = Field(None, description="Difficulty level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_seed: Optional[str] = Field(None, description="Original seed used for generation")
    cycle_id: Optional[int] = Field(None, description="Cycle ID when generated")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "context": self.context,
            "category": self.category,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "source_seed": self.source_seed,
            "cycle_id": self.cycle_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticPair":
        """Create from dictionary."""
        data = data.copy()
        if "id" in data and isinstance(data["id"], str):
            try:
                data["id"] = UUID(data["id"])
            except ValueError:
                pass  # Keep as string if not a valid UUID
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class QualityScores(BaseModel):
    """Quality scores for a synthetic pair."""
    
    coherence: float = Field(..., ge=0, le=10, description="Coherence score 0-10")
    accuracy: float = Field(..., ge=0, le=10, description="Accuracy score 0-10")
    helpfulness: float = Field(..., ge=0, le=10, description="Helpfulness score 0-10")
    overall: float = Field(..., ge=0, le=10, description="Overall quality score 0-10")
    
    @field_validator("coherence", "accuracy", "helpfulness", "overall")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is within bounds."""
        return max(0.0, min(10.0, float(v)))


class JudgmentResult(BaseModel):
    """Result of quality judgment on a synthetic pair."""
    
    pair_id: Union[str, UUID] = Field(..., description="ID of the judged pair")
    scores: QualityScores = Field(..., description="Quality scores")
    passed: bool = Field(..., description="Whether the pair passed quality threshold")
    judge_model: str = Field(..., description="Model used for judging")
    judgment_reasoning: str = Field(default="", description="Reasoning for the judgment")
    judged_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def quality_tier(self) -> str:
        """Get quality tier based on overall score."""
        if self.scores.overall >= 8:
            return "high"
        elif self.scores.overall >= 6:
            return "medium"
        else:
            return "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pair_id": str(self.pair_id),
            "scores": {
                "coherence": self.scores.coherence,
                "accuracy": self.scores.accuracy,
                "helpfulness": self.scores.helpfulness,
                "overall": self.scores.overall,
            },
            "passed": self.passed,
            "judge_model": self.judge_model,
            "judgment_reasoning": self.judgment_reasoning,
            "judged_at": self.judged_at.isoformat() if self.judged_at else None,
            "quality_tier": self.quality_tier,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgmentResult":
        """Create from dictionary."""
        data = data.copy()
        if "pair_id" in data and isinstance(data["pair_id"], str):
            try:
                data["pair_id"] = UUID(data["pair_id"])
            except ValueError:
                pass  # Keep as string if not a valid UUID
        if "judged_at" in data and isinstance(data["judged_at"], str):
            data["judged_at"] = datetime.fromisoformat(data["judged_at"])
        if "scores" in data:
            data["scores"] = QualityScores(**data["scores"])
        return cls(**data)


class CycleState(BaseModel):
    """State of a flywheel cycle."""
    
    cycle_id: int = Field(..., description="Unique cycle identifier")
    status: str = Field(default="pending", description="Cycle status")
    seeds: List[str] = Field(default_factory=list, description="Seeds used for generation")
    generated_pairs: List[SyntheticPair] = Field(default_factory=list, description="Generated pairs")
    judgments: List[JudgmentResult] = Field(default_factory=list, description="Judgment results")
    passed_pairs: List[SyntheticPair] = Field(default_factory=list, description="Pairs that passed")
    passed_judgments: List[JudgmentResult] = Field(default_factory=list, description="Judgments for passed pairs")
    eval_metrics: Dict[str, Any] = Field(default_factory=dict, description="Evaluation metrics")
    dataset_path: Optional[str] = Field(None, description="Path to saved dataset")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Generated artifacts")
    start_time: Optional[datetime] = Field(None, description="Cycle start time")
    end_time: Optional[datetime] = Field(None, description="Cycle end time")
    duration_seconds: Optional[float] = Field(None, description="Cycle duration in seconds")
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if not self.judgments:
            return 0.0
        passed = sum(1 for j in self.judgments if j.passed)
        return passed / len(self.judgments)
    
    @property
    def avg_quality_score(self) -> float:
        """Calculate average quality score."""
        if not self.judgments:
            return 0.0
        return sum(j.scores.overall for j in self.judgments) / len(self.judgments)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "seeds": self.seeds,
            "generated_pairs": [p.to_dict() for p in self.generated_pairs],
            "judgments": [j.to_dict() for j in self.judgments],
            "passed_pairs": [p.to_dict() for p in self.passed_pairs],
            "passed_judgments": [j.to_dict() for j in self.passed_judgments],
            "eval_metrics": self.eval_metrics,
            "dataset_path": self.dataset_path,
            "artifacts": self.artifacts,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "pass_rate": self.pass_rate,
            "avg_quality_score": self.avg_quality_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleState":
        """Create from dictionary."""
        data = data.copy()
        
        if "generated_pairs" in data:
            data["generated_pairs"] = [
                SyntheticPair.from_dict(p) if isinstance(p, dict) else p
                for p in data["generated_pairs"]
            ]
        if "judgments" in data:
            data["judgments"] = [
                JudgmentResult.from_dict(j) if isinstance(j, dict) else j
                for j in data["judgments"]
            ]
        if "passed_pairs" in data:
            data["passed_pairs"] = [
                SyntheticPair.from_dict(p) if isinstance(p, dict) else p
                for p in data["passed_pairs"]
            ]
        if "passed_judgments" in data:
            data["passed_judgments"] = [
                JudgmentResult.from_dict(j) if isinstance(j, dict) else j
                for j in data["passed_judgments"]
            ]
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        
        # Remove computed properties if present
        data.pop("pass_rate", None)
        data.pop("avg_quality_score", None)
        
        return cls(**data)


class FlywheelConfig(BaseModel):
    """Configuration for the flywheel."""
    
    # OpenRouter settings
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API key")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openrouter_model: str = Field("qwen/qwen3-8b:free", description="Model for generation")
    
    # Ollama settings
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field("gemma4:4b", description="Model for judging")
    
    # Quality thresholds
    quality_min_score: float = Field(7.0, ge=0, le=10, description="Minimum quality score")
    quality_min_coherence: float = Field(6.0, ge=0, le=10, description="Minimum coherence score")
    quality_min_accuracy: float = Field(6.0, ge=0, le=10, description="Minimum accuracy score")
    quality_min_helpfulness: float = Field(6.0, ge=0, le=10, description="Minimum helpfulness score")
    
    # Flywheel settings
    max_cycles: int = Field(10, ge=1, description="Maximum cycles to run")
    min_pass_rate: float = Field(0.5, ge=0, le=1, description="Minimum pass rate to continue")
    
    # Paths
    data_dir: str = Field("./data", description="Data directory")
    checkpoint_dir: str = Field("./data/checkpoints", description="Checkpoint directory")
    report_output_dir: str = Field("./reports", description="Report output directory")
    notebook_output_dir: str = Field("./notebooks", description="Notebook output directory")
    
    # HuggingFace
    huggingface_token: Optional[str] = Field(None, description="HuggingFace API token")
    
    # Training
    trainer_base_model: str = Field("unsloth/llama-3-8b-bnb-4bit", description="Base model for training")
    
    # Generator settings
    generator_max_tokens: int = Field(2048, ge=1, description="Max tokens for generation")
    generator_temperature: float = Field(0.7, ge=0, le=2, description="Generation temperature")
    generator_top_p: float = Field(0.9, ge=0, le=1, description="Generation top-p")
    generator_timeout: int = Field(120, ge=1, description="Generation timeout in seconds")
    
    # Judge settings
    judge_max_tokens: int = Field(1024, ge=1, description="Max tokens for judging")
    judge_temperature: float = Field(0.3, ge=0, le=2, description="Judging temperature")
    judge_timeout: int = Field(60, ge=1, description="Judging timeout in seconds")
    
    # A2A settings
    a2a_host: str = Field("0.0.0.0", description="A2A server host")
    a2a_port: int = Field(8080, ge=1, le=65535, description="A2A server port")
    
    # Logging
    log_level: str = Field("INFO", description="Log level")
    log_format: str = Field("json", description="Log format")
