"""Core data models - Pydantic models for SyntheticPair, JudgmentResult, CycleState, FlywheelConfig."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SyntheticPair(BaseModel):
    """A synthetic training data pair (instruction/output)."""
    
    id: Union[str, UUID] = Field(default_factory=uuid4)
    instruction: str = Field(default="", description="The instruction or question")
    input: Optional[str] = Field(default=None, description="Optional input context")
    output: str = Field(default="", description="The response or answer")
    context: Optional[str] = Field(default=None, description="Additional context")
    category: Optional[str] = Field(default=None, description="Category of the pair")
    difficulty: Optional[str] = Field(default=None, description="Difficulty level")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_seed: Optional[str] = Field(default=None, description="Original seed used for generation")
    cycle_id: Optional[int] = Field(default=None, description="Cycle ID when generated")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
    
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
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace('Z', '+00:00'))
        return cls(**data)


class QualityScores(BaseModel):
    """Quality scores for a synthetic pair."""
    
    coherence: float = Field(default=5.0, ge=0, le=10, description="Coherence score 0-10")
    accuracy: float = Field(default=5.0, ge=0, le=10, description="Accuracy score 0-10")
    helpfulness: float = Field(default=5.0, ge=0, le=10, description="Helpfulness score 0-10")
    overall: float = Field(default=5.0, ge=0, le=10, description="Overall quality score 0-10")
    
    @field_validator("coherence", "accuracy", "helpfulness", "overall")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is within bounds."""
        return max(0.0, min(10.0, float(v)))


class RubricRef(BaseModel):
    """Reference to the rubric a judgment was produced under."""

    name: str = ""
    version: int = 1


class JudgmentResult(BaseModel):
    """Result of quality judgment on a synthetic pair."""

    pair_id: Union[str, UUID] = Field(..., description="ID of the judged pair")
    scores: QualityScores = Field(default_factory=QualityScores, description="Quality scores")
    passed: bool = Field(default=False, description="Whether the pair passed quality threshold")
    judge_model: str = Field(default="", description="Model used for judging")
    judgment_reasoning: str = Field(default="", description="Reasoning for the judgment")
    judged_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rubric: Optional[RubricRef] = Field(default=None, description="Rubric used for judgment")
    tag: Optional[str] = Field(default=None, description="User tag to distinguish judgment runs")

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to attributes."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
    
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
            "rubric": self.rubric.model_dump() if self.rubric else None,
            "tag": self.tag,
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
            data["judged_at"] = datetime.fromisoformat(data["judged_at"].replace('Z', '+00:00'))
        if "scores" in data:
            scores_data = data["scores"]
            # Handle partial scores - fill in defaults
            if isinstance(scores_data, dict):
                scores_data = {
                    "coherence": scores_data.get("coherence", scores_data.get("overall", 5.0)),
                    "accuracy": scores_data.get("accuracy", scores_data.get("overall", 5.0)),
                    "helpfulness": scores_data.get("helpfulness", scores_data.get("overall", 5.0)),
                    "overall": scores_data.get("overall", 5.0),
                }
            data["scores"] = QualityScores(**scores_data)
        if "rubric" in data and isinstance(data["rubric"], dict):
            data["rubric"] = RubricRef(**data["rubric"])
        data.pop("quality_tier", None)
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
    dataset_path: Optional[str] = Field(default=None, description="Path to saved dataset")
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Generated artifacts")
    start_time: Optional[datetime] = Field(default=None, description="Cycle start time")
    end_time: Optional[datetime] = Field(default=None, description="Cycle end time")
    duration_seconds: Optional[float] = Field(default=None, description="Cycle duration in seconds")
    timing: Dict[str, Any] = Field(default_factory=dict, description="Timing information")
    
    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if not self.generated_pairs:
            return 0.0
        passed = len(self.passed_pairs)
        return passed / len(self.generated_pairs)
    
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
            data["start_time"] = datetime.fromisoformat(data["start_time"].replace('Z', '+00:00'))
        if "end_time" in data and isinstance(data["end_time"], str):
            data["end_time"] = datetime.fromisoformat(data["end_time"].replace('Z', '+00:00'))
        
        # Remove computed properties if present
        data.pop("pass_rate", None)
        data.pop("avg_quality_score", None)
        
        return cls(**data)


class FlywheelConfig(BaseModel):
    """Configuration for the flywheel."""
    
    # OpenRouter settings
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openrouter_model: str = Field(default="qwen/qwen3-8b:free", description="Model for generation")
    
    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="gemma4:4b", description="Model for judging")
    
    # Quality thresholds
    quality_min_score: float = Field(default=7.0, ge=0, le=10, description="Minimum quality score")
    quality_min_coherence: float = Field(default=6.0, ge=0, le=10, description="Minimum coherence score")
    quality_min_accuracy: float = Field(default=6.0, ge=0, le=10, description="Minimum accuracy score")
    quality_min_helpfulness: float = Field(default=6.0, ge=0, le=10, description="Minimum helpfulness score")
    
    # Flywheel settings
    max_cycles: int = Field(default=10, ge=1, description="Maximum cycles to run")
    min_pass_rate: float = Field(default=0.5, ge=0, le=1, description="Minimum pass rate to continue")
    
    # Paths
    data_dir: str = Field(default="./data", description="Data directory")
    checkpoint_dir: str = Field(default="./data/checkpoints", description="Checkpoint directory")
    report_output_dir: str = Field(default="./reports", description="Report output directory")
    notebook_output_dir: str = Field(default="./notebooks", description="Notebook output directory")
    
    # HuggingFace
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    
    # Training
    trainer_base_model: str = Field(default="unsloth/llama-3-8b-bnb-4bit", description="Base model for training")
    
    # Generator settings
    generator_max_tokens: int = Field(default=2048, ge=1, description="Max tokens for generation")
    generator_temperature: float = Field(default=0.7, ge=0, le=2, description="Generation temperature")
    generator_top_p: float = Field(default=0.9, ge=0, le=1, description="Generation top-p")
    generator_timeout: int = Field(default=120, ge=1, description="Generation timeout in seconds")
    
    # Judge settings
    judge_max_tokens: int = Field(default=1024, ge=1, description="Max tokens for judging")
    judge_temperature: float = Field(default=0.3, ge=0, le=2, description="Judging temperature")
    judge_timeout: int = Field(default=180, ge=1, description="Judging timeout in seconds")
    
    # A2A settings
    a2a_host: str = Field(default="0.0.0.0", description="A2A server host")
    a2a_port: int = Field(default=8080, ge=1, le=65535, description="A2A server port")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")


class Label(BaseModel):
    """A human/auto label applied to a pair."""

    pair_id: Union[str, UUID] = Field(..., description="ID of the labeled pair")
    status: str = Field(..., description="approved|rejected|needs_edit|skipped")
    tag: Optional[str] = Field(default=None, description="Free-form tag")
    note: Optional[str] = Field(default=None, description="Labeler note")
    labeler: str = Field(default="user", description="Who/what produced the label")
    labeled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": str(self.pair_id),
            "status": self.status,
            "tag": self.tag,
            "note": self.note,
            "labeler": self.labeler,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Label":
        data = data.copy()
        if "labeled_at" in data and isinstance(data["labeled_at"], str):
            data["labeled_at"] = datetime.fromisoformat(data["labeled_at"].replace("Z", "+00:00"))
        return cls(**data)


class ValidationIssue(BaseModel):
    """A single validation issue found for a pair."""

    pair_id: Union[str, UUID]
    check: str
    severity: str = Field(default="warning", description="error|warning|info")
    message: str = ""
    field: Optional[str] = None


class ValidationReport(BaseModel):
    """Aggregate validation results for a dataset."""

    dataset: str
    total_pairs: int = 0
    counts: Dict[str, int] = Field(default_factory=dict)
    issues: List[ValidationIssue] = Field(default_factory=list)
    produced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total_pairs": self.total_pairs,
            "counts": self.counts,
            "issues": [i.model_dump() | {"pair_id": str(i.pair_id)} for i in self.issues],
            "produced_at": self.produced_at.isoformat(),
        }


class DatasetMeta(BaseModel):
    """Metadata about a user-ingested dataset."""

    name: str
    source: str = Field(default="user", description="jsonl|json|csv|hf|user|generated")
    row_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = Field(default_factory=list)
    mapping: Dict[str, str] = Field(default_factory=dict)
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "row_count": self.row_count,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "mapping": self.mapping,
            "checksum": self.checksum,
        }
