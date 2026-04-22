"""Configuration management for Synthetic Data Flywheel."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    
    # OpenRouter Configuration (Generator)
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    openrouter_model: str = Field(
        default="qwen/qwen3-8b",
        description="Model to use for generation"
    )
    
    # Generator settings
    generator_max_tokens: int = Field(default=2048, ge=1, le=8192)
    generator_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    generator_top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    generator_timeout: int = Field(default=60, ge=1)
    
    # Ollama Configuration (Judge)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="gemma4:4b",
        description="Model to use for judgment"
    )
    
    # Judge settings
    judge_max_tokens: int = Field(default=1024, ge=1, le=4096)
    judge_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    judge_timeout: int = Field(default=30, ge=1)
    
    # Quality thresholds
    quality_min_score: float = Field(default=7.0, ge=0.0, le=10.0)
    quality_min_coherence: float = Field(default=6.0, ge=0.0, le=10.0)
    quality_min_accuracy: float = Field(default=6.0, ge=0.0, le=10.0)
    quality_min_helpfulness: float = Field(default=6.0, ge=0.0, le=10.0)
    
    # HuggingFace Configuration
    hf_token: Optional[str] = Field(None, description="HuggingFace API token")
    hf_dataset_namespace: Optional[str] = Field(None, description="Dataset namespace")
    hf_private_datasets: bool = Field(default=False)
    
    # Flywheel Engine Configuration
    flywheel_max_cycles: int = Field(default=10, ge=1)
    flywheel_samples_per_cycle: int = Field(default=100, ge=1)
    flywheel_seed_ratio: float = Field(default=0.2, ge=0.0, le=1.0)
    flywheel_checkpoint_interval: int = Field(default=1, ge=1)
    
    # Feedback settings
    feedback_failure_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    feedback_max_failures_per_cycle: int = Field(default=50, ge=1)
    feedback_diversity_window: int = Field(default=5, ge=1)
    
    # Training Configuration
    training_output_dir: str = Field(default="./data/training")
    training_batch_size: int = Field(default=4, ge=1)
    training_learning_rate: float = Field(default=2e-4, gt=0)
    training_num_epochs: int = Field(default=3, ge=1)
    training_max_seq_length: int = Field(default=2048, ge=1)
    training_lora_r: int = Field(default=16, ge=1)
    training_lora_alpha: int = Field(default=32, ge=1)
    training_lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    
    # Evaluation Configuration
    eval_batch_size: int = Field(default=8, ge=1)
    eval_max_samples: int = Field(default=1000, ge=1)
    eval_metrics: str = Field(default="accuracy,perplexity,f1")
    
    # A2A Agent Configuration
    a2a_host: str = Field(default="0.0.0.0")
    a2a_port: int = Field(default=8080, ge=1, le=65535)
    a2a_log_level: str = Field(default="info")
    
    # Report Generation
    report_output_dir: str = Field(default="./reports")
    report_template_dir: str = Field(default="./templates")
    report_auto_open: bool = Field(default=False)
    
    # Logging & Debugging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    debug: bool = Field(default=False)
    
    @property
    def eval_metrics_list(self) -> List[str]:
        """Parse evaluation metrics string into list."""
        return [m.strip() for m in self.eval_metrics.split(",") if m.strip()]
    
    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists():
                return current
            current = current.parent
        return Path(os.environ.get("FLYWHEEL_PROJECT_ROOT", "."))
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get checkpoint directory path."""
        return self.project_root / "data" / "checkpoints"
    
    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        """Get raw data directory path."""
        return self.project_root / "data" / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        """Get processed data directory path."""
        return self.project_root / "data" / "processed"
    
    @property
    def reports_dir(self) -> Path:
        """Get reports directory path."""
        return self.project_root / "reports"
    
    @property
    def templates_dir(self) -> Path:
        """Get templates directory path."""
        return self.project_root / "templates"
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        dirs = [
            self.checkpoint_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.templates_dir,
            Path(self.training_output_dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def to_flywheel_config(self) -> "FlywheelConfig":
        """Convert settings to FlywheelConfig."""
        from synthetic_data_flywheel.models import FlywheelConfig
        
        return FlywheelConfig(
            max_cycles=self.flywheel_max_cycles,
            samples_per_cycle=self.flywheel_samples_per_cycle,
            seed_ratio=self.flywheel_seed_ratio,
            checkpoint_interval=self.flywheel_checkpoint_interval,
            checkpoint_dir=str(self.checkpoint_dir),
            quality_min_score=self.quality_min_score,
            quality_min_coherence=self.quality_min_coherence,
            quality_min_accuracy=self.quality_min_accuracy,
            quality_min_helpfulness=self.quality_min_helpfulness,
            feedback_failure_threshold=self.feedback_failure_threshold,
            feedback_max_failures_per_cycle=self.feedback_max_failures_per_cycle,
            feedback_diversity_window=self.feedback_diversity_window,
            dataset_namespace=self.hf_dataset_namespace,
            dataset_private=self.hf_private_datasets,
            training_output_dir=self.training_output_dir,
            training_batch_size=self.training_batch_size,
            training_learning_rate=self.training_learning_rate,
            training_num_epochs=self.training_num_epochs,
            training_max_seq_length=self.training_max_seq_length,
            training_lora_r=self.training_lora_r,
            training_lora_alpha=self.training_lora_alpha,
            training_lora_dropout=self.training_lora_dropout,
            eval_batch_size=self.eval_batch_size,
            eval_max_samples=self.eval_max_samples,
            log_level=self.log_level,
            debug=self.debug,
        )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings from environment (clears cache)."""
    get_settings.cache_clear()
    return get_settings()


def get_flywheel_config() -> "FlywheelConfig":
    """Get FlywheelConfig from current settings."""
    return get_settings().to_flywheel_config()
