"""Configuration management - Load from .env, validate configs, defaults."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # OpenRouter settings
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen3-8b:free"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:4b"
    
    # Quality thresholds
    quality_min_score: float = 7.0
    quality_min_coherence: float = 6.0
    quality_min_accuracy: float = 6.0
    quality_min_helpfulness: float = 6.0
    
    # Flywheel settings
    max_cycles: int = 10
    min_pass_rate: float = 0.5
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./data/checkpoints"
    report_output_dir: str = "./reports"
    notebook_output_dir: str = "./notebooks"
    
    # HuggingFace
    huggingface_token: Optional[str] = None

    # Anthropic (for anthropic judge backend)
    anthropic_api_key: Optional[str] = None

    # Judge / data platform
    default_judge_backend: str = "ollama"
    judge_concurrency: int = 4
    rubrics_dir: str = "./rubrics"
    user_data_dir: str = "./data/user"
    validation_dir: str = "./data/validation"
    labels_dir: str = "./data/labels"
    judgments_dir: str = "./data/judgments"
    pii_policy: str = "warn"  # strict|warn|off
    
    # Training
    trainer_base_model: str = "unsloth/llama-3-8b-bnb-4bit"
    
    # Generator settings
    generator_max_tokens: int = 2048
    generator_temperature: float = 0.7
    generator_top_p: float = 0.9
    generator_timeout: int = 120
    
    # Judge settings
    judge_max_tokens: int = 1024
    judge_temperature: float = 0.3
    judge_timeout: int = 600
    
    # A2A settings
    a2a_host: str = "0.0.0.0"
    a2a_port: int = 8080
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
