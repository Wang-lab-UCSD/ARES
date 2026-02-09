"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LLMModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    provider: str = Field(description="LLM provider: openai, anthropic, or gemini")
    model: str = Field(description="Model name/ID")
    api_key_env: str = Field(description="Environment variable containing API key")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"openai", "anthropic", "gemini"}
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of {allowed}, got {v}")
        return v.lower()

    def get_api_key(self) -> str:
        """Get API key from environment variable."""
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"API key not found in environment variable: {self.api_key_env}")
        return key


class LLMConfig(BaseModel):
    """Configuration for all LLM models used in the pipeline."""

    hypothesis_model: LLMModelConfig
    coding_model: LLMModelConfig
    summary_model: LLMModelConfig


class ExecutionConfig(BaseModel):
    """Configuration for code execution."""

    type: str = Field(default="jupyter", description="Execution type: jupyter")
    timeout_seconds: int = Field(default=300, ge=0)  # 0 means no timeout
    max_retries: int = Field(default=5, ge=0)


class PipelineConfig(BaseModel):
    """Configuration for the pipeline behavior."""

    max_iterations: int = Field(default=10, gt=0)
    log_dir: str = Field(default="logs")
    output_dir: str = Field(default="outputs")
    save_intermediate: bool = Field(default=True)


class CostConfig(BaseModel):
    """Configuration for cost tracking and limiting."""

    enabled: bool = Field(default=True, description="Enable cost tracking and blocking")
    per_call_limit_usd: float = Field(
        default=10.0, gt=0,
        description="Maximum cost for a single API call in USD (prevents runaway token usage)"
    )
    session_limit_usd: float = Field(
        default=50.0, gt=0,
        description="Maximum total cost for the entire session in USD"
    )
    warn_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Fraction of session limit at which to warn (0.8 = 80%)"
    )


class InteractiveConfig(BaseModel):
    """Configuration for interactive approval mode."""

    enabled: bool = Field(default=False, description="Enable interactive approval")
    approve_hypotheses: bool = Field(default=True, description="Require approval for hypotheses")
    approve_code: bool = Field(default=True, description="Require approval for code")
    use_rich: bool = Field(default=True, description="Use rich library for enhanced UI")
    max_regeneration_attempts: int = Field(
        default=3, ge=1,
        description="Maximum attempts after rejection before skipping"
    )


class Config(BaseModel):
    """Root configuration object."""

    llm: LLMConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    interactive: InteractiveConfig = Field(default_factory=InteractiveConfig)


class DataManifest(BaseModel):
    """Data manifest describing available data for analysis."""

    finding: str = Field(description="The scientific finding to investigate (X predicts Y)")
    context: str = Field(default="", description="Additional context about the finding")
    data: dict[str, Any] = Field(default_factory=dict, description="Available data paths")
    tools: list[str] = Field(default_factory=list, description="Available CLI tools")


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def load_data_manifest(manifest_path: str | Path) -> DataManifest:
    """Load data manifest from YAML file."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Data manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    return DataManifest.model_validate(data)
