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
    max_tokens: int | None = Field(default=None, description="Max output tokens. None = use API default (no explicit limit). Set for models with low defaults (e.g. DeepSeek: 8192).")
    reasoning_effort: str | None = Field(default=None, description="Reasoning effort for OpenAI reasoning models: 'low', 'medium', or 'high'. None = API default (high).")
    base_url: str | None = Field(
        default=None,
        description="Optional custom API base URL (e.g. for Anthropic-compatible third-party endpoints)",
    )

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
    review_model: LLMModelConfig | None = Field(
        default=None,
        description="Optional dedicated model for code/hypothesis review. Falls back to coding_model if not set.",
    )


class ExecutionConfig(BaseModel):
    """Configuration for code execution."""

    type: str = Field(default="jupyter", description="Execution type: jupyter")
    timeout_seconds: int = Field(default=300, ge=0)  # 0 means no timeout
    max_retries: int = Field(default=5, ge=0)


class PipelineConfig(BaseModel):
    """Configuration for the pipeline behavior."""

    max_iterations: int = Field(default=10, gt=0)
    max_consecutive_hypothesis_rejections: int = Field(
        default=6, ge=1,
        description="After this many consecutive reviewer rejections, synthesize from supported evidence if any, else stop."
    )
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
    """Data manifest describing available data for analysis.

    Supports a ``base_manifest`` field that points to a shared YAML file (e.g.,
    cell-line-wide data).  The base is loaded first, then the pair-specific
    manifest is deep-merged on top — pair-specific keys override base keys.
    """

    base_manifest: str | None = Field(
        default=None,
        description="Optional path to a base manifest YAML. Resolved relative to this manifest file. "
                    "The base 'data' dict is deep-merged under this manifest's 'data' dict.",
    )
    finding: str = Field(description="The scientific finding to investigate (X predicts Y)")
    context: str = Field(default="", description="Additional context about the finding")
    investigation_objective: str = Field(
        default="",
        description="Goal of the investigation (e.g. discover and synthesize a coherent mechanism, possibly new). When set, prompts emphasize discovery and allow convergence on a new named mechanism.",
    )
    data: dict[str, Any] = Field(default_factory=dict, description="Available data paths")
    tools: list[str] = Field(default_factory=list, description="Available CLI tools")
    execution_requirements_path: str | None = Field(
        default=None,
        description="Optional path to a requirements file (e.g. requirements.txt) listing Python packages available in the execution environment. Relative paths are resolved from the manifest file's directory. Used to restrict generated code to only import listed packages.",
    )
    causal_capable_data: bool = Field(
        default=False,
        description="When True, the run includes data that can support causal inference (e.g. Perturb-seq, time-series, knockdown, CRISPRi). Convergence then requires the but-for test. When False (default), data are observational only; convergence is allowed on the best-supported mechanism given the data without requiring causality.",
    )


def load_config(config_path: str | Path) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config.model_validate(data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (override wins on conflicts)."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_data_manifest(manifest_path: str | Path) -> DataManifest:
    """Load data manifest from YAML file.

    Optionally supports ``base_manifest`` — if present, the base file is loaded
    first and the pair-specific manifest is deep-merged on top. If absent, the
    manifest is loaded as-is (fully backwards compatible).
    """
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Data manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    base_path = data.pop("base_manifest", None)
    if base_path:
        base_resolved = (manifest_path.parent / base_path).resolve()
        if not base_resolved.exists():
            raise FileNotFoundError(f"Base manifest not found: {base_resolved}")
        with open(base_resolved) as f:
            base_data = yaml.safe_load(f)
        data = _deep_merge(base_data, data)

    return DataManifest.model_validate(data)


def parse_requirements_package_names(requirements_path: str | Path) -> list[str]:
    """Parse a requirements-style file and return package names (no version specifiers).

    Skips comments, empty lines, and editable installs. Used to tell the coding
    agent which Python packages are available in the execution environment.
    """
    path = Path(requirements_path)
    if not path.exists():
        return []
    names = []
    for line in path.read_text().splitlines():
        line = line.strip().split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        # Skip editable: -e git+... or -e .
        if line.startswith("-e ") or " -e " in line:
            continue
        # First word before ==, >=, <=, <, >, or space
        for sep in ("==", ">=", "<=", ">", "<", " ", "[", "]"):
            if sep in line:
                line = line.split(sep)[0].strip()
                break
        if line and not line.startswith("-"):
            names.append(line)
    return names


# Default packages available in the execution environment when no requirements file is specified.
# Matches the "Bioinformatics (for code execution)" section of the project requirements.txt.
DEFAULT_EXECUTION_PACKAGES = [
    "numpy", "pandas", "scipy", "biopython", "pyBigWig", "pyfaidx", "pybedtools", "matplotlib",
]
