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
    subscription: bool = Field(
        default=False,
        description="If True, API cost is $0 (e.g. MiniMax subscription plan). Tokens are still tracked but not billed.",
    )
    service_tier: str | None = Field(
        default=None,
        description=(
            "Gemini and native OpenAI: service tier ('flex', 'standard', 'priority'). 'flex' runs on spare "
            "capacity at roughly half price, in exchange for queueing and more 429s (the retry loop absorbs "
            "those). Gemini defaults to 'flex'; OpenAI sends nothing unless you set this, because the same "
            "provider class also talks to DeepSeek, GLM-5 and MiniMax, which reject the unknown parameter. "
            "Tracker prices are standard-tier for the GPT rows, so halve the relevant cost_tracker.py entry "
            "when you run a model on flex."
        ),
    )
    thinking_level: str | None = Field(
        default=None,
        description="Gemini only: reasoning effort ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH'). Lower levels cut thinking tokens (billed as output). Provider default is 'MEDIUM'.",
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
        description="Optional dedicated model for hypothesis review. Falls back to hypothesis_model if not set.",
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


def flatten_manifest_data(
    data: dict[str, Any],
    manifest_dir: Path | None = None,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Flatten a manifest ``data`` section into the ``data_files`` dict the kernel receives.

    The manifest's ``data`` field is typed ``dict[str, Any]`` so that a manifest can group
    files however its author likes, but only two shapes actually reach generated code:

        data:
          chipseq:                      # category -> {name: path}
            TF_A_peaks: "/path/a.bed"   #   visible as data_files["TF_A_peaks"]
                                        #             and data_files["chipseq.TF_A_peaks"]
          genome_fasta: "/path/hg38.fa" # category -> path, visible as data_files["genome_fasta"]

    Anything nested a third level deep is silently invisible to the pipeline, which is why this
    function reports it rather than dropping it quietly: a manifest that validates cleanly can
    still hide half its files from every hypothesis.

    Args:
        data: The manifest's ``data`` section.
        manifest_dir: Directory the manifest was loaded from. Relative paths are resolved
            against it, matching what the kernel gets; ``None`` leaves values untouched.

    Not every non-string is a mistake. A manifest legitimately carries lists and numbers next to
    its paths -- ``chromhmm_promoter_state`` is a list of state names, ``string.min_score`` an int
    -- and those reach the agents through the manifest itself, which is rendered into the prompts.
    They are reported as ``"info"`` rather than ``"lost"`` so a dry run can tell the two apart:
    a path buried a level too deep is invisible everywhere and needs fixing; a list of ChromHMM
    states is doing exactly what it should.

    Returns:
        ``(flat, dropped)``. ``flat`` maps every key the kernel will see to its resolved path.
        ``dropped`` holds ``(dotted_path, severity, reason)``:

        - ``"lost"`` — no part of the pipeline can reach the entry.
        - ``"shadowed"`` — the entry is reachable as ``category.name`` but a later category
          claimed the same bare name, so ``data_files[name]`` points somewhere else.
        - ``"info"`` — a non-path value, which travels in the prompts rather than ``data_files``.
    """
    def resolved(value: str) -> str:
        if manifest_dir is None or os.path.isabs(value):
            return value
        candidate = manifest_dir / value
        return str(candidate.resolve()) if candidate.exists() else value

    flat: dict[str, str] = {}
    dropped: list[tuple[str, str, str]] = []
    # Which category last claimed each bare key, so a second claimant can be reported.
    bare_owner: dict[str, str] = {}

    for category, items in (data or {}).items():
        if isinstance(items, dict):
            for name, value in items.items():
                if isinstance(value, str):
                    p = resolved(value)
                    if name in bare_owner:
                        # Two categories use the same leaf name -- `data_root` under both
                        # all_tf_chipseq and additional_chipseq, say. The qualified keys stay
                        # distinct, but the bare alias can only hold one, and the last one
                        # written wins. Code that reaches for data_files["data_root"] then gets
                        # whichever category happened to come later in the YAML, with nothing to
                        # signal that another tree was meant.
                        dropped.append((
                            f"{bare_owner[name]}.{name}",
                            "shadowed",
                            f'bare key "{name}" is also defined by {category}; '
                            f'data_files["{name}"] resolves to that one, so reach this entry '
                            f'as data_files["{bare_owner[name]}.{name}"]',
                        ))
                    bare_owner[name] = category
                    flat[name] = p
                    flat[f"{category}.{name}"] = p
                elif isinstance(value, dict):
                    dropped.append((
                        f"{category}.{name}",
                        "lost",
                        "nested one level too deep — data_files reads only data.<category>.<name>, "
                        f"so the {len(value)} entr(ies) under it reach nothing",
                    ))
                else:
                    dropped.append((
                        f"{category}.{name}",
                        "info",
                        f"{type(value).__name__} — not a path, so not in data_files; still "
                        "readable from the manifest in the prompts",
                    ))
        elif isinstance(items, str):
            flat[category] = resolved(items)
        elif isinstance(items, dict):  # unreachable, kept for symmetry with the branch above
            continue
        else:
            dropped.append((
                category,
                "info",
                f"{type(items).__name__} at the top level — not a path, so not in data_files",
            ))

    return flat, dropped


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
