"""Interactive model selection for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelOption:
    """Represents an available model option."""
    provider: str
    model_id: str
    display_name: str
    api_key_env: str
    description: str
    recommended_for: list[str]  # e.g., ["hypothesis", "coding", "summary"]


# Available models as of January 2026
AVAILABLE_MODELS: list[ModelOption] = [
    # OpenAI models
    ModelOption(
        provider="openai",
        model_id="gpt-5-2",
        display_name="GPT-5.2",
        api_key_env="OPENAI_API_KEY",
        description="Latest flagship model for professional work",
        recommended_for=["hypothesis", "summary"],
    ),
    ModelOption(
        provider="openai",
        model_id="gpt-5-2-codex",
        display_name="GPT-5.2 Codex",
        api_key_env="OPENAI_API_KEY",
        description="Optimized for agentic coding tasks",
        recommended_for=["coding"],
    ),
    # Anthropic models
    ModelOption(
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        display_name="Claude Opus 4.5",
        api_key_env="ANTHROPIC_API_KEY",
        description="Most capable Claude model for complex tasks",
        recommended_for=["hypothesis", "coding"],
    ),
    ModelOption(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20251101",
        display_name="Claude Sonnet 4.5",
        api_key_env="ANTHROPIC_API_KEY",
        description="Balanced performance and speed",
        recommended_for=["summary"],
    ),
    # Google models
    ModelOption(
        provider="gemini",
        model_id="gemini-3-pro-preview",
        display_name="Gemini 3 Pro",
        api_key_env="GEMINI_API_KEY",
        description="State-of-the-art reasoning and multimodal",
        recommended_for=["hypothesis"],
    ),
    ModelOption(
        provider="gemini",
        model_id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash",
        api_key_env="GEMINI_API_KEY",
        description="Fast and cost-effective",
        recommended_for=["summary"],
    ),
]


def get_models_for_agent(agent_type: str) -> list[ModelOption]:
    """Get available models, with recommended ones first.

    Args:
        agent_type: One of "hypothesis", "coding", "summary"

    Returns:
        List of ModelOption, recommended ones first
    """
    recommended = []
    others = []

    for model in AVAILABLE_MODELS:
        if agent_type in model.recommended_for:
            recommended.append(model)
        else:
            others.append(model)

    return recommended + others


def display_model_options(agent_type: str, agent_description: str) -> None:
    """Display model options for an agent type.

    Args:
        agent_type: One of "hypothesis", "coding", "summary"
        agent_description: Human-readable description of what the agent does
    """
    models = get_models_for_agent(agent_type)

    print(f"\nSelect model for {agent_type.upper()} AGENT ({agent_description}):")

    for i, model in enumerate(models, 1):
        recommended = " (Recommended)" if agent_type in model.recommended_for else ""
        print(f"  [{i}] {model.provider.capitalize()}: {model.display_name}{recommended}")
        print(f"      {model.description}")


def prompt_model_selection(agent_type: str, agent_description: str) -> ModelOption:
    """Prompt user to select a model for an agent.

    Args:
        agent_type: One of "hypothesis", "coding", "summary"
        agent_description: Human-readable description

    Returns:
        Selected ModelOption
    """
    models = get_models_for_agent(agent_type)
    display_model_options(agent_type, agent_description)

    while True:
        try:
            choice = input("> ").strip()
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
            print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            raise SystemExit(1)


def interactive_model_selection() -> dict[str, ModelOption]:
    """Run interactive model selection for all agents.

    Returns:
        Dictionary mapping agent type to selected ModelOption
    """
    print("\n" + "=" * 50)
    print("MODEL SELECTION")
    print("=" * 50)

    selections = {}

    agent_configs = [
        ("hypothesis", "reasoning, hypothesis generation"),
        ("coding", "code generation, debugging"),
        ("summary", "result interpretation"),
    ]

    for agent_type, description in agent_configs:
        selections[agent_type] = prompt_model_selection(agent_type, description)

    # Display summary
    print("\n" + "-" * 50)
    print("Starting pipeline with:")
    for agent_type, model in selections.items():
        print(f"  - {agent_type.capitalize()}: {model.display_name} ({model.provider})")
    print("-" * 50 + "\n")

    return selections


def selections_to_config(selections: dict[str, ModelOption]) -> dict:
    """Convert model selections to config dictionary format.

    Args:
        selections: Dictionary from interactive_model_selection()

    Returns:
        Config dictionary compatible with Config model
    """
    return {
        "llm": {
            "hypothesis_model": {
                "provider": selections["hypothesis"].provider,
                "model": selections["hypothesis"].model_id,
                "api_key_env": selections["hypothesis"].api_key_env,
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "coding_model": {
                "provider": selections["coding"].provider,
                "model": selections["coding"].model_id,
                "api_key_env": selections["coding"].api_key_env,
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            "summary_model": {
                "provider": selections["summary"].provider,
                "model": selections["summary"].model_id,
                "api_key_env": selections["summary"].api_key_env,
                "temperature": 0.3,
                "max_tokens": 2048,
            },
        },
        "execution": {
            "type": "jupyter",
            "timeout_seconds": 300,
            "max_retries": 3,
        },
        "pipeline": {
            "max_iterations": 10,
            "log_dir": "logs",
            "output_dir": "outputs",
            "save_intermediate": True,
        },
    }
