"""Base classes for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Message:
        return cls(role=Role.ASSISTANT, content=content)


class LLMResponse(BaseModel):
    """Response from an LLM."""

    content: str
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    raw_response: Any = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        """Initialize the provider.

        Args:
            model: Model name/ID to use
            api_key: API key for authentication
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.default_temperature = kwargs.get("temperature", 0.7)
        self.default_max_tokens = kwargs.get("max_tokens", 4096)

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion for the given messages.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (uses default if None)
            max_tokens: Maximum tokens to generate (uses default if None)
            **kwargs: Additional provider-specific options

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    async def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON completion for the given messages.

        Args:
            messages: List of messages in the conversation
            schema: Optional JSON schema for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Returns:
            Parsed JSON response as a dictionary
        """
        pass

    def _get_temperature(self, temperature: float | None) -> float:
        """Get temperature, using default if not specified."""
        return temperature if temperature is not None else self.default_temperature

    def _get_max_tokens(self, max_tokens: int | None) -> int:
        """Get max tokens, using default if not specified."""
        return max_tokens if max_tokens is not None else self.default_max_tokens


def create_provider(
    provider_name: str,
    model: str,
    api_key: str,
    **kwargs: Any,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_name: Name of the provider (openai, anthropic, gemini)
        model: Model name/ID
        api_key: API key
        **kwargs: Additional provider-specific options

    Returns:
        Configured LLMProvider instance
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        from src.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model, api_key, **kwargs)
    elif provider_name == "anthropic":
        from src.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model, api_key, **kwargs)
    elif provider_name == "gemini":
        from src.llm.gemini_provider import GeminiProvider
        return GeminiProvider(model, api_key, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
