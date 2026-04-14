"""OpenAI LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


# Retry configuration
# Linear backoff: delay[i] = RETRY_DELAY_BASE * (i + 1)
# Total window: sum(5, 10, 15, ..., 50) = 275s (~4.5 min), tuned to ride out
# MiniMax/Z.AI multi-minute 5xx windows seen in the Apr 13 runs (529 overloaded_error).
MAX_RETRIES = 10
RETRY_DELAY_BASE = 5.0


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Errors that should trigger a retry.
    #
    # InternalServerError is the base class for all 5xx responses in the openai
    # SDK (500, 502, 503, 504). We added it after the 2026-04-09 ETS2/YY1 run
    # died at iter 7 with HTTP 500 / Cloudflare 520 (Z.AI origin outage) —
    # without this, a single transient 5xx kills the whole pipeline. A second
    # concurrent job died for the same reason 8 minutes later, proving it was
    # a provider-side issue rather than anything in our code.
    RETRYABLE_ERRORS = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        ConnectionError,
    )

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        client_kwargs: dict = {"api_key": api_key, "timeout": 600.0}
        base_url = kwargs.get("base_url")
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.logger = get_logger("openai")
        # Third-party OpenAI-compatible APIs (DeepSeek, GLM-5, etc.) use the legacy
        # "max_tokens" parameter name; native OpenAI uses "max_completion_tokens".
        self._max_tokens_param = "max_tokens" if base_url is not None else "max_completion_tokens"
        self.default_reasoning_effort: str | None = kwargs.get("reasoning_effort")

    async def _retry_with_backoff(self, operation, operation_name: str):
        """Execute an operation with exponential backoff retry on transient errors.

        Args:
            operation: Async callable to execute
            operation_name: Name for logging purposes

        Returns:
            Result of the operation

        Raises:
            The last exception if all retries fail
        """
        last_error = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await operation()
            except self.RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE * (attempt + 1)
                    self.logger.warning(
                        f"{operation_name} failed, retrying in {delay}s",
                        {"error": str(e), "attempt": attempt + 1, "max_retries": MAX_RETRIES}
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"{operation_name} failed after {MAX_RETRIES + 1} attempts",
                        {"error": str(e)}
                    )
        raise last_error

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to OpenAI format."""
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using OpenAI API."""
        self.logger.debug("Sending completion request", {
            "model": self.model,
            "message_count": len(messages),
        })

        # Check cost before making API call
        self._check_cost(messages)

        # Build request params - DO NOT set max_completion_tokens for OpenAI models
        # as they use internal reasoning tokens that count against this budget
        request_params = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }
        # Only add temperature if explicitly set (gpt-5-mini only supports default)
        if temperature is not None:
            request_params["temperature"] = temperature
        # Set max_completion_tokens only when an explicit limit is configured.
        # GPT-5.2+ must NOT have this set (reasoning tokens eat the budget).
        # Other models (DeepSeek, GLM-5, etc.) need it set to avoid API defaults that are too low.
        effective_max = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max is not None:
            request_params[self._max_tokens_param] = effective_max
        if self.default_reasoning_effort is not None:
            request_params["reasoning_effort"] = self.default_reasoning_effort
        request_params.update(kwargs)

        async def _make_request():
            return await self.client.chat.completions.create(**request_params)

        try:
            response = await self._retry_with_backoff(_make_request, "Completion")

            # Debug: log full response structure for GPT-5.2 compatibility
            message = response.choices[0].message
            self.logger.debug("Response message structure", {
                "content_type": type(message.content).__name__ if message.content else "None",
                "content_length": len(message.content) if message.content else 0,
                "has_tool_calls": bool(getattr(message, "tool_calls", None)),
                "finish_reason": response.choices[0].finish_reason,
                "model": response.model,
            })

            content = message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

            # Extract cached token count from response
            cached_tokens = 0
            if response.usage and hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0

            # Record actual usage for cost tracking
            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"], cached_tokens)

            self.logger.debug("Received completion", {"usage": usage, "content_preview": content[:200] if content else "(empty)"})

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            self.logger.error("Completion failed", {"error": str(e)})
            raise

    async def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON completion using OpenAI API."""
        self.logger.debug("Sending JSON completion request", {
            "model": self.model,
            "has_schema": schema is not None,
        })

        # Add JSON instruction to system message if not present
        json_messages = list(messages)
        if schema:
            schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            if json_messages and json_messages[0].role == Role.SYSTEM:
                json_messages[0] = Message.system(json_messages[0].content + schema_instruction)
            else:
                json_messages.insert(0, Message.system(f"You must respond with valid JSON.{schema_instruction}"))

        # Check cost before making API call
        self._check_cost(json_messages)

        # Build request params - DO NOT set max_completion_tokens for OpenAI models
        request_params = {
            "model": self.model,
            "messages": self._convert_messages(json_messages),
            "response_format": {"type": "json_object"},
        }
        # Only add temperature if explicitly set (gpt-5-mini only supports default)
        if temperature is not None:
            request_params["temperature"] = temperature
        effective_max = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max is not None:
            request_params[self._max_tokens_param] = effective_max
        if self.default_reasoning_effort is not None:
            request_params["reasoning_effort"] = self.default_reasoning_effort
        request_params.update(kwargs)

        async def _make_request():
            return await self.client.chat.completions.create(**request_params)

        try:
            response = await self._retry_with_backoff(_make_request, "JSON completion")

            # Record actual usage for cost tracking
            if response.usage:
                cached_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                self._record_usage(response.usage.prompt_tokens, response.usage.completion_tokens, cached_tokens)

            content = response.choices[0].message.content or "{}"

            # Parse JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error("Failed to parse JSON response", {
                    "error": str(e),
                    "content": content[:500],
                })
                raise ValueError(f"Invalid JSON response: {e}")

            self.logger.debug("Received JSON completion", {
                "keys": list(result.keys()) if isinstance(result, dict) else "not a dict",
            })

            return result

        except Exception as e:
            self.logger.error("JSON completion failed", {"error": str(e)})
            raise
