"""OpenAI LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # seconds, will be multiplied by attempt number


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    # Errors that should trigger a retry
    RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, ConnectionError)

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        # Set long timeout for GPT-5.2 which uses reasoning tokens
        self.client = AsyncOpenAI(api_key=api_key, timeout=600.0)
        self.logger = get_logger("openai")

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
        # Only add max_completion_tokens if explicitly passed (not from defaults)
        if max_tokens is not None:
            request_params["max_completion_tokens"] = max_tokens
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

            # Record actual usage for cost tracking
            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

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
        if max_tokens is not None:
            request_params["max_completion_tokens"] = max_tokens
        request_params.update(kwargs)

        async def _make_request():
            return await self.client.chat.completions.create(**request_params)

        try:
            response = await self._retry_with_backoff(_make_request, "JSON completion")

            # Record actual usage for cost tracking
            if response.usage:
                self._record_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

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
