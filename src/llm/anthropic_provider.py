"""Anthropic (Claude) LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # seconds, will be multiplied by attempt number


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider.

    Requires the anthropic package: pip install anthropic
    """

    # Errors that should trigger a retry (will be populated when anthropic is imported)
    RETRYABLE_ERRORS = (ConnectionError,)

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.logger = get_logger("anthropic")
        self._client = None
        self._setup_retryable_errors()

    def _setup_retryable_errors(self):
        """Set up retryable error types after anthropic is imported."""
        try:
            from anthropic import APIConnectionError, APITimeoutError, RateLimitError
            AnthropicProvider.RETRYABLE_ERRORS = (
                APIConnectionError, APITimeoutError, RateLimitError, ConnectionError
            )
        except ImportError:
            pass  # Will use default ConnectionError only

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

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert Message objects to Anthropic format.

        Returns:
            Tuple of (system_message, conversation_messages)
        """
        system_msg = None
        conv_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_msg = msg.content
            else:
                conv_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return system_msg, conv_messages

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Anthropic API."""
        client = self._get_client()

        self.logger.debug("Sending completion request", {
            "model": self.model,
            "message_count": len(messages),
        })

        system_msg, conv_messages = self._convert_messages(messages)

        # Check cost before making API call
        self._check_cost(messages)

        # Build request params
        max_tokens_value = self._get_max_tokens(max_tokens)
        temp_value = self._get_temperature(temperature)

        async def _make_request():
            return await client.messages.create(
                model=self.model,
                max_tokens=max_tokens_value,
                system=system_msg or "",
                messages=conv_messages,
                temperature=temp_value,
                **kwargs,
            )

        try:
            response = await self._retry_with_backoff(_make_request, "Completion")

            content = response.content[0].text if response.content else ""
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            # Record actual usage for cost tracking
            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

            self.logger.debug("Received completion", {"usage": usage})

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
        """Generate a JSON completion using Anthropic API."""
        self.logger.debug("Sending JSON completion request", {
            "model": self.model,
            "has_schema": schema is not None,
        })

        # Add JSON instruction
        json_messages = list(messages)
        schema_instruction = ""
        if schema:
            schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        if json_messages and json_messages[0].role == Role.SYSTEM:
            json_messages[0] = Message.system(
                json_messages[0].content +
                "\n\nYou must respond with valid JSON only, no other text." +
                schema_instruction
            )
        else:
            json_messages.insert(0, Message.system(
                "You must respond with valid JSON only, no other text." + schema_instruction
            ))

        response = await self.complete(
            json_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Parse JSON from response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            content = "\n".join(lines[1:-1])

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse JSON response", {
                "error": str(e),
                "content": content[:500],
            })
            raise ValueError(f"Invalid JSON response: {e}")

        return result
