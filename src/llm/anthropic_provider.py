"""Anthropic (Claude) LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    RETRYABLE_ERRORS = (ConnectionError,)

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.logger = get_logger("anthropic")
        self._client = None
        self._base_url: str | None = kwargs.get("base_url", None)
        self._setup_retryable_errors()

    def _setup_retryable_errors(self):
        try:
            from anthropic import APIConnectionError, APITimeoutError, RateLimitError
            AnthropicProvider.RETRYABLE_ERRORS = (
                APIConnectionError, APITimeoutError, RateLimitError, ConnectionError
            )
        except ImportError:
            pass

    async def _retry_with_backoff(self, operation, operation_name: str):
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
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                client_kwargs: dict = {"api_key": self.api_key}
                if self._base_url is not None:
                    client_kwargs["base_url"] = self._base_url
                self._client = AsyncAnthropic(**client_kwargs)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
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

    def _extract_text_content(self, content_blocks: list[Any]) -> str:
        """Extract plain text from mixed Anthropic content blocks.

        Skips ThinkingBlock and other non-text blocks safely.
        """
        if not content_blocks:
            return ""

        text_parts = []

        for block in content_blocks:
            # SDK object style
            block_type = getattr(block, "type", None)
            if block_type == "text":
                txt = getattr(block, "text", None)
                if txt:
                    text_parts.append(txt)
                continue

            # Dict style fallback
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
                continue

            # Generic fallback for odd compatible providers
            txt = getattr(block, "text", None)
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt)

        return "".join(text_parts).strip()

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()

        self.logger.debug("Sending completion request", {
            "model": self.model,
            "message_count": len(messages),
        })

        system_msg, conv_messages = self._convert_messages(messages)

        self._check_cost(messages)

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

            content = self._extract_text_content(response.content)
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            self._record_usage(usage["prompt_tokens"], usage["completion_tokens"])

            self.logger.debug("Received completion", {
                "usage": usage,
                "content_blocks": [getattr(b, "type", type(b).__name__) for b in (response.content or [])],
            })

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
        self.logger.debug("Sending JSON completion request", {
            "model": self.model,
            "has_schema": schema is not None,
        })

        json_messages = list(messages)
        schema_instruction = ""
        if schema:
            schema_instruction = (
                f"\n\nRespond with valid JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )

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

        content = response.content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) >= 3:
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