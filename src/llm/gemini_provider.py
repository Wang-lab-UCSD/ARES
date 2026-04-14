"""Google Gemini LLM provider implementation (google-genai SDK)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


# Retry configuration
# Linear backoff: delay[i] = RETRY_DELAY_BASE * (i + 1)
# Total window: sum(5, 10, 15, ..., 50) = 275s (~4.5 min), tuned to ride out
# Gemini flex-tier queueing delays and transient 5xx outages.
MAX_RETRIES = 10
RETRY_DELAY_BASE = 5.0

# HTTP statuses we consider transient and worth retrying
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Uses the google-genai SDK (replaces the deprecated google-generativeai).
    Install with: pip install google-genai

    Supports per-call cost controls specific to reasoning models:
      - service_tier: 'flex' cuts cost ~50% at the price of lower priority.
      - thinking_level: 'MEDIUM' reduces internal reasoning tokens (which are
        billed at output rate) vs the default HIGH for Pro models.
    Both can be overridden via config kwargs or per-call kwargs.
    """

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.logger = get_logger("gemini")
        self._client = None

        # Cost-reduction defaults for Gemini 3.x reasoning models. Override
        # by setting service_tier / thinking_level in the llm config block,
        # or per-call via complete()/complete_json() kwargs. A None value
        # from config falls back to these defaults (not to None).
        self.service_tier = kwargs.get("service_tier") or "flex"
        self.thinking_level = kwargs.get("thinking_level") or "MEDIUM"

    def _get_client(self):
        """Lazy initialization of the genai Client."""
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert Message objects to (system_instruction, contents) for genai.

        Returns:
            system_instruction: single string or None
            contents: list of {"role": ..., "parts": [{"text": ...}]} dicts
        """
        system_instruction: str | None = None
        contents: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Concatenate multiple system messages if present
                if system_instruction is None:
                    system_instruction = msg.content
                else:
                    system_instruction += "\n\n" + msg.content
            elif msg.role == Role.USER:
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == Role.ASSISTANT:
                contents.append({"role": "model", "parts": [{"text": msg.content}]})

        return system_instruction, contents

    def _is_retryable(self, exc: BaseException) -> bool:
        """Return True if the exception looks like a transient API error."""
        if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True
        try:
            from google.genai.errors import APIError, ServerError
        except ImportError:
            return False
        if isinstance(exc, ServerError):
            return True
        if isinstance(exc, APIError):
            status = getattr(exc, "code", None) or getattr(exc, "status", None)
            if isinstance(status, int) and status in RETRYABLE_STATUS_CODES:
                return True
        return False

    async def _retry_with_backoff(self, operation, operation_name: str):
        """Execute an async operation with exponential backoff on transient errors."""
        last_error: BaseException | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return await operation()
            except BaseException as e:  # noqa: BLE001 — re-raised if not retryable
                if not self._is_retryable(e):
                    raise
                last_error = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY_BASE * (attempt + 1)
                    self.logger.warning(
                        f"{operation_name} failed, retrying in {delay}s",
                        {"error": str(e), "attempt": attempt + 1, "max_retries": MAX_RETRIES},
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"{operation_name} failed after {MAX_RETRIES + 1} attempts",
                        {"error": str(e)},
                    )
        assert last_error is not None
        raise last_error

    def _build_config(
        self,
        *,
        system_instruction: str | None,
        temperature: float | None,
        max_tokens: int | None,
        response_mime_type: str | None,
        service_tier_override: str | None,
        thinking_level_override: str | None,
    ):
        """Assemble a GenerateContentConfig with cost-reduction options applied."""
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": self._get_temperature(temperature),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Only set max_output_tokens when explicitly configured. When None,
        # let the API use its own default (avoids capping reasoning-heavy
        # models at the 4096 fallback).
        effective_max = max_tokens if max_tokens is not None else self.default_max_tokens
        if effective_max is not None:
            config_kwargs["max_output_tokens"] = effective_max

        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        # Thinking level: downgrade from default HIGH to MEDIUM on Pro models
        # to reduce thinking-token cost (thinking tokens bill at output rate).
        thinking_level = thinking_level_override or self.thinking_level
        if thinking_level:
            level_enum = getattr(types.ThinkingLevel, thinking_level.upper(), None)
            if level_enum is not None:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level_enum)
            else:
                self.logger.warning(
                    "Unknown thinking_level, ignoring",
                    {"thinking_level": thinking_level},
                )

        # Service tier: 'flex' cuts cost ~50% at the price of lower priority.
        service_tier = service_tier_override or self.service_tier
        if service_tier:
            try:
                config_kwargs["service_tier"] = types.ServiceTier(service_tier.lower())
            except ValueError:
                self.logger.warning(
                    "Unknown service_tier, ignoring",
                    {"service_tier": service_tier},
                )

        return types.GenerateContentConfig(**config_kwargs)

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using the Gemini API."""
        self.logger.debug("Sending completion request", {
            "model": self.model,
            "message_count": len(messages),
        })

        self._check_cost(messages)

        try:
            client = self._get_client()
            system_instruction, contents = self._convert_messages(messages)

            response_mime_type = kwargs.pop("response_mime_type", None)
            service_tier_override = kwargs.pop("service_tier", None)
            thinking_level_override = kwargs.pop("thinking_level", None)

            config = self._build_config(
                system_instruction=system_instruction,
                temperature=temperature,
                max_tokens=max_tokens,
                response_mime_type=response_mime_type,
                service_tier_override=service_tier_override,
                thinking_level_override=thinking_level_override,
            )

            async def _make_request():
                return await client.aio.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

            response = await self._retry_with_backoff(_make_request, "Completion")

            content = response.text or ""

            # IMPORTANT: Gemini reasoning models (3.1 Pro, etc.) bill thinking
            # tokens at the output-token rate, but the API reports them in
            # `thoughts_token_count` SEPARATELY from `candidates_token_count`.
            # Both must be summed for output-token cost, or the tracker will
            # underestimate by 5-10x for reasoning-heavy calls.
            prompt_tokens = 0
            completion_tokens = 0
            thoughts_tokens = 0
            cached_tokens = 0
            if response.usage_metadata is not None:
                prompt_tokens = response.usage_metadata.prompt_token_count or 0
                completion_tokens = response.usage_metadata.candidates_token_count or 0
                thoughts_tokens = response.usage_metadata.thoughts_token_count or 0
                cached_tokens = response.usage_metadata.cached_content_token_count or 0

            billable_output_tokens = completion_tokens + thoughts_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thoughts_tokens": thoughts_tokens,
                "total_tokens": prompt_tokens + billable_output_tokens,
            }

            self._record_usage(prompt_tokens, billable_output_tokens, cached_tokens)

            self.logger.debug("Received completion")

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                raw_response=response,
            )

        except Exception as e:
            self.logger.error("Completion failed", {"error": str(e)})
            raise

    # How many times to re-call the LLM when JSON parsing fails
    JSON_PARSE_RETRIES = 2

    async def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON completion using the Gemini API.

        Retries the full LLM call up to JSON_PARSE_RETRIES times when the
        response is not valid JSON (truncated output, extra prose, etc.).
        """
        self.logger.debug("Sending JSON completion request", {
            "model": self.model,
            "has_schema": schema is not None,
        })

        # Add JSON instruction to the system message
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

        last_parse_error: Exception | None = None
        content = ""

        for json_attempt in range(1 + self.JSON_PARSE_RETRIES):
            response = await self.complete(
                json_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_mime_type="application/json",
                **kwargs,
            )

            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            # Attempt 1: direct parse
            try:
                parsed = json.loads(content)
                # Gemini sometimes wraps the response object in a JSON array:
                # [{...}] instead of {...}. Unwrap single-element arrays so
                # callers always get a dict back.
                if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                return parsed
            except json.JSONDecodeError as e:
                last_parse_error = e

            # Attempt 2: extract {...} substring and clean trailing commas
            extracted = self._extract_json_object(content)
            if extracted is not None:
                try:
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass

            if json_attempt < self.JSON_PARSE_RETRIES:
                delay = RETRY_DELAY_BASE * (json_attempt + 1)
                self.logger.warning(
                    "JSON parse failed, retrying LLM call",
                    {
                        "error": str(last_parse_error),
                        "attempt": json_attempt + 1,
                        "max_retries": self.JSON_PARSE_RETRIES,
                        "content_preview": content[:300],
                    },
                )
                await asyncio.sleep(delay)

        self.logger.error("Failed to parse JSON response after retries", {
            "error": str(last_parse_error),
            "content": content[:500],
        })
        raise ValueError(f"Invalid JSON response after {1 + self.JSON_PARSE_RETRIES} attempts: {last_parse_error}")

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract a likely JSON object from a text blob.

        Gemini may prepend/append commentary even when instructed to return JSON only.
        This tries to salvage the first top-level {...} block.
        """
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1].strip()
        # Quick sanity check: avoid returning huge non-JSON with no quotes/colons
        if ":" not in candidate:
            return None
        # Remove common trailing commas before } or ]
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return candidate
