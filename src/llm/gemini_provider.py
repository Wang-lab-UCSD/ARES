"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2.0  # seconds, will be multiplied by attempt number


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Requires the google-generativeai package: pip install google-generativeai
    """

    # Errors that should trigger a retry
    RETRYABLE_ERRORS = (ConnectionError, TimeoutError)

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.logger = get_logger("gemini")
        self._client = None
        self._setup_retryable_errors()

    def _setup_retryable_errors(self):
        """Set up retryable error types after google-generativeai is imported."""
        try:
            from google.api_core.exceptions import (
                ServiceUnavailable, DeadlineExceeded, ResourceExhausted
            )
            GeminiProvider.RETRYABLE_ERRORS = (
                ServiceUnavailable, DeadlineExceeded, ResourceExhausted,
                ConnectionError, TimeoutError
            )
        except ImportError:
            pass  # Will use default errors only

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
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Convert Message objects to Gemini format.

        Returns:
            Tuple of (system_instruction, conversation_history)
        """
        system_instruction = None
        history = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            elif msg.role == Role.USER:
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == Role.ASSISTANT:
                history.append({"role": "model", "parts": [msg.content]})

        return system_instruction, history

    async def complete(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Gemini API."""
        self.logger.debug("Sending completion request", {
            "model": self.model,
            "message_count": len(messages),
        })

        # Check cost before making API call
        self._check_cost(messages)

        try:
            import google.generativeai as genai

            system_instruction, history = self._convert_messages(messages)

            # Create model with system instruction if present
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
            else:
                model = self._get_client()

            # Configure generation
            # Some google-generativeai versions support response_mime_type="application/json"
            # which improves structured output reliability. We pass it through when provided.
            response_mime_type = kwargs.pop("response_mime_type", None)
            try:
                if response_mime_type:
                    generation_config = genai.GenerationConfig(
                        temperature=self._get_temperature(temperature),
                        max_output_tokens=self._get_max_tokens(max_tokens),
                        response_mime_type=response_mime_type,
                    )
                else:
                    generation_config = genai.GenerationConfig(
                        temperature=self._get_temperature(temperature),
                        max_output_tokens=self._get_max_tokens(max_tokens),
                    )
            except TypeError:
                # Older library version: ignore response_mime_type
                generation_config = genai.GenerationConfig(
                    temperature=self._get_temperature(temperature),
                    max_output_tokens=self._get_max_tokens(max_tokens),
                )

            # Start chat with history (excluding the last user message)
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])

            # Send the last message with retry
            last_message = history[-1]["parts"][0] if history else ""

            async def _make_request():
                return await chat.send_message_async(
                    last_message,
                    generation_config=generation_config,
                )

            response = await self._retry_with_backoff(_make_request, "Completion")

            content = response.text or ""

            # Gemini doesn't provide detailed token usage in the same way
            # Try to get usage metadata if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            # Record actual usage for cost tracking (may be 0 if not provided)
            self._record_usage(prompt_tokens, completion_tokens)

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
        """Generate a JSON completion using Gemini API.

        Retries the full LLM call up to JSON_PARSE_RETRIES times when the
        response is not valid JSON (truncated output, extra prose, etc.).
        """
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

        last_parse_error: Exception | None = None

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
                return json.loads(content)
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
