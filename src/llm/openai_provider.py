"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)
        self.logger = get_logger("openai")

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

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(messages),
                temperature=self._get_temperature(temperature),
                max_tokens=self._get_max_tokens(max_tokens),
                **kwargs,
            )

            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }

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

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self._convert_messages(json_messages),
                temperature=self._get_temperature(temperature),
                max_tokens=self._get_max_tokens(max_tokens),
                response_format={"type": "json_object"},
                **kwargs,
            )

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
