"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any

from src.llm.base import LLMProvider, LLMResponse, Message, Role
from src.utils.logging import get_logger


class GeminiProvider(LLMProvider):
    """Google Gemini API provider.

    Note: This is a stub implementation. Full implementation requires
    the google-generativeai package and API access.
    """

    def __init__(self, model: str, api_key: str, **kwargs: Any):
        super().__init__(model, api_key, **kwargs)
        self.logger = get_logger("gemini")
        self._client = None

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
            generation_config = genai.GenerationConfig(
                temperature=self._get_temperature(temperature),
                max_output_tokens=self._get_max_tokens(max_tokens),
            )

            # Start chat with history (excluding the last user message)
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])

            # Send the last message
            last_message = history[-1]["parts"][0] if history else ""
            response = await chat.send_message_async(
                last_message,
                generation_config=generation_config,
            )

            content = response.text or ""

            # Gemini doesn't provide detailed token usage in the same way
            usage = {
                "prompt_tokens": 0,  # Not provided by Gemini
                "completion_tokens": 0,
                "total_tokens": 0,
            }

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

    async def complete_json(
        self,
        messages: list[Message],
        schema: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a JSON completion using Gemini API."""
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
