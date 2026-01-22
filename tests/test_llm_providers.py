"""Tests for LLM providers."""

import pytest

from src.llm.base import Message, Role, LLMResponse, create_provider


class TestMessage:
    """Tests for Message class."""

    def test_create_system_message(self):
        """Test creating system message."""
        msg = Message.system("You are helpful.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "You are helpful."

    def test_create_user_message(self):
        """Test creating user message."""
        msg = Message.user("Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_create_assistant_message(self):
        """Test creating assistant message."""
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.ASSISTANT
        assert msg.content == "Hi there!"


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_create_response(self):
        """Test creating response."""
        response = LLMResponse(
            content="Hello!",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

        assert response.content == "Hello!"
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 0  # Not automatically calculated


class TestCreateProvider:
    """Tests for create_provider factory."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        provider = create_provider(
            provider_name="openai",
            model="gpt-4o",
            api_key="test-key",
        )

        assert provider.model == "gpt-4o"
        assert provider.api_key == "test-key"

    def test_create_anthropic_provider(self):
        """Test creating Anthropic provider."""
        provider = create_provider(
            provider_name="anthropic",
            model="claude-3-opus",
            api_key="test-key",
        )

        assert provider.model == "claude-3-opus"

    def test_create_gemini_provider(self):
        """Test creating Gemini provider."""
        provider = create_provider(
            provider_name="gemini",
            model="gemini-pro",
            api_key="test-key",
        )

        assert provider.model == "gemini-pro"

    def test_invalid_provider(self):
        """Test that invalid provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(
                provider_name="invalid",
                model="test",
                api_key="test",
            )

    def test_provider_case_insensitive(self):
        """Test that provider name is case insensitive."""
        provider = create_provider(
            provider_name="OpenAI",
            model="gpt-4o",
            api_key="test",
        )

        assert provider.model == "gpt-4o"
