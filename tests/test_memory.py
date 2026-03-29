"""Tests for memory and conversation management."""

import pytest

from src.llm.base import Role
from src.memory.conversation import ConversationMemory, ConversationTurn, PipelineState


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_init_with_system_message(self):
        """Test initialization with system message."""
        memory = ConversationMemory(system_message="You are helpful.")
        assert memory.system_message == "You are helpful."

    def test_add_user_message(self):
        """Test adding user messages."""
        memory = ConversationMemory()
        memory.add_user_message("Hello")
        memory.add_user_message("How are you?")

        assert len(memory) == 2

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        memory = ConversationMemory()
        memory.add_assistant_message("Hello!")

        messages = memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == Role.ASSISTANT

    def test_get_messages_with_system(self):
        """Test getting messages includes system message."""
        memory = ConversationMemory(system_message="Be helpful.")
        memory.add_user_message("Hello")

        messages = memory.get_messages(include_system=True)
        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER

    def test_get_messages_without_system(self):
        """Test getting messages excludes system when requested."""
        memory = ConversationMemory(system_message="Be helpful.")
        memory.add_user_message("Hello")

        messages = memory.get_messages(include_system=False)
        assert len(messages) == 1
        assert messages[0].role == Role.USER

    def test_get_last_n_messages(self):
        """Test getting last N messages."""
        memory = ConversationMemory()
        for i in range(5):
            memory.add_user_message(f"Message {i}")

        messages = memory.get_last_n_messages(3, include_system=False)
        assert len(messages) == 3
        assert messages[0].content == "Message 2"

    def test_clear(self):
        """Test clearing conversation history."""
        memory = ConversationMemory(system_message="System")
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi")

        memory.clear()

        assert len(memory) == 0
        assert memory.system_message == "System"  # System message preserved

    def test_metadata_tracking(self):
        """Test that metadata is tracked on turns."""
        memory = ConversationMemory()
        memory.add_user_message("Hello", iteration=1, stage="initial")

        # Access internal turns to check metadata
        assert memory._turns[0].metadata["iteration"] == 1
        assert memory._turns[0].metadata["stage"] == "initial"


class TestPipelineState:
    """Tests for PipelineState class."""

    def test_init(self):
        """Test basic initialization."""
        state = PipelineState(
            finding="X predicts Y",
            context="Some context",
        )

        assert state.finding == "X predicts Y"
        assert state.current_iteration == 0
        assert not state.converged

    def test_add_hypothesis(self):
        """Test adding hypotheses."""
        state = PipelineState(finding="X predicts Y")
        state.add_hypothesis({"name": "H1", "rationale": "Because..."})
        state.add_hypothesis({"name": "H2", "rationale": "Maybe..."})

        assert len(state.hypotheses) == 2
        assert state.hypotheses[0]["id"] == 0
        assert state.hypotheses[1]["id"] == 1

    def test_add_evidence(self):
        """Test adding evidence."""
        state = PipelineState(finding="X predicts Y")
        state.current_iteration = 1
        state.current_hypothesis_index = 0

        state.add_evidence({
            "findings": ["Found correlation"],
            "support_level": "SUPPORTS",
        })

        assert len(state.evidence) == 1
        assert state.evidence[0]["iteration"] == 1
        assert state.evidence[0]["hypothesis_id"] == 0

    def test_mark_converged(self):
        """Test marking pipeline as converged."""
        state = PipelineState(finding="X predicts Y")
        state.mark_converged(
            reason="Hypothesis confirmed",
            confidence=0.9,
            conclusion="X causes Y through mechanism Z",
        )

        assert state.converged
        assert state.convergence_reason == "Hypothesis confirmed"
        assert state.confidence_level == 0.9
        assert state.conclusion == "X causes Y through mechanism Z"
        assert state.run_status == "converged"
        assert not state.synthesized

    def test_mark_stopped(self):
        """Test marking pipeline as stopped without convergence."""
        state = PipelineState(finding="X predicts Y")
        state.mark_stopped(
            reason="Max iterations reached",
            confidence=0.4,
            conclusion="Best-supported interpretation so far",
        )

        assert not state.converged
        assert state.stop_reason == "Max iterations reached"
        assert state.confidence_level == 0.4
        assert state.conclusion == "Best-supported interpretation so far"
        assert state.run_status == "stopped"
        assert not state.synthesized

    def test_history_summary_empty(self):
        """Test history summary when empty."""
        state = PipelineState(finding="X predicts Y")
        summary = state.get_history_summary()

        assert "No hypotheses have been tested" in summary

    def test_history_summary_with_data(self):
        """Test history summary with tested hypotheses."""
        state = PipelineState(finding="X predicts Y")
        state.tested_hypotheses.append({
            "iteration": 1,
            "name": "H1 name",
            "result": "SUPPORTS",
            "evidence_summary": "Found correlation",
        })

        summary = state.get_history_summary()

        assert "Iteration 1" in summary
        assert "H1 name" in summary

    def test_serialization(self):
        """Test state serialization and deserialization."""
        state = PipelineState(
            finding="X predicts Y",
            context="Context here",
            current_iteration=3,
        )
        state.add_hypothesis({"name": "H1"})

        # Serialize
        data = state.to_dict()
        assert data["finding"] == "X predicts Y"
        assert data["current_iteration"] == 3

        # Deserialize
        restored = PipelineState.from_dict(data)
        assert restored.finding == "X predicts Y"
        assert restored.current_iteration == 3
        assert len(restored.hypotheses) == 1
