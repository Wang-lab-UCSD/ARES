"""Conversation history management for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from src.llm.base import Message, Role


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        """Convert to an LLM Message."""
        return Message(role=self.role, content=self.content)


class ConversationMemory:
    """Manages conversation history for an agent.

    Features:
    - Maintains full conversation history
    - Supports system message
    - Can export to LLM message format
    - Tracks metadata per turn (iteration, hypothesis, etc.)
    """

    def __init__(self, system_message: str | None = None):
        """Initialize conversation memory.

        Args:
            system_message: Optional system message to prepend to all conversations
        """
        self._system_message = system_message
        self._turns: list[ConversationTurn] = []

    @property
    def system_message(self) -> str | None:
        """Get the system message."""
        return self._system_message

    @system_message.setter
    def system_message(self, value: str) -> None:
        """Set the system message."""
        self._system_message = value

    def add_user_message(self, content: str, **metadata: Any) -> None:
        """Add a user message to the conversation."""
        self._turns.append(ConversationTurn(
            role=Role.USER,
            content=content,
            metadata=metadata,
        ))

    def add_assistant_message(self, content: str, **metadata: Any) -> None:
        """Add an assistant message to the conversation."""
        self._turns.append(ConversationTurn(
            role=Role.ASSISTANT,
            content=content,
            metadata=metadata,
        ))

    def get_messages(self, include_system: bool = True) -> list[Message]:
        """Get all messages in LLM format.

        Args:
            include_system: Whether to include the system message

        Returns:
            List of Message objects
        """
        messages = []

        if include_system and self._system_message:
            messages.append(Message.system(self._system_message))

        for turn in self._turns:
            messages.append(turn.to_message())

        return messages

    def get_last_n_messages(self, n: int, include_system: bool = True) -> list[Message]:
        """Get the last N messages.

        Args:
            n: Number of recent messages to include
            include_system: Whether to include the system message

        Returns:
            List of Message objects
        """
        messages = []

        if include_system and self._system_message:
            messages.append(Message.system(self._system_message))

        for turn in self._turns[-n:]:
            messages.append(turn.to_message())

        return messages

    def get_summary(self) -> str:
        """Get a brief summary of the conversation history."""
        if not self._turns:
            return "No conversation history."

        user_count = sum(1 for t in self._turns if t.role == Role.USER)
        assistant_count = sum(1 for t in self._turns if t.role == Role.ASSISTANT)

        return f"Conversation with {user_count} user messages and {assistant_count} assistant messages."

    def clear(self) -> None:
        """Clear all conversation history (keeps system message)."""
        self._turns = []

    def __len__(self) -> int:
        """Get the number of turns."""
        return len(self._turns)


class PipelineState(BaseModel):
    """Persistent state for the pipeline across iterations."""

    finding: str
    context: str = ""
    current_iteration: int = 0
    max_iterations: int = 10

    # Hypothesis tracking
    hypotheses: list[dict[str, Any]] = []
    current_hypothesis_index: int = 0
    tested_hypotheses: list[dict[str, Any]] = []

    # File familiarization output (from iteration 0, before main loop)
    file_summaries: dict[str, str] = {}

    # Evidence accumulation
    evidence: list[dict[str, Any]] = []

    # Convergence state
    converged: bool = False
    convergence_reason: str | None = None
    confidence_level: float = 0.0
    # Completion status (distinguish true convergence vs synthesized stop)
    run_status: str = "running"  # running | converged | synthesized_stop | stopped
    stop_reason: str | None = None
    synthesized: bool = False

    # Named mechanism at convergence (item 17)
    mechanism_category_number: int | None = None
    mechanism_category_name: str | None = None

    # Final output
    conclusion: str | None = None

    def add_hypothesis(self, hypothesis: dict[str, Any]) -> None:
        """Add a new hypothesis, skipping duplicates by name."""
        name = hypothesis.get("name", "")
        if name:
            existing_names = {h.get("name", "") for h in self.hypotheses}
            if name in existing_names:
                return
        hypothesis["id"] = len(self.hypotheses)
        hypothesis["iteration"] = self.current_iteration
        self.hypotheses.append(hypothesis)

    def add_evidence(self, evidence: dict[str, Any]) -> None:
        """Add new evidence from an experiment."""
        evidence["iteration"] = self.current_iteration
        evidence["hypothesis_id"] = self.current_hypothesis_index
        self.evidence.append(evidence)

    def mark_converged(
        self,
        reason: str,
        confidence: float,
        conclusion: str,
        mechanism_category_number: int | None = None,
        mechanism_category_name: str | None = None,
    ) -> None:
        """Mark the pipeline as converged."""
        self.converged = True
        self.convergence_reason = reason
        self.confidence_level = confidence
        self.conclusion = conclusion
        self.mechanism_category_number = mechanism_category_number
        self.mechanism_category_name = mechanism_category_name
        self.run_status = "converged"
        self.stop_reason = None
        self.synthesized = False

    def mark_synthesized_stop(
        self,
        reason: str,
        confidence: float,
        conclusion: str,
        mechanism_category_number: int | None = None,
        mechanism_category_name: str | None = None,
    ) -> None:
        """Stop early and synthesize from supported evidence.

        This is NOT true convergence: the convergence criteria were not fully satisfied,
        but the pipeline cannot progress (e.g. repeated duplicate hypothesis rejections).
        """
        self.converged = False
        self.convergence_reason = None
        self.stop_reason = reason
        self.confidence_level = confidence
        self.conclusion = conclusion
        self.mechanism_category_number = mechanism_category_number
        self.mechanism_category_name = mechanism_category_name
        self.run_status = "synthesized_stop"
        self.synthesized = True

    def mark_stopped(
        self,
        reason: str,
        confidence: float,
        conclusion: str,
    ) -> None:
        """Mark the pipeline as stopped without convergence."""
        self.converged = False
        self.convergence_reason = None
        self.stop_reason = reason
        self.confidence_level = confidence
        self.conclusion = conclusion
        self.mechanism_category_number = None
        self.mechanism_category_name = None
        self.run_status = "stopped"
        self.synthesized = False

    def get_history_summary(self) -> str:
        """Get a summary of all iterations for the hypothesis model."""
        if not self.tested_hypotheses:
            return "No hypotheses have been tested yet."

        parts = []
        for hypo in self.tested_hypotheses:
            parts.append(f"Iteration {hypo.get('iteration', '?')}:")
            parts.append(f"  Hypothesis: {hypo.get('name', 'N/A')}")
            parts.append(f"  Result: {hypo.get('result', 'N/A')}")
            data_used = ', '.join(hypo.get('required_data', [])) or 'not specified'
            parts.append(f"  Data used: {data_used}")
            parts.append(f"  Evidence: {hypo.get('evidence_summary', 'N/A')}")
            simpler = hypo.get('simpler_supported_explanation')
            if simpler:
                parts.append(f"  Residual signal: {simpler}")
            conv_reasoning = hypo.get('convergence_reasoning')
            if conv_reasoning:
                # Keep it short — full reasoning can be long
                parts.append(f"  Convergence feedback: {conv_reasoning[:400]}")
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        """Create from dictionary."""
        return cls.model_validate(data)
