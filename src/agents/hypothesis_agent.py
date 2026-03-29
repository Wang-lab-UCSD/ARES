"""Hypothesis generation and refinement agent."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider, Message
from src.memory.conversation import ConversationMemory, PipelineState
from src.prompts.hypothesis import (
    HYPOTHESIS_SYSTEM_PROMPT,
    build_refinement_prompt,
    build_regeneration_prompt,
)
from src.utils.logging import get_logger


REFINEMENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["REFINE", "NEW_HYPOTHESIS", "TECHNICAL_ERROR", "INSUFFICIENT_DATA"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "technical_issues": {"type": "array", "items": {"type": "string"}},
        "hypotheses": {"type": "array"},
    },
    "required": ["decision", "confidence", "reasoning"],
}

REGENERATION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "hypothesis": {"type": "object"},
        "reasoning": {"type": "string"},
    },
    "required": ["hypothesis"],
}


class HypothesisAgent:
    """Agent for generating and refining scientific hypotheses.

    This agent:
    - Generates the first mechanism hypothesis (no prior history) and refines
      subsequent ones based on experimental results, using a single unified prompt.
    - Maintains conversation memory for context
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the hypothesis agent.

        Args:
            llm: LLM provider for generating hypotheses
        """
        self.llm = llm
        self.logger = get_logger("hypothesis_agent")

    async def refine_hypotheses(
        self,
        state: PipelineState,
        last_hypothesis: dict[str, Any] | None,
        last_result: dict[str, Any] | None,
        data_manifest: dict[str, Any],
        group_summary: str | None = None,
        encourage_different_mechanism: bool = False,
        unused_biology_layers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate or refine hypotheses based on experimental results.

        On the first iteration, pass last_hypothesis=None and last_result=None
        to generate the initial mechanism hypothesis with no prior history.

        Args:
            state: Current pipeline state
            last_hypothesis: The hypothesis that was just tested, or None on first iteration
            last_result: Results from testing the hypothesis, or None on first iteration
            data_manifest: Available data and tools
            group_summary: Summary of completed hypothesis group for synthesis
            encourage_different_mechanism: If True, prompt asks to prefer a different mechanism category when at least one hypothesis already SUPPORTS
            unused_biology_layers: When set, expression/conservation data are available but unused in any supported result; prefer a hypothesis that uses one.

        Returns:
            Dictionary containing decision and updated hypotheses
        """
        self.logger.info("Refining hypotheses", {
            "iteration": state.current_iteration,
            "last_hypothesis": last_hypothesis.get("name", "N/A") if last_hypothesis else "None (first iteration)",
            "group_complete": group_summary is not None,
            "encourage_different_mechanism": encourage_different_mechanism,
            "unused_biology_layers": unused_biology_layers,
        })

        prompt = build_refinement_prompt(
            finding=state.finding,
            history_summary=state.get_history_summary(),
            last_hypothesis=last_hypothesis,
            last_result=last_result,
            data_manifest=data_manifest,
            group_summary=group_summary,
            encourage_different_mechanism=encourage_different_mechanism,
            unused_biology_layers=unused_biology_layers,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(HYPOTHESIS_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                schema=REFINEMENT_RESPONSE_SCHEMA,
            )

            self.logger.info("Refinement decision", {
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to refine hypotheses", {"error": str(e)})
            raise

    def get_next_hypothesis(
        self,
        hypotheses: list[dict[str, Any]],
        tested_ids: set[int] | None = None,
        last_tested: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Get the next hypothesis to test.

        Prefers untested hypotheses from the same group as the last tested one,
        so related sub-hypotheses are tested together before moving on.

        Args:
            hypotheses: List of available hypotheses
            tested_ids: Set of already tested hypothesis IDs
            last_tested: The last successfully tested hypothesis (for group affinity)

        Returns:
            Next hypothesis to test, or None if all tested
        """
        tested_ids = tested_ids or set()

        untested = [
            h for h in hypotheses
            if h.get("id", hypotheses.index(h)) not in tested_ids
        ]
        if not untested:
            return None

        # If last tested hypothesis had a group, prefer same-group hypotheses
        last_group = last_tested.get("group") if last_tested else None
        if last_group:
            same_group = [h for h in untested if h.get("group") == last_group]
            if same_group:
                return sorted(same_group, key=lambda h: h.get("priority", 999))[0]

        # Fallback: sort all untested by priority
        return sorted(untested, key=lambda h: h.get("priority", 999))[0]

    async def regenerate_hypothesis(
        self,
        state: PipelineState,
        rejected_hypothesis: dict[str, Any],
        feedback: str,
        data_manifest: dict[str, Any],
        rejection_category: str | None = None,
        reviewer_rejected: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Regenerate a hypothesis that was rejected by review.

        Args:
            state: Current pipeline state
            rejected_hypothesis: The hypothesis that was rejected
            feedback: Reviewer's feedback on why it was rejected
            data_manifest: Available data and tools
            rejection_category: Structured category from reviewer ("wrong_mechanism")
            reviewer_rejected: All hypotheses rejected by reviewer this session (with feedback)

        Returns:
            New hypothesis dict, or None if generation failed
        """
        self.logger.info("Regenerating hypothesis", {
            "rejected": rejected_hypothesis.get("name", "N/A"),
            "feedback": feedback[:200],
        })

        prompt = build_regeneration_prompt(
            finding=state.finding,
            rejected_hypothesis=rejected_hypothesis,
            feedback=feedback,
            tested_hypotheses=state.tested_hypotheses,
            data_manifest=data_manifest,
            rejection_category=rejection_category,
            reviewer_rejected=reviewer_rejected,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(HYPOTHESIS_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                schema=REGENERATION_RESPONSE_SCHEMA,
            )

            hypothesis = result.get("hypothesis")
            if hypothesis:
                # Assign a new ID
                max_id = max(
                    (h.get("id", 0) for h in state.hypotheses),
                    default=-1,
                )
                hypothesis["id"] = max_id + 1
                hypothesis["iteration"] = state.current_iteration

                self.logger.info("Regenerated hypothesis", {
                    "name": hypothesis.get("name", "N/A"),
                })
                return hypothesis

            self.logger.warning("No hypothesis in regeneration response")
            return None

        except Exception as e:
            self.logger.error("Failed to regenerate hypothesis", {"error": str(e)})
            return None

