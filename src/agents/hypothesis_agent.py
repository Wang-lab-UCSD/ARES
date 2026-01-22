"""Hypothesis generation and refinement agent."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider, Message
from src.memory.conversation import ConversationMemory, PipelineState
from src.prompts.hypothesis import (
    HYPOTHESIS_SYSTEM_PROMPT,
    build_initial_hypothesis_prompt,
    build_refinement_prompt,
    build_convergence_check_prompt,
)
from src.utils.logging import get_logger


class HypothesisAgent:
    """Agent for generating and refining scientific hypotheses.

    This agent:
    - Generates initial hypotheses based on a scientific finding
    - Refines hypotheses based on experimental results
    - Decides when to converge or continue iterating
    - Maintains conversation memory for context
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the hypothesis agent.

        Args:
            llm: LLM provider for generating hypotheses
        """
        self.llm = llm
        self.memory = ConversationMemory(system_message=HYPOTHESIS_SYSTEM_PROMPT)
        self.logger = get_logger("hypothesis_agent")

    async def generate_initial_hypotheses(
        self,
        finding: str,
        context: str,
        data_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate initial hypotheses for a scientific finding.

        Args:
            finding: The scientific finding to explain (X predicts Y)
            context: Additional context about the research
            data_manifest: Available data and tools

        Returns:
            Dictionary containing hypotheses and recommendation
        """
        self.logger.info("Generating initial hypotheses", {"finding": finding[:100]})

        prompt = build_initial_hypothesis_prompt(finding, context, data_manifest)
        self.memory.add_user_message(prompt, stage="initial")

        try:
            result = await self.llm.complete_json(
                self.memory.get_messages(),
                temperature=0.7,
            )

            self.memory.add_assistant_message(
                str(result),
                stage="initial",
                hypothesis_count=len(result.get("hypotheses", [])),
            )

            self.logger.info("Generated hypotheses", {
                "count": len(result.get("hypotheses", [])),
                "recommended_first": result.get("recommended_first", 0),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to generate hypotheses", {"error": str(e)})
            raise

    async def refine_hypotheses(
        self,
        state: PipelineState,
        last_hypothesis: dict[str, Any],
        last_result: dict[str, Any],
        data_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        """Refine hypotheses based on experimental results.

        Args:
            state: Current pipeline state
            last_hypothesis: The hypothesis that was just tested
            last_result: Results from testing the hypothesis
            data_manifest: Available data and tools

        Returns:
            Dictionary containing decision and updated hypotheses
        """
        self.logger.info("Refining hypotheses", {
            "iteration": state.current_iteration,
            "last_hypothesis": last_hypothesis.get("name", "N/A"),
        })

        prompt = build_refinement_prompt(
            finding=state.finding,
            history_summary=state.get_history_summary(),
            last_hypothesis=last_hypothesis,
            last_result=last_result,
            data_manifest=data_manifest,
        )

        self.memory.add_user_message(
            prompt,
            stage="refinement",
            iteration=state.current_iteration,
        )

        try:
            result = await self.llm.complete_json(
                self.memory.get_messages(),
                temperature=0.7,
            )

            self.memory.add_assistant_message(
                str(result),
                stage="refinement",
                iteration=state.current_iteration,
                decision=result.get("decision"),
            )

            self.logger.info("Refinement decision", {
                "decision": result.get("decision"),
                "confidence": result.get("confidence"),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to refine hypotheses", {"error": str(e)})
            raise

    async def check_convergence(
        self,
        state: PipelineState,
    ) -> dict[str, Any]:
        """Check if the pipeline should stop iterating.

        Args:
            state: Current pipeline state

        Returns:
            Dictionary with convergence decision
        """
        self.logger.info("Checking convergence", {
            "iteration": state.current_iteration,
            "evidence_count": len(state.evidence),
        })

        prompt = build_convergence_check_prompt(
            finding=state.finding,
            history_summary=state.get_history_summary(),
            total_evidence=state.evidence,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(HYPOTHESIS_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                temperature=0.3,
            )

            self.logger.info("Convergence check result", {
                "should_continue": result.get("should_continue"),
                "confidence": result.get("confidence"),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to check convergence", {"error": str(e)})
            raise

    def get_next_hypothesis(
        self,
        hypotheses: list[dict[str, Any]],
        tested_ids: set[int] | None = None,
    ) -> dict[str, Any] | None:
        """Get the next hypothesis to test.

        Args:
            hypotheses: List of available hypotheses
            tested_ids: Set of already tested hypothesis IDs

        Returns:
            Next hypothesis to test, or None if all tested
        """
        tested_ids = tested_ids or set()

        # Sort by priority
        sorted_hypos = sorted(hypotheses, key=lambda h: h.get("priority", 999))

        for hypo in sorted_hypos:
            hypo_id = hypo.get("id", hypotheses.index(hypo))
            if hypo_id not in tested_ids:
                return hypo

        return None

    def reset_memory(self) -> None:
        """Reset conversation memory for a new run."""
        self.memory.clear()
