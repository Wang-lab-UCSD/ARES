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
    build_regeneration_prompt,
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
        group_summary: str | None = None,
    ) -> dict[str, Any]:
        """Refine hypotheses based on experimental results.

        Args:
            state: Current pipeline state
            last_hypothesis: The hypothesis that was just tested
            last_result: Results from testing the hypothesis
            data_manifest: Available data and tools
            group_summary: Summary of completed hypothesis group for synthesis

        Returns:
            Dictionary containing decision and updated hypotheses
        """
        self.logger.info("Refining hypotheses", {
            "iteration": state.current_iteration,
            "last_hypothesis": last_hypothesis.get("name", "N/A"),
            "group_complete": group_summary is not None,
        })

        prompt = build_refinement_prompt(
            finding=state.finding,
            history_summary=state.get_history_summary(),
            last_hypothesis=last_hypothesis,
            last_result=last_result,
            data_manifest=data_manifest,
            group_summary=group_summary,
        )

        self.memory.add_user_message(
            prompt,
            stage="refinement",
            iteration=state.current_iteration,
        )

        try:
            result = await self.llm.complete_json(
                self.memory.get_messages(),
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
    ) -> dict[str, Any] | None:
        """Regenerate a hypothesis that was rejected by review.

        Args:
            state: Current pipeline state
            rejected_hypothesis: The hypothesis that was rejected
            feedback: Reviewer's feedback on why it was rejected
            data_manifest: Available data and tools

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
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(HYPOTHESIS_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
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

    def reset_memory(self) -> None:
        """Reset conversation memory for a new run."""
        self.memory.clear()
