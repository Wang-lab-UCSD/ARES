"""Hypothesis review agent.

Reviews proposed hypotheses for novelty (vs prior iterations), mechanism vs.
characterization, and verification-plan relevance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.base import LLMProvider, Message
from src.prompts.review import (
    HYPOTHESIS_REVIEW_SYSTEM_PROMPT,
    build_hypothesis_review_prompt,
)
from src.utils.logging import get_logger


@dataclass
class ReviewResult:
    """Result of a hypothesis review."""

    approved: bool
    issues_found: list[str]
    reasoning: str
    rejection_category: str | None = None  # "wrong_mechanism" or None
    revised_prediction: str | None = None
    revised_verification_plan: list[str] | None = None


class HypothesisReviewAgent:
    """Agent for reviewing proposed hypotheses before code generation.

    This agent:
    - Checks the hypothesis for duplicates against prior iterations
    - Checks that it proposes a causal mechanism (not characterization)
    - Checks that the verification plan tests the stated mechanism
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the review agent.

        Args:
            llm: LLM provider for hypothesis review
        """
        self.llm = llm
        self.logger = get_logger("review_agent")

    async def review_hypothesis(
        self,
        hypothesis: dict[str, Any],
        tested_hypotheses: list[dict[str, Any]],
    ) -> ReviewResult:
        """Review a hypothesis before code generation.

        Checks for duplicates and vague predictions.

        Args:
            hypothesis: The hypothesis to review
            tested_hypotheses: Previously tested hypotheses

        Returns:
            ReviewResult with approval status and feedback
        """
        self.logger.info("Reviewing hypothesis", {
            "hypothesis": hypothesis.get("name", "N/A"),
        })

        prompt = build_hypothesis_review_prompt(hypothesis, tested_hypotheses)

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(HYPOTHESIS_REVIEW_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                max_tokens=self.llm.default_max_tokens,
            )

            # Support both old (approved: bool) and new (decision: str) formats
            decision = result.get("decision")
            if decision is not None:
                approved = decision in ("approve", "approve_with_revision")
            else:
                approved = result.get("approved", True)
                decision = "approve" if approved else "reject"

            issues = result.get("issues", [])
            reasoning = result.get("reasoning", "")
            rejection_category = result.get("rejection_category") if decision == "reject" else None

            # Extract revision fields for approve_with_revision
            revised_prediction = None
            revised_verification_plan = None
            if decision == "approve_with_revision":
                revised_prediction = result.get("revised_prediction")
                revised_verification_plan = result.get("revised_verification_plan")

            self.logger.info("Hypothesis review complete", {
                "decision": decision,
                "approved": approved,
                "issues_count": len(issues),
                "rejection_category": rejection_category,
                "reasoning": reasoning[:200],
            })

            if issues:
                for issue in issues:
                    self.logger.warning("Hypothesis review issue", {"issue": issue})

            return ReviewResult(
                approved=approved,
                issues_found=issues,
                reasoning=reasoning,
                rejection_category=rejection_category,
                revised_prediction=revised_prediction,
                revised_verification_plan=revised_verification_plan,
            )

        except Exception as e:
            self.logger.error("Hypothesis review failed", {"error": str(e)})
            return ReviewResult(
                approved=False,
                issues_found=["REVIEW_UNAVAILABLE: LLM call failed — hypothesis will be re-queued"],
                reasoning=f"Review failed: {str(e)}",
            )
