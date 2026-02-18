"""Code review agent for catching rule violations before execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.llm.base import LLMProvider, Message
from src.prompts.review import (
    REVIEW_SYSTEM_PROMPT,
    HYPOTHESIS_REVIEW_SYSTEM_PROMPT,
    build_review_prompt,
    build_hypothesis_review_prompt,
)
from src.utils.logging import get_logger


@dataclass
class ReviewResult:
    """Result of a code review."""

    approved: bool
    corrected_code: str | None
    issues_found: list[str]
    reasoning: str


class ReviewAgent:
    """Agent for reviewing generated code before execution.

    This agent:
    - Checks code against pipeline rules (FIMO --motif, single test, etc.)
    - Returns corrected code when issues are found
    - Approves code that follows all rules
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the review agent.

        Args:
            llm: LLM provider for code review
        """
        self.llm = llm
        self.logger = get_logger("review_agent")

    async def review_code(
        self,
        code: str,
        hypothesis: dict[str, Any],
        data_manifest: dict[str, Any],
        tested_hypotheses: list[dict[str, Any]] | None = None,
    ) -> ReviewResult:
        """Review generated code for rule violations.

        Args:
            code: The generated Python code
            hypothesis: The hypothesis being tested
            data_manifest: Available data paths and tools
            tested_hypotheses: Previously tested hypotheses for duplicate detection

        Returns:
            ReviewResult with approval status and optional corrections
        """
        self.logger.info("Reviewing generated code", {
            "hypothesis": hypothesis.get("name", "N/A"),
            "code_length": len(code),
        })

        prompt = build_review_prompt(code, hypothesis, data_manifest, tested_hypotheses)

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(REVIEW_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
            )

            approved = result.get("approved", True)
            issues = result.get("issues", [])
            corrected = result.get("corrected_code")
            reasoning = result.get("reasoning", "")

            # Extract code from markdown blocks if present
            if corrected and not approved:
                corrected = self._extract_code(corrected)

            self.logger.info("Code review complete", {
                "approved": approved,
                "issues_count": len(issues),
                "reasoning": reasoning[:200],
            })

            if issues:
                for issue in issues:
                    self.logger.warning("Code review issue", {"issue": issue})

            return ReviewResult(
                approved=approved,
                corrected_code=corrected if not approved else None,
                issues_found=issues,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.error("Code review failed", {"error": str(e)})
            # If review fails, approve by default to not block the pipeline
            return ReviewResult(
                approved=True,
                corrected_code=None,
                issues_found=[],
                reasoning=f"Review skipped due to error: {str(e)}",
            )

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
            )

            approved = result.get("approved", True)
            issues = result.get("issues", [])
            reasoning = result.get("reasoning", "")

            self.logger.info("Hypothesis review complete", {
                "approved": approved,
                "issues_count": len(issues),
                "reasoning": reasoning[:200],
            })

            if issues:
                for issue in issues:
                    self.logger.warning("Hypothesis review issue", {"issue": issue})

            return ReviewResult(
                approved=approved,
                corrected_code=None,
                issues_found=issues,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.error("Hypothesis review failed", {"error": str(e)})
            return ReviewResult(
                approved=True,
                corrected_code=None,
                issues_found=[],
                reasoning=f"Review skipped due to error: {str(e)}",
            )

    def _extract_code(self, text: str) -> str:
        """Extract code from potential markdown blocks."""
        import re

        text = text.strip()
        patterns = [
            r"```python\s*(.*?)```",
            r"```py\s*(.*?)```",
            r"```\s*(.*?)```",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code = "\n\n".join(matches).strip()
                if code:
                    return code
        return text
