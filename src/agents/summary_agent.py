"""Result summarization agent."""

from __future__ import annotations

from typing import Any

from src.llm.base import LLMProvider, Message
from src.execution.jupyter_executor import ExecutionResult
from src.prompts.summary import (
    SUMMARY_SYSTEM_PROMPT,
    CONVERGENCE_CHECK_SYSTEM_PROMPT,
    build_result_summary_prompt,
    build_final_report_prompt,
    build_convergence_check_prompt,
)
from src.utils.logging import get_logger


class SummaryAgent:
    """Agent for summarizing experimental results.

    This agent:
    - Interprets code execution results
    - Determines support level for hypotheses
    - Generates final reports
    """

    def __init__(self, llm: LLMProvider):
        """Initialize the summary agent.

        Args:
            llm: LLM provider for summarization
        """
        self.llm = llm
        self.logger = get_logger("summary_agent")

    async def summarize_results(
        self,
        hypothesis: dict[str, Any],
        code: str,
        execution_result: ExecutionResult,
    ) -> dict[str, Any]:
        """Summarize execution results for a hypothesis test.

        Args:
            hypothesis: The hypothesis that was tested
            code: The code that was executed
            execution_result: Result from code execution

        Returns:
            Dictionary containing summary and interpretation
        """
        self.logger.info("Summarizing results", {
            "hypothesis": hypothesis.get("name", "N/A"),
            "execution_success": execution_result.success,
        })

        # Format execution output
        execution_output = execution_result.get_display_output()

        prompt = build_result_summary_prompt(
            hypothesis=hypothesis,
            code=code,
            execution_result=execution_output,
            outputs=execution_result.outputs,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(SUMMARY_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                # temperature removed for gpt-5-mini compatibility
            )

            self.logger.info("Result summary", {
                "support_level": result.get("support_level"),
                "confidence": result.get("confidence"),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to summarize results", {"error": str(e)})
            # Return a basic error summary
            return {
                "execution_success": False,
                "findings": [],
                "statistics": {},
                "support_level": "ERROR",
                "confidence": 0.0,
                "reasoning": f"Failed to summarize: {str(e)}",
                "issues": [str(e)],
                "summary": f"Error during summarization: {str(e)}",
            }

    async def check_convergence(
        self,
        finding: str,
        tested_hypotheses: list[dict[str, Any]],
        last_result: dict[str, Any],
        raw_output: str | None = None,
    ) -> dict[str, Any]:
        """Independently check whether the evidence warrants convergence.

        Uses a fresh context (no accumulated history) to evaluate convergence
        against the mechanism taxonomy and three convergence criteria. Intentionally
        separate from HypothesisAgent to prevent motivated reasoning.

        Args:
            finding: Original scientific finding
            tested_hypotheses: All tested hypotheses with results
            last_result: Summary of the most recent result
            raw_output: Raw stdout from the last execution (will be truncated)

        Returns:
            Dictionary with converged (bool), mechanism_category_number,
            mechanism_category_name, confidence, conclusion, reasoning
        """
        self.logger.info("Running independent convergence check", {
            "iterations_tested": len(tested_hypotheses),
            "last_support_level": last_result.get("support_level"),
        })

        prompt = build_convergence_check_prompt(
            finding=finding,
            tested_hypotheses=tested_hypotheses,
            last_result=last_result,
            raw_output=raw_output,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(CONVERGENCE_CHECK_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
            )

            self.logger.info("Convergence check complete", {
                "converged": result.get("converged", False),
                "mechanism_category_number": result.get("mechanism_category_number"),
                "confidence": result.get("confidence"),
                "reasoning": result.get("reasoning"),
            })

            return result

        except Exception as e:
            self.logger.error("Convergence check failed", {"error": str(e)})
            # On failure, do not converge — safe default
            return {
                "converged": False,
                "mechanism_category_number": None,
                "mechanism_category_name": None,
                "confidence": 0.0,
                "conclusion": "",
                "reasoning": f"Convergence check failed: {str(e)}",
            }

    async def generate_final_report(
        self,
        finding: str,
        context: str,
        history_summary: str,
        conclusion: str,
        all_evidence: list[dict[str, Any]],
    ) -> str:
        """Generate the final report for the investigation.

        Args:
            finding: Original scientific finding
            context: Research context
            history_summary: Summary of all iterations
            conclusion: Final conclusion
            all_evidence: All accumulated evidence

        Returns:
            Markdown-formatted final report
        """
        self.logger.info("Generating final report")

        prompt = build_final_report_prompt(
            finding=finding,
            context=context,
            history_summary=history_summary,
            conclusion=conclusion,
            all_evidence=all_evidence,
        )

        try:
            response = await self.llm.complete(
                [
                    Message.system(SUMMARY_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                # temperature removed for gpt-5-mini compatibility
            )

            self.logger.info("Generated final report", {
                "length": len(response.content),
            })

            return response.content

        except Exception as e:
            self.logger.error("Failed to generate report", {"error": str(e)})
            # Return a basic error report
            return f"""# Investigation Report

## Error

Failed to generate full report: {str(e)}

## Finding

{finding}

## Conclusion

{conclusion}
"""

    def format_evidence_for_report(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format evidence list for final report generation.

        Args:
            evidence: Raw evidence from pipeline state

        Returns:
            Formatted evidence suitable for report
        """
        formatted = []

        for e in evidence:
            formatted.append({
                "iteration": e.get("iteration", "?"),
                "hypothesis": e.get("hypothesis_name", "N/A"),
                "findings": e.get("findings", []),
                "support_level": e.get("support_level", "N/A"),
                "confidence": e.get("confidence", 0.0),
            })

        return formatted
