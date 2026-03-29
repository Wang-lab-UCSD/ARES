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


RESULT_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "execution_success": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "support_level": {
            "type": "string",
            "enum": ["SUPPORTS", "REFUSES", "INCONCLUSIVE", "ERROR", "UNTESTABLE"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["support_level", "confidence", "reasoning", "summary"],
}

CONVERGENCE_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "converged": {"type": "boolean"},
        "mechanism_category_number": {"type": ["integer", "null"]},
        "mechanism_category_name": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "conclusion": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["converged", "confidence", "reasoning"],
}


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
                schema=RESULT_SUMMARY_SCHEMA,
            )

            self.logger.info("Result summary", {
                "support_level": result.get("support_level"),
                "confidence": result.get("confidence"),
            })

            return result

        except Exception as e:
            self.logger.error("Failed to summarize results", {"error": str(e)})
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
        investigation_objective: str | None = None,
        available_biology_layers: list[str] | None = None,
        used_biology_layers: list[str] | None = None,
        causal_capable_data: bool = False,
    ) -> dict[str, Any]:
        """Independently check whether the evidence warrants convergence.

        Uses a fresh context (no accumulated history) to evaluate convergence
        against the mechanism taxonomy. When causal_capable_data is False (observational
        only), convergence is allowed on the best-supported mechanism without requiring
        a but-for test.

        Args:
            finding: Original scientific finding
            tested_hypotheses: All tested hypotheses with results
            last_result: Summary of the most recent result
            raw_output: Raw stdout from the last execution (will be truncated)
            investigation_objective: Optional goal (discovery/synthesis); when set, convergence on a new named mechanism is acceptable.
            available_biology_layers: Manifest data categories for expression/conservation (e.g. rnaseq, phyloP) when present.
            used_biology_layers: Among SUPPORTED hypotheses, which of those categories were used.
            causal_capable_data: When True, require but-for/causality. When False, observational mode: converge on best-supported mechanism without requiring causality.

        Returns:
            Dictionary with converged (bool), mechanism_category_number,
            mechanism_category_name, confidence, conclusion, reasoning
        """
        self.logger.info("Running independent convergence check", {
            "iterations_tested": len(tested_hypotheses),
            "last_support_level": last_result.get("support_level"),
            "available_biology_layers": available_biology_layers,
            "used_biology_layers": used_biology_layers,
            "causal_capable_data": causal_capable_data,
        })

        prompt = build_convergence_check_prompt(
            finding=finding,
            tested_hypotheses=tested_hypotheses,
            last_result=last_result,
            raw_output=raw_output,
            investigation_objective=investigation_objective,
            available_biology_layers=available_biology_layers,
            used_biology_layers=used_biology_layers,
            causal_capable_data=causal_capable_data,
        )

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(CONVERGENCE_CHECK_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                schema=CONVERGENCE_CHECK_SCHEMA,
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
        converged: bool,
        run_status: str | None = None,
        synthesized: bool = False,
    ) -> str:
        """Generate the final report for the investigation.

        Args:
            finding: Original scientific finding
            context: Research context
            history_summary: Summary of all iterations
            conclusion: Final conclusion
            all_evidence: All accumulated evidence
            converged: Whether the run met strict convergence
            run_status: Terminal run status
            synthesized: Whether the run stopped via synthesis

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
            converged=converged,
            run_status=run_status,
            synthesized=synthesized,
        )

        try:
            response = await self.llm.complete(
                [
                    Message.system(SUMMARY_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                # temperature removed for gpt-5-mini compatibility
            )

            report = response.content

            report = self._apply_outcome_guard(report, all_evidence)

            self.logger.info("Generated final report", {
                "length": len(report),
            })

            return report

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

    def _apply_outcome_guard(
        self,
        report: str,
        all_evidence: list[dict[str, Any]],
    ) -> str:
        """Cross-check the LLM-generated report against recorded support levels.

        If the report text claims a hypothesis was SUPPORTED but the evidence
        record says otherwise (REFUSES / ERROR / INCONCLUSIVE / UNTESTABLE),
        append a correction notice and an authoritative outcome table so the
        reader is not misled.
        """
        outcome_map: dict[str, str] = {}
        for e in all_evidence:
            name = e.get("hypothesis", e.get("hypothesis_name", ""))
            level = e.get("support_level", "N/A")
            if name:
                outcome_map[name] = level

        supported_names = {
            n for n, lvl in outcome_map.items() if lvl == "SUPPORTS"
        }

        mismatches: list[str] = []
        report_upper = report.upper()
        for name, level in outcome_map.items():
            if level == "SUPPORTS":
                continue
            name_upper = name.upper()
            idx = report_upper.find(name_upper)
            while idx != -1:
                window = report_upper[idx:idx + len(name_upper) + 200]
                if "SUPPORT" in window and "REFUSE" not in window:
                    mismatches.append(name)
                    break
                idx = report_upper.find(name_upper, idx + 1)

        if not mismatches:
            return report

        self.logger.warning(
            "Report-outcome mismatch detected; appending correction",
            {"mismatched_hypotheses": mismatches},
        )

        table_lines = [
            "\n\n---\n",
            "## Authoritative Hypothesis Outcomes\n",
            "*Auto-generated from recorded evidence — supersedes any conflicting "
            "narrative above.*\n",
            "| Hypothesis | Recorded Outcome |",
            "|---|---|",
        ]
        for name, level in outcome_map.items():
            table_lines.append(f"| {name} | **{level}** |")

        if mismatches:
            table_lines.append("")
            table_lines.append(
                f"> **Note**: The narrative above may incorrectly describe "
                f"the following as SUPPORTED when they were not: "
                f"{', '.join(mismatches)}. "
                f"Please rely on the table above for authoritative outcomes."
            )

        return report + "\n".join(table_lines) + "\n"

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
