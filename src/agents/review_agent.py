"""Code review agent for catching rule violations before execution."""

from __future__ import annotations

import re
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
    warning_issues: list[str]
    blocker_issues: list[str]
    performance_blocker_issues: list[str]
    reasoning: str
    rejection_category: str | None = None  # "wrong_mechanism" or None


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

        # Static checks are run by the orchestrator as a pre-gate before this call,
        # so code reaching here has already passed static screening.
        # Collect warnings (non-blocking) to include in the result, but skip the
        # blocking static path — it would never trigger here.
        static_warnings = self.static_review_code(code).warning_issues

        prompt = build_review_prompt(code, hypothesis, data_manifest, tested_hypotheses)

        try:
            result = await self.llm.complete_json(
                [
                    Message.system(REVIEW_SYSTEM_PROMPT),
                    Message.user(prompt),
                ],
                max_tokens=self.llm.default_max_tokens,
            )

            # If the model returned an empty response (e.g. reasoning tokens exhausted
            # the max_tokens budget), treat it as a failed review rather than auto-approving.
            if not result:
                raise ValueError("Review model returned empty JSON (likely token budget exhausted)")

            approved = result.get("approved", True)
            issues = result.get("issues", [])
            corrected = result.get("corrected_code")
            reasoning = result.get("reasoning", "")
            blocker_issues, performance_blocker_issues, warning_issues = self._classify_issues(issues)
            warning_issues = static_warnings + warning_issues  # merge static warnings
            approved = approved and not blocker_issues and not performance_blocker_issues

            # Extract code from markdown blocks if present
            if corrected and not approved:
                corrected = self._extract_code(corrected)

            # Run static check on the LLM's corrected code before returning it.
            # The LLM reviewer doesn't know the static rules, so its corrected code
            # can introduce new pd.qcut(), merge(on='name'), or ENCFF violations.
            # If static fails on the corrected code, merge those issues in and clear
            # corrected_code — the model will receive all issues in one fix prompt
            # instead of discovering them one rejection at a time.
            if corrected and not approved:
                static_on_corrected = self.static_review_code(corrected)
                if not static_on_corrected.approved:
                    extra = static_on_corrected.blocker_issues + static_on_corrected.performance_blocker_issues
                    self.logger.warning("LLM corrected_code failed static check — merging issues, clearing corrected_code", {
                        "static_blockers": extra,
                    })
                    blocker_issues = blocker_issues + static_on_corrected.blocker_issues
                    performance_blocker_issues = performance_blocker_issues + static_on_corrected.performance_blocker_issues
                    warning_issues = warning_issues + static_on_corrected.warning_issues
                    corrected = None  # Force fresh generation with merged issue list

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
                issues_found=blocker_issues + performance_blocker_issues + warning_issues,
                warning_issues=warning_issues,
                blocker_issues=blocker_issues,
                performance_blocker_issues=performance_blocker_issues,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.error("Code review failed", {"error": str(e)})
            return ReviewResult(
                approved=False,
                corrected_code=None,
                issues_found=["REVIEW_UNAVAILABLE: LLM call failed — static check will run"],
                warning_issues=[],
                blocker_issues=["REVIEW_UNAVAILABLE: LLM call failed — static check will run"],
                performance_blocker_issues=[],
                reasoning=f"Review failed: {str(e)}",
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
                max_tokens=self.llm.default_max_tokens,
            )

            approved = result.get("approved", True)
            issues = result.get("issues", [])
            reasoning = result.get("reasoning", "")
            rejection_category = result.get("rejection_category") if not approved else None

            self.logger.info("Hypothesis review complete", {
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
                corrected_code=None,
                issues_found=issues,
                warning_issues=[],
                blocker_issues=issues if not approved else [],
                performance_blocker_issues=[],
                reasoning=reasoning,
                rejection_category=rejection_category,
            )

        except Exception as e:
            self.logger.error("Hypothesis review failed", {"error": str(e)})
            return ReviewResult(
                approved=False,
                corrected_code=None,
                issues_found=["REVIEW_UNAVAILABLE: LLM call failed — hypothesis will be re-queued"],
                warning_issues=[],
                blocker_issues=["REVIEW_UNAVAILABLE: LLM call failed — hypothesis will be re-queued"],
                performance_blocker_issues=[],
                reasoning=f"Review failed: {str(e)}",
            )

    @staticmethod
    def static_review_code(
        code: str,
    ) -> "ReviewResult":
        """Static rule-based code check, used as fallback when LLM review is unavailable.

        Checks the four highest-value rules that are syntactically detectable
        without an LLM call.

        Args:
            code: The Python code to check

        Returns:
            ReviewResult with approval status
        """
        warning_issues: list[str] = []
        blocker_issues: list[str] = []
        performance_blocker_issues: list[str] = []

        # Only rules that prevent hours of wasted compute or produce silently wrong results
        # that the runtime fix loop cannot detect.  Everything else is left to runtime.

        # Rule 1: FIMO without --motif — scans all 800+ motifs, takes 2+ hours.
        _fimo_cli = re.search(r"""['"]\s*fimo\s*['"]|\bfimo\s+--""", code)
        if _fimo_cli and not re.search(r"--motif\b", code):
            performance_blocker_issues.append("FIMO called without --motif flag — would scan all 800+ motifs (2+ hours)")

        # Rule 2: FIMO or bedtools getfasta inside a loop — N× expensive per peak.
        lines = code.split("\n")
        in_loop = False
        loop_indent = 0
        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if re.match(r"(for |while )", stripped):
                in_loop = True
                loop_indent = indent
            elif in_loop and stripped and not stripped.startswith("#") and indent <= loop_indent:
                in_loop = False
            if in_loop and re.search(r"\b(fimo|bedtools getfasta)\b", line):
                performance_blocker_issues.append("Expensive tool (fimo or bedtools getfasta) called inside a loop — will run once per peak")
                break

        # Rule 5: FIMO without --no-pgc — silently reports chromosome names instead of peak IDs.
        # This produces wrong join keys; the error only appears downstream as zero matches,
        # which the model cannot diagnose from a traceback alone.
        if _fimo_cli and not re.search(r"--no-pgc", code):
            blocker_issues.append(
                "FIMO called without --no-pgc — FIMO will parse genomic coordinates "
                "from FASTA headers, reporting chromosomes instead of peak IDs in "
                "sequence_name. Add --no-pgc to every fimo invocation."
            )

        # Rule 5b: FIMO on full genome FASTA — takes hours, will time out.
        if _fimo_cli and "run_fimo_on_peaks" not in code:
            if re.search(r"""fasta['"]\]|genome\.fa|hg38\.fa|\.genome\.""", code):
                performance_blocker_issues.append(
                    "FIMO appears to scan the full genome FASTA — this takes hours and "
                    "will time out (15-minute execution limit). Use run_fimo_on_peaks() "
                    "to scan within peak windows."
                )

        # Rule 7: bw.values() per-base extraction — extremely slow for thousands of peaks.
        # Exception: FFT/autocorrelation analyses genuinely need per-base arrays.
        if re.search(r"bw\.values\s*\(", code):
            if not re.search(r"fft|autocorr|periodicity|nucleosome.phas|np\.fft|scipy\.signal", code, re.IGNORECASE):
                performance_blocker_issues.append(
                    "pyBigWig bw.values() extracts per-base signal — very slow for many peaks. "
                    "Use extract_bigwig_signals() or extract_summit_signals() from src.utils.bioio instead."
                )

        # Rule 12: STRING API call — rate-limited (HTTP 429) on shared cluster IPs.
        # Local files are always available; this error is not self-correcting.
        if re.search(r"string-db\.org|stringdb\.org", code, re.IGNORECASE):
            blocker_issues.append(
                "Do NOT call the STRING API (string-db.org) — rate-limited (HTTP 429) on shared cluster IPs. "
                "Use local files from data_files['string_links'] and data_files['string_aliases'] instead."
            )

        # Hardcoded ENCFF IDs — silently opens the wrong file; no traceback.
        if re.search(r"ENCFF[A-Z0-9]{6}", code):
            blocker_issues.append(
                "Hardcoded ENCFF ID detected — use data_files['key'] instead. "
                "The pipeline injects data_files into the kernel at runtime."
            )

        # data_files/data_manifest redefinition — silently overrides injected paths.
        if re.search(r"(?m)^\s*(data_files|data_manifest)\s*=", code):
            blocker_issues.append(
                "Local reassignment of data_files/data_manifest detected. "
                "These are injected by the pipeline at runtime — do not redefine them."
            )

        issues = blocker_issues + performance_blocker_issues + warning_issues
        if blocker_issues or performance_blocker_issues:
            return ReviewResult(
                approved=False,
                corrected_code=None,
                issues_found=issues,
                warning_issues=warning_issues,
                blocker_issues=blocker_issues,
                performance_blocker_issues=performance_blocker_issues,
                reasoning="Static rule check failed: " + "; ".join(issues),
            )

        return ReviewResult(
            approved=True,
            corrected_code=None,
            issues_found=warning_issues,
            warning_issues=warning_issues,
            blocker_issues=[],
            performance_blocker_issues=[],
            reasoning="Static rule check passed",
        )

    @staticmethod
    def _classify_issues(issues: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Split free-form review issues into blockers, performance blockers, and warnings."""
        blocker_patterns = [
            r"mismatch",
            r"wrong statistical test|incorrect primary test|invalid test",
            r"runtime failure|hard crash|will fail at runtime|undefined|unboundlocalerror",
            r"wrong.*column|mis-assign|misclassif|parsing is incorrect|column selection by position",
            r"wrong motif|motif id|fimo.*parse|sequence_name parsing|--no-pgc",
            r"shell-wrapped cli|bash -lc|shell=True",
            r"missing required|does not implement|not actually enforced|does not match the hypothesis",
            r"invalid because|not a valid test|wrong universe|wrong denominator",
            r"incorrect endpoint|api endpoints are incorrect",
            r"bug / wrong tool invocation|invalid because .* -ibam",
            r"external http api call.*without retry|rate limit.*429",
            r"nan values in coordinate|cannot convert float nan",
        ]
        performance_patterns = [
            r"would scan all 800\+ motifs",
            r"expensive tool .* inside a loop",
            r"extremely slow|excessive runtime|runtime blowup|api abuse risk",
            r"pybigwig bw\.values|per-base signal",
            r"called inside a loop .*slow for many intervals",
            r"bw\.stats\(\) called directly|bw\.stats.*invalid interval|extract_bigwig_signals.*extract_summit_signals",
        ]
        warning_patterns = [
            r"redundant|wasted compute|duplicate .*sort|not incorrect, but unnecessary",
            r"low risk|no change required|minor but real|rare but possible",
            r"brittle|safer to|for safety|nondetermin",
        ]

        blockers: list[str] = []
        perf_blockers: list[str] = []
        warnings: list[str] = []
        for issue in issues:
            text = issue.lower()
            if any(re.search(pat, text) for pat in performance_patterns):
                perf_blockers.append(issue)
            elif any(re.search(pat, text) for pat in blocker_patterns):
                blockers.append(issue)
            elif any(re.search(pat, text) for pat in warning_patterns):
                warnings.append(issue)
            else:
                blockers.append(issue)
        return blockers, perf_blockers, warnings

    def _extract_code(self, text: str) -> str:
        """Extract code from potential markdown blocks."""
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
