"""Result summarization agent."""

from __future__ import annotations

import re
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


# Gene/protein symbol pattern: 2-10 uppercase letters/digits, starting with a letter.
# Catches things like SP1, NFYA, EP300, CREBBP, TP53, ZNF263. Excludes pure numbers
# and very short tokens that produce too many false positives.
_GENE_SYMBOL_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

# Common English / technical words that match the gene-symbol regex but are not genes.
# Used to filter false positives from the fabrication check.
_GENE_SYMBOL_STOPWORDS = {
    # Statistical / scientific
    "FC", "P", "Q", "R", "RHO", "D", "N", "SD", "SEM", "CI", "OR", "RR", "ROC", "AUC",
    "SUPPORTS", "REJECTS", "INCONCLUSIVE", "ERROR", "UNTESTABLE", "PASS", "FAIL",
    "TRUE", "FALSE", "NULL", "NONE", "NA", "NAN", "INF",
    # File formats / tools
    "BED", "GTF", "TSV", "CSV", "JSON", "FASTA", "FA", "BAM", "VCF", "BIGWIG",
    "FIMO", "MEME", "BEDTOOLS", "SAMTOOLS", "STRING", "GO", "KEGG",
    "JASPAR", "HOCOMOCO", "ENCODE", "ENCFF", "GENCODE",
    # Genomic concepts (commonly used as generic words, not gene symbols)
    "TF", "TFA", "TFB", "PWM", "DNA", "RNA", "MRNA", "TSS", "TTS", "CDS", "UTR",
    "PPI", "QC", "IR", "ID", "IDS", "ENSP", "ENSG", "HGNC",
    "CHIP", "CHIPSEQ", "RNASEQ", "ATAC", "DNASE", "WGBS",
    "CHROMHMM", "PHYLOP", "TPM", "FPKM", "FDR",
    # Histone marks / chromatin (these aren't strictly stopwords but the pattern
    # picks them up; they're usually descriptive context, not "named partners")
    "H3K27AC", "H3K27ME3", "H3K4ME3", "H3K9ME3", "H3K36ME3", "H3K4ME1",
    # Cell lines / common single-letter or short tokens
    "K562", "HEPG2", "GM12878", "HELA", "MCF7", "HG38", "HG19", "MM10",
    "USA", "UK", "EU",
    # Misc
    "T", "U", "V", "W", "X", "Y", "Z",  # single letters
    "BP", "KB", "MB", "GB",
    "AND", "OR", "NOT", "BUT", "IF", "THE", "FOR", "TO", "OF", "IN", "ON", "AT",
    "USE", "USED", "NEW", "OLD", "ALL", "ANY", "ONE", "TWO",
}


def _extract_gene_symbols(text: str) -> set[str]:
    """Extract gene-symbol-like tokens from text, filtering common stopwords."""
    if not text:
        return set()
    candidates = set(_GENE_SYMBOL_PATTERN.findall(text))
    return {c for c in candidates if c not in _GENE_SYMBOL_STOPWORDS}


def _detect_fabricated_entities(
    summary_text: str,
    findings: list[str],
    raw_output: str | None,
) -> list[str]:
    """Find gene-symbol-like tokens in the LLM's summary/findings that don't
    appear in the raw execution output.

    Returns a list of fabricated tokens (empty if none found, or if raw_output
    is missing). Used as a deterministic safety net for the fabrication check
    in evaluate_result.
    """
    if not raw_output:
        return []

    # Combine all text the LLM evaluator produced
    summary_pool = " ".join([summary_text or ""] + [f or "" for f in (findings or [])])
    summary_symbols = _extract_gene_symbols(summary_pool)
    if not summary_symbols:
        return []

    raw_symbols = _extract_gene_symbols(raw_output)
    fabricated = sorted(summary_symbols - raw_symbols)
    return fabricated


RESULT_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "execution_success": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "support_level": {
            "type": "string",
            "enum": ["SUPPORTS", "REJECTS", "INCONCLUSIVE", "ERROR", "UNTESTABLE"],
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

    async def evaluate_result(
        self,
        hypothesis: dict[str, Any],
        coding_result: dict[str, Any],
        raw_output: str | None = None,
    ) -> dict[str, Any]:
        """Independently evaluate the coding agent's execution output.

        The coding agent executes code and reports raw statistics. This method
        is the authoritative scorer — it determines the support level, confidence,
        summary, and key findings that go into the narrative and state.

        Args:
            hypothesis: The hypothesis that was tested
            coding_result: The coding agent's raw result (may include self-score for reference)
            raw_output: Raw stdout from code execution

        Returns:
            Dictionary with support_level, confidence, summary, findings, reasoning
        """
        self.logger.info("Evaluating result", {
            "hypothesis": hypothesis.get("name", ""),
            "coding_agent_level": coding_result.get("support_level"),
        })

        raw_section = ""
        if raw_output:
            truncated = raw_output[-6000:] if len(raw_output) > 6000 else raw_output
            raw_section = f"\n## Raw Execution Output (last 6000 chars)\n```\n{truncated}\n```"

        predictions = hypothesis.get("prediction", "N/A")
        verification_plan = hypothesis.get("verification_plan", [])
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(verification_plan)) if verification_plan else "N/A"

        prompt = f"""You are an independent scientific reviewer evaluating the results of a
bioinformatics hypothesis test. The coding agent executed the verification plan and
produced statistics. Your job is to determine the result.

## Hypothesis
**Name**: {hypothesis.get('name', 'N/A')}
**Mechanism**: {hypothesis.get('scratchpad', {}).get('mechanism', 'N/A')}
**Predictions**: {predictions}
**Verification Plan**:
{plan_text}

## What the Coding Agent Actually Did
**Analysis performed**: {coding_result.get('analysis', 'N/A')}

## Coding Agent's Raw Output
**Self-scored support level** (for reference only): {coding_result.get('support_level', 'N/A')}
**Coding agent summary**: {coding_result.get('summary', 'N/A')}
**Coding agent evidence**: {coding_result.get('evidence', 'N/A')}
{raw_section}

## Evaluation Rules
Check each prediction against the reported evidence:
- SUPPORTS: ALL key predictions pass their stated thresholds (FC, p-value, r, Cohen's d).
- REJECTS: At least one key prediction clearly fails (wrong direction or far below threshold).
- INCONCLUSIVE: Mixed results — some predictions pass, some fail, or results are ambiguous.
  Use this when the evidence is genuinely mixed, not when it clearly fails.

**Artifact-check special case** (when `group=artifact-check` or iteration 0):
SUPPORTS means the signal IS an artifact: TF_A NOT expressed AND motif NOT enriched.
If TF_A IS expressed (TPM ≥ 0.5) OR its motif IS enriched (≥1.2-fold), the signal is
genuine and you MUST set REJECTS, regardless of how the prediction is phrased. Ignore
any "if genuine: X" clause in compound predictions — that describes the negation of
the hypothesis, not the hypothesis itself.

## Sub-prediction discipline (CRITICAL)

Predictions in this pipeline are often compound — e.g., "STRING will reveal a direct
interaction OR shared partners; AND overlap > 10%; AND signal higher at co-bound sites".
Decompose every compound prediction into its sub-predictions and evaluate EACH ONE
separately. Then aggregate:

- **All sub-predictions pass** → SUPPORTS
- **All sub-predictions fail** → REJECTS
- **Some pass, some fail** → INCONCLUSIVE (this is the default for mixed results)

A compound prediction joined by "AND" requires ALL parts to pass for SUPPORTS. A part
joined by "OR" only requires one of the OR-options to pass — but if the prediction was
"A AND (B OR C)", you still need A to pass.

**Common error to avoid**: cherry-picking the strongest sub-prediction and ignoring the
failed ones. Example from a real run: prediction was "STRING reveals direct interaction
(score>=700) OR shared partners (score>=400); AND overlap >10%; AND signal higher at
co-bound sites." The result was: STRING direct=NOT FOUND (FAILED), shared partners=6
found (passed), overlap=14% (passed), signal FC=1.21 (passed). The verdict was wrongly
called SUPPORTS because the cherry-picked summary only mentioned the passing parts.
The CORRECT verdict is INCONCLUSIVE because the direct-interaction sub-prediction
failed and the shared-partners sub-prediction is a weaker substitute (the original
hypothesis named direct PPI as the primary mechanism).

When you call SUPPORTS, your `reasoning` must walk through every sub-prediction and
confirm each one passed. If you skip a sub-prediction in the walk-through, you are
cherry-picking.

## CONTROL OVERRIDE (critical)
If the raw output contains a control or comparator that tests whether the
mechanism-specific component adds explanatory power beyond a simpler baseline, and
the mechanism-specific group is NOT detectably stronger than the simpler baseline,
the named mechanism is NOT supported. Set support_level to REJECTS (or INCONCLUSIVE
if the control is underpowered), and fill `simpler_supported_explanation` with the
explanation that the data DO support.

Examples:
- anchor+YY1 not stronger than non-anchor+YY1 → the anchor/3D component is not
  supported; only the YY1-associated explanation remains.
- motif+ vs motif- groups differ, but the same effect persists after removing the
  proposed mediator → the mediator-specific mechanism is not supported.

Look for phrases in the raw output like "entirely explained by", "no additional
effect", "control shows", or comparisons where the mechanism-specific group is not
significantly different from the simpler-explanation group.

## Interpretation ceiling
Before setting support_level, answer two questions:
1. What is the strongest claim these results justify?
2. What simpler explanation is still compatible with the control results?
Your `summary` and `reasoning` must not exceed that ceiling. State the ceiling in
the `interpretation_ceiling` field.

## Fabrication check (CRITICAL)
The coding agent sometimes hardcodes biological entities as string literals — e.g., a
"Notable shared partners: EP300, CREBBP, JUN" line that was typed by the model rather
than computed from the data. Before trusting any specific named entities (gene symbols,
protein names, pathway names) in the coding agent's summary or findings:

1. **Cross-reference each named entity against the raw execution output above.** If a
   gene name appears in the summary but is NOT printed anywhere in the raw stdout,
   it is fabricated. The coding agent MUST derive every named entity from a printed
   variable in the same block.
2. **If you find fabricated names**: downgrade confidence by at least 0.2 and explicitly
   note the fabrication in the `reasoning` field. Strip the fabricated names from your
   own `summary` and `findings` — only report what the raw output actually shows.
3. **Common red flag**: a tidy 5-10 item list of well-known cofactors (EP300, CREBBP,
   TP53, JUN, FOS, etc.) that does not appear in the printed FOR loop output. The
   coding agent will sometimes guess at "biologically plausible" names instead of
   reading the actual computation result.

Write a 2-3 sentence summary of the key findings. Use ONLY entities that appear
in the raw output.

**Include exact numerical magnitudes for every quantitative claim.** When reporting
a "significant" effect, include the actual means AND effect size AND p-value in the
summary — not just the verbal framing. Write
  `"phyloP mean 0.051 (motif+) vs 0.025 (motif-), FC=2.01, p=4e-05"`
rather than
  `"significantly higher conservation"`.

The downstream hypothesis agent does NOT see the raw execution output — it only sees
your summary and findings. If you strip the magnitudes, the hypothesis agent will
amplify a statistically-significant-but-biologically-marginal effect into a strong
claim ("highly conserved", "strongly methylated"), which then drives the next
hypothesis down a wrong path. This already happened in a previous run: a phyloP
result of "0.05 vs 0.025 (both near-neutral, p=4e-05)" was summarized as
"significantly higher vertebrate conservation", which became "highly conserved
across vertebrates" in the next iteration's rationale. Don't let that happen.

**Subjective intensifiers are BANNED unless attached to a specific number.** Do not
write "highly", "strongly", "markedly", "substantially", "dramatically", "extensively",
"robustly" etc. without an accompanying numerical magnitude on the same line. Prefer
neutral phrasing like "FC=1.51 (moderate increase)" or "r=0.15 (weak positive)" when
the number itself isn't self-explanatory.

Each bullet in `findings` must include the relevant numbers: sample sizes (n), means
or medians for each group, effect size (FC, Cohen's d, OR, or Spearman r), and the
p-value. A finding without numbers is not a finding.

## Verification step coverage (CRITICAL — do not selectively report)

The verification plan above lists every step the coding agent was supposed to run.
For EACH step in the plan, your `findings` MUST include at least one bullet that
reports its result, even if the result is negative, null, or inconvenient for the
verdict you would prefer.

- If the step ran a tool (STRING, FIMO, GO enrichment, bedtools, bigWig, etc.) and
  found nothing, report it: "STRING direct ATF6-REST interaction: NOT FOUND
  (combined_score=0)" or "STRING shared partners (score>=400): HDAC1 (994), CREB1
  (731), JUN (655), TP53 (718), PPARG (577), SP1 (428)".
- If the step ran but the prediction it tested was falsified, report the falsifying
  number AND lower the support_level accordingly. A SUPPORTS verdict that omits a
  failed prediction is **selective reporting** and is forbidden.
- If the raw output contains a result for a verification step, but you do NOT
  include it in `findings`, you are stripping evidence. The downstream agents will
  never see it. Treat this as a hard rule: every executed step must produce a
  finding bullet.
- Named third-party molecules from STRING / co-IP / shared-partner queries MUST be
  surfaced verbatim in the findings, with their scores. These are downstream leads
  that the convergence check needs to see (see "Untested molecular lead" rule).

Respond in JSON:
{{
    "support_level": "SUPPORTS" | "REJECTS" | "INCONCLUSIVE",
    "confidence": 0.0 to 1.0,
    "summary": "2-3 sentence summary with the key numbers inline",
    "findings": ["each finding must include n, means, effect size, and p-value"],
    "reasoning": "explanation of how each prediction was evaluated, citing numbers",
    "interpretation_ceiling": "strongest claim these results justify; the simpler explanation still compatible",
    "control_override_applied": true | false,
    "simpler_supported_explanation": "REQUIRED when control_override_applied is true. The explanation the data DO support after downgrading the named mechanism. null otherwise."
}}"""

        try:
            evaluation = await self.llm.complete_json(
                [Message.user(prompt)],
                schema={
                    "type": "object",
                    "properties": {
                        "support_level": {"type": "string", "enum": ["SUPPORTS", "REJECTS", "INCONCLUSIVE"]},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "summary": {"type": "string"},
                        "findings": {"type": "array", "items": {"type": "string"}},
                        "reasoning": {"type": "string"},
                        "interpretation_ceiling": {"type": "string"},
                        "control_override_applied": {"type": "boolean"},
                        "simpler_supported_explanation": {"type": ["string", "null"]},
                    },
                    "required": ["support_level", "confidence", "summary", "findings", "reasoning"],
                },
            )

            # Deterministic fabrication check: extract gene-symbol-like tokens
            # from the LLM's summary/findings and verify they appear in raw_output.
            # Catches cases where the coding agent invented names AND the LLM
            # evaluator failed to notice.
            fabricated = _detect_fabricated_entities(
                summary_text=evaluation.get("summary", ""),
                findings=evaluation.get("findings", []),
                raw_output=raw_output,
            )
            if fabricated:
                self.logger.warning("Fabricated entities detected in evaluation", {
                    "fabricated_tokens": fabricated[:20],
                    "count": len(fabricated),
                })
                fabrication_note = (
                    f"[FABRICATION WARNING] {len(fabricated)} entity name(s) in this "
                    f"evaluation do not appear in the raw execution output: "
                    f"{', '.join(fabricated[:10])}"
                    f"{'...' if len(fabricated) > 10 else ''}. "
                    f"These were likely hardcoded by the coding agent rather than "
                    f"computed from data."
                )
                # Append warning to reasoning so it shows up in narrative/state
                existing_reasoning = evaluation.get("reasoning", "")
                evaluation["reasoning"] = existing_reasoning + "\n\n" + fabrication_note
                # Lower confidence — the entities cited cannot be trusted
                old_conf = evaluation.get("confidence", 0.5)
                evaluation["confidence"] = max(0.0, old_conf - 0.2)

            self.logger.info("Result evaluation complete", {
                "coding_agent_level": coding_result.get("support_level"),
                "summary_agent_level": evaluation.get("support_level"),
                "confidence": evaluation.get("confidence"),
                "fabricated_entities": len(fabricated),
                "reasoning": evaluation.get("reasoning", "")[:200],
            })

            return evaluation

        except Exception as e:
            self.logger.error("Result evaluation failed", {"error": str(e)})
            # On failure, fall back to coding agent's judgment
            return {
                "support_level": coding_result.get("support_level", "UNKNOWN"),
                "confidence": coding_result.get("confidence", 0.0),
                "summary": coding_result.get("summary", ""),
                "findings": coding_result.get("findings", []),
                "reasoning": f"Evaluation failed, using coding agent's judgment: {str(e)}",
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
        biology_gate_status: dict[str, Any] | None = None,
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
            used_biology_layers: Among tested non-QC hypotheses, which of those categories were used.
            biology_gate_status: Deterministic 5a/5b status computed by the orchestrator.
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
            "biology_gate_status": biology_gate_status,
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
            biology_gate_status=biology_gate_status,
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
        record says otherwise (REJECTS / ERROR / INCONCLUSIVE / UNTESTABLE),
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
