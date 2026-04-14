"""Prompt templates for result summarization."""

from __future__ import annotations

from typing import Any

SUMMARY_SYSTEM_PROMPT = """You are a scientific result interpreter specializing in bioinformatics. Your role is to:

1. Analyze experimental results and code outputs
2. Determine whether results support, refuse, or are inconclusive for a hypothesis
3. Extract key findings and statistics
4. Identify any issues or anomalies in the results
5. Prepare concise summaries for the hypothesis refinement process
6. When writing the final report, go beyond the mechanistic explanation (why the ML rule works) to speculate on the likely **biological purpose** of that rule: why the cell might use it and what function or selective advantage it could provide. This is interpretation, not a finding — phrase it explicitly as speculation (e.g., "may serve to…", "consistent with a role in…", "one possibility is that…"). Do NOT state biological purpose as established fact.

Be objective and precise in your interpretations.

=== SUPPORT LEVEL CRITERIA (you MUST follow these rules) ===

Evaluate support_level by comparing the results against the hypothesis's **prediction**
field. The prediction states what should be observed if the hypothesis is true — your
job is to check whether that specific prediction was confirmed.

**SUPPORTS** — ALL of the following must be true:
  1. The predicted effect exists in the data
  2. p < 0.05
  3. fold change >= 1.2 OR Cohen's d >= 0.3
  If all three are met, set support_level = "SUPPORTS" and confidence >= 0.8.
  Note: ChIP-seq and epigenomic enrichment analyses routinely produce 1.2–1.5× fold
  changes that are biologically meaningful and reproducible. Do not require 1.5× as
  a hard threshold for these data types.
  For co-occupancy: if fold enrichment over background is ≥10-fold and p < 0.05, the
  prediction is SUPPORTED even if the absolute overlap percentage is below the hypothesis
  threshold — low absolute percentage with high fold enrichment means the co-occupancy is
  real but both TFs cover a small fraction of the genome.

**INCONCLUSIVE** — The effect is real but weak:
  1. p < 0.05
  2. 1.1 <= fold change < 1.2 OR 0.15 <= Cohen's d < 0.3
  Set support_level = "INCONCLUSIVE" and confidence 0.4-0.7.

**REJECTS** — ANY of the following:
  1. The predicted effect is absent or reversed (e.g., depletion instead of enrichment)
  2. p >= 0.05 with adequate sample size (N >= 30)
  3. fold change < 1.1 AND Cohen's d < 0.15
  If clearly refused, set support_level = "REJECTS" and confidence >= 0.7.

  **STRING exception**: When a hypothesis is tested solely via STRING and no interaction is
  found, set REJECTS but with confidence 0.5-0.7 (not 1.0). STRING is biased toward
  well-studied proteins — absence of a STRING edge lowers the prior on PPI but does not
  definitively disprove it. If other genomic evidence supports PPI (co-occupancy, signal
  correlation), note this in reasoning.

**ERROR** — Technical failure prevented analysis (code crashed, wrong file format, etc.)

**CONTROL OVERRIDE** — If the raw output contains a control or comparator that tests
whether the mechanism-specific component adds explanatory power beyond a simpler
baseline explanation, and the mechanism-specific group is NOT detectably stronger than
the simpler baseline, the named mechanism is NOT supported. Examples:
- anchor+YY1 not stronger than non-anchor+YY1 → the anchor/3D component is not
  supported; only the YY1-associated explanation remains.
- motif-positive and motif-negative groups differ, but the same effect persists after
  removing the proposed mediator → the mediator-specific mechanism is not supported.
In these cases, set support_level to REJECTS or INCONCLUSIVE (not SUPPORTS) for the
named mechanism, and fill simpler_supported_explanation with the explanation that the
data DO support.

**Interpretation ceiling** — Before setting support_level, answer two questions:
1. What is the strongest claim these results justify?
2. What simpler explanation is still compatible with the control results?
Your final wording must not exceed that ceiling.

IMPORTANT: Apply these criteria to the hypothesis's stated prediction, not to a
reframed version of the question. If the prediction says "X is enriched in group A
compared to group B", evaluate whether that specific enrichment exists — do not
reinterpret it as "X exists at all" or "X explains all of the association."

IMPORTANT: When support_level is SUPPORTS with confidence >= 0.8, do NOT include
phrases like "additional analyses are required", "more controls needed", "further
validation required", or "should be addressed in follow-up" in your summary. These
phrases cause the pipeline to keep iterating unnecessarily. If you want to note
limitations, state them as observations (e.g., "GC content was not controlled for")
rather than action items (e.g., "GC content controls should be added").
"""


CONVERGENCE_CHECK_SYSTEM_PROMPT = """You are an independent scientific adjudicator. Decide whether the accumulated evidence is sufficient to converge on a named mechanism. You have no stake in the outcome — be skeptical.

## Modes
- **Causal-capable** (Perturb-seq, time-series, knockdown): apply ALL 5 criteria including #3 (but-for).
- **Observational** (ChIP-seq, bulk RNA-seq, epigenomics): apply criteria 1, 2, 4, 5. Skip #3. The conclusion MUST state "best-supported interpretation given observational data; causality not established."

## Apply criteria in order. Stop at the first failure.

**Special case — artifact-check SUPPORTS**: If the iteration-0 artifact check returned SUPPORTS (TF_A not expressed AND motif not enriched), converge IMMEDIATELY on "technical artifact". Skip criteria 1-5.

**1. Mechanism SUPPORTS prerequisite**
At least one tested hypothesis must have `support_level = "SUPPORTS"` AND test a specific molecular mechanism (PPI, motif, chromatin, 3D, sequence-intrinsic, motif grammar, cofactor proxy, etc.).

The following do NOT count as mechanism SUPPORTS — if these are the only SUPPORTS results, set converged=false:
- Iteration-0 artifact check REJECTS (it's a QC pass, not a mechanism)
- Functional-characterization hypotheses (GO enrichment, RNA-seq target analysis, phyloP conservation, or any hypothesis whose group is `functional-characterization`). GO enrichment is near-guaranteed to find SOME enriched term — it satisfies 5b but never the prerequisite. If GO is the only SUPPORTS in the history, set converged=false.
- Promising sub-findings inside a REJECTS or INCONCLUSIVE hypothesis. The overall verdict must be SUPPORTS.

**2. Mechanism quality** — the supported claim must not reduce to co-occurrence
The mechanism must name a specific molecular event AND the test must distinguish it from "both bind active chromatin / shared regulatory context." Reject convergence if:
- The claim is "TF_A signal is higher at co-bound sites" with no control for chromatin context, accessibility, or shared partners. This is co-occurrence, not a mechanism — even if the hypothesis name adds words like "modulates", "enhances", or "stabilizes".
- The claim is "TF_B motif is enriched / score correlates with TF_A signal" with no localization test (motif-center-to-summit distance, central enrichment vs flanks). This re-validates the ML association, not the mechanism.
- The claim is "direct binding" without a peak-centered localization signal. Without summit-centered motif positioning, the strongest defensible claim is "sequence-intrinsic association" or "motif-associated occupancy."
- A simpler explanation (YY1 alone, enhancer context, GC content, accessibility) is fully consistent with the same data. Use cautious wording instead and set INCONCLUSIVE-mode at the convergence level.

**3. But-for test** (causal-capable mode only)
Ask: "If the proposed causal agent were absent, would the data look different?" If the answer is "not necessarily" (could reflect passive co-occupancy or a confound), do not converge. Skip in observational mode.

**4. Cross-layer consistency** — two independent MEASUREMENT modalities
The mechanism must be supported by directional evidence from two independent measurements (ChIP-seq + DNase-seq, ChIP-seq + RNA-seq, ChIP-seq + phyloP, ChIP-seq + Hi-C, motif + histone marks, etc.).

**The following are NOT independent layers** — they are derived from the same data:
- ChIP-seq peak overlap + GO enrichment of those peaks → ONE layer (ChIP-seq), not two. GO is a downstream annotation.
- Peak overlap + motif enrichment in those same peaks → ONE layer.
- Multiple statistics on the same ChIP-seq data → ONE layer.
A run with only "peak overlap + motif scan + GO" satisfies cross-layer consistency for ZERO layers beyond ChIP-seq. Set converged=false unless you can name a truly independent measurement.

**5. Biology layers — both 5a and 5b required**
- **5a. STRING/PPI**: at least one hypothesis must have queried STRING (any result). STRING is always accessible; not optional.
- **5b. Functional characterization**: if the manifest has rnaseq or phyloP, at least one hypothesis must have run GO/pathway enrichment, RNA-seq target analysis, or phyloP conservation. (GO satisfies 5b but never criterion 1 — see prerequisite above.)
- The iter-0 artifact check's RNA-seq usage (TPM lookup) does NOT count toward 5b.

## Untested molecular lead block
If any SUPPORTS result explicitly names a third-party molecule (cofactor, shared interactor, chromatin mark) that was identified but NOT independently tested in a follow-up hypothesis, do NOT converge. Example: STRING finds TRIM28 as a shared partner → the pipeline must test TRIM28 occupancy at co-bound sites before converging on a cofactor-mediated mechanism.

## Output
Set `mechanism_category_number` to null (the taxonomy is reference only — never force a result into a numbered bucket). Put a specific descriptive name in `mechanism_category_name`. The mechanism can be anything that fits the evidence.
"""


def _truncate_output(text: str, max_chars: int = 50000) -> str:
    """Truncate large output while preserving useful information.

    Keeps the beginning (usually setup/imports) and end (usually results/conclusions).

    Args:
        text: The text to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated text with indicator if truncation occurred
    """
    if len(text) <= max_chars:
        return text

    # Keep first 30% and last 70% (results are usually at the end)
    head_chars = int(max_chars * 0.3)
    tail_chars = int(max_chars * 0.7)

    head = text[:head_chars]
    tail = text[-tail_chars:]

    truncated_chars = len(text) - max_chars
    return (
        f"{head}\n\n"
        f"[... TRUNCATED {truncated_chars:,} characters ({truncated_chars // 4:,} tokens approx) ...]\n"
        f"[... Showing last {tail_chars:,} characters which typically contain results ...]\n\n"
        f"{tail}"
    )


def build_convergence_check_prompt(
    finding: str,
    tested_hypotheses: list[dict[str, Any]],
    last_result: dict[str, Any],
    raw_output: str | None = None,
    max_raw_chars: int = 8000,
    investigation_objective: str | None = None,
    available_biology_layers: list[str] | None = None,
    used_biology_layers: list[str] | None = None,
    biology_gate_status: dict[str, Any] | None = None,
    causal_capable_data: bool = False,
) -> str:
    """Build prompt for the independent convergence check.

    Args:
        finding: Original scientific finding
        tested_hypotheses: All hypotheses tested so far with results
        last_result: Summary of the most recent result
        raw_output: Raw stdout from the last code execution (will be truncated)
        max_raw_chars: Maximum characters of raw output to include
        investigation_objective: Optional goal (discovery/synthesis); when set, convergence on a new named mechanism is acceptable.
        available_biology_layers: Top-level manifest data keys that provide expression/conservation (e.g. rnaseq, phyloP) when present.
        used_biology_layers: Among tested non-QC hypotheses, which of those layers were used (from required_data).
        biology_gate_status: Deterministic 5a/5b status computed by the orchestrator.
        causal_capable_data: When True, data can support causal inference (e.g. Perturb-seq, time-series); require but-for test. When False, observational only; allow convergence on best-supported mechanism without causality.

    Returns:
        Formatted prompt string
    """
    if causal_capable_data:
        modality_section = (
            "\n## Data modality\n\n"
            "This run includes **causal-capable data** (e.g. perturbation, time-series). "
            "Apply ALL FIVE convergence criteria, including the **but-for test** (criterion 3). "
            "Do not converge unless the evidence supports a causal mechanism.\n"
        )
        criteria_instruction = "Apply ALL FIVE convergence criteria strictly (including but-for / causality)."
    else:
        modality_section = (
            "\n## Data modality\n\n"
            "This run uses **observational data only** (e.g. ChIP-seq, bulk RNA-seq, epigenomics). "
            "No perturbation or time-series data are present, so the data **cannot establish causality**. "
            "Use **OBSERVATIONAL** convergence: do NOT require the but-for test. "
            "Converge when criteria 1, 2, 4, and 5 are met and the evidence supports the **best-supported mechanistic interpretation** given the data (e.g. co-occupancy, promoter platform, retention). "
            "When converged=true, the conclusion MUST state that this is the best-supported interpretation given observational data and that causality is not established.\n"
        )
        criteria_instruction = (
            "Apply criteria 1, 2, 4, and 5. Do NOT require the but-for test (criterion 3). "
            "Converge on the best-supported mechanism given the data; state in the conclusion that causality is not established."
        )

    objective_section = ""
    if investigation_objective and investigation_objective.strip():
        objective_section = f"\n## Investigation Objective\n\n{investigation_objective.strip()}\n"
    history_lines = []
    for i, th in enumerate(tested_hypotheses):
        data_used = ", ".join(th.get("required_data", [])) or "not specified"
        history_lines.append(
            f"### Iteration {i + 1}: {th.get('name', 'N/A')}\n"
            f"- Result: {th.get('result', 'N/A')}\n"
            f"- Data used: {data_used}\n"
            f"- Evidence: {str(th.get('evidence_summary', 'N/A'))[:300]}\n"
            f"- Decision: {th.get('decision', 'N/A')}\n"
        )
    history_text = "\n".join(history_lines) if history_lines else "No hypotheses tested yet."

    raw_section = ""
    if raw_output:
        truncated = _truncate_output(raw_output, max_raw_chars)
        raw_section = f"\n## Raw Execution Output (last experiment)\n\n```\n{truncated}\n```\n"

    biology_section = ""
    if available_biology_layers is not None and used_biology_layers is not None:
        manifest_avail = [k for k in (available_biology_layers or []) if k != "string"]
        avail_str = ", ".join(manifest_avail) if manifest_avail else "none"
        used = ", ".join(used_biology_layers) if used_biology_layers else "none"
        gate = biology_gate_status or {}
        string_checked = gate.get("string_attempted", "string" in (used_biology_layers or []))
        functional_5b_attempted = gate.get("functional_5b_attempted", False)
        functional_5b_sources = gate.get("functional_5b_sources", [])
        qc_excluded = gate.get("qc_excluded_layers", [])
        functional_sources_text = ", ".join(functional_5b_sources) if functional_5b_sources else "none"
        qc_excluded_text = ", ".join(qc_excluded) if qc_excluded else "none"
        biology_section = (
            "\n## Biology layers (expression / conservation / PPI)\n\n"
            f"- **STRING/PPI** (always required): {'✓ checked' if string_checked else '✗ NOT yet checked'}\n"
            f"- **Expression/conservation layers available in manifest**: {avail_str}\n"
            f"- **Biology layers used in any hypothesis (any result)**: {used}\n\n"
            "### Deterministic criterion-5 status (authoritative)\n"
            f"- **5a satisfied outside QC**: {'✓ yes' if string_checked else '✗ no'}\n"
            f"- **5b satisfied outside QC**: {'✓ yes' if functional_5b_attempted else '✗ no'}\n"
            f"- **Non-QC hypotheses that count toward 5b**: {functional_sources_text}\n"
            f"- **QC-only layers explicitly excluded from 5b**: {qc_excluded_text}\n\n"
            "Treat the deterministic 5a/5b status above as authoritative. Do NOT infer that "
            "iteration-0 artifact-check TPM lookup or motif QC satisfies 5b, even if the "
            "history text mentions RNA-seq.\n"
            "Criterion 5a: STRING must be checked in at least one hypothesis (any result). "
            "If STRING was never attempted, set converged=false.\n"
            "Criterion 5b: If rnaseq or phyloP is listed above as available, at least one hypothesis must have "
            "addressed functional relevance via rnaseq, phyloP, OR GO/pathway enrichment (any result). "
            "Verify from the evidence summaries that the analysis was actually performed — a layer listed in "
            "required_data but absent from the evidence summary does NOT satisfy 5b. "
            "IMPORTANT: Using RNA-seq only to check TF expression (TPM) in the artifact check does NOT satisfy 5b. "
            "5b requires linking the mechanism to gene expression, conservation, or pathway function. "
            "Set converged=false if none of rnaseq/phyloP/GO was genuinely analyzed for functional relevance.\n"
        )

    prompt = f"""# Scientific Finding Under Investigation

{finding}
{objective_section}
{modality_section}
## Hypothesis Testing History

{history_text}
{biology_section}
## Latest Result Summary

**Support level**: {last_result.get('support_level', 'N/A')}
**Confidence**: {last_result.get('confidence', 'N/A')}
**Summary**: {last_result.get('summary', 'N/A')}
**Findings**: {last_result.get('findings', [])}
{raw_section}
## Task

Evaluate whether the accumulated evidence is sufficient to declare convergence on a named mechanism (or best-supported interpretation in observational mode).

{criteria_instruction}

PREREQUISITE: At least one hypothesis must have support_level = "SUPPORTS". If none do, set converged=false immediately.
1. Statistical support: at least one hypothesis with support_level = "SUPPORTS" (not just promising numbers inside a REJECTS result)
2. Named mechanism: a specific molecular process with a clear causal/mechanistic interpretation, not merely correlation. The mechanism can be anything that fits the evidence—it need not match any predefined category. Set mechanism_category_number to null and put a descriptive name in mechanism_category_name.
3. But-for test (only when causal-capable data: causal, not merely correlational)
4. Cross-layer consistency (consistent directional support from >= 2 independent omics layers)
5. Biology layers — two independent sub-requirements (BOTH must be met):
   5a. STRING/PPI (always required): at least one hypothesis must have tested STRING (any result). Always required regardless of manifest.
   5b. Functional characterization (when available): if rnaseq or phyloP is in the manifest, at least one hypothesis must have addressed functional relevance via rnaseq, phyloP, OR GO/pathway enrichment analysis (any result). STRING does NOT substitute. Checking TF expression (TPM) in an artifact check does NOT count — 5b requires linking the mechanism to gene expression levels, conservation scores, or pathway function.

When converged=true, always fill mechanism_category_name with a descriptive mechanism name. Set mechanism_category_number to null (the taxonomy is for reference only). In observational mode, the conclusion must state that causality is not established.

Respond in JSON format:
{{
    "converged": true | false,
    "mechanism_category_number": null or integer 1-12 (use null unless the mechanism clearly matches a taxonomy example),
    "mechanism_category_name": "<string; when converged, provide a descriptive mechanism name>",
    "confidence": 0.0-1.0,
    "conclusion": "<one-paragraph mechanistic interpretation if converged, else empty string. For observational data: use association language only ('co-occupancy data suggest…', 'consistent with…', 'associated with…'). Forbidden: 'proves', 'demonstrates', 'drives', 'causes', 'enables'. Must state that causality is not established from observational data.>",
    "reasoning": "<explanation of why each criterion is or is not met>"
}}
"""
    return prompt


def build_result_summary_prompt(
    hypothesis: dict[str, Any],
    code: str,
    execution_result: str,
    outputs: list[dict[str, Any]] | None = None,
    max_output_chars: int = 50000,
) -> str:
    """Build prompt for summarizing execution results.

    Args:
        hypothesis: The hypothesis being tested
        code: The code that was executed
        execution_result: Combined stdout/stderr from execution
        outputs: Additional outputs (data frames, etc.)
        max_output_chars: Maximum characters of execution output to include

    Returns:
        Formatted prompt string
    """
    # Truncate execution result if too large
    truncated_result = _truncate_output(execution_result, max_output_chars)

    outputs_section = ""
    if outputs:
        outputs_section = "\n## Additional Outputs\n"
        for i, out in enumerate(outputs):
            if "text/plain" in out:
                outputs_section += f"\n### Output {i+1}\n```\n{out['text/plain'][:2000]}\n```\n"

    prompt = f"""# Hypothesis Being Tested

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}
**Rationale**: {hypothesis.get('rationale', 'N/A')}

# Code Executed

```python
{code}
```

# Execution Output

```
{truncated_result}
```
{outputs_section}

# Task

Analyze the execution results and provide a summary. Determine:

1. **Success**: Did the code execute successfully?
2. **Findings**: What were the key numerical results?
3. **Support Level**: Does this support, refuse, or is inconclusive for the hypothesis?
4. **Confidence**: How confident are you in this interpretation?
5. **Issues**: Were there any data quality issues or anomalies?

Respond in JSON format:
{{
    "execution_success": true | false,
    "findings": [
        "Key finding 1",
        "Key finding 2",
        ...
    ],
    "statistics": {{
        "relevant_stat_name": value,
        ...
    }},
    "support_level": "SUPPORTS" | "REJECTS" | "INCONCLUSIVE" | "ERROR",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of the interpretation",
    "simpler_supported_explanation": "REQUIRED when control override fires (mechanism-specific component not supported but a simpler explanation holds). null otherwise.",
    "issues": ["Any issues or concerns"],
    "summary": "One paragraph summary suitable for the hypothesis model"
}}
"""
    return prompt


def build_final_report_prompt(
    finding: str,
    context: str,
    history_summary: str,
    conclusion: str,
    all_evidence: list[dict[str, Any]],
    converged: bool,
    run_status: str | None = None,
    synthesized: bool = False,
) -> str:
    """Build prompt for generating the final report.

    Args:
        finding: Original scientific finding
        context: Research context
        history_summary: Summary of all iterations
        conclusion: Final conclusion
        all_evidence: All accumulated evidence
        converged: Whether the run strictly converged
        run_status: Terminal run status
        synthesized: Whether the run stopped via synthesis

    Returns:
        Formatted prompt string
    """
    evidence_text = ""
    for i, e in enumerate(all_evidence):
        evidence_text += f"""
### Iteration {e.get('iteration', i+1)}
- **Hypothesis**: {e.get('hypothesis', 'N/A')}
- **Findings**: {e.get('findings', 'N/A')}
- **Support Level**: {e.get('support_level', 'N/A')}
"""

    prompt = f"""# Final Report Generation

## Original Finding

{finding}

## Context

{context}

## Investigation Summary

{history_summary}

## Evidence Collected
{evidence_text}

## Conclusion

{conclusion}

## Run Status

- converged: {converged}
- run_status: {run_status or "unknown"}
- synthesized: {synthesized}

# Task

Generate a comprehensive final report summarizing the investigation. The report should include:

1. **Executive Summary**: Brief overview of the finding and conclusion
2. **Methodology**: How the investigation was conducted
3. **Key Findings**: Most important discoveries
4. **Evidence Summary**: Supporting evidence for the conclusion
5. **Synthesis / mechanism comparison**: If multiple hypotheses were tested, provide a short integrative narrative: which mechanism(s) are best supported, which were ruled out, and how they relate (e.g. a small comparison table). The mechanism can be anything that fits the evidence—it need not match any predefined category. Prefer a concise comparison table (e.g. mechanism vs evidence vs outcome) when there are several hypotheses.
6. **Biological purpose (speculative)**: Go one step beyond the mechanism: speculate on the **biological purpose** (functional interpretation) of the supported rule. Why might the cell use this rule? What selective or functional advantage could it provide? Consider e.g. precise transcriptional control, cell-type or developmental identity, stress/dynamic response, promoter robustness, or other plausible rationales.

   **LANGUAGE RULES for this section (strictly enforced)**:
   - Every sentence must use hedged phrasing. Required starters: "may serve to…", "consistent with a role in…", "one possibility is that…", "this arrangement could allow…", "suggests a model in which…".
   - **Forbidden causal words**: "proves", "demonstrates", "establishes", "drives", "causes", "enables" (unless preceded by "may" or "could"), "the mechanism is", "TF_X acts as a [role]" stated as fact.
   - At the end of this section, add one sentence explicitly stating: "These functional interpretations are speculative; the available data are observational and do not establish causality."
   - Do not invent evidence; base this only on the supported mechanism and the known biological roles of the molecules involved.
7. **Confidence Level**: How confident we are in the conclusion
8. **Limitations**: What we couldn't determine or potential issues
9. **Recommendations**: Suggested follow-up experiments if any

**CRITICAL — causal language**: This run uses observational data (ChIP-seq, bulk RNA-seq, epigenomics). Observational data show **association**, not causation. Throughout the entire report:
- Use: "co-occupancy data suggest…", "consistent with…", "associated with…", "the best-supported interpretation is…", "the evidence is consistent with a model in which…"
- **Forbidden**: "proves", "demonstrates", "establishes", "X drives Y", "X causes Y", "X enables Y" (unless qualified with "may" or "could"), "the mechanism is" stated as settled fact.
- In the Executive Summary and Conclusion sections, add one explicit sentence: "Because these data are observational, the proposed mechanism represents the best-supported interpretation and does not establish causality."

**CRITICAL — outcome accuracy**: When describing whether a hypothesis was supported or
refused, you MUST use the exact **Support Level** recorded in the "Evidence Collected"
section above (SUPPORTS, REJECTS, INCONCLUSIVE, ERROR, UNTESTABLE). Do NOT upgrade a
REJECTS/ERROR/INCONCLUSIVE result to "supported" or "confirmed" in the narrative. If no
hypothesis achieved SUPPORTS, state that clearly. Misrepresenting outcomes is the single
most harmful error this report can contain.

**CRITICAL — run status accuracy**: The final report MUST respect the run status above.
If `converged` is false, do NOT present the outcome as settled or fully validated. Use
language like "best-supported interpretation so far", "non-converged run", or
"provisional conclusion". If `run_status` is `stopped` or `synthesized_stop`, say that
explicitly in the Executive Summary and Confidence sections.

Format the report in Markdown.
"""
    return prompt
