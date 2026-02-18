"""Prompt templates for hypothesis generation."""

from __future__ import annotations

from typing import Any

HYPOTHESIS_SYSTEM_PROMPT = """You are a scientific hypothesis generation assistant specializing in bioinformatics. You investigate findings of the form: "ML model says TF_B's motif predicts TF_A's binding — why?"

## Investigation Framework

Follow this phased approach. Each phase builds on the previous one's results.

**Phase 1 — Confirm the association.**
Is the co-occurrence real? Measure enrichment of TF_B motif or peaks in TF_A peaks vs shuffled controls. The magnitude matters: 2x enrichment vs 28x enrichment imply fundamentally different explanations.

**Phase 2 — Sequence vs protein.**
Is the ML model detecting the DNA sequence pattern, or actual TF_B protein binding nearby?
Scan TF_A peaks for TF_B motif occurrences (FIMO), independently of TF_B ChIP-seq signal.
If TF_B motif appears in TF_A peaks that lack TF_B protein signal → sequence effect.
If TF_B protein is consistently co-bound → protein-mediated effect.

**Phase 3 — Identify the mechanism.** This depends on Phase 2 results:
- If SEQUENCE effect: check motif nesting/containment (does TF_B's longer motif contain TF_A's core binding sequence as a substring?), shared sequence features (GC content, repeat elements)
- If PROTEIN effect: spatial analysis (distance distribution between peaks — bimodal suggests two modes of interaction), chromatin state analysis (DNase-seq, histone marks at co-bound vs solo sites), functional annotation (promoter vs enhancer, gene expression differences)

**Phase 4 — Test competing mechanisms.**
Whatever the leading explanation, test at least one alternative to rule it out. This is required for convergence.

## Key Principles

- Each result should directly motivate the next hypothesis. Surprising or contradictory results are the most valuable clues — lean into them, don't treat them as failures.
- Stratification from early results (e.g., proximal vs distal groups) should carry forward into later hypotheses.
- The final conclusion must synthesize ALL results (including rejected hypotheses), not just the one that was supported.
"""


def build_initial_hypothesis_prompt(
    finding: str,
    context: str,
    data_manifest: dict[str, Any],
) -> str:
    """Build the initial prompt for hypothesis generation.

    Args:
        finding: The scientific finding to explain (X predicts Y)
        context: Additional context about the research
        data_manifest: Available data and tools

    Returns:
        Formatted prompt string
    """
    # Format available data
    data_section = _format_data_manifest(data_manifest)

    prompt = f"""# Scientific Finding to Investigate

{finding}

# Context

{context}

# Available Data and Tools

{data_section}

# Task

Generate hypotheses for **Phase 1 and Phase 2 only** (see the investigation framework in
your system instructions). Do NOT generate mechanism hypotheses (Phase 3+) yet — those
depend on what Phases 1-2 reveal and will be generated later.

Phase 1 hypotheses confirm the association: is TF_B motif/binding enriched in TF_A peaks?
Phase 2 hypotheses discriminate: is it a sequence pattern effect or actual protein co-binding?

**Computational constraint**: Prefer analytical tests (Fisher's exact, Mann-Whitney) over
permutations. If a permutation test is truly needed, specify at most 100 replicates.

Each hypothesis must have exactly ONE prediction tested by exactly ONE statistical test.
If a phase has multiple testable aspects, split them into separate hypotheses and give
them all the same `group` string so they are tested together.

**Prediction quality** — each prediction must specify:
- What metric is measured
- The comparison (group A vs group B)
- The expected direction and approximate magnitude
- The statistical test

Examples:
- GOOD: "TF_A peaks will overlap TF_B peaks at >= 5x the rate expected by chance
  (bedtools shuffle, 100 permutations, Fisher's exact p < 0.05)"
- BAD: "TF_B motif will be found in TF_A peaks" (no comparison, no magnitude)
- BAD: "TF_B peaks are closer to TSS AND have higher GC%" (two tests — split)

For each hypothesis, provide:
1. **Hypothesis Name**: A brief descriptive name
2. **Rationale**: The biological reasoning behind this hypothesis
3. **Prediction**: Quantitative, falsifiable (see examples above)
4. **Verification Plan**: Steps leading to ONE statistical test

Respond in JSON format:
{{
    "hypotheses": [
        {{
            "name": "Hypothesis name",
            "group": "mechanism-slug",
            "rationale": "Biological reasoning",
            "prediction": "Quantitative, falsifiable prediction with metric, comparison, and expected direction/magnitude",
            "verification_plan": ["Step 1", "Step 2", ...],
            "priority": 1,
            "required_data": ["data_key_1", "data_key_2"]
        }},
        ...
    ],
    "recommended_first": 0,
    "reasoning": "Why this hypothesis should be tested first"
}}

Note on `group`: Hypotheses that are sub-parts of the same broad mechanism MUST share the
same `group` string (a short kebab-case slug, e.g. "promoter-mechanism"). The pipeline will
test all hypotheses in a group before moving on. Hypotheses exploring independent mechanisms
should have different group values.
"""
    return prompt


def build_refinement_prompt(
    finding: str,
    history_summary: str,
    last_hypothesis: dict[str, Any],
    last_result: dict[str, Any],
    data_manifest: dict[str, Any],
    group_summary: str | None = None,
) -> str:
    """Build prompt for refining hypotheses based on results.

    Args:
        finding: Original scientific finding
        history_summary: Summary of all previous iterations
        last_hypothesis: The most recently tested hypothesis
        last_result: Results from testing the last hypothesis
        data_manifest: Available data and tools
        group_summary: Summary of a completed hypothesis group for synthesis

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest)

    prompt = f"""# Scientific Finding Under Investigation

{finding}

# Previous Iterations

{history_summary}

# Latest Hypothesis Tested

**Name**: {last_hypothesis.get('name', 'N/A')}
**Rationale**: {last_hypothesis.get('rationale', 'N/A')}
**Prediction**: {last_hypothesis.get('prediction', 'N/A')}

# Experimental Results

{last_result.get('summary', 'No summary available')}

**Evidence**:
{last_result.get('evidence', 'No evidence recorded')}

**Interpretation**:
{last_result.get('interpretation', 'No interpretation')}

# Available Data and Tools

{data_section}

{"" if not group_summary else group_summary}

# Task

Based on these results, decide how to proceed.

INVESTIGATION PHASE GUIDANCE:

Refer to the investigation framework (Phase 1 → 2 → 3 → 4) in your system instructions.
- If Phase 1 (association confirmation) just completed: move to Phase 2 (sequence vs protein).
- If Phase 2 just completed: move to Phase 3 (mechanism identification). Choose the
  mechanism path based on what Phase 2 revealed — sequence effect vs protein effect.
- If Phase 3 just completed: move to Phase 4 (test a competing mechanism).
- If Phase 4 just completed: you may have enough to converge.

**"Use the surprise" rule**: Your next hypothesis MUST be directly motivated by the specific
result from the latest experiment. State explicitly: (1) what the last result revealed,
(2) what question it opens up, (3) how your new hypothesis addresses that question.
Do NOT generate a hypothesis unrelated to what was just observed.

CONVERGENCE GUIDELINES (read carefully before deciding):

You may choose CONVERGED only when ALL three conditions are met:

Key distinction — a **mechanism** hypothesis explains WHY the finding exists (a causal
process), not just THAT it exists (an association). For example, "REST motif contains
ATF6's core binding sequence" is a mechanism; "ATF6 and REST peaks co-occur" is an
observation. Your conclusion must be a mechanism.

1. A mechanism hypothesis has statistical support (p < 0.05, clear effect size).
   An observation (e.g., "X and Y co-occur") does NOT count.
2. At least one alternative mechanism hypothesis has been tested and ruled out.
   Ruling out an observation-level hypothesis does NOT satisfy this condition.
3. Your conclusion explains WHY the finding exists. If multiple mechanisms are
   independently supported, describe how they contribute together.

If condition 1 is met but conditions 2 or 3 are not, use NEW_HYPOTHESIS to test
alternative mechanisms or explore WHY the association exists.

Do NOT choose REFINE just to add more permutations, stricter matching, or additional
control analyses on the same hypothesis. A good conclusion acknowledges limitations
without requiring them to be resolved first. REFINE is for when results are ambiguous
or partially contradictory, NOT for when results are clear but could theoretically be
made more rigorous.

Decisions:

1. **CONVERGED**: All three conditions above are met (mechanism supported, alternative mechanism ruled out, conclusion explains WHY)
2. **REFINE**: If results partially support the hypothesis but need refinement
3. **NEW_HYPOTHESIS**: If results refute the hypothesis and a new one is needed
4. **TECHNICAL_ERROR**: If the code had bugs, parsing errors, or technical failures that prevented proper analysis. This is NOT convergence - we need to fix the code and retry.
5. **INSUFFICIENT_DATA**: If the biological data truly cannot test the hypotheses (missing data files, wrong data type, etc.)

IMPORTANT: Do NOT choose CONVERGED or INSUFFICIENT_DATA if there were technical errors in the code execution. Technical failures (parsing errors, wrong column names, file format issues, etc.) should be marked as TECHNICAL_ERROR so the code can be fixed and re-run.

Respond in JSON format:
{{
    "decision": "CONVERGED" | "REFINE" | "NEW_HYPOTHESIS" | "TECHNICAL_ERROR" | "INSUFFICIENT_DATA",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation for the decision",
    "technical_issues": ["List of specific technical issues to fix, if TECHNICAL_ERROR"],
    "hypotheses": [
        // ONLY include the ONE hypothesis to test next. Do NOT repeat previously tested
        // hypotheses or list all candidates. Return exactly one hypothesis here.
        {{
            "name": "Hypothesis name",
            "group": "mechanism-slug",
            "rationale": "Biological reasoning",
            "prediction": "Quantitative, falsifiable prediction with metric, comparison, and expected direction/magnitude",
            "verification_plan": ["Step 1", "Step 2", ...],
            "priority": 1,
            "required_data": ["data_key_1", "data_key_2"]
        }}
    ],
    "conclusion": "Final conclusion if CONVERGED or INSUFFICIENT_DATA"
}}
"""
    return prompt


def build_convergence_check_prompt(
    finding: str,
    history_summary: str,
    total_evidence: list[dict[str, Any]],
) -> str:
    """Build prompt to check if we should stop iterating.

    Args:
        finding: Original scientific finding
        history_summary: Summary of all iterations
        total_evidence: All accumulated evidence

    Returns:
        Formatted prompt string
    """
    evidence_text = "\n".join([
        f"- Iteration {e.get('iteration', '?')}: {e.get('summary', 'N/A')}"
        for e in total_evidence
    ])

    prompt = f"""# Scientific Finding

{finding}

# Investigation History

{history_summary}

# Accumulated Evidence

{evidence_text}

# Task

Evaluate whether we have reached a satisfactory conclusion or should continue investigating.

Consider:
1. Is there a consistent pattern in the evidence?
2. Have we tested the most plausible hypotheses?
3. Is additional iteration likely to yield new insights?

Respond in JSON format:
{{
    "should_continue": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "Explanation for the decision",
    "conclusion": "Current best explanation for the finding"
}}
"""
    return prompt


def build_regeneration_prompt(
    finding: str,
    rejected_hypothesis: dict[str, Any],
    feedback: str,
    tested_hypotheses: list[dict[str, Any]],
    data_manifest: dict[str, Any],
) -> str:
    """Build prompt for regenerating a rejected hypothesis.

    Args:
        finding: Original scientific finding
        rejected_hypothesis: The hypothesis that was rejected by review
        feedback: Reviewer's feedback on why it was rejected
        tested_hypotheses: All previously tested hypotheses
        data_manifest: Available data and tools

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest)

    tested_section = ""
    if tested_hypotheses:
        parts = []
        for t in tested_hypotheses:
            parts.append(f"- **{t.get('name', '?')}**: {t.get('prediction', '?')} → {t.get('result', '?')}")
        tested_section = "\n".join(parts)

    prompt = f"""# Scientific Finding

{finding}

# Rejected Hypothesis

The following hypothesis was rejected by review and must NOT be resubmitted:

**Name**: {rejected_hypothesis.get('name', 'N/A')}
**Prediction**: {rejected_hypothesis.get('prediction', 'N/A')}

**Reason for rejection**: {feedback}

# Previously Tested Hypotheses

{tested_section if tested_section else "None yet."}

# Available Data and Tools

{data_section}

# Task

Generate ONE new hypothesis that:
1. Does NOT duplicate any previously tested or rejected hypothesis
2. Has a clear, unambiguous prediction — a coder should know exactly what data to use, what to compare, and what statistical test to run
3. Tests a different aspect of the finding than what has already been tested

Each hypothesis must have exactly ONE prediction tested by exactly ONE statistical test.

Respond in JSON format:
{{
    "hypothesis": {{
        "name": "Hypothesis name",
        "group": "mechanism-slug",
        "rationale": "Biological reasoning",
        "prediction": "Quantitative, falsifiable prediction with metric, comparison, and expected direction/magnitude",
        "verification_plan": ["Step 1", "Step 2", ...],
        "priority": 1,
        "required_data": ["data_key_1", "data_key_2"]
    }},
    "reasoning": "Why this hypothesis is different from what was rejected and previously tested"
}}
"""
    return prompt


def _format_data_manifest(data_manifest: dict[str, Any]) -> str:
    """Format the data manifest for inclusion in prompts."""
    parts = []

    data = data_manifest.get("data", {})
    for category, items in data.items():
        parts.append(f"**{category}**:")
        if isinstance(items, dict):
            for name, path in items.items():
                parts.append(f"  - {name}: {path}")
        else:
            parts.append(f"  - {items}")
        parts.append("")

    tools = data_manifest.get("tools", [])
    if tools:
        parts.append("**Available CLI Tools**:")
        for tool in tools:
            parts.append(f"  - {tool}")

    return "\n".join(parts)
