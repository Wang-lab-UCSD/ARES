"""Prompt templates for hypothesis generation."""

from __future__ import annotations

from typing import Any

HYPOTHESIS_SYSTEM_PROMPT = """You are a scientific hypothesis generation assistant specializing in bioinformatics. You investigate findings of the form: "ML model says TF_B's motif predicts TF_A's binding — why?"

## How to Approach Mechanism Exploration

Your goal is to find a causal explanation — not just describe what is observed.

Before proposing a hypothesis, reason from two directions:

1. **From biology**: What do you already know about TF_A and TF_B? Given their known functions and the cell type, what causal relationship is plausible?

2. **From the data**: Given what previous iterations have shown, what is the simplest explanation that accounts for ALL observations — including surprising or contradictory ones?

Then ask: what single test would most efficiently distinguish between the top competing explanations?

One diagnostic question is usually worth asking early, because it eliminates half the search space at once: **Is it the DNA sequence pattern or the actual TF_B protein that the ML model is detecting?** If TF_B protein is absent from TF_A sites, all protein-interaction mechanisms are eliminated immediately. But this is a suggestion, not a requirement — use your judgment based on what you already know.

## Key Principles

- The final conclusion must synthesize ALL results (including refuted hypotheses), not just the one that was supported.
- Surprising or contradictory results are the most valuable clues — build on them.
- A good conclusion describes a causal sequence: what happens first, what it causes, and why.
- Your ultimate goal is to identify which specific mechanism class from a named taxonomy explains the finding — you will see the full taxonomy when evaluating convergence.
"""


def build_initial_hypothesis_prompt(
    finding: str,
    context: str,
    data_manifest: dict[str, Any],
    file_summaries: dict[str, str] | None = None,
) -> str:
    """Build the initial prompt for hypothesis generation.

    Args:
        finding: The scientific finding to explain (X predicts Y)
        context: Additional context about the research
        data_manifest: Available data and tools
        file_summaries: Pre-run file inspection output (from iteration 0)

    Returns:
        Formatted prompt string
    """
    # Format available data
    data_section = _format_data_manifest(data_manifest, file_summaries)

    prompt = f"""# Scientific Finding to Investigate

{finding}

# Context

{context}

# Available Data and Tools

{data_section}

# Task

Before selecting data for your hypothesis, review all available files in the manifest above — pay balanced attention to each data category so you don't overlook information that may be directly relevant.

Generate exactly ONE hypothesis to confirm the association is real — that TF_B's motif
or binding is enriched at TF_A's binding locations relative to a matched background.
Choose the single most efficient statistical test for this confirmation.

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
    "reasoning": "Why this is the most efficient association confirmation test"
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
    file_summaries: dict[str, str] | None = None,
) -> str:
    """Build prompt for refining hypotheses based on results.

    Args:
        finding: Original scientific finding
        history_summary: Summary of all previous iterations
        last_hypothesis: The most recently tested hypothesis
        last_result: Results from testing the last hypothesis
        data_manifest: Available data and tools
        group_summary: Summary of a completed hypothesis group for synthesis
        file_summaries: Pre-run file inspection output (from iteration 0)

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest, file_summaries)

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

INVESTIGATION GUIDANCE:

You are in the **mechanism exploration phase**. The association has already been confirmed.

**Hard rule — no characterization**: Do NOT propose hypotheses that describe what the data
looks like. Every hypothesis must propose a specific causal mechanism: a molecular event
that explains WHY TF_B predicts TF_A's binding.

A valid mechanism hypothesis must:
1. Name a specific causal agent or molecular event (e.g. pioneer opening, motif containment, protein tethering, 3D proximity)
2. Predict a direction of effect
3. Be falsifiable — a negative result would rule out this mechanism

Invalid (characterization — do NOT propose):
- "Are the shared sites at promoters or enhancers?" — describes where, not why
- "Does the correlation hold genome-wide?" — confirms the association at larger scale
- "What chromatin states do co-occupied sites fall in?" (unless predicting a specific pioneer/accessibility mechanism)

Valid (causal mechanism — propose these):
- "TF_B acts as a pioneer factor: co-occupied sites should be enriched in closed chromatin (chromHMM heterochromatin states) relative to TF_A-only sites"
- "TF_B's motif contains TF_A's core binding sequence: literal substring match rate should exceed PWM match rate"
- "TF_B and TF_A are tethered via protein-protein interaction: TF_A signal at TF_B sites should drop when TF_B motif is absent"

Your hypothesis should also be motivated by previous results — build on what you've learned,
don't ignore it. But advancing toward mechanism takes priority over following up on details.

CONVERGENCE GUIDELINE:

You may choose CONVERGED only when the evidence supports a specific named mechanism from
the taxonomy below. You must cite the mechanism category number and name in your response.

## Mechanism Taxonomy

1.  **Direct TF-TF protein contacts** — physical binding between TF_A and TF_B proteins (pull-down, co-IP evidence)
2.  **DNA sequence-mediated** — motif containment, motif similarity, shared core binding sequence
3.  **Chromatin/accessibility** — pioneer factor activity, nucleosome remodeling, ATAC-seq enrichment
4.  **Cofactor sharing** — both TFs recruit the same intermediate protein or complex
5.  **3D genome architecture** — chromatin loop anchors, TAD co-boundaries, Hi-C proximity
6.  **Phase separation / condensates** — co-recruitment into super-enhancer condensates
7.  **Post-translational modifications** — phospho/acetyl events that enable co-binding
8.  **Gene regulatory network** — shared target gene programs, co-regulated gene sets
9.  **Binding kinetics** — cooperative binding or competitive exclusion at shared sites
10. **RNA-mediated** — eRNA or lncRNA scaffolding that co-localizes TFs
11. **Molecular crowding** — high local TF concentration at active loci draws both TFs
12. **Paralog / family cross-binding** — TF_B binds TF_A's motif due to structural similarity

To declare CONVERGED, the evidence must identify the mechanism class that is operating —
not just show that the two TFs co-occur or that a signal is higher at co-bound sites.
Co-occurrence alone maps to no category and does not warrant convergence.

If you cannot map your evidence to a specific numbered category, do NOT converge.
Continue with a new hypothesis that would distinguish between candidate categories.

Do NOT choose REFINE just to add more permutations, stricter matching, or additional
control analyses on the same hypothesis. A good conclusion acknowledges limitations
without requiring them to be resolved first. REFINE is for when results are ambiguous
or partially contradictory, NOT for when results are clear but could theoretically be
made more rigorous.

Decisions:

1. **CONVERGED**: Evidence supports a specific mechanism from the taxonomy above. You must populate `mechanism_category_number` (integer 1-12) and `mechanism_category_name` in your response.
2. **REFINE**: Results partially support a hypothesis but are ambiguous — retry with a cleaner test.
3. **NEW_HYPOTHESIS**: Results refute the current hypothesis — propose a different mechanism.
4. **TECHNICAL_ERROR**: Code had bugs, parsing errors, or technical failures that prevented proper analysis. Fix and retry — do NOT use this decision to converge.
5. **INSUFFICIENT_DATA**: The available data truly cannot test the mechanism classes (wrong data type, missing files).

IMPORTANT: Do NOT choose CONVERGED or INSUFFICIENT_DATA if there were technical errors in the code execution.

Respond in JSON format:
{{
    "decision": "CONVERGED" | "REFINE" | "NEW_HYPOTHESIS" | "TECHNICAL_ERROR" | "INSUFFICIENT_DATA",
    "mechanism_category_number": null,
    "mechanism_category_name": null,
    "confidence": 0.0-1.0,
    "reasoning": "Explanation for the decision",
    "technical_issues": ["List of specific technical issues to fix, if TECHNICAL_ERROR"],
    "hypotheses": [
        // ONLY include the ONE hypothesis to test next. Do NOT repeat previously tested
        // hypotheses or list all candidates. Return exactly one hypothesis here.
        // Omit this field when decision is CONVERGED or INSUFFICIENT_DATA.
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

To decide whether to stop: check if the accumulated evidence supports a specific mechanism
from the taxonomy below. If yes, cite the category number and name. If no mechanism has
been identified, recommend continuing unless iterations are exhausted.

## Mechanism Taxonomy

1.  **Direct TF-TF protein contacts** — physical binding between TF_A and TF_B proteins
2.  **DNA sequence-mediated** — motif containment, motif similarity, shared core binding sequence
3.  **Chromatin/accessibility** — pioneer factor activity, nucleosome remodeling, ATAC-seq enrichment
4.  **Cofactor sharing** — both TFs recruit the same intermediate protein or complex
5.  **3D genome architecture** — chromatin loop anchors, TAD co-boundaries, Hi-C proximity
6.  **Phase separation / condensates** — co-recruitment into super-enhancer condensates
7.  **Post-translational modifications** — phospho/acetyl events that enable co-binding
8.  **Gene regulatory network** — shared target gene programs, co-regulated gene sets
9.  **Binding kinetics** — cooperative binding or competitive exclusion at shared sites
10. **RNA-mediated** — eRNA or lncRNA scaffolding that co-localizes TFs
11. **Molecular crowding** — high local TF concentration at active loci draws both TFs
12. **Paralog / family cross-binding** — TF_B binds TF_A's motif due to structural similarity

Respond in JSON format:
{{
    "should_continue": true | false,
    "mechanism_category_number": null,
    "mechanism_category_name": null,
    "confidence": 0.0-1.0,
    "reasoning": "Explanation for the decision",
    "conclusion": "Current best explanation for the finding (required when should_continue is false)"
}}

When should_continue is false, you must populate mechanism_category_number and mechanism_category_name
if the evidence supports a named mechanism. If no mechanism was identified, set both to null and
explain in reasoning.
"""
    return prompt


def build_regeneration_prompt(
    finding: str,
    rejected_hypothesis: dict[str, Any],
    feedback: str,
    tested_hypotheses: list[dict[str, Any]],
    data_manifest: dict[str, Any],
    file_summaries: dict[str, str] | None = None,
) -> str:
    """Build prompt for regenerating a rejected hypothesis.

    Args:
        finding: Original scientific finding
        rejected_hypothesis: The hypothesis that was rejected by review
        feedback: Reviewer's feedback on why it was rejected
        tested_hypotheses: All previously tested hypotheses
        data_manifest: Available data and tools
        file_summaries: Pre-run file inspection output (from iteration 0)

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest, file_summaries)

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


def _format_data_manifest(
    data_manifest: dict[str, Any],
    file_summaries: dict[str, str] | None = None,
) -> str:
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

    if file_summaries and file_summaries.get("all_files"):
        parts.append("")
        parts.append("**File Previews (from pre-run inspection)**:")
        parts.append("```")
        preview = file_summaries["all_files"]
        if len(preview) > 3000:
            preview = preview[:3000] + "\n... [truncated]"
        parts.append(preview)
        parts.append("```")

    return "\n".join(parts)
