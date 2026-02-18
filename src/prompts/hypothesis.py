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

Generate initial hypotheses to investigate this finding. Start by confirming the association
is real and diagnosing whether it is a sequence-level or protein-level effect — these are
the most diagnostic early questions and will inform all subsequent hypotheses.

Do NOT generate mechanism hypotheses yet — those depend on what early results reveal.

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

INVESTIGATION GUIDANCE:

Refer to the scientific reasoning approach in your system instructions.

**Hypothesis prioritization**: Before proposing your next hypothesis, ask yourself:
does this hypothesis move closer to explaining WHY the finding exists (a causal
mechanism), or does it further characterize WHAT the finding looks like (more detail
about the association)?

- If it explains WHY → propose it.
- If it characterizes WHAT → skip it, unless it is necessary to disambiguate
  between two competing mechanism hypotheses.

Examples of WHAT (skip these):
- "Is the overlap directional?" — describes the phenomenon more precisely
- "Are the shared sites at promoters or enhancers?" — describes where, not why
- "Does the correlation hold genome-wide?" — confirms the phenomenon at larger scale

Examples of WHY (propose these):
- "Does TF_A compete with TF_B for the same binding site?" — proposes a causal process
- "Does TF_A recruit a co-factor that blocks TF_B?" — proposes a molecular event
- "Is the motif similarity due to TF_B's motif containing TF_A's core sequence?" — explains the root cause

Your hypothesis should also be motivated by previous results — build on what you've learned,
don't ignore it. But advancing toward mechanism takes priority over following up on details.

CONVERGENCE GUIDELINES (read carefully before deciding):

You may choose CONVERGED only when ALL three conditions are met:

Key distinction — a **mechanism** hypothesis proposes a causal sequence: what molecular
event happens first, what it causes next, and why one leads to the other.

**The test**: Can you describe a before/after — what happens at the molecular level when
the mechanism is active vs inactive? If your conclusion is a list of correlated
observations (even detailed ones), it is NOT a mechanism.

**Example — NOT a mechanism** (phenomenon):
"TF_A and TF_B co-bind at promoters via shared E-box motif in a dose-dependent manner."
This describes what happens, where, and how much — but not why.

**Example — IS a mechanism** (causal process):
"TF_B acts as a pioneer factor that opens chromatin at E-box promoters; TF_A then binds
the accessible E-box as a secondary occupant. The ML model detects TF_B's motif as
predictive because TF_B's prior binding is a prerequisite for TF_A access."
This proposes a causal sequence (TF_B opens → TF_A follows) and is testable: if true,
removing TF_B should reduce TF_A binding.

Your conclusion must be a mechanism, not a phenomenon.

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
