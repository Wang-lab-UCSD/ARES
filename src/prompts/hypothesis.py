"""Prompt templates for hypothesis generation."""

from __future__ import annotations

from typing import Any

HYPOTHESIS_SYSTEM_PROMPT = """You are a molecular biologist specializing in transcriptional regulation, working with bioinformatics tools. You investigate findings of the form: "ML model says TF_B's motif predicts TF_A's binding — why?"

You are given two TFs: TF_A is the primary TF with ChIP-seq or CUT&TAG binding experiments; the motif of TF_B is found to be the most predictive feature to TF_A's binding signals. This is a surprising observation. You aim to find the molecular mechanisms to explain this.

## How to Approach Mechanism Exploration

Your goal is to find a causal explanation — not just describe what is observed.

Before proposing a hypothesis, reason from two directions:

1. **From biology**: What do you already know about TF_A and TF_B? Given their known functions and the cell type, what causal relationship is plausible?

2. **From the data**: Given what previous iterations have shown, what is the simplest molecular mechanism that accounts for ALL observations — including surprising or contradictory ones?

Then ask: what single test would most efficiently distinguish between the top competing explanations?

One diagnostic question is usually worth asking early, because it eliminates half the search space at once: **Is it the DNA sequence pattern or the actual TF_B protein that the ML model is detecting?** If TF_B protein is absent from TF_A sites, all protein-interaction mechanisms are eliminated immediately. But this is a suggestion, not a requirement — use your judgment based on what you already know.

## Key Principles

- The final conclusion must synthesize ALL results (including refused hypotheses), not just the one that was supported.
- Surprising or contradictory results are the most valuable clues — build on them.
- A good conclusion describes a causal sequence: what happens first, what it causes, and why.
- Your ultimate goal is to identify a specific mechanism that explains why TF_B is identified as the most predictive feature of TF_A's binding signals.
"""


def build_refinement_prompt(
    finding: str,
    history_summary: str,
    last_hypothesis: dict[str, Any] | None,
    last_result: dict[str, Any] | None,
    data_manifest: dict[str, Any],
    group_summary: str | None = None,
    file_summaries: dict[str, str] | None = None,
) -> str:
    """Build prompt for refining hypotheses based on results.

    Also used for the first iteration (last_hypothesis=None, last_result=None),
    in which case no prior hypothesis section is shown.

    Args:
        finding: Original scientific finding
        history_summary: Summary of all previous iterations
        last_hypothesis: The most recently tested hypothesis, or None on first iteration
        last_result: Results from testing the last hypothesis, or None on first iteration
        data_manifest: Available data and tools
        group_summary: Summary of a completed hypothesis group for synthesis
        file_summaries: Pre-run file inspection output (from iteration 0)

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest, file_summaries)

    if last_hypothesis is not None and last_result is not None:
        latest_section = f"""# Latest Hypothesis Tested

**Name**: {last_hypothesis.get('name', 'N/A')}
**Rationale**: {last_hypothesis.get('rationale', 'N/A')}
**Prediction**: {last_hypothesis.get('prediction', 'N/A')}

# Experimental Results

{last_result.get('summary', 'No summary available')}

**Evidence**:
{last_result.get('evidence', 'No evidence recorded')}

**Interpretation**:
{last_result.get('interpretation', 'No interpretation')}"""
    else:
        latest_section = """# First Iteration

No hypotheses have been tested yet. This is the start of the investigation.

Before selecting data for your hypothesis, review all available files in the manifest above — pay balanced attention to each data category so you don't overlook information that may be directly relevant."""

    prompt = f"""# Scientific Finding Under Investigation

{finding}

# Previous Iterations

{history_summary if history_summary else "None yet."}

{latest_section}

# Available Data and Tools

{data_section}

{group_summary or ""}

# Task

Based on these results, decide how to proceed.

INVESTIGATION GUIDANCE:

You are in the **mechanism exploration phase**.

**Hard rule — no characterization**: Do NOT propose hypotheses that describe what the data
looks like. Every hypothesis must propose a specific causal mechanism: a molecular event
that explains WHY TF_B predicts TF_A's binding — such as TF_B modifying chromatin or
recruiting a co-factor to facilitate TF_A's binding.

A valid mechanism hypothesis must:
1. Name a specific causal mechanism or molecular event (e.g. pioneer opening, motif containment, protein tethering, 3D proximity)
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

Do NOT choose REFINE just to add more permutations, stricter matching, or additional
control analyses on the same hypothesis. A good conclusion acknowledges limitations
without requiring them to be resolved first. REFINE is for when results are ambiguous
or partially contradictory, NOT for when results are clear but could theoretically be
made more rigorous.

Decisions:

1. **REFINE**: Results partially support a hypothesis but are ambiguous — retry with a cleaner test.
2. **NEW_HYPOTHESIS**: Results refuse the current hypothesis — propose a different mechanism.
3. **TECHNICAL_ERROR**: Code had bugs, parsing errors, or technical failures that prevented proper analysis. Fix and retry.
4. **INSUFFICIENT_DATA**: The available data truly cannot test this mechanism class (wrong data type, missing files).

IMPORTANT: Do NOT choose INSUFFICIENT_DATA if there were technical errors in the code execution.

Respond in JSON format:
{{
    "decision": "REFINE" | "NEW_HYPOTHESIS" | "TECHNICAL_ERROR" | "INSUFFICIENT_DATA",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation for the decision",
    "technical_issues": ["List of specific technical issues to fix, if TECHNICAL_ERROR"],
    "hypotheses": [
        // ONLY include the ONE hypothesis to test next. Do NOT repeat previously tested
        // hypotheses or list all candidates. Return exactly one hypothesis here.
        // Omit this field when decision is INSUFFICIENT_DATA.
        {{
            "scratchpad": {{
                "mechanism": "The specific molecular event (not a category name)",
                "causal_chain": "Event A → Event B → measurable outcome C",
                "prediction_if_mechanism": "What the data would show if this mechanism operates",
                "prediction_if_co_occupancy_only": "What the data would show if TF_B and TF_A simply co-occur at active sites with no causal relationship",
                "distinguishable": "YES or NO — and why. If NO, redesign the prediction field below."
            }},
            "name": "Hypothesis name",
            "group": "mechanism-slug",
            "rationale": "Biological reasoning",
            "prediction": "Quantitative, falsifiable prediction with metric, comparison, and expected direction/magnitude",
            "verification_plan": ["Step 1", "Step 2", ...],
            "priority": 1,
            "required_data": ["data_key_1", "data_key_2"]
        }}
    ]
}}
"""
    return prompt


def build_regeneration_prompt(
    finding: str,
    rejected_hypothesis: dict[str, Any],
    feedback: str,
    tested_hypotheses: list[dict[str, Any]],
    data_manifest: dict[str, Any],
    rejection_category: str | None = None,
    reviewer_rejected: list[dict[str, Any]] | None = None,
    file_summaries: dict[str, str] | None = None,
) -> str:
    """Build prompt for regenerating a rejected hypothesis.

    Args:
        finding: Original scientific finding
        rejected_hypothesis: The hypothesis that was rejected by review
        feedback: Reviewer's feedback on why it was rejected
        tested_hypotheses: All previously tested hypotheses
        data_manifest: Available data and tools
        rejection_category: Structured category from reviewer ("wrong_mechanism" or "flawed_test")
        reviewer_rejected: All hypotheses rejected by reviewer this session (with feedback)
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

    # Build list of all reviewer-rejected hypotheses (excluding the current one which is shown separately)
    prior_rejected_section = ""
    if reviewer_rejected:
        current_name = rejected_hypothesis.get("name", "")
        prior = [r for r in reviewer_rejected if r.get("name") != current_name]
        if prior:
            parts = []
            for r in prior:
                parts.append(
                    f"- **{r.get('name', '?')}**: {r.get('prediction', '?')}\n"
                    f"  Rejected because: {r.get('feedback', '?')}"
                )
            prior_rejected_section = "\n# Previously Rejected Hypotheses (do NOT repeat these)\n\n" + "\n".join(parts)

    # Build diagnosis section based on structured rejection category
    if rejection_category == "wrong_mechanism":
        diagnosis_section = """# Diagnosis: Wrong Mechanism

The reviewer determined that the MECHANISM itself is the problem — it is a duplicate of a
prior test, a characterization (not causal), fails the directional counterfactual, or its
answer is already implied by prior results.

**Action**: ABANDON this mechanism entirely. Propose a DIFFERENT causal mechanism."""

    elif rejection_category == "flawed_test":
        diagnosis_section = """# Diagnosis: Flawed Test Design

The reviewer determined that the mechanism is reasonable, but the prediction is unclear,
the verification plan has a flawed null model, or the test logic does not match the
hypothesis claim.

**Action**: KEEP the same mechanism. Redesign the test with a different experimental
approach — different comparison groups, different statistical test, or different data."""

    else:
        # Fallback when category is unavailable (e.g. data-availability rejection from
        # orchestrator, or reviewer LLM didn't produce the field)
        diagnosis_section = """# Diagnosis — Read the Rejection Reason Carefully

Before generating a replacement, determine WHY the hypothesis was rejected:

**(A) Wrong mechanism** — The rejection says the mechanism is a duplicate, a characterization
(not causal), the directional counterfactual fails (mechanism and co-occupancy predict the
same direction), or the prediction's answer is implied by the setup. In this case, ABANDON
the mechanism and propose a different one.

**(B) Flawed test design** — The rejection says the mechanism is reasonable but the prediction
is unclear, the verification plan has a flawed null model, or the test logic does not match
the hypothesis claim. In this case, KEEP the same mechanism and redesign the test with a
different experimental approach.

**(C) Missing data** — The rejection says the required data files or tools are not available
in the manifest. In this case, KEEP the mechanism idea but redesign the test to use only
data that IS available, or propose a different mechanism if no viable test exists."""

    prompt = f"""# Scientific Finding

{finding}

# Rejected Hypothesis

The following hypothesis was rejected and must NOT be resubmitted without addressing the rejection reason:

**Name**: {rejected_hypothesis.get('name', 'N/A')}
**Prediction**: {rejected_hypothesis.get('prediction', 'N/A')}

**Reason for rejection**: {feedback}
{prior_rejected_section}
# Previously Tested Hypotheses

{tested_section if tested_section else "None yet."}

# Available Data and Tools

{data_section}

{diagnosis_section}

# Task

Generate ONE new hypothesis that:
1. Does NOT duplicate any previously tested or rejected hypothesis
2. Has a clear, unambiguous prediction — a coder should know exactly what data to use, what to compare, and what statistical test to run
3. Tests a genuinely new question — either a different mechanism OR the same mechanism with a fundamentally different test design (if the rejection was about test design, not the mechanism itself)

**Hard rule**: Every hypothesis must propose a specific CAUSAL mechanism — a molecular event that explains WHY TF_B predicts TF_A's binding. Characterization hypotheses (describing what data looks like, confirming co-occurrence, or re-stating the original finding at a different scale) are NOT valid.

Each hypothesis must have exactly ONE prediction tested by exactly ONE statistical test.

Respond in JSON format:
{{
    "hypothesis": {{
        "scratchpad": {{
            "mechanism": "The specific molecular event (not a category name)",
            "causal_chain": "Event A → Event B → measurable outcome C",
            "prediction_if_mechanism": "What the data would show if this mechanism operates",
            "prediction_if_co_occupancy_only": "What the data would show if TF_B and TF_A simply co-occur at active sites with no causal relationship",
            "distinguishable": "YES or NO — and why. If NO, redesign the prediction field below."
        }},
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
