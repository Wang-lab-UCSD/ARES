"""Prompt templates for hypothesis generation."""

from __future__ import annotations

from typing import Any

HYPOTHESIS_SYSTEM_PROMPT = """You are a scientific hypothesis generation assistant specializing in bioinformatics and molecular biology. Your role is to:

1. Generate testable hypotheses to explain scientific findings
2. Provide clear rationale based on biological knowledge
3. Design verification approaches that can be implemented with available data
4. Evaluate evidence and refine hypotheses iteratively

When generating hypotheses:
- Consider multiple possible mechanisms (sequence-based, chromatin-based, regulatory)
- Make specific, testable predictions
- Prioritize hypotheses by biological plausibility
- Be explicit about what data would support or refute each hypothesis

When evaluating results:
- Assess whether the evidence supports, refutes, or is inconclusive for the hypothesis
- Consider alternative explanations
- Suggest refinements or new hypotheses based on findings
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

Generate 3-5 testable hypotheses to explain this finding. For each hypothesis, provide:

1. **Hypothesis Name**: A brief descriptive name
2. **Rationale**: The biological reasoning behind this hypothesis
3. **Prediction**: What we should observe if this hypothesis is true
4. **Verification Plan**: Specific steps to test this hypothesis using the available data

Prioritize the hypotheses by biological plausibility and recommend which one to test first.

Respond in JSON format:
{{
    "hypotheses": [
        {{
            "name": "Hypothesis name",
            "rationale": "Biological reasoning",
            "prediction": "Expected observation if true",
            "verification_plan": ["Step 1", "Step 2", ...],
            "priority": 1,
            "required_data": ["data_key_1", "data_key_2"]
        }},
        ...
    ],
    "recommended_first": 0,
    "reasoning": "Why this hypothesis should be tested first"
}}
"""
    return prompt


def build_refinement_prompt(
    finding: str,
    history_summary: str,
    last_hypothesis: dict[str, Any],
    last_result: dict[str, Any],
    data_manifest: dict[str, Any],
) -> str:
    """Build prompt for refining hypotheses based on results.

    Args:
        finding: Original scientific finding
        history_summary: Summary of all previous iterations
        last_hypothesis: The most recently tested hypothesis
        last_result: Results from testing the last hypothesis
        data_manifest: Available data and tools

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

# Task

Based on these results, decide how to proceed.

CONVERGENCE GUIDELINES (read carefully before deciding):

You may choose CONVERGED only when ALL three conditions are met:
1. The current hypothesis has statistical support (p < 0.05, clear effect size).
2. At least one alternative hypothesis has been tested and ruled out or compared.
3. Your conclusion addresses WHY the finding exists (a mechanism), not just THAT it
   exists (an association). "X and Y significantly co-occur" is an observation, not
   a mechanism — it does not satisfy this condition.

If condition 1 is met but conditions 2 or 3 are not, use NEW_HYPOTHESIS to test
alternative explanations or explore the mechanism behind the association.

Do NOT choose REFINE just to add more permutations, stricter matching, or additional
control analyses on the same hypothesis. A good conclusion acknowledges limitations
without requiring them to be resolved first. REFINE is for when results are ambiguous
or partially contradictory, NOT for when results are clear but could theoretically be
made more rigorous.

Decisions:

1. **CONVERGED**: All three convergence conditions above are met (statistical support, alternatives tested, mechanism explained)
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
            "rationale": "Biological reasoning",
            "prediction": "Expected observation if true",
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
