"""Prompt templates for code review."""

from __future__ import annotations

from typing import Any

REVIEW_SYSTEM_PROMPT = """You are a code reviewer for bioinformatics analysis scripts. Your job is to catch common mistakes and rule violations BEFORE the code runs, saving hours of wasted computation.

You will receive a Python script and the hypothesis it is testing. Review the code against the checklist below and either approve it or return a corrected version.

Be practical: only flag real problems that would cause incorrect results or excessive runtime. Do not flag style issues, missing comments, or minor inefficiencies."""


def build_review_prompt(
    code: str,
    hypothesis: dict[str, Any],
    data_manifest: dict[str, Any],
    tested_hypotheses: list[dict[str, Any]] | None = None,
) -> str:
    """Build the code review prompt.

    Args:
        code: The generated Python code to review
        hypothesis: The hypothesis being tested
        data_manifest: Available data paths
        tested_hypotheses: Previously tested hypotheses for duplicate detection

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_brief(data_manifest)

    # Format previously tested hypotheses
    tested_section = ""
    if tested_hypotheses:
        parts = []
        for t in tested_hypotheses:
            parts.append(f"- **{t.get('name', '?')}**: {t.get('prediction', '?')} → {t.get('result', '?')}")
        tested_section = "\n".join(parts)

    prompt = f"""# Code to Review

```python
{code}
```

# Hypothesis Being Tested

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}
**Verification Plan**: {hypothesis.get('verification_plan', 'N/A')}

# Previously Tested Hypotheses

{tested_section if tested_section else "None yet."}

# Available Data

{data_section}

# Review Checklist

Check the code for these specific issues:

**1. FIMO --motif flag**
If the code calls `fimo` and the hypothesis only needs specific motif(s), the command MUST include `--motif <MOTIF_ID>`. Without it, FIMO scans all 800+ motifs in JASPAR which takes hours instead of seconds.
- If `--motif` is missing and the hypothesis names specific motifs → FIX: add `--motif`
- If the hypothesis genuinely requires scanning all motifs → APPROVE as-is

**2. Single statistical test**
The code should perform ONE primary statistical test that directly answers the hypothesis prediction. If it performs multiple unrelated tests (e.g., distance + GC% + expression), remove the extras and keep only the one matching the prediction.
- Data loading, FIMO scanning, file parsing are fine — those are setup, not extra tests
- One Fisher's test OR one Mann-Whitney test is the goal

**3. Column selection**
When reading tabular data (TSV/CSV), columns should be selected by name (e.g., `df['TPM']`), not by position (e.g., `columns[0]`, `value_cols[0]`). Positional selection often picks the wrong column.

**4. Expensive operations in loops**
These operations MUST NOT appear inside any for/while loop:
- `bedtools getfasta`
- `fimo`
- `pyBigWig` / `bigWigAverageOverBed`
They should be called ONCE, outside all loops.

**5. Permutation cap**
The pipeline enforces a hard cap of 100 permutation replicates for runtime reasons. If the code
uses 100 permutations but the hypothesis specifies more (e.g., 1000), this is CORRECT behavior —
do NOT flag it as an issue.

**6. Obvious bugs**
- Wrong file paths (not matching the data manifest)
- Missing imports
- Using chromosome names instead of peak IDs for counting

**7. Code matches hypothesis**
Read the hypothesis prediction carefully. Does the code actually test that specific claim? For example, if the hypothesis says "the REST motif (21bp) contains the ATF6 motif (5bp) as a substring," the code must scan the motif sequence itself — NOT scan broad peak regions (hundreds of bp) for the pattern, which would test a different question. If the code tests something different from what the prediction states → REJECT (do not fix — the mismatch is too fundamental).

**8. FIMO p-value vs q-value filtering**
When code filters FIMO output for peak-level motif analysis, prefer `p-value < 1e-4` over
q-value. FIMO's q-value applies genome-wide multiple testing correction that may be overly
conservative for short peak regions — legitimate hits in control/shuffled regions can get
q > 0.05, producing zero results. If the code filters on q-value and gets zero or
suspiciously few hits, flag it and suggest switching to p-value filtering.

# Task

Review the code against the checklist above. Respond in JSON format:

{{
    "approved": true,
    "issues": [],
    "corrected_code": null,
    "reasoning": "Code follows all rules"
}}

OR if issues are found:

{{
    "approved": false,
    "issues": ["Issue 1 description", "Issue 2 description"],
    "corrected_code": "... the full corrected Python code ...",
    "reasoning": "Explanation of what was fixed"
}}

If you fix the code, return the COMPLETE corrected script (not just the changed lines).
"""
    return prompt


HYPOTHESIS_REVIEW_SYSTEM_PROMPT = """You review scientific hypotheses before they are tested. Your job is to catch duplicates and association/characterization hypotheses BEFORE expensive code generation and execution.

You have exactly TWO checklist items. Reject ONLY when one of them is violated. Do NOT evaluate statistical methodology, test design, control groups, or causal inference validity — that is not your job."""


def build_hypothesis_review_prompt(
    hypothesis: dict[str, Any],
    tested_hypotheses: list[dict[str, Any]],
) -> str:
    """Build the hypothesis review prompt.

    Args:
        hypothesis: The hypothesis to review
        tested_hypotheses: Previously tested hypotheses

    Returns:
        Formatted prompt string
    """
    tested_section = ""
    if tested_hypotheses:
        parts = []
        for t in tested_hypotheses:
            parts.append(
                f"- **{t.get('name', '?')}**: {t.get('prediction', '?')} → {t.get('result', '?')}"
            )
        tested_section = "\n".join(parts)

    # Format scratchpad if present (hypothesis agent's counterfactual analysis)
    scratchpad = hypothesis.get("scratchpad")
    if scratchpad and isinstance(scratchpad, dict):
        scratchpad_section = f"""
**Hypothesis Agent's Counterfactual Analysis (scratchpad)**:
- Mechanism: {scratchpad.get('mechanism', 'N/A')}
- Causal chain: {scratchpad.get('causal_chain', 'N/A')}
- Prediction if mechanism operates: {scratchpad.get('prediction_if_mechanism', 'N/A')}
- Prediction if co-occupancy only: {scratchpad.get('prediction_if_co_occupancy_only', 'N/A')}
- Distinguishable?: {scratchpad.get('distinguishable', 'N/A')}"""
    else:
        scratchpad_section = ""

    prompt = f"""# Hypothesis to Review

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}
**Verification Plan**: {hypothesis.get('verification_plan', 'N/A')}
{scratchpad_section}

# Previously Tested Hypotheses

{tested_section if tested_section else "None yet (first hypothesis)."}

# Review Checklist

You have exactly TWO checks. Apply ONLY these. Do not invent additional criteria.

**1. Already answered**
Has this question already been answered — either directly or by logical implication — by a
prior tested hypothesis? This includes:
- Same question reworded (e.g. "A overlaps B" vs "B overlaps A")
- Same mechanism class with a different proxy (e.g. prior SUPPORTS "open chromatin" →
  "H3K27ac enriched at co-bound sites" is the same claim)
- Logical inverse of a confirmed result (e.g. prior SUPPORTS enrichment → testing
  depletion is redundant)

If the prior list is empty, this check cannot trigger — approve.

**2. Mechanism vs. association**
The hypothesis must name a specific causal mechanism — a molecular event explaining WHY
TF_B predicts TF_A binding. Reject if it merely describes the data (where, what, how much)
or re-confirms co-occurrence without proposing a mechanism.

BAD (characterization or co-occurrence re-statement — reject):
- "Are the shared sites at promoters or enhancers?" — describes where, not why
- "Does the correlation hold genome-wide?" — confirms co-occurrence at larger scale
- "What chromatin states do co-occupied sites fall in?" — describes, does not explain
- "Is TF_A enrichment higher where TF_B is present?" — re-states the original finding

GOOD (causal mechanism — approve):
- "TF_B acts as a pioneer factor: co-occupied sites should be enriched in closed chromatin
  (ChromHMM heterochromatin states) relative to TF_A-only sites" — tests a molecular event
- "TF_B's motif contains TF_A's core binding sequence: literal substring match rate should
  exceed PWM match rate" — tests a sequence-level mechanism
- "TF_B and TF_A are tethered via protein-protein interaction: TF_A signal at TF_B sites
  should drop when TF_B motif is absent" — tests a physical interaction mechanism

Do NOT reject for:
- Statistical methodology concerns (test design, controls, bias, confounding)
- Missing implementation details (file paths, column names, tool parameters)
- Low discriminating power (same-direction counterfactual is advisory, not a reject)

# Task

Review the hypothesis against ONLY the two checks above. Respond in JSON:

{{
    "approved": true,
    "issues": [],
    "reasoning": "Hypothesis is novel and names a causal mechanism"
}}

OR if one of the two checks fails:

{{
    "approved": false,
    "rejection_category": "wrong_mechanism",
    "issues": ["Which check failed and why"],
    "reasoning": "Why this hypothesis should not be tested"
}}
"""
    return prompt


def _format_data_brief(data_manifest: dict[str, Any]) -> str:
    """Format data manifest briefly for review context."""
    parts = []
    data = data_manifest.get("data", {})
    for category, items in data.items():
        if isinstance(items, dict):
            for name, path in items.items():
                parts.append(f"- `{name}`: `{path}`")
        else:
            parts.append(f"- `{items}`")
    return "\n".join(parts)
