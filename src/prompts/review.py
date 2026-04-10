"""Prompt templates for hypothesis review."""

from __future__ import annotations

from typing import Any


HYPOTHESIS_REVIEW_SYSTEM_PROMPT = """You review scientific hypotheses before they are tested. Your job is to catch duplicates, association/characterization hypotheses, and verification steps that are irrelevant to the stated mechanism — all BEFORE expensive code generation and execution.

You have exactly THREE checklist items. Reject ONLY when one of them is violated. Do NOT evaluate statistical methodology (test choice, power, controls, causal inference validity) — that is not your job. Your job is to check whether the hypothesis is novel, causal, and whether each verification step is testing the right thing for the stated mechanism."""


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
            iter_num = t.get('iteration', '?')
            name = t.get('name', '?')
            mechanism = t.get('scratchpad', {}).get('mechanism', '') if isinstance(t.get('scratchpad'), dict) else ''
            prediction = t.get('prediction', '?')
            verification = t.get('verification_plan', [])
            verif_text = "; ".join(verification) if isinstance(verification, list) else str(verification)
            result = t.get('result', '?')
            evidence = t.get('evidence_summary', '')
            data_used = t.get('required_data', [])
            data_text = ", ".join(data_used[:6]) if data_used else 'unspecified'
            if len(data_used) > 6:
                data_text += f" (+{len(data_used)-6} more)"

            parts.append(
                f"### Iteration {iter_num}: {name}\n"
                f"- **Mechanism class**: {mechanism or 'N/A'}\n"
                f"- **Prediction**: {prediction}\n"
                f"- **What was actually tested**: {verif_text}\n"
                f"- **Data layers used**: {data_text}\n"
                f"- **Result**: {result}\n"
                f"- **Evidence**: {evidence}\n"
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

You have exactly THREE checks. Apply ONLY these. Do not invent additional criteria.

**1. Already answered**
Read the prior iterations carefully — the **prediction**, **what was actually tested**,
**data layers used**, and **evidence** are all shown above. Then ask:

> "Given what has already been measured and observed, does this new hypothesis genuinely
>  ask a different question, or is it re-measuring something already established?"

A hypothesis is a duplicate (REJECT) if:
- It tests the same observable using the same data layers as a prior iteration, even if
  the metric is different (e.g., overlap % vs signal at overlapping sites, or correlation
  vs fold-change between groups — both measure "are A and B co-bound")
- It rebrands a previously-measured phenomenon as a "new mechanism" without actually
  proposing a new molecular event (e.g., calling co-binding "quantitative modulation"
  when no new causal step has been proposed)
- It would produce evidence already present in the prior result (e.g., if iter 2 already
  showed REST signal is enriched at ATF6 peaks, asking "is ATF6 signal correlated with
  REST signal at co-bound peaks" is the same observation viewed differently)
- It is the logical inverse of a confirmed result (e.g., prior SUPPORTS enrichment → testing
  depletion is redundant)

A hypothesis is novel (APPROVE) if:
- It tests a fundamentally different molecular event (e.g., switching from "do they co-bind"
  to "is the motif a sequence proxy for a third factor")
- It uses a fundamentally different data modality (e.g., moving from ChIP-seq overlap to
  RNA-seq expression, phyloP conservation, or Hi-C looping) to test the same mechanism class
- It proposes a specific molecular event (steric competition, motif containment, indirect
  recruitment via cofactor X, etc.) that introduces a new causal step beyond what's been measured

**CRITICAL EXCEPTION**: Testing the same mechanism class using a fundamentally different
data modality (e.g. ChIP-seq → RNA-seq → phyloP) is NOT a duplicate — it is REQUIRED for
cross-layer convergence. APPROVE these.

**Check the rebranding trap**: The hypothesis agent sometimes rebrands an INCONCLUSIVE
result by changing the hypothesis name and prediction wording while testing the same
underlying observation. Look at the **data layers** and **what was actually tested** in the
verification plan, not just the hypothesis name. If the new hypothesis would compute on the
same data as a prior iteration to answer the same biological question, it's a duplicate.

If the prior list is empty, this check cannot trigger — approve.

**2. Mechanism vs. association**
The hypothesis must name a specific causal mechanism — a molecular event explaining WHY
TF_B predicts TF_A binding. Reject if it merely describes the data (where, what, how much)
or re-confirms co-occurrence without proposing a mechanism.

**EXCEPTION 1**: If no hypotheses have been tested yet (first hypothesis) AND the hypothesis
checks signal authenticity (TF_A expression + motif enrichment), APPROVE it — this is a
required QC step before mechanism testing, not idle characterization.

**EXCEPTION 2**: If the new hypothesis tests functional relevance via GO enrichment, RNA-seq
expression, or phyloP conservation, APPROVE it when EITHER condition holds:
  (a) A prior mechanism hypothesis has SUPPORTS status and this tests functional relevance
      of that mechanism, OR
  (b) Multiple mechanism hypotheses have been tested (>=3) regardless of their results —
      functional characterization at this stage provides convergence-required biological
      context even when no single mechanism has been confirmed yet.

BAD (characterization or co-occurrence re-statement — reject):
- "Are the shared sites at promoters or enhancers?" — describes where, not why
- "Does the correlation hold genome-wide?" — confirms co-occurrence at larger scale
- "What chromatin states do co-occupied sites fall in?" — describes, does not explain
- "Is TF_A enrichment higher where TF_B is present?" — re-states the original finding
- "TF_B motif is enriched at TF_A binding sites" or "TF_B motif score correlates with TF_A
  signal" — the ML model is a regression model that already established TF_B PWM score predicts
  TF_A binding signal. Confirming the motif is present or correlated is re-validating the ML
  input, not explaining WHY. A mechanism must explain the causal relationship behind the
  correlation (e.g., motif similarity, protein interaction, chromatin context).

GOOD (causal mechanism — approve):
- "TF_B acts as a pioneer factor: co-occupied sites should be enriched in closed chromatin
  (ChromHMM heterochromatin states) relative to TF_A-only sites" — tests a molecular event
- "TF_B's motif contains TF_A's core binding sequence: literal substring match rate should
  exceed PWM match rate" — tests a sequence-level mechanism
- "TF_B and TF_A are tethered via protein-protein interaction: TF_A signal at TF_B sites
  should drop when TF_B motif is absent" — tests a physical interaction mechanism

**3. Verification plan — each step must directly test the stated mechanism**

For each step in `verification_plan`, ask: "If this step came back negative, would it falsify or significantly undermine this specific hypothesis?" If the answer is no for any step, reject and name the offending step(s).

Common irrelevant steps to catch:
- STRING/PPI query in a hypothesis about DNA-sequence or chromatin architecture — protein interaction data cannot falsify a DNA-level mechanism
- GO enrichment or gene expression in a hypothesis about motif co-occurrence or pioneer activity — functional annotation describes context, does not test the mechanism
- ChromHMM or TSS-distance annotation in a hypothesis about protein-protein interaction — genomic context does not falsify a PPI claim

Also reject if any step requires a **genome-wide FIMO scan** or any operation on the full genome
FASTA — these will time out. Motif scans must be scoped to peak regions, not the whole genome.

Do NOT reject for:
- The number of steps (1–3 steps are all fine)
- Statistical methodology concerns (test design, controls, bias, confounding)
- Missing implementation details (file paths, column names, tool parameters)
- Low discriminating power (same-direction counterfactual is advisory, not a reject)

# Task

Review the hypothesis against ONLY the three checks above. Respond in JSON:

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
