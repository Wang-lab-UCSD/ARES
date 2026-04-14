"""Prompt templates for hypothesis review."""

from __future__ import annotations

from typing import Any


HYPOTHESIS_REVIEW_SYSTEM_PROMPT = """You review scientific hypotheses before they are tested. Your job is to catch duplicates, association/characterization hypotheses, and verification steps that are irrelevant to the stated mechanism — all BEFORE expensive code generation and execution.

You have exactly FOUR checklist items. Reject ONLY when one of them is violated. Do NOT evaluate statistical methodology (test choice, power, controls, causal inference validity) — that is not your job. Your job is to check whether the hypothesis is novel, causal, discriminative, and whether each verification step is testing the right thing for the stated mechanism."""


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

You have exactly FOUR checks. Apply ONLY these. Do not invent additional criteria.

**1. Already answered**
Read the prior iterations carefully — the **prediction**, **what was actually tested**,
**data layers used**, and **evidence** are all shown above. Then ask:

> "Does the new hypothesis ask a different BIOLOGICAL QUESTION, or is it re-measuring
>  something already established with a different metric?"

The key distinction: **same question** = duplicate. **Different question at the same loci** = novel.

**MANDATORY BEFORE REJECTING ON CHECK 1**: You MUST explicitly answer these two questions
in your reasoning before you may reject:
  (a) "What specific molecule, mark, or data modality does the new hypothesis measure
       that was NOT measured in any prior iteration?" — If you can name one (e.g., TRIM28,
       H3K27ac, WGBS methylation, RNA-seq expression, motif containment/substring), the
       hypothesis is NOVEL and you MUST NOT reject on Check 1.
  (b) "What is the exact prior iteration whose test is equivalent?" — Name the iteration
       number and explain what measurement is identical (not just related).
If you cannot answer (a) with "none — every molecule/mark/modality was already tested"
AND answer (b) with a specific iteration, you may not reject on Check 1.

In TF-pair investigations, almost every hypothesis starts from peak overlap between TF_A
and TF_B — that is the fundamental design pattern, NOT a sign of duplication. The question
is what NEW molecule, mark, or functional readout is being measured at those loci.

A hypothesis is a duplicate (REJECT) if:
- It measures the same relationship with a different metric (e.g., overlap % vs signal
  correlation vs fold-change — all asking "do TF_A and TF_B co-bind?")
- It rebrands a previously-measured phenomenon without a new molecular player (e.g.,
  calling co-binding "quantitative modulation" when no new causal step is proposed)
- It is the logical inverse of a confirmed result

A hypothesis is novel (APPROVE) if it does ANY of these — even if the starting point
is TF_A/TF_B peak overlap:
- **Introduces a new molecular player**: e.g., measures TRIM28, RNF2, H3K27me3, or
  another factor at co-bound sites. Prior test measured TF_A signal; new test measures
  a third protein/mark = different question.
- **Tests a different causal pathway**: e.g., prior test asked "does TF_B protein recruit
  TF_A?" (protein tethering); new test asks "does TF_B create accessible chromatin for
  TF_A?" (chromatin remodeling). Different molecular event = novel.
- **Uses a different data modality**: e.g., ChIP-seq → RNA-seq, phyloP, Hi-C, WGBS.
  This is REQUIRED for cross-layer convergence.
- **Adds a conditional slice or exclusion**: e.g., "test at sites where TF_B protein is
  absent," "compare repressed vs active chromatin," "control for GC content."
- **Measures a new genomic feature**: e.g., DNA methylation, motif spacing, loop anchors,
  conservation scores — even if measured at the same peaks.

**Check the rebranding trap**: The hypothesis agent sometimes rebrands an INCONCLUSIVE
result by changing the name while testing the same underlying relationship. Look at WHAT
IS BEING MEASURED, not the hypothesis name. If the new hypothesis computes the same
metric on the same molecules as a prior iteration, it's a duplicate — even if the
mechanism label changed. But if it measures a DIFFERENT molecule or mark, it's novel.

If the prior list is empty, this check cannot trigger — approve.

**2. Mechanism vs. association**
The hypothesis must name a specific causal mechanism — a molecular event explaining WHY
TF_B motif predicts TF_A binding affinity. Reject if it merely describes the data (where, what, how much)
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

For each step in `verification_plan`, ask: "If this step came back negative, would it falsify or significantly undermine this specific hypothesis?" Flag any step where the answer is no, and name the offending step(s).

Also flag any step requiring a **genome-wide FIMO scan** or any operation on the full genome
FASTA — these will time out. Motif scans must be scoped to peak regions, not the whole genome.

**Special rule for direct DNA-binding / direct-recognition claims**:
If the hypothesis claims that TF_A directly binds, directly recognizes, cross-binds, or
mimics TF_B's motif sequence, the verification plan must include at least one **peak-centered
localization test** asking whether the motif itself is positioned at the TF_A ChIP-seq summit
or tightly concentrated around it. Acceptable examples:
- motif-center-to-summit distance distribution
- central enrichment versus peak flanks
- narrow motif centering around the summit in peak-centered windows

Motif enrichment, motif-positive vs motif-negative signal differences, motif score
correlation, or merely scanning summit-centered windows are NOT sufficient by themselves
for a direct-binding claim. If this localization test is missing, prefer
**approve_with_revision** — either add the localization step or weaken the mechanism to a
sequence-intrinsic / motif-associated occupancy claim.

**Decision rule**: If the mechanism itself is novel (passes Check 1) and causal (passes Check 2),
but the verification plan has one or more irrelevant steps or a disallowed operation,
prefer **approve_with_revision** — remove or replace the offending step(s) while keeping
the mechanism. Only hard-reject on Check 3 if the ENTIRE plan is irrelevant to the stated
mechanism (not just one fixable step).

Do NOT reject for:
- The number of steps (1–3 steps are all fine)
- Statistical methodology concerns (test design, controls, bias, confounding)
- Missing implementation details (file paths, column names, tool parameters)

**4. Discriminativeness — would a positive result distinguish this mechanism from the strongest alternative?**

Identify the strongest plausible alternative explanation for a positive result. Then ask:

> "If this hypothesis is supported, would that result distinguish the proposed mechanism
>  from this alternative?"

- **Approve with revision** (PREFERRED) if the biological idea is sound but the test could be made more discriminative by adding one contrast, exclusion, or conditional check. Suggest the revision.
- **Approve** if the test already includes a discriminating element (e.g., testing at TF_B-free sites, comparing against a matched null, or conditioning on a specific chromatin state).
- **Reject** ONLY if a positive result would be equally explained by a generic alternative (e.g., GC-content bias, general active-chromatin co-occurrence, or the ML model's own training signal) AND no reasonable revision could fix it. If you can think of a contrast or control that would make the test discriminative, use approve_with_revision instead.

For direct-binding / direct-recognition hypotheses, "discriminative" includes distinguishing
"the motif marks these loci" from "TF_A directly binds the motif." A positive result based
only on motif enrichment, motif-positive > motif-negative signal, or motif-score correlation
does NOT distinguish direct binding from a broader sequence-context proxy. Require a
localization-style narrowing test (motif centering / summit distance / central enrichment)
before approving a direct-binding claim as sufficiently discriminative.

EXCEPTION: First-iteration artifact checks and functional characterization hypotheses (GO, RNA-seq, phyloP) are exempt from this check.

# Task

Review the hypothesis against the four checks above. Respond in JSON with one of three decisions:

**APPROVE** — hypothesis is novel, causal, and discriminative:
{{
    "decision": "approve",
    "issues": [],
    "reasoning": "Hypothesis is novel and names a causal mechanism",
    "leading_alternative": "The strongest alternative explanation considered",
    "discrimination_rationale": "Why the test distinguishes the proposed mechanism from the alternative",
    "interpretation_ceiling": "The strongest interpretation a positive result could justify"
}}

**APPROVE WITH REVISION** — biological idea is sound, but the test needs one more contrast/exclusion to be discriminative. Provide the revised prediction and verification plan:
{{
    "decision": "approve_with_revision",
    "issues": ["What needs strengthening and why"],
    "reasoning": "The biological idea is sound but the test is not yet discriminative enough",
    "leading_alternative": "The alternative that the current test cannot rule out",
    "discrimination_rationale": "How the revision fixes the discrimination gap",
    "interpretation_ceiling": "The strongest interpretation the revised test could justify",
    "revised_prediction": "The improved prediction with the added contrast/exclusion",
    "revised_verification_plan": ["Step 1...", "Step 2...", "Step 3 (the added discriminating test)..."]
}}

**REJECT** — a check fundamentally fails (true duplicate, pure characterization, irrelevant plan, or fundamentally non-discriminative):
{{
    "decision": "reject",
    "rejection_category": "duplicate" | "wrong_mechanism" | "irrelevant_plan" | "not_discriminative",
    "check1_new_molecule_or_modality": "REQUIRED — name the new molecule/mark/modality introduced by this hypothesis, or 'none' if truly nothing new",
    "check1_equivalent_prior_iteration": "REQUIRED — the specific prior iteration number whose test is equivalent, or 'none' if no exact equivalent",
    "issues": ["Which check failed and why"],
    "reasoning": "Why this hypothesis should not be tested",
    "leading_alternative": "The alternative that the test cannot rule out",
    "discrimination_rationale": "Why the test fails to discriminate",
    "interpretation_ceiling": "What a positive result would actually show (weaker than claimed)"
}}

Set `rejection_category` to:
- `"duplicate"` if Check 1 failed (same biological question as a prior iteration)
- `"wrong_mechanism"` if Check 2 failed (characterization, not a causal mechanism)
- `"irrelevant_plan"` if Check 3 failed (verification plan does not test the stated mechanism)
- `"not_discriminative"` if Check 4 failed (generic alternative not ruled out, unfixable)

Prefer "approve_with_revision" over "reject" when the biological idea is sound. Only reject when the hypothesis is a true duplicate, pure characterization, or fundamentally untestable.
**Self-check before submitting REJECT**: If `check1_new_molecule_or_modality` is anything other than "none", you MUST NOT reject on Check 1 — switch to approve or approve_with_revision.
"""
    return prompt
