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
    encourage_different_mechanism: bool = False,
    unused_biology_layers: list[str] | None = None,
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
        encourage_different_mechanism: If True, instruct to prefer a different mechanism category when at least one hypothesis already SUPPORTS
        unused_biology_layers: When set (e.g. ["rnaseq", "phyloP"]), expression/conservation data are available but no supported hypothesis used them; prefer proposing a hypothesis that uses one so the conclusion can address biological purpose.

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_manifest(data_manifest)
    different_mechanism_guidance = ""
    has_supports = encourage_different_mechanism  # True when at least one SUPPORTS exists
    biology_layers_guidance = ""

    # Determine if 5b is the only thing blocking convergence
    expr_cons_unused = []
    if unused_biology_layers:
        expr_cons_unused = [k for k in unused_biology_layers if k in ("rnaseq", "phyloP")]
    _5b_is_blocker = has_supports and bool(expr_cons_unused)

    # Show diversity guidance only when 5b is NOT the remaining blocker —
    # otherwise it conflicts with the "do functional characterization NOW" instruction.
    if encourage_different_mechanism and not _5b_is_blocker:
        different_mechanism_guidance = """
**Diversity after support**: At least one hypothesis has already **SUPPORTS**. Prefer proposing a hypothesis that tests a **different** mechanism (e.g. pioneer vs tethering vs chromatin modifier—these are examples of what "different mechanism" means, not a prescribed list; propose whatever mechanism best fits the data), so the conclusion can compare or synthesize across mechanisms. *Exception*: If you are testing a previously supported mechanism using a fundamentally NEW data modality (like moving from ChIP to RNA-seq or phyloP for cross-layer validation), that is highly encouraged and not considered repetitive.

"""

    if unused_biology_layers:
        layers_desc = {
            "rnaseq": "expression (RNA-seq)",
            "phyloP": "conservation (phyloP)",
            "string": "protein-protein interaction network (STRING DB)",
        }
        string_unused = "string" in unused_biology_layers
        # expr_cons_unused already computed above
        nudge_lines = []
        if string_unused:
            nudge_lines.append(
                "⚠️  **STRING/PPI has not been checked yet (required for convergence).** "
                "When you propose a protein-interaction hypothesis, include a STRING query as the first "
                "verification step. Do NOT add STRING to non-PPI hypotheses (DNA-sequence, chromatin, "
                "motif similarity) — the review agent will reject it as irrelevant.\n"
                "Prefer `load_string_links(data_files['links_file'])` (local file, no rate limits) over "
                "`fetch_shared_partners()` (API, has rate limits across parallel jobs).\n"
                "**STRING score tiers** (use the appropriate tier for the hypothesis):\n"
                "- >=900: direct physical contact (high-confidence experimental binding)\n"
                "- >=700: strong interaction (use for direct TF-TF contact hypotheses)\n"
                "- >=400: moderate interaction (use for shared partners, cofactor bridging, "
                "and non-DBD protein recruitment)\n"
                "- <150: background / not mechanistically informative\n"
                "**Always check BOTH**: (1) direct interaction at >=700, AND (2) shared interactors "
                "at >=400. A PPI hypothesis should report both results — direct edge AND shared "
                "partners — even if the direct edge is absent. Shared partners like HDAC1, EP300, "
                "or cohesin subunits often mediate indirect cooperation.\n"
                "**IMPORTANT**: Absence of STRING evidence does NOT eliminate PPI mechanisms. "
                "STRING is biased toward well-studied factors — many real interactions have not "
                "been assayed. A negative STRING result should lower the prior on PPI (make it "
                "less likely), not definitively refute it. If other evidence supports PPI "
                "(e.g., high co-occupancy, signal correlation at co-bound sites), the mechanism "
                "remains plausible despite no STRING edge."
            )
        if expr_cons_unused and has_supports:
            nudge_lines.append(
                "⚠️  **Functional characterization is the ONLY remaining convergence criterion.** "
                "You already have a supported mechanism. Do NOT propose a new mechanism — instead, "
                "you MUST propose a hypothesis that tests functional relevance of the supported mechanism "
                "using ONE of:\n"
                "- **GO / pathway enrichment (preferred)**: identify genes near co-bound peaks, run "
                "`run_go_enrichment(gene_list)` from `src.utils.bioio`, and report which biological processes "
                "or pathways are enriched. This characterizes what the TF pair *does* biologically.\n"
                "- **RNA-seq**: link co-occupancy or mechanism to gene expression levels "
                "(e.g. 'genes near co-bound sites show higher expression')\n"
                "- **phyloP**: test evolutionary conservation at co-bound sites "
                "(e.g. 'co-bound motif pairs are more conserved than solo-bound motifs')\n"
                "Any one of these satisfies the requirement. Do NOT explore other mechanisms until this is done."
            )
        elif expr_cons_unused:
            nudge_lines.append(
                "**Functional characterization not yet done.** Prioritize proposing new causal mechanism "
                "hypotheses — do not default to functional characterization.\n\n"
                "**Exception — deadlock-breaking synthesis**: If multiple mechanism hypotheses (>=3) have "
                "already been tested without achieving SUPPORTS, you MAY propose a functional-characterization "
                "hypothesis as a synthesis step using ONE of:\n"
                "- **GO / pathway enrichment**: identify genes near co-bound peaks, run "
                "`run_go_enrichment(gene_list)` from `src.utils.bioio`, and report which biological processes "
                "or pathways are enriched.\n"
                "- **RNA-seq**: link co-occupancy to gene expression levels.\n"
                "- **phyloP**: test evolutionary conservation at co-bound sites.\n"
                "This is a deliberate fallback to provide biological context, not a substitute for finding "
                "a supported mechanism."
            )
        biology_layers_guidance = "\n" + "\n\n".join(nudge_lines) + "\n\n" if nudge_lines else ""

    if last_hypothesis is not None and last_result is not None:
        support_level = last_result.get("support_level", "N/A")
        confidence = last_result.get("confidence", "N/A")

        findings = last_result.get("findings") or []
        if isinstance(findings, list):
            findings_text = "\n".join(f"- {f}" for f in findings) if findings else "No detailed findings recorded."
        else:
            findings_text = str(findings)

        issues = last_result.get("issues") or []
        if isinstance(issues, list):
            issues_text = "\n".join(f"- {i}" for i in issues) if issues else "None recorded."
        else:
            issues_text = str(issues)

        if findings:
            surprising_intro = (
                "Below are observations that may be surprising, strong, or partially contradictory.\n"
                "Your next hypothesis MUST treat at least one of these as a phenomenon to explain,\n"
                "not just re-ask the original question.\n\n"
                "**High-value lead — motif-score-vs-signal correlation**: If any finding below\n"
                "reports a Spearman or Pearson correlation |r| >= 0.15 between the TF_B motif\n"
                "score and TF_A signal (in EITHER direction, even if the hypothesis predicted\n"
                "the opposite sign), this is a strong candidate for a sequence-intrinsic mechanism.\n"
                "Consider testing whether the correlation persists at TF_A peaks that lack TF_B\n"
                "protein (no ChIP-seq overlap) — this is the single most diagnostic test for\n"
                "distinguishing protein-mediated vs sequence-intrinsic mechanisms. However, weigh\n"
                "this against other evidence before committing: if there is strong PPI evidence\n"
                "(STRING score >= 700) or other leads, those may be more productive to pursue first."
            )
        else:
            surprising_intro = "No specific high-signal findings were extracted; focus on clarifying the mechanism."

        latest_section = f"""# Latest Hypothesis Tested

**Name**: {last_hypothesis.get('name', 'N/A')}
**Rationale**: {last_hypothesis.get('rationale', 'N/A')}
**Prediction**: {last_hypothesis.get('prediction', 'N/A')}

**Support level**: {support_level} (confidence {confidence})

# Experimental Results

{last_result.get('summary', 'No summary available')}

**Evidence**:
{last_result.get('evidence', 'No evidence recorded')}

**Interpretation**:
{last_result.get('interpretation', 'No interpretation')}

## Surprising / high-signal observations from this result

{surprising_intro}

{findings_text}

**Issues / anomalies noted**:
{issues_text}"""
    else:
        latest_section = """# First Iteration

No hypotheses have been tested yet. This is the start of the investigation.

Before selecting data for your hypothesis, review all available files in the manifest above — pay balanced attention to each data category so you don't overlook information that may be directly relevant.

**First hypothesis MUST be a standalone artifact check — do NOT combine it with
any mechanism test.** The hypothesis asks: "Is the ChIP-seq signal a technical artifact?"

Check 1 (Expression): Is TF_A expressed (TPM >= 0.5)?
Check 2 (Motif): Is TF_A's known motif enriched at TF_A peaks (≥2-fold over background,
p < 0.05)? Note: 1.2-2.0 fold is ambiguous (some direct + mostly indirect binding);
≥2-fold indicates confident direct binding.

Interpretation — the hypothesis predicts the signal is an artifact, so:
- YES expression + YES motif (≥2-fold): signal is genuine, not an artifact → REJECTS.
  High-confidence direct binding. Proceed normally.
- YES expression + WEAK motif (1.2-2.0 fold): signal is likely genuine → REJECTS.
  Hypothesize protein interaction or cooperative binding mechanisms.
- YES expression + NO motif (<1.2-fold): signal is genuine but indirect → REJECTS.
  Hypothesize indirect/tethered binding. Look for partner motifs.
- NO expression + YES motif: ambiguous → INCONCLUSIVE. Proceed with caution.
- NO expression + NO motif: signal IS a technical artifact → SUPPORTS.
  The pipeline will converge on "technical artifact" and stop.

Only call SUPPORTS (artifact confirmed) when BOTH checks fail. A single failing check
should redirect the hypothesis, not confirm artifact. Do NOT add co-occupancy, chromatin,
or any other mechanism steps to this hypothesis."""

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
{different_mechanism_guidance}
{biology_layers_guidance}
Your task: propose a testable causal mechanism hypothesis.

**Hard rule — no characterization**: Do NOT propose hypotheses that describe what the data
looks like. Every hypothesis must propose a specific causal mechanism: a molecular event
that explains WHY TF_B predicts TF_A's binding — such as TF_B modifying chromatin or
recruiting a co-factor to facilitate TF_A's binding.

A valid mechanism hypothesis must:
1. Name a specific causal mechanism or molecular event (e.g. pioneer opening, motif containment, protein tethering, 3D proximity)
2. Predict a direction of effect
3. Be falsifiable — a negative result would rule out this mechanism

**Verification plan — start from 1, justify each addition.**

Default: **1 step** — the single most decisive test that, if negative, falsifies the mechanism.
Add a second step only if the mechanism has a second genuinely independent, non-redundant component that the first step cannot capture.
Add a third step only if there is a third such component. 3 is the hard cap.

Before writing each step, ask: "If this came back negative, would it falsify the hypothesis?" If no, it does not belong. Do not add steps to be thorough, to characterize context, or to satisfy convergence criteria — add them only if they are independently falsifying.

Check type alignment (only use when directly required by the mechanism):
- Spatial co-occurrence or overlap fractions → for co-binding / composite-element mechanisms
- Signal enrichment comparison → for mechanisms predicting quantitative differences
- Negative control slice ("TF_B present, TF_A absent") → for any mechanism claiming specificity
- STRING/PPI → only for protein-interaction mechanisms, never for DNA-sequence or chromatin hypotheses
- ChromHMM / TSS-distance / GO enrichment → only when the prediction explicitly involves chromatin state, promoter proximity, or gene function

**Forbidden operations in verification plans** — these will time out and waste the entire iteration:
- Genome-wide FIMO scans (scanning hg38 takes hours). Always scan within peak regions only.
- Any operation on the full genome FASTA. Always scope to peak regions or specific intervals.
- Genome-wide shuffled background with FIMO. Use bedtools shuffle + scan the shuffled peaks, not the whole genome.

When relevant, always check **both conditional directions** (e.g., "% of SP1 peaks with NFYA"
AND "% of NFYA peaks with SP1"). An asymmetric relationship (NFYA almost always with SP1,
but SP1 not always with NFYA) is a strong mechanistic signal that would be missed by one
direction alone.

Invalid (characterization — do NOT propose):
- "Are the shared sites at promoters or enhancers?" — describes where, not why
- "Does the correlation hold genome-wide?" — confirms the association at larger scale
- "What chromatin states do co-occupied sites fall in?" (unless predicting a specific pioneer/accessibility mechanism)

Valid (causal mechanism — propose these):
- "TF_B acts as a pioneer factor: co-occupied sites should be enriched in closed chromatin (chromHMM heterochromatin states) relative to TF_A-only sites; AND TF_B-only sites should be more accessible than background"
- "TF_B's motif contains TF_A's core binding sequence: literal substring match rate should exceed PWM match rate; check also if TF_B signal at co-bound sites predicts TF_A signal"
- "TF_B and TF_A are tethered via protein-protein interaction: check co-occupancy fraction from both sides AND whether TF_A signal drops at TF_B sites when TF_B motif is absent"

Tool note: When referencing FIMO motif significance in your prediction, prefer p-value
(e.g., "p < 1e-4") over q-value. For peak-level motif analysis, FIMO's q-value applies
genome-wide multiple testing correction that may be overly conservative for short peak regions.

Your hypothesis should also be motivated by previous results — build on what you've learned,
don't ignore it. But advancing toward mechanism takes priority over following up on details.

**Artifact check**: If the first iteration SUPPORTS (TF_A not expressed AND motif not enriched),
the signal is a confirmed artifact — the pipeline will stop. If the first iteration REJECTS
(signal is genuine), proceed with mechanism hypotheses. If INCONCLUSIVE, proceed with caution.
When REJECTS with weak motif (1.2-2.0 fold), prioritize indirect/tethering mechanisms.

**No-DBD adjustment**: If the QC check found TF_A has NO motif in the database (no DNA-binding
domain), adjust ALL subsequent hypothesis thresholds:
- Co-occupancy: expect <20% overlap with any single TF (not 30-50%) — non-DBD proteins are
  recruited indirectly and will never show high overlap with a single partner.
- STRING: use the >=400 tier (moderate interaction) for shared partners and bridging proteins,
  not >=700 — indirect recruiters interact through cofactors at lower confidence thresholds.
- When searching for recruiting partners, do NOT restrict to TFs with similar motifs. Instead
  use a two-stage screen: (1) loop through all_tf_chipseq directories, calculate peak overlap
  fraction with TF_A for each TF (bedtools intersect is fast, ~1s per TF), rank by overlap;
  (2) check STRING only for the top 10-20 TFs by overlap. This avoids 352 STRING queries while
  still casting a wide net.

When the last result **REJECTS** or is **INCONCLUSIVE** but its findings include strong or
surprising patterns, your next hypothesis MUST propose a mechanism that EXPLAINS those
patterns. Examples of surprising patterns that MUST be followed up:
- Large fold enrichment despite low absolute percentage (e.g., 108-fold co-occupancy at 23% overlap)
- Correlation in the unexpected direction (e.g., negative DNase-motif correlation)
- Effect in the opposite direction from prediction (e.g., higher methylation where lower was expected)
- Positive signal correlation between TFs despite no STRING interaction

Do NOT discard these as "failed predictions." They are the most informative results — they
tell you the mechanism is different from what you hypothesized, not that there is no mechanism.

Your next hypothesis MUST propose a mechanism
that explains those patterns themselves. Do NOT simply re-ask the same question with
slightly different thresholds or proxies. Treat the surprising observations as a new
phenomenon to explain. Even if the overall hypothesis was rejected, you should use any positive, interesting signals as a foundation for your next hypothesis so that the final conclusion weaves these iterations into a coherent story.

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
                "distinguishable": "YES or NO — and why. If NO, redesign the prediction field below.",
                "multi_facet_plan": "Start with 1: what is the single most decisive test? Then: is there a second independent, non-redundant component the first step cannot capture? Only if yes, add it. Repeat for a third (hard cap). Do not add steps to be thorough or to check convergence criteria."
            }},
            "name": "Hypothesis name",
            "group": "mechanism-slug",
            "rationale": "Biological reasoning",
            "prediction": "Falsifiable prediction with metric, comparison groups, expected direction, and threshold. Use the pipeline's standard support thresholds: p < 0.05 AND fold change >= 1.2 (or Cohen's d >= 0.3, or Spearman/Pearson r >= 0.2). For peak overlap thresholds, calibrate to the mechanism: cooperative binding >= 30%, protein tethering/indirect recruitment >= 10%, unbiased screen >= 15%. Do NOT invent aggressive thresholds (>2.0 fold, >30% overlap for non-cooperative mechanisms, r > 0.4) — these cause genuine but moderate biological signals to be labeled REJECTS.",
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
) -> str:
    """Build prompt for regenerating a rejected hypothesis.

    Args:
        finding: Original scientific finding
        rejected_hypothesis: The hypothesis that was rejected by review
        feedback: Reviewer's feedback on why it was rejected
        tested_hypotheses: All previously tested hypotheses
        data_manifest: Available data and tools
        rejection_category: Structured category from reviewer ("wrong_mechanism")
        reviewer_rejected: All hypotheses rejected by reviewer this session (with feedback)

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
    if rejection_category in ("wrong_mechanism", "already_answered"):
        diagnosis_section = """# Diagnosis: Wrong Mechanism or Duplicate

The reviewer determined that this hypothesis is a duplicate of a prior test, already
implied by prior results, or a characterization (not causal).

**Action**: ABANDON this mechanism entirely and propose a DIFFERENT causal mechanism.
*Exception*: If you are trying to validate a prior mechanism using a completely new data modality (e.g. RNA-seq, phyloP) to achieve cross-layer convergence, make sure the new hypothesis explicitly centers on the new modality in its prediction so the reviewer recognizes it as a cross-layer validation.

**CRITICAL RULE**: If your hypotheses have been repeatedly rejected for being duplicates, you MUST propose a completely different experiment. Do not just reword the same idea or tweak thresholds. Change the modality, change the mechanism, or both."""

    else:
        # Fallback when category is unavailable (e.g. data-availability rejection from
        # orchestrator, or reviewer LLM didn't produce the field)
        diagnosis_section = """# Diagnosis — Read the Rejection Reason Carefully

Before generating a replacement, determine WHY the hypothesis was rejected:

**(A) Wrong mechanism / Duplicate** — The rejection says the mechanism is a duplicate, a characterization
(not causal), or already implied by prior results. In this case, ABANDON the mechanism and
propose a different one. *Unless* you are doing cross-layer validation (e.g. adding RNA-seq), in which case ensure the prediction focuses heavily on the new modality.

**(B) Missing data** — The rejection says the required data files or tools are not available
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

Tool note: When referencing FIMO motif significance in your prediction, prefer p-value
(e.g., "p < 1e-4") over q-value. For peak-level motif analysis, FIMO's q-value applies
genome-wide multiple testing correction that may be overly conservative for short peak regions.

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
        "prediction": "Falsifiable prediction with metric, comparison groups, expected direction, and threshold. Use the pipeline's standard support thresholds: p < 0.05 AND fold change >= 1.2 (or Cohen's d >= 0.3, or r >= 0.2). For overlap: cooperative >= 30%, tethering >= 10%, unbiased screen >= 15%. Do NOT invent aggressive thresholds.",
        "verification_plan": ["Step 1", "Step 2", ...],
        "priority": 1,
        "required_data": ["data_key_1", "data_key_2"]
    }},
    "reasoning": "Why this hypothesis is different from what was rejected and previously tested"
}}
"""
    return prompt


def _canonical_required_data_keys(data_manifest: dict[str, Any]) -> list[str]:
    """Return canonical required_data keys (category.key) so the LLM uses exact keys."""
    keys: list[str] = []
    data = data_manifest.get("data", {})
    for category, items in data.items():
        if isinstance(items, dict):
            for leaf_key in items:
                keys.append(f"{category}.{leaf_key}")
    return sorted(keys)


def _format_data_manifest(
    data_manifest: dict[str, Any],
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

    # List canonical keys so the LLM uses them in required_data instead of guessing
    # (e.g. ChromHMM is under epigenome, so use epigenome.chromhmm not annotations.chromhmm)
    canonical = _canonical_required_data_keys(data_manifest)
    if canonical:
        parts.append(
            "**required_data keys** (use these exact strings in your hypothesis): "
            + ", ".join(canonical)
        )
        parts.append("")

    tools = data_manifest.get("tools", [])
    if tools:
        parts.append("**Available CLI Tools**:")
        for tool in tools:
            parts.append(f"  - {tool}")

    return "\n".join(parts)
