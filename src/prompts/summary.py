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

**REFUSES** — ANY of the following:
  1. The predicted effect is absent or reversed (e.g., depletion instead of enrichment)
  2. p >= 0.05 with adequate sample size (N >= 30)
  3. fold change < 1.1 AND Cohen's d < 0.15
  If clearly refused, set support_level = "REFUSES" and confidence >= 0.7.

  **STRING exception**: When a hypothesis is tested solely via STRING and no interaction is
  found, set REFUSES but with confidence 0.5-0.7 (not 1.0). STRING is biased toward
  well-studied proteins — absence of a STRING edge lowers the prior on PPI but does not
  definitively disprove it. If other genomic evidence supports PPI (co-occupancy, signal
  correlation), note this in reasoning.

**ERROR** — Technical failure prevented analysis (code crashed, wrong file format, etc.)

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


CONVERGENCE_CHECK_SYSTEM_PROMPT = """You are an independent scientific adjudicator for a bioinformatics hypothesis testing pipeline. Your sole job is to evaluate whether the accumulated experimental evidence is sufficient to declare convergence on a named causal mechanism (or, in observational mode, the best-supported mechanistic interpretation given the data).

You have NO stake in any particular outcome. You were not involved in generating the hypotheses. Evaluate the evidence skeptically and objectively.

## Two convergence modes

**FULL (causal-capable data)**  
When the user prompt states that the run includes data that can support causal inference (e.g. Perturb-seq, time-series, knockdown), apply ALL FIVE criteria below, including the but-for test. Converge only when the evidence supports a causal mechanism.

**OBSERVATIONAL (observational data only)**  
When the user prompt states that the run uses only observational data (ChIP-seq, bulk RNA-seq, epigenomics, etc.) and cannot establish causality, do NOT require the but-for test. Instead, declare converged=true when criteria 1, 2, 4, and 5 are met and the evidence supports a named mechanism that is the *best interpretation given the data* (e.g. co-occupancy, promoter platform, chromatin priming, retention). The conclusion MUST state that this is the best-supported interpretation given observational data and that causality is not established.

## Convergence Criteria

**PREREQUISITE — at least one SUPPORTS**: Convergence is ONLY possible when at least one
hypothesis in the testing history has support_level = "SUPPORTS". If every hypothesis so
far has been REFUSES, INCONCLUSIVE, ERROR, or UNTESTABLE, you MUST set converged=false
regardless of how interesting the individual findings are. Promising sub-findings inside a
REFUSES result do NOT count — the hypothesis must have been formally SUPPORTED as a whole.

**Signal authenticity QC (special handling)**:
The first hypothesis typically checks whether TF_A's ChIP-seq signal is genuine (expression +
motif enrichment). Treat this result as follows:
- QC SUPPORTS (signal is genuine): this does NOT satisfy the SUPPORTS prerequisite above —
  it is a quality check, not a mechanism. The pipeline still needs a mechanism SUPPORTS.
- QC INCONCLUSIVE (weak expression or weak motif enrichment): proceed with caution. The signal
  is likely real but low-confidence. The pipeline still needs a mechanism SUPPORTS.
- QC REFUTES (TF_A not expressed AND motif not enriched): the signal is a technical artifact.
  The pipeline MAY converge immediately on "technical artifact" as the conclusion — criteria
  2-5 are waived because there is no biological mechanism to characterize.

The QC hypothesis's use of RNA-seq (TPM lookup) or motif scanning does NOT satisfy ANY
convergence criterion — not 5a (STRING/PPI), not 5b (functional characterization), not
criterion 1 (statistical support for a mechanism). Criterion 5b requires linking the mechanism
to gene function (GO enrichment, expression at target genes, or conservation) — not just
checking if TF_A is
expressed. Do not count QC data usage toward any convergence criterion.

1. **Statistical support**: At least one hypothesis with support_level = "SUPPORTS" (i.e., p < 0.05 with fold change >= 1.2 OR Cohen's d >= 0.3, and the predicted effect confirmed). Do NOT cherry-pick individual statistics from a REFUSES result to satisfy this criterion — the overall support_level must be SUPPORTS.

2. **Named mechanism**: A mechanism that: (1) names a specific molecular process, (2) states a clear causal chain (or, in observational mode, a clear mechanistic interpretation), and (3) is not merely a re-description of correlation. The mechanism can be **anything** that fits the evidence—it need not match any predefined category. The taxonomy below is for **reference only** (to illustrate what "mechanism" means in terms of specificity); do NOT constrain convergence to those categories.

3. **But-for test** (apply ONLY when causal-capable data are available): Ask — "If the proposed causal agent were absent, would the data look different?" If the answer is "not necessarily" (because the result could reflect passive co-occurrence, shared active chromatin, or any confound), do NOT converge. When the run is observational-only, skip this criterion.

4. **Cross-layer consistency**: The proposed mechanism must be supported by consistent directional evidence from at least two independent omics layers (e.g., ChIP-seq + DNase-seq, or motif analysis + histone marks, or Hi-C + expression). A single data type is not sufficient — convergence requires cross-validation across independent measurement modalities.

5. **Biology layers — two independent sub-requirements, BOTH must be satisfied**:

   **5a. STRING/PPI (always required)**: At least one hypothesis must have tested the STRING protein–protein interaction network — the result can be SUPPORTS, REFUTES, or INCONCLUSIVE. STRING is an external API that is always accessible; it is not contingent on the manifest. Set converged=false if STRING/PPI was never attempted in any hypothesis.

   **5b. Functional characterization (required when available)**: If the manifest provides expression (rnaseq) or conservation (phyloP) data, at least one hypothesis must have addressed functional relevance using **one of** the following approaches — result can be SUPPORTS, REFUTES, or INCONCLUSIVE:
   - **rnaseq**: link co-occupancy or mechanism to gene expression levels
   - **phyloP**: test evolutionary conservation at co-bound sites
   - **GO / pathway enrichment**: run GO term or pathway enrichment on genes associated with the mechanism (e.g. genes near co-bound peaks), to characterize the biological processes the TF pair regulates

   The purpose is functional insight — understanding what the mechanism does biologically. Any one of these three approaches satisfies 5b. Set converged=false only if rnaseq or phyloP was available but none of these was ever attempted.

   Note: STRING satisfies 5a (PPI check) but does NOT substitute for 5b. Both sub-requirements must be met independently.

   **Escape-hatch characterization**: When multiple mechanism hypotheses have been tested without
   achieving SUPPORTS, a functional-characterization hypothesis (GO enrichment, RNA-seq, phyloP)
   may be tested as a deadlock-breaking synthesis step. This satisfies criterion 5b (functional
   characterization was attempted). However, it does NOT satisfy the PREREQUISITE (at least one
   mechanism SUPPORTS) — a SUPPORTS result on a characterization-only hypothesis is NOT a
   mechanism SUPPORTS. The pipeline must still find a supported mechanism to converge.

## Mechanism Taxonomy (examples only — for understanding what "mechanism" means)

The following list illustrates the level of specificity required: a mechanism names a molecular process and causal chain, not just correlation. Use it to judge whether a proposed mechanism is sufficiently specific. The evidence may support a mechanism that matches one of these, or something entirely different—do NOT constrain convergence to these categories.

1. Direct TF–TF contacts
These involve physical interaction between TF_A and TF_B
1.1 Homodimerization and heterodimerization (bZIP, bHLH, nuclear receptors, etc.) to change DNA-binding specificity, affinity, or regulatory output.
1.2 Higher-order oligomerization (tetramers, arrays) to create multivalent binding and sharpen responses or cooperativity.
1.3 Partner switching, where the same TF forms different heterodimers with distinct functions or motif preferences.
1.4 Interface masking/unmasking, where binding of one TF exposes or hides activation domains or NLS/NES of another TF.

2. DNA-mediated interplay (on naked or sparsely nucleosomal DNA)
TFs influence each other through how they bind to DNA.
2.1 Classic cooperative binding at nearby sites via direct protein–protein contacts while both are DNA-bound.
2.2 Composite motifs where the relative spacing/orientation of sites encodes a preferred TF pair geometry (DNA-guided cooperativity).
2.3 Competitive binding to overlapping or partially overlapping sites (mutual exclusion on DNA).
2.4 DNA-mediated allostery: TF_B changes local DNA shape/dynamics, which alters another TF_A's binding at a distance (tens of bp).
2.5 Facilitated diffusion and "antenna" effects, where multiple weak sites or STRs promote 1D sliding and sharing of a region by several TFs.

3. Nucleosome and chromatin-based mechanisms
Interactions are mediated by nucleosomes and chromatin state rather than direct TF contact.
3.1 Pioneer activity: TF_B binds nucleosomal DNA and open chromatin, enabling binding of secondary TFs.
3.2 Nucleosome-displacement cooperativity: many TFs (same or different) collectively compete with nucleosomes; each TF helps others by contributing to nucleosome eviction without strong pairwise contacts.
3.3 Nucleosome-positioning competition, where TF_B shifts nucleosome positions, exposing or occluding sites for TF_A.
3.4 Chromatin-modifier recruitment: TF_B recruits HATs, HDACs, methyltransferases, remodelers, etc.; multiple TFs converge on the same complexes and thereby influence each other's access/activity.

4. Cofactors, Mediator, and general machinery
TFs "talk" through shared cofactors and the transcription apparatus.
4.1 Shared coactivators/corepressors (e.g., p300/CBP, Mediator, BRD4, HDAC complexes) recruited by TF_B.
4.2 Competition ("squelching") for limiting cofactors, where one TF sequesters coactivators away from another TF's targets.
4.3 Scaffolding via Mediator and PIC components: different TFs engage overlapping subunits, bringing their bound enhancers/promoters into a common transcriptional hub.

5. 3D genome and nuclear organization
Spatial genome architecture mediates functional TF interactions.
5.1 Enhancer–promoter looping brings TFs from distant elements into proximity, allowing combinatorial action without direct DNA adjacency.
5.2 Multi-enhancer hubs/super-enhancers where many TF-bound regions cluster and jointly regulate a gene set.
5.3 Nuclear microenvironments ("transcription factories"), where Pol II and cofactors are enriched; TFs that target the same factory effectively interact by sharing this environment.

6. Phase separation and condensates
Collective, multivalent interactions in dense clusters.
6.1 LLPS-driven transcriptional condensates at super-enhancers, formed by TF IDRs plus cofactors (Med1, BRD4, etc.), concentrate many TFs and promote emergent cooperativity.
6.2 Condensate-mediated buffering or inhibition, where clustering can also trap TFs and reduce effective activity at certain loci.
6.3 Genomic binding clustering tendency: TFs with strong condensate propensity tend to exhibit highly clustered binding profiles, shaping how multiple TFs co-occupy regions.

7. Post-translational and signaling crosstalk
One TF affects another via modifications or signaling pathways.
7.1 Kinase/phosphatase recruitment: TF A brings a kinase that modifies TF B, changing B's DNA binding, localization, or cofactor affinity.
7.2 Other PTMs (acetylation, methylation, SUMOylation, ubiquitination) on TFs, often written or erased by enzymes targeted by partner TFs.
7.3 Proteolytic processing or truncation of regulatory regions that alter interaction surfaces or activation domains.
7.4 Allosteric regulation by metabolites or small molecules that are under control of other TFs (metabolic and stress-responsive circuits).

8. Indirect regulatory network interactions
Interactions via gene-regulatory logic rather than physical proximity.
8.1 Feedforward and feedback loops where TF_B controls TF_A's expression levels, which in turn changes binding competition and combinatorial occupancy.
8.2 Logical integration at promoters/enhancers (AND/OR/NAND-like behavior) based on requirement for multiple TFs, even if they never contact physically but act on distinct steps (chromatin opening vs Pol II recruitment).
8.3 Network-level emergent cooperativity, where many weak, context-dependent TF–DNA interactions together produce robust expression patterns.

9. Dynamic and kinetic mechanisms
Interactions emerge from binding dynamics and non-equilibrium behavior.
9.1 Kinetic synergy: TF_A and TF_B, acting at different transcriptional steps (initiation, pause release, elongation), combine to produce super-additive transcriptional output.
9.2 Temporal ordering and pulse-based interactions, where early TFs set chromatin or cofactor states that gate later TF binding.
9.3 Rapid binding–unbinding and "hit-and-run" behavior of certain TFs that prime loci for others without long residence time.

10. RNA-Mediated Interactions (The "Hidden" Partner)
RNA is now recognized as a major scaffold for TF interactions.
10.1 eRNA Scaffolding: Enhancer RNAs (eRNAs) can act as decoys or scaffolds that stabilize the binding of TFs and cofactors (like p300 or Mediator) at specific loci.
10.2 RNA-TF Allostery: Binding to small nuclear RNAs or lncRNAs can induce conformational changes in a TF, altering its DNA-binding affinity.

11. Molecular Crowding & Volume Exclusion
11.1 Entropy-Driven Binding: High concentrations of macromolecules in the nucleus (like nucleoli or heterochromatin) physically "push" TFs toward open chromatin, increasing their effective concentration at regulatory sites without specific chemical affinity.

12. Evolutionary & Paralog Interactions
12.1 Paralog Balancing: Many TFs have paralogs (e.g., the GATA family). Their interactions are often governed by a balance of concentration; if one paralog is mutated, the other can "buffer" or take over the site, but with different regulatory kinetics.

## Common False Convergence Patterns — Do NOT converge on these

- **"All hypotheses REFUSES but the findings are interesting"** — if no hypothesis achieved support_level = "SUPPORTS", you CANNOT converge. Interesting sub-findings within a REFUSES result mean the pipeline should refine the hypothesis (e.g., drop the failed prediction, keep the successful ones) and test again, not declare convergence.
- "Co-bound sites are in active chromatin" — correlation. Active sites attract many TFs. Does NOT establish mechanism unless pioneer activity is shown (mechanism #3 requires the pioneer to OPEN the site, not merely be present at already-open sites).
- "TF_B signal is higher where TF_A is present" — restates the original finding. Not a mechanism.
- "TF_B motif is enriched at TF_A binding sites" or "TF_B motif score correlates with TF_A signal" — the ML model is a regression model where TF_B PWM score already predicts TF_A binding signal. Re-confirming the motif is present or correlated at TF_A peaks is re-validating the ML input-output relationship, not a mechanism. A mechanism must explain WHY TF_B motif predicts TF_A binding (e.g., motif similarity, protein interaction, shared chromatin context).
- "The effect is small but real" — effect sizes below threshold (fold change < 1.2 AND Cohen's d < 0.3) do not meet the statistical support criterion.
- "Data cannot answer the question" — insufficient data is NOT convergence.

Co-occurrence alone does not warrant convergence. The mechanism can be anything that fits the evidence. When converged, set mechanism_category_number to null and put a descriptive name in mechanism_category_name. The taxonomy is illustrative only; do not force the mechanism into a category.
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
        used_biology_layers: Among SUPPORTED hypotheses, which of those layers were used (from required_data).
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
        string_checked = "string" in (used_biology_layers or [])
        biology_section = (
            "\n## Biology layers (expression / conservation / PPI)\n\n"
            f"- **STRING/PPI** (always required): {'✓ checked' if string_checked else '✗ NOT yet checked'}\n"
            f"- **Expression/conservation layers available in manifest**: {avail_str}\n"
            f"- **Biology layers used in any hypothesis (any result)**: {used}\n\n"
            "Criterion 5a: STRING must be checked in at least one hypothesis (any result). "
            "If STRING was never attempted, set converged=false.\n"
            "Criterion 5b: If rnaseq or phyloP is listed above as available, at least one hypothesis must have "
            "addressed functional relevance via rnaseq, phyloP, OR GO/pathway enrichment (any result). "
            "Verify from the evidence summaries that the analysis was actually performed — a layer listed in "
            "required_data but absent from the evidence summary does NOT satisfy 5b. "
            "Set converged=false if none of rnaseq/phyloP/GO was genuinely analyzed.\n"
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
1. Statistical support: at least one hypothesis with support_level = "SUPPORTS" (not just promising numbers inside a REFUSES result)
2. Named mechanism: a specific molecular process with a clear causal/mechanistic interpretation, not merely correlation. The mechanism can be anything that fits the evidence—it need not match any predefined category. Set mechanism_category_number to null and put a descriptive name in mechanism_category_name.
3. But-for test (only when causal-capable data: causal, not merely correlational)
4. Cross-layer consistency (consistent directional support from >= 2 independent omics layers)
5. Biology layers — two independent sub-requirements (BOTH must be met):
   5a. STRING/PPI (always required): at least one hypothesis must have tested STRING (any result). Always required regardless of manifest.
   5b. Functional characterization (when available): if rnaseq or phyloP is in the manifest, at least one hypothesis must have addressed functional relevance via rnaseq, phyloP, OR GO/pathway enrichment analysis (any result). STRING does NOT substitute.

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
    "support_level": "SUPPORTS" | "REFUSES" | "INCONCLUSIVE" | "ERROR",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of the interpretation",
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
section above (SUPPORTS, REFUSES, INCONCLUSIVE, ERROR, UNTESTABLE). Do NOT upgrade a
REFUSES/ERROR/INCONCLUSIVE result to "supported" or "confirmed" in the narrative. If no
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
