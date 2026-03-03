"""Prompt templates for result summarization."""

from __future__ import annotations

from typing import Any

SUMMARY_SYSTEM_PROMPT = """You are a scientific result interpreter specializing in bioinformatics. Your role is to:

1. Analyze experimental results and code outputs
2. Determine whether results support, refuse, or are inconclusive for a hypothesis
3. Extract key findings and statistics
4. Identify any issues or anomalies in the results
5. Prepare concise summaries for the hypothesis refinement process

Be objective and precise in your interpretations.

=== SUPPORT LEVEL CRITERIA (you MUST follow these rules) ===

Evaluate support_level by comparing the results against the hypothesis's **prediction**
field. The prediction states what should be observed if the hypothesis is true — your
job is to check whether that specific prediction was confirmed.

**SUPPORTS** — ALL of the following must be true:
  1. The predicted effect exists in the data
  2. p < 0.05
  3. fold change >= 1.5 OR Cohen's d >= 0.4
  If all three are met, set support_level = "SUPPORTS" and confidence >= 0.8.

**INCONCLUSIVE** — The effect is real but modest:
  1. p < 0.05
  2. 1.2 <= fold change < 1.5 OR 0.2 <= Cohen's d < 0.4
  Set support_level = "INCONCLUSIVE" and confidence 0.4-0.7.

**REFUSES** — ANY of the following:
  1. The predicted effect is absent or reversed (e.g., depletion instead of enrichment)
  2. p >= 0.05 with adequate sample size (N >= 30)
  3. fold change < 1.2 AND Cohen's d < 0.2
  If clearly refused, set support_level = "REFUSES" and confidence >= 0.7.

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


CONVERGENCE_CHECK_SYSTEM_PROMPT = """You are an independent scientific adjudicator for a bioinformatics hypothesis testing pipeline. Your sole job is to evaluate whether the accumulated experimental evidence is sufficient to declare convergence on a named causal mechanism.

You have NO stake in any particular outcome. You were not involved in generating the hypotheses. Evaluate the evidence skeptically and objectively.

## Convergence Criteria

Declare converged=true ONLY when ALL FOUR conditions are met:

1. **Statistical support**: At least one result shows p < 0.05 with meaningful effect size (fold change >= 1.5 OR Cohen's d >= 0.4).

2. **Named mechanism**: The evidence maps to a specific numbered category from the mechanism taxonomy. Co-occurrence and correlation do NOT qualify — you must identify a concrete molecular or structural mechanism.

3. **But-for test**: Ask — "If the proposed causal agent were absent, would the data look different?" If the answer is "not necessarily" (because the result could reflect passive co-occurrence, shared active chromatin, or any confound), do NOT converge.

4. **Cross-layer consistency**: The proposed mechanism must be supported by consistent directional evidence from at least two independent omics layers (e.g., ChIP-seq + DNase-seq, or motif analysis + histone marks, or Hi-C + expression). A single data type is not sufficient — convergence requires cross-validation across independent measurement modalities.

## Mechanism Taxonomy

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

- "Co-bound sites are in active chromatin" — correlation. Active sites attract many TFs. Does NOT establish mechanism unless pioneer activity is shown (mechanism #3 requires the pioneer to OPEN the site, not merely be present at already-open sites).
- "TF_B signal is higher where TF_A is present" — restates the original finding. Not a mechanism.
- "The effect is small but real" — effect sizes below threshold (fold change < 1.5 AND Cohen's d < 0.4) do not meet the statistical support criterion.
- "Data cannot answer the question" — insufficient data is NOT convergence.

Co-occurrence alone maps to no category and does not warrant convergence. Alternatively, if the evidence supports a mechanism not listed here, convergence is permitted provided all three criteria are met AND the proposed mechanism: (1) names a specific molecular process (e.g., a named enzymatic activity, a structural interaction, a defined signal transduction step), (2) states a clear causal chain (what acts on what, in what order), and (3) is not merely a re-description of the observed correlation.
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
) -> str:
    """Build prompt for the independent convergence check.

    Args:
        finding: Original scientific finding
        tested_hypotheses: All hypotheses tested so far with results
        last_result: Summary of the most recent result
        raw_output: Raw stdout from the last code execution (will be truncated)
        max_raw_chars: Maximum characters of raw output to include

    Returns:
        Formatted prompt string
    """
    history_lines = []
    for i, th in enumerate(tested_hypotheses):
        history_lines.append(
            f"### Iteration {i + 1}: {th.get('name', 'N/A')}\n"
            f"- Result: {th.get('result', 'N/A')}\n"
            f"- Evidence: {str(th.get('evidence_summary', 'N/A'))[:300]}\n"
            f"- Decision: {th.get('decision', 'N/A')}\n"
        )
    history_text = "\n".join(history_lines) if history_lines else "No hypotheses tested yet."

    raw_section = ""
    if raw_output:
        truncated = _truncate_output(raw_output, max_raw_chars)
        raw_section = f"\n## Raw Execution Output (last experiment)\n\n```\n{truncated}\n```\n"

    prompt = f"""# Scientific Finding Under Investigation

{finding}

## Hypothesis Testing History

{history_text}

## Latest Result Summary

**Support level**: {last_result.get('support_level', 'N/A')}
**Confidence**: {last_result.get('confidence', 'N/A')}
**Summary**: {last_result.get('summary', 'N/A')}
**Findings**: {last_result.get('findings', [])}
{raw_section}
## Task

Evaluate whether the accumulated evidence is sufficient to declare convergence on a named causal mechanism.

Apply ALL FOUR convergence criteria strictly:
1. Statistical support (p < 0.05, fold change >= 1.5 or Cohen's d >= 0.4)
2. Named mechanism from taxonomy (cite category number)
3. But-for test (causal, not merely correlational)
4. Cross-layer consistency (consistent directional support from >= 2 independent omics layers)

Respond in JSON format:
{{
    "converged": true | false,
    "mechanism_category_number": <integer 1-12, or null if not converged>,
    "mechanism_category_name": "<string, or null if not converged>",
    "confidence": 0.0-1.0,
    "conclusion": "<one-paragraph causal conclusion if converged, else empty string>",
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
) -> str:
    """Build prompt for generating the final report.

    Args:
        finding: Original scientific finding
        context: Research context
        history_summary: Summary of all iterations
        conclusion: Final conclusion
        all_evidence: All accumulated evidence

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

# Task

Generate a comprehensive final report summarizing the investigation. The report should include:

1. **Executive Summary**: Brief overview of the finding and conclusion
2. **Methodology**: How the investigation was conducted
3. **Key Findings**: Most important discoveries
4. **Evidence Summary**: Supporting evidence for the conclusion
5. **Confidence Level**: How confident we are in the conclusion
6. **Limitations**: What we couldn't determine or potential issues
7. **Recommendations**: Suggested follow-up experiments if any

Format the report in Markdown.
"""
    return prompt
