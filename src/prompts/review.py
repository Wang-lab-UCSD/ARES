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

**1. FIMO --motif and --no-pgc flags**
If the code calls `fimo` directly as a CLI command (via subprocess):
- The command MUST include `--motif <MOTIF_ID>` when the hypothesis only needs specific motif(s). Without it, FIMO scans all 800+ motifs in JASPAR which takes hours instead of seconds.
- The command MUST include `--no-pgc`. Without it, FIMO parses FASTA headers as genomic coordinates and reports `sequence_name` as chromosome (e.g. chr7) instead of your peak ID (e.g. ATF3_000000). Merging FIMO output to peaks by peak_id then matches nothing → 0 motif hits even when motifs are present.
- If `--motif` is missing and the hypothesis names specific motifs → FIX: add `--motif`
- If `--no-pgc` is missing → FIX: add `--no-pgc` so sequence_name stays as the FASTA header (peak ID).
- If the hypothesis genuinely requires scanning all motifs → APPROVE as-is for --motif only; still require --no-pgc.
- **EXCEPTION: If the code uses `run_fimo_on_peaks()` from `src.utils.bioio`, do NOT flag missing flags.** That helper always passes `--no-pgc` and `--motif` internally. Asking the model to bypass the helper and call FIMO manually is counterproductive.

**2. Focused statistical testing**
The code should have ONE primary statistical test that directly answers the hypothesis prediction.
However, additional closely related tests ARE ALLOWED when:
- They are part of the same verification plan (e.g., testing in promoter AND enhancer strata as specified by the hypothesis)
- They serve as internal controls or specificity checks explicitly required by the prediction
- They share the same input data and test the same mechanism from complementary angles

Only REJECT for this rule if the code runs truly UNRELATED tests (e.g., distance + GC% + expression when the hypothesis only asks about one of them). Do NOT reject for having stratified tests, meta-analyses across strata, or matched controls that the hypothesis verification plan explicitly requires.
- Data loading, FIMO scanning, file parsing are fine — those are setup, not extra tests

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
- `data_files` is injected by the pipeline runtime before the script runs. Do NOT flag
  `data_files` as undefined merely because it is not created inside the script.
  Only flag this area if the script redefines `data_files` / `data_manifest` locally,
  hardcodes paths or ENCFF IDs, or uses keys that do not exist in the manifest.
- Using `bash -lc`, `shell=True`, or `>` redirection for tool commands such as
  `bedtools`, `samtools`, `cut`, or `fimo`. This is a real runtime bug because a
  login shell can reset `PATH` and lose the active conda environment. Prefer
  `subprocess.run([...], stdout=fh, text=True, check=True)` with `shell=False`.

**7. Code matches hypothesis**
Read the hypothesis prediction carefully. Does the code actually test that specific claim? For example, if the hypothesis says "the REST motif (21bp) contains the ATF6 motif (5bp) as a substring," the code must scan the motif sequence itself — NOT scan broad peak regions (hundreds of bp) for the pattern, which would test a different question. If the code tests something different from what the prediction states → REJECT (do not fix — the mismatch is too fundamental).

**8. FIMO p-value vs q-value filtering**
When code filters FIMO output for peak-level motif analysis, prefer `p-value < 1e-4` over
q-value. FIMO's q-value applies genome-wide multiple testing correction that may be overly
conservative for short peak regions — legitimate hits in control/shuffled regions can get
q > 0.05, producing zero results. If the code filters on q-value and gets zero or
suspiciously few hits, flag it and suggest switching to p-value filtering.

**9. Shared parser helpers**
For these fragile parsing tasks, prefer the shared helpers in `src.utils.bioio` over
hand-rolled parsing code:
- narrowPeak summit parsing → `read_narrowpeak`
- FIMO TSV parsing → `parse_fimo_tsv`
- ChromHMM intersect parsing → `parse_chromhmm_intersect`
- named-column RNA-seq loading → `load_rnaseq_with_gene_id`
- `bedtools closest -d` parsing → `parse_bedtools_closest`
- STRING protein interaction files → `load_string_links`

If the script re-implements one of these parsers manually and the replacement is brittle
(e.g. positional column slicing, manual summit fallback, fixed-width truncation), REJECT
and suggest using the shared helper instead.

**10. Known-correct patterns — DO NOT flag these**

The following patterns are correct by design. Flagging them wastes review attempts:

- **`read_narrowpeak()` returns `summit` as an absolute genomic coordinate** (computed as
  `start + peak_offset`), NOT a raw offset. It also handles negative peak offsets
  (peak = -1) by falling back to the interval midpoint. The `summit` column is ALWAYS
  a valid absolute genomic coordinate. Do NOT flag summit calculations as incorrect
  when the code uses `read_narrowpeak()` or `make_integer_summit_windows()`.

- **`make_integer_summit_windows()` handles the `summit` column correctly.** When a column
  named `summit` exists, it treats it as an absolute genomic coordinate (which is what
  `read_narrowpeak()` provides). Do NOT flag this as an offset/coordinate confusion.

- **`read_narrowpeak()` does NOT return a `name` column.** It returns:
  `chrom, start, end, score, strand, signal_value, p_value, q_value, peak,
  peak_offset, summit`. The narrowPeak `name` field (col 4) is dropped because ENCODE
  files use `'.'` for all peaks. Code cannot merge on `name` because the column does
  not exist.

- **Fabricating `peak_00000` sequential IDs to match FIMO output is CORRECT.**
  `run_fimo_on_peaks()` auto-assigns sequential IDs (`peak_00000`, `peak_00001`, ...)
  whenever the name column is missing or uniform. Specifically: (a) if the input has
  fewer than 4 columns (BED3), the helper treats names as all `'.'`; (b) if col 4 exists
  but is uniform (e.g. all `'.'` in ENCODE files), sequential IDs are assigned. In both
  cases `fimo_hits['peak_id']` contains `peak_00000`-style values. Code that assigns
  `peaks["_peak_id"] = [f"peak_{{i:05d}}" for i in range(len(peaks))]` and then joins
  on `_peak_id` is the correct pattern. Do NOT flag BED3 input or `peak_00000` IDs as
  a mismatch — the helper handles this internally.

- **`intersect_peaks(..., mode='flag'/'count')` preserves input row order AND column names
  when given DataFrames.** When the A input is a DataFrame, the output is a copy of that
  DataFrame with an extra `overlaps_b` (flag) or `overlap_count` (count) column appended.
  All original columns (`chrom`, `start`, `end`, `summit`, etc.) are preserved. The code
  CAN safely reference `result["chrom"]`, `result["start"]`, `result["summit"]`, etc.
  Do NOT flag column-name mismatches or row-order alignment for these modes.

- **`count_overlapping_peaks()` always returns a dict**, never a float or scalar. The
  return value has keys: `'fraction'`, `'n_query'`, `'n_overlapping'`, `'n_nonoverlapping'`.
  Access results as `result['fraction']`, `result['n_overlapping']`, etc. Do NOT flag
  `count_overlapping_peaks()` as potentially returning a float or ambiguous type.

- **`count_overlapping_peaks()` and `intersect_peaks()` are coordinate-based join
  helpers.** They use `bedtools intersect` internally. If the code uses these instead of
  `.merge()`, that is CORRECT. Do NOT suggest replacing them with pandas merges.

- **`extract_bigwig_signals()` defaults to `chrom='chrom'`, `start='start'`, `end='end'`.**
  Both explicit calls (`extract_bigwig_signals(df, bw_path, chrom='chrom', ...)`) and
  implicit calls (`extract_bigwig_signals(df, bw_path)`) are valid — the column names
  are the same. Do NOT flag "inconsistent" explicit vs implicit calls to this function
  as a real issue.

- **NEVER suggest `.merge(on='name')` in corrected_code.** ENCODE narrowPeak files have
  `'.'` as the name for ALL peaks. Merging on `name` produces a cartesian product and
  will be rejected by the static checker. Use `intersect_peaks()` or
  `count_overlapping_peaks()` instead.

- **`load_string_links(path, min_score=400)` is the correct way to load STRING files.**
  STRING protein-protein interaction files use variable whitespace separators and
  inconsistent headers. `load_string_links()` handles all format variants robustly.
  Do NOT flag this helper as incorrect or suggest `pd.read_csv(sep=' ')` instead.
  Code that calls `load_string_links()` is correct by design — do NOT reject it.

- **Genome-wide FIMO scans are NOT feasible.** Running FIMO on the full hg38 genome FASTA
  takes hours and will time out (the execution limit is 15 minutes). Do NOT require or
  suggest genome-wide FIMO scans. Instead, scanning the union of relevant peak sets
  (e.g., all ChIP-seq peaks merged together) is the correct approximation for finding
  motif instances across the genome. Scanning only within relevant peak windows using
  `run_fimo_on_peaks()` is also acceptable. Do NOT reject code solely because it scans
  peak windows instead of the full genome.

# Task

**Read the ENTIRE script against ALL checklist items before writing your response.**
Do not stop after finding the first problem. The coding model gets exactly one chance to fix
everything you report; returning issues one at a time wastes a review attempt per bug and
the pipeline hits its rejection cap before receiving a clean script.

Scan for every applicable checklist violation, collect them all, then respond once with the
complete list and a single corrected script that fixes all of them.

Respond in JSON format:

{{
    "approved": true,
    "issues": [],
    "corrected_code": null,
    "reasoning": "Code follows all rules"
}}

OR if issues are found:

{{
    "approved": false,
    "issues": ["Issue 1 description", "Issue 2 description", "Issue 3 description"],
    "corrected_code": "... the full corrected Python code with ALL issues fixed ...",
    "reasoning": "Explanation of every fix made"
}}

If you fix the code, return the COMPLETE corrected script (not just the changed lines),
with ALL issues addressed — not just the first one you noticed.
Never "fix" code by adding a local `data_files = {...}` or `data_manifest = {...}` block;
those are injected by the pipeline runtime and must remain external.
"""
    return prompt


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

You have exactly THREE checks. Apply ONLY these. Do not invent additional criteria.

**1. Already answered**
Has this question already been answered — either directly or by logical implication — by a
prior tested hypothesis? This includes:
- Same question reworded (e.g. "A overlaps B" vs "B overlaps A")
- Same mechanism class using the exact same data layers/modalities as a prior test.
- Logical inverse of a confirmed result (e.g. prior SUPPORTS enrichment → testing
  depletion is redundant)

**CRITICAL EXCEPTION**: Testing the *same* mechanism class using a fundamentally *different data modality* (e.g. moving from ChIP-seq binding to RNA-seq expression, phyloP conservation, or Hi-C looping) is NOT a duplicate. In fact, it is REQUIRED for cross-layer convergence. If the new hypothesis tests an existing mechanism idea but uses RNA-seq, phyloP, or a different major data layer to provide cross-layer validation, you MUST APPROVE it.

If the prior list is empty, this check cannot trigger — approve.

**2. Mechanism vs. association**
The hypothesis must name a specific causal mechanism — a molecular event explaining WHY
TF_B predicts TF_A binding. Reject if it merely describes the data (where, what, how much)
or re-confirms co-occurrence without proposing a mechanism.

**EXCEPTION**: If prior hypotheses already SUPPORT a mechanism AND the new hypothesis tests
functional relevance of that supported mechanism (via GO enrichment, RNA-seq expression, or
phyloP conservation), APPROVE it — this is required for convergence, not idle characterization.

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
