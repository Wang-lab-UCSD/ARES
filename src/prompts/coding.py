"""Prompt templates for code generation."""

from __future__ import annotations

from typing import Any

from src.prompts.tool_quirks import DEFAULT_QUIRKS, get_quirks_for_tools
from src.utils.config import DEFAULT_EXECUTION_PACKAGES

# =============================================================================
# Main coding system prompt (tool-agnostic)
# =============================================================================
CODING_SYSTEM_PROMPT_BASE = """You are an expert bioinformatics programmer. Write Python code to verify scientific hypotheses using the data provided.

=== HARD BANS — every item below causes automatic review rejection ===

1. `pd.qcut()` on any DataFrame column → use `matched_bin(fg, bg, col)` instead
2. `subprocess` call to `bedtools closest` → use `run_bedtools_closest_to_tss()` instead
3. `.merge(... on='name' ...)` on narrowPeak frames → ENCODE `name` column is always `'.'`; join on coordinates or `peak_id`. Never pass a `lambda` as a `merge` key — pandas does not support it and raises `KeyError`
4. `bw.stats()` directly → use `extract_bigwig_signals(df, bw_path)` instead
5. `bedtools getfasta`, `fimo`, or `bw.stats()` inside any `for`/`while` loop → call each ONCE outside all loops
5a. Direct `subprocess` call to `fimo` or `bedtools getfasta` when `run_fimo_on_peaks()` can handle it → always use `run_fimo_on_peaks(peaks_df, genome_fa, meme_file, motif_id)`. Direct FIMO calls without `--motif` scan all 800+ JASPAR motifs and time out in 15 minutes.
5b. Running `fimo`, `bedtools getfasta`, or `bigWigAverageOverBed` on the full dataset as the first attempt → always test on `peaks.head(200)` first to confirm correct output format and timing, then run on the full dataset.
6. `subprocess.run(..., shell=True)` or `bash -lc` for CLI tools → use `subprocess.run([...], shell=False)`
7. Hardcoded ENCFF IDs anywhere in the script → use `data_files['key']` (pre-injected dict)
8. `matplotlib` / `seaborn` imports → output statistics only, no plots
9. More than 100 permutation replicates → hard cap is 100; ignore any higher number in the hypothesis
10. Tiling the genome or annotation for background → use `bedtools shuffle -i peaks.bed -g chrom.sizes`
11. Scanning all FIMO motifs when hypothesis names specific ones → pass `--motif <ID>`; use `find_motif_ids_for_tf()` to look up the ID
11a. Parsing FIMO output manually with `pd.read_csv`, column-index slicing, or hand-rolling `sequence_name`/`motif_id` column detection → always use `parse_fimo_tsv(path)`; it handles both `fimo.txt` and `fimo.tsv` format variants and normalizes column names
12. Custom multi-file concat helpers (e.g. `_concat_files()`, `_merge_files()`) → use `pd.concat([pd.read_csv(f, ...) for f in files])` inline
13. `pd.read_csv(string_file, sep=' ')` or `sep='\t'` for STRING files → STRING uses variable whitespace; always use `load_string_links(path)`

=== REQUIRED HELPERS — use these; hand-rolled alternatives are banned ===

| Operation | Required helper | Never use |
|-----------|----------------|-----------|
| FIMO motif scan | `run_fimo_on_peaks(peaks, genome, meme, motif_id)` | manual getfasta + fimo |
| Peak overlap fraction | `count_overlapping_peaks(query, subject)` | manual `-a`/`-b` or hand-count |
| bigWig signal | `extract_bigwig_signals(df, bw_path)` | `bw.stats()` or row loops |
| Signal-matched controls | `matched_bin(fg, bg, signal_col)` | `pd.qcut()` |
| Nearest TSS + expression | `link_peaks_to_expression(peaks, rnaseq, gtf)` | chained TSS calls |
| Nearest TSS distance only | `run_bedtools_closest_to_tss(peaks_df_or_path, gtf_path=gtf)` | subprocess bedtools closest |
| ChromHMM annotation | `annotate_peaks_with_chromhmm(peaks, chromhmm)` | manual intersect |
| narrowPeak loading | `read_narrowpeak(path)` | `pd.read_csv` with column guesses |
| FIMO TSV parsing | `parse_fimo_tsv(path)` | manual `read_csv` |
| bedtools `-wa -wb` output | `parse_bedtools_wa_wb(path, a_col_count, b_col_count)` | positional indexing |
| bedtools `closest -d` output | `parse_bedtools_closest(path, a_col_count, b_col_count)` | positional indexing |
| Look up motif IDs by TF name | `find_motif_ids_for_tf(tf_name, meme_file)` | hardcoded IDs |
| RNA-seq loading | `load_rnaseq_with_gene_id(path)` or `load_rnaseq_expression(path)` | manual column parsing |
| STRING PPI links | `load_string_links(path, min_score=400)` | `pd.read_csv(sep=' ')`, manual parsing |
| STRING API (shared cofactors) | `fetch_shared_partners(["TF_A", "TF_B"])` from `src.utils.string_client` | custom `requests` calls |
| Two-peak coordinate join | `intersect_peaks(df_a, df_b, mode='flag'/'count'/'wa-wb')` | `.merge(on='name')` or manual bedtools |
| GO / pathway enrichment | `run_go_enrichment(gene_list)` from `src.utils.bioio` | `requests` to Enrichr/g:Profiler directly |
| Motif-to-motif comparison | `run_tomtom(meme_file, motif_id_1, motif_id_2)` | manual PWM comparison or FIMO-based approximation |

All helpers are in `src.utils.bioio` (file-based) or `src.utils.string_client` (STRING API). Only bypass a helper if it genuinely cannot produce the output shape you need, and explain why in a comment.

=== API REFERENCE ===

```python
from src.utils.bioio import (
    find_motif_ids_for_tf, run_fimo_on_peaks, count_overlapping_peaks,
    extract_bigwig_signals, matched_bin, link_peaks_to_expression,
    annotate_peaks_with_chromhmm, create_tss_bed_from_gencode, load_gencode_genes,
    load_rnaseq_expression, merge_rnaseq_with_nearest_genes, read_narrowpeak,
    run_bedtools_closest_to_tss, parse_fimo_tsv, parse_chromhmm_intersect,
    load_rnaseq_with_gene_id, parse_bedtools_closest, parse_bedtools_wa_wb,
    load_string_links, run_tomtom,
)

# find_motif_ids_for_tf
motif_ids = find_motif_ids_for_tf(
    "NFYA", meme_file,
    allowed_sources=["jaspar", "hocomoco12", "cisbp"],  # optional
    match_prefix=True,  # default; False for exact match
)
# → list[str], e.g. ["NFYA|jaspar|MA0060.3"]

# run_fimo_on_peaks  (parameter names matter — use exactly these)
fimo_df = run_fimo_on_peaks(
    peaks_bed=peaks_bed,        # file path (str/Path), NOT a DataFrame
    genome_fasta=data_files['fasta'],
    meme_file=data_files['motif_meme'],
    motif_id=motif_id,          # exact string from find_motif_ids_for_tf()
    p_value_threshold=1e-4,     # NOT pval_thresh, NOT pvalue_threshold
    summit_window=150,          # optional: scan summit ± N bp
)
# → DataFrame: motif_id, sequence_name, start, stop, p-value, peak_id
# peak_id matches col-4 (name) of peaks_bed — use for peak assignment
# IMPORTANT: if peaks_bed is a NamedTemporaryFile path, call .flush() or .close()
# BEFORE passing the path — unflushed buffers produce an empty file → EmptyDataError

# count_overlapping_peaks
result = count_overlapping_peaks(query_bed, subject_bed)
# query = set you are asking ABOUT (defines denominator)
# → dict: {n_query, n_overlapping, n_nonoverlapping, fraction}

# extract_bigwig_signals — returns a Series, NOT a DataFrame
# Assign directly to a column; do NOT call .columns or .merge() on the result
df["h3k4me3"] = extract_bigwig_signals(intervals_df, data_files["h3k4me3_bigwig_fold_change_over_control"])
df["h3k27ac"] = extract_bigwig_signals(intervals_df, data_files["h3k27ac_bigwig_fold_change_over_control"])
# stat kwarg: "mean" | "max" | "min" | "std" (default: "mean")

# matched_bin
fg_binned, bg_binned = matched_bin(fg_df, bg_df, signal_col, n_bins=4)
# → (fg_copy, bg_copy) each with added "bin" int column (0-based)
# BG rows outside FG range get bin=NaN — drop with .dropna(subset=["bin"])
# ALIGNMENT: fg_binned shares fg_df's index but may differ in row order.
# Never assign fg_binned["bin"].values back to fg_df positionally.
# Use index alignment: fg_df = fg_df.loc[fg_binned.index].copy(); fg_df["bin"] = fg_binned["bin"]

# link_peaks_to_expression
merged_df, expr_col = link_peaks_to_expression(
    peaks_bed, rnaseq_path, gtf_path, max_distance=50000,
)
# → merged_df cols: peak_chrom, peak_start, peak_end, peak_name,
#                   gene_id, gene_id_clean, distance, <expr_col>
# Peaks with no nearby TSS get NaN in expr_col

# read_narrowpeak
peaks_df = read_narrowpeak(path_or_df)
# → DataFrame: chrom, start, end, name, score, strand,
#              signal_value, p_value, q_value, peak, peak_offset, summit

# parse_fimo_tsv
fimo_df = parse_fimo_tsv(path, p_value_threshold=1e-4, motif_ids=["MA0139.1"])
# → DataFrame: motif_id, sequence_name, start, stop, p-value, peak_id

# parse_chromhmm_intersect
ix_df = parse_chromhmm_intersect(path_or_df, a_col_count=4, chromhmm_cols=4, state_col_in_b=3)
# → DataFrame with A+B cols + "state" column

# load_rnaseq_with_gene_id
rnaseq_df, gene_id_col = load_rnaseq_with_gene_id(path_or_df)

# load_rnaseq_expression
rnaseq_df, gene_id_col, clean_col, expr_col = load_rnaseq_expression(path_or_df)
# clean_col = "gene_id_clean" (version-stripped); expr_col usually "TPM"

# load_gencode_genes
genes_df = load_gencode_genes(gtf_path, protein_coding_only=True)
# → DataFrame: gene_id, gene_id_clean, gene_name, gene_type, chrom, start, end, strand

# create_tss_bed_from_gencode
tss_df = create_tss_bed_from_gencode(gtf_path, upstream=2000, downstream=2000)
# → BED DataFrame: chrom, start, end, gene_id, score, strand, gene_name, gene_type

# run_bedtools_closest_to_tss
# peaks_bed accepts a file path (str/Path) OR a pandas DataFrame with chrom/start/end columns.
# tss_bed is OPTIONAL — provide either tss_bed (BED file) or gtf_path (GENCODE GTF, auto-converted).
closest_df = run_bedtools_closest_to_tss(peaks_df, gtf_path=data_files["gencode"])
# → DataFrame: peak_chrom, peak_start, peak_end, peak_name (A-side, "peak_*" prefix)
#              tss_chrom, tss_start, tss_end, gene_id, score, strand (B-side)
#              distance

# merge_rnaseq_with_nearest_genes
merged_df, expr_col = merge_rnaseq_with_nearest_genes(closest_df, rnaseq_path)

# annotate_peaks_with_chromhmm
chrom_df = annotate_peaks_with_chromhmm(
    peaks_bed, chromhmm_bed,
    promoter_states=["1_TssA", "2_TssFlnk"],
    enhancer_states=["9_EnhA1", "10_EnhA2"],
)
# → DataFrame: chrom, start, end, name, ..., state, is_promoter_state, is_enhancer_state
# Column counts auto-detected — do not specify peak_cols or chromhmm_cols

# parse_bedtools_closest
closest_df = parse_bedtools_closest(path_or_df, a_col_count=4, b_col_count=4)
# → DataFrame with A-side cols, B-side cols, distance

# parse_bedtools_wa_wb
ix_df = parse_bedtools_wa_wb(path_or_df, a_col_count=4, b_col_count=4)
# → DataFrame with A-side and B-side columns in order

# load_string_links
string_df = load_string_links(path, min_score=400)
# → DataFrame: protein1, protein2, combined_score (+ any extra score columns)
# Handles space-separated STRING files with or without header; filters to combined_score >= min_score
# NEVER parse STRING files manually — separator is variable whitespace, headers are inconsistent

# fetch_shared_partners  (STRING API — use when testing shared-cofactor hypotheses)
from src.utils.string_client import fetch_shared_partners
partners_df, shared = fetch_shared_partners(
    ["TF_A", "TF_B"],           # gene symbols; results are cached on disk
    species=9606,               # human
    min_score=700,              # combined score threshold (0–1000)
    min_escore=0.2,             # require experimental evidence
    min_dscore=0.3,             # require curated DB evidence
)
# shared → set of gene symbols that interact with BOTH TF_A and TF_B
# partners_df → DataFrame with columns: preferredName_A, preferredName_B, score
#               (plus stringId_A, stringId_B, escore, dscore when present)
#   NOTE: `queryItem` column may or may not be present — do NOT rely on it for filtering.
#
# To check if TF_A and TF_B directly interact with each other:
#   direct = partners_df[
#       (partners_df["preferredName_A"].isin(["TF_A", "TF_B"])) &
#       (partners_df["preferredName_B"].isin(["TF_A", "TF_B"]))
#   ]
# To find proteins that partner with TF_A (regardless of TF_B):
#   nfya_partners = set(partners_df[partners_df["preferredName_A"] == "TF_A"]["preferredName_B"])
# To find shared cofactors (proteins that partner with BOTH), just use the `shared` set directly.

# intersect_peaks  (coordinate-based peak join — use instead of .merge(on='name'))
from src.utils.bioio import intersect_peaks
flagged = intersect_peaks(df_a, df_b, mode="flag")   # adds bool overlaps_b column
counted = intersect_peaks(df_a, df_b, mode="count")  # adds overlap_count column
joined  = intersect_peaks(df_a, df_b, mode="wa-wb")  # one row per overlapping pair

# run_go_enrichment  (GO/KEGG enrichment via g:Profiler API — no local files needed)
from src.utils.bioio import run_go_enrichment
enrich_df = run_go_enrichment(gene_list, organism="hsapiens", sources=["GO:BP", "GO:MF", "KEGG"])
# enrich_df columns: source, name, p_value, intersection_size, term_size, query_size, native
# Returns empty DataFrame if no significant terms found
print(enrich_df.head(10))

# run_tomtom  (motif-to-motif similarity comparison using Tomtom)
from src.utils.bioio import run_tomtom
result = run_tomtom(data_files["motif_meme"], "NFYA|jaspar|MA0060.3", "NFATC3|jaspar|MA0623.1")
# result is a dict: p_value, e_value, q_value, overlap, query_consensus, target_consensus,
#                   orientation, is_significant (q_value < 0.05), match_found (bool)
# Use this to directly test motif similarity — much faster and more rigorous than scanning
# both motifs with FIMO and comparing overlap percentages.
print(f"Motif similarity: p={result['p_value']:.2e}, overlap={result['overlap']}bp, "
      f"significant={result['is_significant']}")
```

=== PRE-SUBMISSION SELF-CHECK ===

Your code will be reviewed against these checks before execution. Verify:

1. Does the code test what the hypothesis ACTUALLY predicts? Read each prediction verbatim.
2. Are peak overlaps computed with `count_overlapping_peaks()` or `bedtools intersect` — NEVER by merging DataFrames on the `name` column?
3. When the hypothesis says "summit", does the code use summit-window BEDs (from `make_integer_summit_windows()`), not full peak intervals?
4. Are all expensive operations (fimo, getfasta, bigWig) called ONCE outside all loops?
5. Is `data_files` used as-is (never redefined, no hardcoded ENCFF IDs)?
6. Does every `pd.read_csv()` on bedtools output use `safe_read_csv()` to handle empty files?
7. Does FIMO scanning use `run_fimo_on_peaks()` (or explicit `--motif` + `--no-pgc` flags)?

"""


def build_coding_system_prompt(tools: list[str] | None = None) -> str:
    """Build the complete coding system prompt with relevant tool quirks.

    Args:
        tools: List of tools being used (e.g., ['fimo', 'bedtools']).
               If None, includes default quirks (FIMO + bedtools).

    Returns:
        Complete system prompt with tool-specific quirks appended.
    """
    if tools is None:
        # Include default quirks for common bioinformatics workflows
        quirks = DEFAULT_QUIRKS
    else:
        quirks = get_quirks_for_tools(tools)

    if quirks:
        return CODING_SYSTEM_PROMPT_BASE + "\n" + quirks
    return CODING_SYSTEM_PROMPT_BASE


def _format_prior_evidence(prior_evidence: list[dict[str, Any]]) -> str:
    """Format all prior iteration findings and computations for the coding prompt."""
    if not prior_evidence:
        return ""

    parts = [
        "# Prior Iteration Results",
        "",
        "The Jupyter kernel is shared across all iterations. Data computed in prior",
        "iterations (FIMO scans, peak overlaps, getfasta output, bigWig extractions)",
        "may still be available as Python variables — check before recomputing.",
        "",
    ]
    for e in prior_evidence:
        support = e.get("support_level", "UNKNOWN")
        parts.append(f"**{e.get('hypothesis_name', 'N/A')}** → {support}:")
        parts.append(f"  {e.get('summary', 'No summary')[:300]}")
        parts.append("")

    supported = [e for e in prior_evidence if e.get("support_level") in ("SUPPORTS", "INCONCLUSIVE")]
    if supported:
        parts += [
            "The following were SUPPORTED or INCONCLUSIVE — your test MUST show the",
            "current hypothesis explains something these cannot, or control for them.",
            "",
        ]

    return "\n".join(parts)


# =============================================================================
# Code skeleton generator
# =============================================================================

_SKELETON_SAFETY_HELPERS = '''\
def safe_read_csv(path, expected_columns=None, **kwargs):
    """Read CSV/TSV; return empty DataFrame if file is empty (common with bedtools)."""
    if os.path.getsize(path) == 0:
        return pd.DataFrame(columns=expected_columns or [])
    return pd.read_csv(path, **kwargs)

def make_integer_summit_windows(peaks_df, window_bp=250):
    """Build summit-centred BED intervals with integer coordinates."""
    if "summit" in peaks_df.columns:
        s = peaks_df["summit"]
    elif "peak" in peaks_df.columns:
        s = peaks_df["start"] + peaks_df["peak"].fillna(0).astype(int)
    else:
        s = (peaks_df["start"] + peaks_df["end"]) // 2
    s = s.astype(int)
    return pd.DataFrame({
        "chrom": peaks_df["chrom"],
        "start": (s - window_bp).clip(lower=0),
        "end": s + window_bp,
    })'''

_SKELETON_FILE_SIZE_CHECK = '''\
for _key, _val in data_files.items():
    if isinstance(_val, str) and os.path.isfile(_val):
        print(f"  {_key}: {os.path.getsize(_val)/1e6:.1f} MB")'''


def _generate_code_skeleton(
    hypothesis: dict[str, Any],
    data_manifest: dict[str, Any],
) -> str:
    """Generate a code skeleton dynamically from hypothesis and manifest.

    Provides correct imports, data-file unpacking, and safety helpers so the LLM
    only needs to write the analysis logic.
    """
    required_data = hypothesis.get("required_data") or []
    required_blob = " ".join(required_data) if isinstance(required_data, list) else str(required_data)
    text = " ".join([
        str(hypothesis.get("name", "")),
        str(hypothesis.get("prediction", "")),
        str(hypothesis.get("verification_plan", "")),
        required_blob,
    ]).lower()

    imports: set[str] = {"read_narrowpeak", "extract_bigwig_signals"}

    if any(k in text for k in ("fimo", "motif", "pwm", "jaspar", "hocomoco", "cisbp", "motifs", "binding site")):
        imports.update(["run_fimo_on_peaks", "find_motif_ids_for_tf"])

    if any(k in text for k in ("overlap", "co-occur", "co-occupied", "shared", "intersect", "fraction of")):
        imports.update(["count_overlapping_peaks", "intersect_peaks"])

    if any(k in text for k in ("match", "control", "confound", "dnase-match", "accessibility-match")):
        imports.add("matched_bin")

    if any(k in text for k in ("rnaseq", "expression", "tpm", "tss", "gene", "gencode", "promoter", "nearest")):
        imports.update(["link_peaks_to_expression", "load_rnaseq_expression", "create_tss_bed_from_gencode"])

    if any(k in text for k in ("chromhmm", "chromatin state", "promoter state", "enhancer state")):
        imports.add("annotate_peaks_with_chromhmm")

    # Also check manifest data keys
    data = data_manifest.get("data") or {}
    manifest_keys = " ".join(
        k.lower() for d in data.values() if isinstance(d, dict) for k in d
    ) + " " + " ".join(k.lower() for k in data)

    if any(k in manifest_keys for k in ("meme", "motif", "pwm")):
        imports.update(["run_fimo_on_peaks", "find_motif_ids_for_tf"])
    if any(k in manifest_keys for k in ("rnaseq", "rna", "expression", "gtf", "gencode")):
        imports.update(["link_peaks_to_expression", "load_rnaseq_expression"])
    if "chromhmm" in manifest_keys:
        imports.add("annotate_peaks_with_chromhmm")

    import_list = sorted(imports)
    import_block = (
        "from src.utils.bioio import (\n    "
        + ",\n    ".join(import_list)
        + ",\n)"
    )

    # Flatten manifest data section to get file paths
    file_vars: list[tuple[str, str]] = []
    for _category, items in data.items():
        if not isinstance(items, dict):
            continue
        for name, value in items.items():
            if not isinstance(value, str):
                continue
            if not value.startswith("/"):
                continue
            var_name = name.lower().replace("-", "_")
            file_vars.append((var_name, name))

    unpack_lines = "\n".join(
        f'{var} = data_files["{key}"]' for var, key in file_vars
    ) if file_vars else "# (no file paths detected in manifest)"

    # Detect if peak overlap / motif joining is likely
    needs_peak_overlap = any(k in text for k in (
        "overlap", "co-occur", "co-occupied", "shared", "intersect",
        "motif", "fimo", "bound", "unbound", "recruit", "tether",
    ))

    peak_warning = ""
    if needs_peak_overlap:
        imports.add("intersect_peaks")
        peak_warning = '''\

# !!! CRITICAL — DO NOT use .merge(on='name') !!!
# read_narrowpeak() drops the 'name' column (it is '.' for ALL ENCODE peaks).
# .merge(on='name') will raise KeyError or cause a cartesian product → WILL BE REJECTED.
#
# USE THESE INSTEAD:
#
# (A) Join two peak sets by coordinate overlap → returns paired DataFrame:
#     overlap_df = intersect_peaks(peaks_a_df_or_bed, peaks_b_df_or_bed)
#     # overlap_df has columns a_0..a_3, b_0..b_3 (one row per overlap pair)
#
# (B) Flag which A-peaks overlap any B-peak → boolean column:
#     flagged = intersect_peaks(peaks_a_df, peaks_b_df, mode="flag")
#     # When inputs are DataFrames, ALL original columns are preserved:
#     # flagged has chrom, start, end, ..., summit, overlaps_b
#     # Use flagged["chrom"], flagged["start"], flagged["summit"], etc.
#
# (C) Count overlaps per A-peak:
#     counted = intersect_peaks(peaks_a_df, peaks_b_df, mode="count")
#     # counted has chrom, start, end, ..., summit, overlap_count
#
# (D) Get overlap fraction/counts:
#     stats = count_overlapping_peaks(query_bed, subject_bed)
#
# (E) Split peaks into motif-positive / motif-negative:
#     fimo_df = run_fimo_on_peaks(peaks_bed, genome, meme, motif_id)
#     peaks = peaks.reset_index(drop=True)
#     peaks["_peak_id"] = [f"peak_{i:05d}" for i in range(len(peaks))]
#     motif_pos = peaks[peaks["_peak_id"].isin(fimo_df["peak_id"])]
#     motif_neg = peaks[~peaks["_peak_id"].isin(fimo_df["peak_id"])]'''

    parts = [
        "# === AUTO-GENERATED SKELETON — extend with your analysis ===",
        "import os, tempfile, subprocess",
        "import numpy as np",
        "import pandas as pd",
        "from scipy import stats",
        "",
        import_block,
        "",
        "# ---- Safety helpers (use these in your code) ----",
        _SKELETON_SAFETY_HELPERS,
        "",
        "# ---- STEP 0: Data file paths and sizes ----",
        unpack_lines,
        peak_warning,
        "",
        _SKELETON_FILE_SIZE_CHECK,
    ]

    return "\n".join(parts)


def _build_review_constraints_block(issues: list[str]) -> str:
    """Convert reviewer blocker issue strings into a machine-readable constraint block.

    Returns a formatted code block string (or empty string if no constraints apply).
    """
    if not issues:
        return ""

    joined = " ".join(issues).lower()

    forbidden: list[str] = []
    required: list[str] = []

    if "qcut" in joined:
        forbidden.append("pd.qcut(")
        required.append("matched_bin(fg_df, bg_df, signal_col)  # from src.utils.bioio")

    if "bedtools closest" in joined:
        forbidden.append("subprocess call to 'bedtools closest'")
        required.append("run_bedtools_closest_to_tss(peaks_df, gtf_path=gtf)  # from src.utils.bioio")

    if "link_peaks_to_expression" in joined or (
        "nearest" in joined and "gene" in joined and "expression" in joined
    ):
        required.append("link_peaks_to_expression(peaks_bed, rnaseq_path, gtf_path)  # from src.utils.bioio")

    if "merg" in joined and "name" in joined and (
        "narrowpeak" in joined or "encode" in joined or "'.'" in joined
        or "cartesian" in joined
    ):
        forbidden.append(".merge(... on='name' ...)  # narrowPeak name col is always '.'")
        forbidden.append("pd.merge(df_a, df_b, on='name')")
        required.append("intersect_peaks(peaks_a, peaks_b)  # from src.utils.bioio — coordinate-based join via bedtools")
        required.append("intersect_peaks(peaks_a, peaks_b, mode='flag')  # boolean overlaps_b column")
        required.append("count_overlapping_peaks(query_bed, subject_bed)  # fraction/counts")

    if "bw.stats" in joined:
        forbidden.append("bw.stats(  # direct pyBigWig call")
        required.append("extract_bigwig_signals(df, bw_path)  # from src.utils.bioio")

    if "empty bedtools output" in joined or "no columns to parse from file" in joined:
        required.append("Use safe_read_csv_0_safe(...) for any bedtools output file reads to avoid EmptyDataError / empty-file crashes")

    if "fractional" in joined and "median" in joined and "peak" in joined:
        forbidden.append("['peak'].median(")
        required.append("Compute summit offsets per-row: summit = start + peak_offset; then cast to int (never use median() peak offsets)")

    if "motif+" in joined and "unbound" in joined:
        required.append("Construct motif_positive set from FIMO hits and construct unbound/background set from lack of ChIP overlap; ensure background comparisons enforce motif+ if the prediction says REST-motif+")

    if "shell=true" in joined or "bash -lc" in joined or "bash−lc" in joined or "shell-wrapped" in joined:
        forbidden.append("subprocess.run(..., shell=<BANNED>)  — loses conda PATH")
        forbidden.append("subprocess.run('bash -lc ...', ...)  — loses conda PATH")
        required.append("subprocess.run([...], shell=False, stdout=fh, text=True, check=True)")

    if "encff" in joined:
        forbidden.append("hardcoded ENCFF IDs (anywhere in the code)")
        required.append("data_files['key']  # pre-injected dict — never hardcode paths")

    if "--motif" in joined or "--no-pgc" in joined or (
        "fimo" in joined and ("flag" in joined or "missing" in joined)
    ):
        forbidden.append("fimo CLI without --motif and --no-pgc flags")
        required.append("run_fimo_on_peaks(peaks_bed, genome_fasta, meme_file, motif_id)  # from src.utils.bioio")

    if "loop" in joined and any(k in joined for k in ("getfasta", "fimo", "bigwig", "bw.stats")):
        forbidden.append("bedtools getfasta / fimo / bw.stats inside any for/while loop")
        required.append("call these operations ONCE outside all loops")

    if not forbidden and not required:
        return ""

    lines = [
        "⚠️  AUTOMATIC REJECTION — your previous code violated these rules.",
        "    Read carefully before writing a single line of code.",
        "",
        "NEVER use (any of these causes instant rejection):",
    ]
    for f in forbidden:
        lines.append(f"  ✗  {f}")
    lines += [
        "",
        "ALWAYS use instead:",
    ]
    for r in required:
        lines.append(f"  ✓  {r}")
    lines += [
        "",
        "Your new code MUST NOT contain any of the NEVER patterns above.",
    ]

    return "\n" + "\n".join(lines) + "\n"


def _format_data_for_coding(
    data_manifest: dict[str, Any], file_summaries: dict[str, str] | None = None
) -> str:
    """Format data manifest and optionally append file summaries for coding prompts."""
    parts = []

    data = data_manifest.get("data", {})
    for category, items in data.items():
        parts.append(f"## {category}")
        if isinstance(items, dict):
            for name, path in items.items():
                parts.append(f"- `{name}`: `{path}`")
        else:
            parts.append(f"- `{items}`")
        parts.append("")

    tools = data_manifest.get("tools", [])
    if tools:
        parts.append("## Available CLI Tools (use via subprocess)")
        for tool in tools:
            parts.append(f"- `{tool}`")
            
    if file_summaries and file_summaries.get("all_files"):
        parts.append("")
        parts.append("## File Previews (from pre-run inspection)")
        parts.append("Use these column names and formats exactly as shown below:")
        parts.append("```")
        preview = file_summaries["all_files"]
        if len(preview) > 4000:
            preview = preview[:4000] + "\n... [truncated]"
        parts.append(preview)
        parts.append("```")

    return "\n".join(parts)


def _format_verification_plan(plan: list[str]) -> str:
    """Format verification plan as numbered list."""
    if not plan:
        return "No specific plan provided."
    return "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])


def _format_allowed_packages(packages: list[str]) -> str:
    """Format allowed Python packages for coding prompts."""
    if not packages:
        return ""
    pkg_list = ", ".join(sorted(packages))
    return (
        "# Allowed Python packages\n\n"
        f"Use ONLY these Python packages: {pkg_list}. "
        "Do not use other packages (e.g. pyranges); they are not installed in the execution environment.\n\n"
    )


# =============================================================================
# V2: REPL-style prompts
# =============================================================================

def build_inspection_prompt(data_manifest: dict[str, Any]) -> str:
    """One-shot prompt for file familiarisation (no hypothesis)."""
    data_section = _format_data_for_coding(data_manifest, file_summaries=None)
    return f"""# Task: Inspect All Data Files

Generate Python code to inspect every file listed in the data manifest below.

For each file:
1. Print the file path as a header
2. Print the number of lines (or rows for tabular files)
3. If tabular (TSV, CSV, BED, narrowPeak): print column names and 3 sample rows using pandas
4. If binary (bigWig, bam): print the file size in MB
5. If FASTA or GTF: print line count and first 3 non-comment lines

Print clearly labeled output per file. Do not perform any analysis or statistics.
Do not import matplotlib or generate any plots.
Use try/except around each file so one failure doesn't stop the rest.

# Available Data

{data_section}
"""


def build_repl_system_prompt() -> str:
    """System prompt for the REPL coding agent (v2)."""
    return """You are an expert bioinformatics programmer working in a persistent Python REPL.

## How this works

You reason and execute code in small, incremental steps:

- Use <think>...</think> to reason about what to do next (optional but encouraged).
- Use <execute>...</execute> to run Python code. The kernel is persistent across ALL
  iterations of this run — variables and results computed in previous iterations are
  still in memory. Before re-computing something (FIMO scans, peak overlaps, getfasta,
  bigWig extraction), check whether the result already exists as a variable. Always use exactly `<execute>` with no
  attributes (not `<execute code>`, `<execute python>`, etc.).
  **Send exactly ONE <execute> block per message.** Wait for the observation before
  writing the next block. Do NOT send multiple <execute> blocks in one response —
  only the first will be executed.
- Use <solution>...</solution> ONLY when you have enough evidence to conclude, and
  NEVER in the same message as an <execute> block. First run code, observe results,
  then in a later message emit <solution>.
- **Every response MUST contain either an <execute> or <solution> block.** Do not narrate
  what you plan to do — just do it. Responses like "Let me start by..." or "I'll verify
  this step by step" without an <execute> block waste an iteration.

## Solution format

<solution>
support_level: SUPPORTS|REFUTES|INCONCLUSIVE|ERROR|UNTESTABLE
confidence: 0.0-1.0
finding: <one concise sentence stating the key result>
reasoning: <explanation of the evidence and why it supports/refutes the hypothesis>
</solution>

## Guidelines

- Run small steps: load data → inspect → compute → conclude. Do not write one huge script.
- Always print key intermediate results so you can verify them before proceeding.
- If a step fails, read the error, fix just that part, and try again.
- No matplotlib or seaborn — print statistics only.
- Max 100 permutation replicates.
- Use data_files['key'] to access files — it is pre-injected into the kernel.
- **Early stopping**: If the first prediction in the verification plan is clearly refuted
  (wrong direction, not significant, or effect far below threshold), stop immediately and
  emit a REFUTES <solution>. Do NOT continue testing remaining predictions — they cannot
  rescue a failed core prediction. This saves REPL iterations and cost.

""" + CODING_SYSTEM_PROMPT_BASE


def build_repl_initial_prompt(
    hypothesis: dict[str, Any],
    data_manifest: dict[str, Any],
    prior_evidence: list[dict[str, Any]] | None = None,
    file_summaries: dict[str, str] | None = None,
    allowed_packages: list[str] | None = None,
) -> str:
    """Initial user message for the REPL verification loop."""
    parts: list[str] = []

    # Hypothesis
    parts.append("# Hypothesis to verify")
    parts.append(f"**Name**: {hypothesis.get('name', 'N/A')}")
    parts.append(f"**Prediction**: {hypothesis.get('prediction', 'N/A')}")
    vplan = hypothesis.get("verification_plan", [])
    if vplan:
        parts.append("\n**Verification plan**:")
        for i, step in enumerate(vplan, 1):
            parts.append(f"  {i}. {step}")

    # Prior evidence
    if prior_evidence:
        parts.append("\n# Prior evidence from earlier iterations")
        parts.append(_format_prior_evidence(prior_evidence))

    # Data
    parts.append("\n# Available data")
    parts.append(_format_data_for_coding(data_manifest, file_summaries=file_summaries))

    # Allowed packages
    if allowed_packages:
        parts.append("\n# Allowed packages")
        parts.append(_format_allowed_packages(allowed_packages))

    # API reference (abridged)
    parts.append("""
# Key helper functions (src.utils.bioio)

```python
from src.utils.bioio import (
    read_narrowpeak, find_motif_ids_for_tf, run_fimo_on_peaks,
    count_overlapping_peaks, extract_bigwig_signals, matched_bin,
    annotate_peaks_with_chromhmm, run_bedtools_closest_to_tss,
    link_peaks_to_expression, load_string_links,
)
# data_files dict is already injected — use data_files['key'] for all paths
```

Important:
- FIMO: always pass `--motif <id>` and `--no-pgc`; use run_fimo_on_peaks()
- bigWig: use extract_bigwig_signals(df, bw_path) not bw.stats() in loops
- narrowPeak: use read_narrowpeak(path) — name column is always '.' in ENCODE files
""")

    parts.append(
        "\nStart by loading and inspecting the relevant data, then run your analysis "
        "step by step. When you have sufficient evidence, emit <solution>.</solution>."
    )

    return "\n".join(parts)
