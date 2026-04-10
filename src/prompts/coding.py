"""Prompt templates for code generation.

The coding prompt is assembled from per-data-category bundles. Each bundle contains:
  - hard_bans: tool-specific banned patterns (added to the global hard bans list)
  - helper_table_rows: rows for the REQUIRED HELPERS table
  - helper_notes: detailed usage notes for non-obvious helpers

`select_helper_bundles(required_data)` maps the hypothesis's `required_data` field
(a list of `category.leaf_key` strings) to bundle names. The orchestrator passes
those bundles into `build_coding_system_prompt()` / `build_repl_system_prompt()`
so each iteration only loads the helpers it actually needs.

Bundles are deterministic — there is no LLM in the dispatch path. The "always"
bundle (critical mistakes, global hard bans, core peak/bigWig helpers) is
unconditionally included.
"""

from __future__ import annotations

from typing import Any

from src.prompts.tool_quirks import DEFAULT_QUIRKS, get_quirks_for_tools
from src.utils.config import DEFAULT_EXECUTION_PACKAGES


# =============================================================================
# Helper bundles — per-data-category groupings of hard bans, table rows, and
# usage notes. The "always" bundle is included unconditionally.
# =============================================================================

# Map from manifest top-level category (e.g. "string", "rnaseq", "epigenome")
# to the bundle name. Multiple categories can map to the same bundle.
#
# This map MUST cover every top-level category that appears in any manifest
# under examples/. The test test_select_helper_bundles_covers_all_manifest_categories
# scans every manifest file and fails if a category is missing — that's the
# canonical defense against silent fallthrough when a new manifest type is added.
_CATEGORY_TO_BUNDLE: dict[str, str] = {
    # ChIP-seq peaks/bigWigs and the genome FASTA are used by almost every
    # iteration — they're handled by the always bundle's core helpers
    # (count_overlapping_peaks, extract_bigwig_signals, intersect_peaks, ...).
    "chipseq": "always",
    "all_tf_chipseq": "always",
    "genome": "always",
    # phyloP is a bigWig conservation track. It uses the same extract_bigwig_signals
    # helper as histone marks, so it doesn't need a separate bundle. Mapped to
    # "always" so phyloP-only iterations don't get the full epigenome bundle.
    "phyloP": "always",
    # Per-category bundles
    "pwm": "fimo",
    "string": "string",
    "rnaseq": "rnaseq",
    "annotations": "gencode",
    "epigenome": "epigenome",
    "hic": "hic",
}


def select_helper_bundles(required_data: list[str] | None) -> list[str]:
    """Map a hypothesis's required_data list to a deduplicated list of bundle names.

    Args:
        required_data: List of `category.leaf_key` strings (e.g. ["chipseq.SP1_peaks",
            "string.links_file"]). If None or empty, all bundles are loaded as a
            safe fallback.

    Returns:
        Sorted list of bundle names with "always" first, then alphabetical.
        Always includes "always".
    """
    if not required_data:
        # Defensive default: load everything if the hypothesis didn't declare its needs
        bundles = set(_CATEGORY_TO_BUNDLE.values())
    else:
        bundles = {"always"}
        unknown_categories: list[str] = []
        for key in required_data:
            category = key.split(".", 1)[0] if "." in key else key
            bundle = _CATEGORY_TO_BUNDLE.get(category)
            if bundle is not None:
                bundles.add(bundle)
            else:
                unknown_categories.append(category)
        if unknown_categories:
            # Surface the silent fallthrough so future manifest additions don't
            # quietly drop their bundles. Use warnings rather than logging so
            # the message shows up even when the orchestrator's logger is not
            # initialised (e.g. in unit tests, ad hoc scripts).
            import warnings
            warnings.warn(
                f"select_helper_bundles: unknown manifest categories "
                f"{sorted(set(unknown_categories))} have no bundle mapping. "
                f"They were ignored — only the 'always' bundle was loaded for them. "
                f"Add an entry to _CATEGORY_TO_BUNDLE in src/prompts/coding.py.",
                RuntimeWarning,
                stacklevel=2,
            )

    bundles.add("always")  # always-loaded regardless

    # Sort with "always" first, then alphabetical
    sorted_bundles = sorted(bundles)
    if "always" in sorted_bundles:
        sorted_bundles.remove("always")
        sorted_bundles.insert(0, "always")
    return sorted_bundles


# =============================================================================
# required_data validator — detect categories mentioned in a hypothesis's
# verification plan / prediction text but not declared in `required_data`.
#
# This is Step 1 of the "validate-then-maybe-fix" rollout: log-only, no auto-fix.
# The orchestrator calls it right after a hypothesis passes review and emits a
# warning if the keyword scan finds a gap. We gather real-run data from these
# warnings before deciding whether to promote this to a deterministic auto-fix.
#
# Design notes:
#   - Matching is intentionally conservative. We fire only on unambiguous
#     signals (tool/data names, domain nouns) — not on passing mentions like
#     "the gene X" or "a motif-bound site in the title".
#   - The validator can only suggest categories, not specific leaf keys. A
#     hypothesis that needs "pwm.*" must have at least one entry whose prefix
#     is "pwm" before the first dot.
#   - False positives are worse than false negatives here: noisy warnings get
#     ignored, but a missed category is recoverable (the hypothesis still runs,
#     just with a larger helper bundle via the `required_data is None` path if
#     everything is missing, or with slightly reduced helpers otherwise).
# =============================================================================

# Keyword → category signals. Each entry is (compiled regex, category).
# Regexes use word boundaries and are case-insensitive to avoid matching
# inside longer words. Order does not matter — all matches are collected.
_REQUIRED_DATA_SIGNALS: list[tuple[str, str]] = [
    # pwm — motif scanning with PWM files
    (r"\bFIMO\b", "pwm"),
    (r"\bMEME\b", "pwm"),
    (r"\bJASPAR\b", "pwm"),
    (r"\bPWM\b", "pwm"),
    (r"\bposition weight matrix\b", "pwm"),
    (r"\bmotif scan(?:ning)?\b", "pwm"),
    (r"\bscan(?:ning)? (?:for )?motifs?\b", "pwm"),
    (r"\bmotif (?:occurrence|match|hit)s?\b", "pwm"),
    # string — protein-protein interactions
    (r"\bSTRING(?:\s+db)?\b", "string"),
    (r"\bPPI\b", "string"),
    (r"\bprotein[- ]protein interaction", "string"),
    (r"\bshared (?:interaction )?partners?\b", "string"),
    (r"\binteractome\b", "string"),
    # annotations (gencode bundle) — gene annotations, GO/pathway enrichment,
    # TSS/nearest-gene assignment
    (r"\bGO enrichment\b", "annotations"),
    (r"\bGO term", "annotations"),
    (r"\bpathway enrichment\b", "annotations"),
    (r"\bg:?Profiler\b", "annotations"),
    (r"\bEnrichr\b", "annotations"),
    (r"\bKEGG\b", "annotations"),
    (r"\bReactome\b", "annotations"),
    (r"\bGSEA\b", "annotations"),
    (r"\bnearest (?:gene|TSS)\b", "annotations"),
    (r"\bclosest (?:gene|TSS)\b", "annotations"),
    (r"\bpromoter assignment\b", "annotations"),
    (r"\bGENCODE\b", "annotations"),
    # rnaseq — expression data
    (r"\bRNA[- ]?seq\b", "rnaseq"),
    (r"\bTPM\b", "rnaseq"),
    (r"\bFPKM\b", "rnaseq"),
    (r"\bexpression level", "rnaseq"),
    (r"\bdifferentially expressed\b", "rnaseq"),
    (r"\bgene expression\b", "rnaseq"),
    # epigenome — ChromHMM states, histone marks, accessibility
    (r"\bChromHMM\b", "epigenome"),
    (r"\bchromatin state", "epigenome"),
    (r"\bH3K\d+(?:me\d*|ac)\b", "epigenome"),  # H3K27ac, H3K4me3, H3K9me3, ...
    (r"\bH2A\.?Z\b", "epigenome"),
    (r"\bhistone mark", "epigenome"),
    (r"\b(?:active|poised|bivalent|repressed) (?:promoter|enhancer|chromatin)",
     "epigenome"),
    (r"\bTssA\b|\bEnhG?\d*\b|\bReprPC\b|\bQuies\b", "epigenome"),  # ChromHMM state labels
    # hic — 3D contacts, loops, TADs
    (r"\bHi[- ]?C\b", "hic"),
    (r"\bchromatin loop", "hic"),
    (r"\bloop anchor", "hic"),
    (r"\bTAD\b", "hic"),
    (r"\btopologically associating domain", "hic"),
    (r"\b3D (?:contact|interaction|genome)", "hic"),
    (r"\bcontact matri", "hic"),
    # phyloP — evolutionary conservation
    (r"\bphyloP\b", "phyloP"),
    (r"\bevolutionary conservation\b", "phyloP"),
    (r"\bsequence conservation\b", "phyloP"),
]


def validate_required_data(hypothesis: dict[str, Any]) -> list[str]:
    """Detect manifest categories implied by the hypothesis text but not declared.

    Scans the hypothesis's `name`, `prediction`, and `verification_plan` text for
    unambiguous keywords (tool names, data-type names, domain nouns) and returns
    the list of manifest categories that the text implies are needed but are
    missing from the hypothesis's `required_data` field.

    Returns an empty list when:
      - every matched signal already has a corresponding `category.*` entry, or
      - no signals match at all.

    This is a pure validator (no side effects). The orchestrator is responsible
    for logging warnings and deciding what to do with the result.

    Args:
        hypothesis: The hypothesis dict, with at least a `required_data` list
            and text fields (`name`, `prediction`, `verification_plan`).

    Returns:
        Sorted list of missing category names (e.g. ["annotations", "pwm"]).
        Empty if nothing is missing.
    """
    import re

    # Collect all declared category prefixes (before the first dot).
    declared_raw = hypothesis.get("required_data") or []
    declared_categories: set[str] = set()
    for key in declared_raw:
        if not isinstance(key, str):
            continue
        declared_categories.add(key.split(".", 1)[0] if "." in key else key)

    # Assemble text to scan. Join verification_plan steps with newlines so
    # per-step keyword boundaries are preserved.
    text_parts: list[str] = [
        str(hypothesis.get("name", "")),
        str(hypothesis.get("prediction", "")),
    ]
    vplan = hypothesis.get("verification_plan") or []
    if isinstance(vplan, list):
        text_parts.extend(str(step) for step in vplan)
    elif isinstance(vplan, str):
        text_parts.append(vplan)
    text = "\n".join(text_parts)

    if not text.strip():
        return []

    suggested: set[str] = set()
    for pattern, category in _REQUIRED_DATA_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            suggested.add(category)

    missing = suggested - declared_categories
    return sorted(missing)


# Each bundle is a dict with:
#   - hard_bans: list[str]              — bullet items appended to the global hard bans
#   - helper_table_rows: list[str]      — markdown table rows (already pipe-formatted)
#   - imports: list[str]                — bare function names for the import statement
#   - helper_notes: str                 — detailed usage block (may be empty)
HELPER_BUNDLES: dict[str, dict[str, Any]] = {
    "always": {
        "hard_bans": [
            "`pd.qcut()` on any DataFrame column → use `matched_bin(fg, bg, col)` instead",
            "`.merge(... on='name' ...)` on narrowPeak frames → ENCODE `name` column is always `'.'`; join on coordinates or `peak_id`. Never pass a `lambda` as a `merge` key — pandas does not support it and raises `KeyError`",
            "Writing `df[['chrom','start','end','peak']]` to a BED file for FIMO → narrowPeak column `peak` is the **summit offset** (a small int like 52, 142, 245), NOT a peak identifier. Using it as col 4 silently collapses thousands of peaks into ~100 unique peak_ids in FIMO output. Either assign `df['peak_id'] = 'peak_' + df.index.astype(str)` first and use `peak_id` as col 4, or pass the DataFrame straight through `run_fimo_on_peaks` without writing your own BED.",
            "`subprocess.run(..., shell=True)` or `bash -lc` for CLI tools → use `subprocess.run([...], shell=False)`",
            "Hardcoded ENCFF IDs anywhere in the script → use `data_files['key']` (pre-injected dict)",
            "Reassigning `data_files = ...` anywhere in your code → `data_files` is pre-injected by the orchestrator at the start of every iteration with the canonical manifest keys (exact case). NEVER overwrite it. If `data_files['SP1_peaks']` fails, the correct fix is to `print(sorted(data_files.keys()))` and use the real key, not to rebuild the dict with your own guesses.",
            "Bare `except:` or `except Exception:` that only `print()`s and continues → silently hides `KeyError`, `NameError`, `AttributeError`, typos, and the very bugs we need to see. Let errors propagate so the REPL loop reports them as observations. If you truly need to test for a variable's existence, use `if 'sp1_peaks' in dir():` instead of try/except.",
            "`matplotlib` / `seaborn` imports → output statistics only, no plots",
            "More than 100 permutation replicates → hard cap is 100; ignore any higher number in the hypothesis",
        ],
        "helper_table_rows": [
            "| Peak overlap fraction | `count_overlapping_peaks(query, subject)` | manual `-a`/`-b` or hand-count |",
            "| bigWig signal | `extract_bigwig_signals(df, bw_path)` | `bw.stats()` or row loops |",
            "| Signal-matched controls | `matched_bin(fg, bg, signal_col)` | `pd.qcut()` |",
            "| narrowPeak loading | `read_narrowpeak(path)` | `pd.read_csv` with column guesses |",
            "| Two-peak coordinate join | `intersect_peaks(df_a, df_b, mode='flag'/'count'/'wa-wb')` | `.merge(on='name')` or manual bedtools |",
            "| bedtools `-wa -wb` output | `parse_bedtools_wa_wb(path, a_col_count, b_col_count)` | positional indexing |",
            "| bedtools `closest -d` output | `parse_bedtools_closest(path, a_col_count, b_col_count)` | positional indexing |",
        ],
        "imports": [
            "read_narrowpeak", "count_overlapping_peaks", "extract_bigwig_signals",
            "matched_bin", "intersect_peaks",
            "parse_bedtools_closest", "parse_bedtools_wa_wb",
        ],
        "helper_notes": """\
# --- count_overlapping_peaks --------------------------------------------------
# Returns dict: {n_query, n_overlapping, n_nonoverlapping, fraction}
# query = set you are asking ABOUT (defines the denominator). The fraction is
# n_overlapping / n_query — see CRITICAL MISTAKE #1 above.

# --- extract_bigwig_signals ---------------------------------------------------
# Returns a pd.Series, NOT a DataFrame. Assign directly to a column; do NOT call
# .columns or .merge() on the result.
#   df["h3k4me3"] = extract_bigwig_signals(intervals_df, data_files["h3k4me3..."])
# stat kwarg: "mean" | "max" | "min" | "std" (default: "mean")
#
# COLUMN NAME DEFAULTS: expects columns named "chrom", "start", "end".
# If your DataFrame uses different names (e.g. "summit_start", "summit_end"),
# you MUST pass them explicitly:
#   extract_bigwig_signals(df, bw, start_col="summit_start", end_col="summit_end")
# Otherwise you will get KeyError: "['start', 'end'] not in index".
#
# PEAK-STRENGTH METRIC CHOICE — read this before comparing signal between peak groups:
#   If your hypothesis is "TF_A binds MORE STRONGLY at sites where TF_B is also bound"
#   (or any per-peak signal intensity comparison), do NOT use
#   `extract_bigwig_signals(..., stat='mean')` on the full peak intervals:
#     - called peaks are already thresholded, so mean fold-change within peaks has
#       limited dynamic range — two groups often come out within 0.01 of each other
#     - mean over wide peaks dilutes the summit signal
#   Better options, in order of preference:
#     1. Use the narrowPeak `signal_value` column (already loaded by read_narrowpeak
#        — it's the peak-call's own quantitative strength score). No bigWig needed:
#           ets2_peaks["signal_value"].groupby(ets2_peaks["overlaps_b"]).mean()
#     2. extract_bigwig_signals(summit_window_df, bw_path, stat='max')
#        where summit_window_df has intervals of (summit - 250, summit + 250).
#        Use read_narrowpeak()'s `summit` column to build the windows.
#   Only use stat='mean' over full peaks for HISTONE MARKS or ACCESSIBILITY tracks
#   (H3K27ac, DNase, etc.), where the peak width is biologically meaningful.

# --- matched_bin --------------------------------------------------------------
# Returns (fg_copy, bg_copy) each with an added "bin" int column. BG rows outside
# FG range get bin=NaN — drop with .dropna(subset=["bin"]).
# ALIGNMENT TRAP: fg_binned shares fg_df's index but may differ in row order.
# NEVER assign fg_binned["bin"].values back to fg_df positionally — use index
# alignment: `fg_df = fg_df.loc[fg_binned.index].copy(); fg_df["bin"] = fg_binned["bin"]`

# --- intersect_peaks ----------------------------------------------------------
# Coordinate-based peak join — use instead of .merge(on='name')
flagged = intersect_peaks(df_a, df_b, mode="flag")   # adds bool overlaps_b column
counted = intersect_peaks(df_a, df_b, mode="count")  # adds overlap_count column
joined  = intersect_peaks(df_a, df_b, mode="wa-wb")  # one row per overlapping pair
# NOTE: the boolean column is ALWAYS named `overlaps_b` — NOT `overlaps_yy1`,
# `overlaps_NFYA`, or any TF-specific name. Use `flagged['overlaps_b']`.
""",
    },

    "fimo": {
        "hard_bans": [
            "`bedtools getfasta`, `fimo`, or `bw.stats()` inside any `for`/`while` loop → call each ONCE outside all loops. Direct `subprocess` calls to these tools when a helper exists are also banned — use `run_fimo_on_peaks()` and `extract_bigwig_signals()`.",
            "Running `fimo`, `bedtools getfasta`, or `bigWigAverageOverBed` on the full dataset as the first attempt → always test on `peaks.head(200)` first to confirm correct output format and timing, then run on the full dataset.",
            "Tiling the genome or annotation for background → use `bedtools shuffle -i peaks.bed -g chrom.sizes`",
        ],
        "helper_table_rows": [
            "| FIMO motif scan | `run_fimo_on_peaks(peaks, genome, meme, motif_id)` | manual getfasta + fimo |",
            "| Look up motif IDs by TF name | `find_motif_ids_for_tf(tf_name, meme_file)` | hardcoded IDs |",
            "| FIMO TSV parsing | `parse_fimo_tsv(path)` | manual `read_csv` |",
            "| Motif-to-motif comparison | `run_tomtom(meme_file, motif_id_1, motif_id_2)` | manual PWM comparison or FIMO-based approximation |",
        ],
        "imports": [
            "find_motif_ids_for_tf", "run_fimo_on_peaks", "parse_fimo_tsv", "run_tomtom",
        ],
        "helper_notes": """\
# --- run_fimo_on_peaks --------------------------------------------------------
# Parameter names matter — use these EXACT spellings:
#   peaks_bed=        # file path (str/Path), NOT a DataFrame
#   genome_fasta=     # NOT genome_fa, NOT fasta
#   meme_file=        # NOT meme, NOT meme_path
#   motif_id=         # exact string from find_motif_ids_for_tf()
#   p_value_threshold=1e-4   # NOT pval_thresh, NOT pvalue_threshold
#   summit_window=150        # optional: scan summit ± N bp (re-windows internally)
# Returns: motif_id, sequence_name, start, stop, p-value, peak_id
# TRAP: if peaks_bed is a NamedTemporaryFile path, call .flush() or .close() BEFORE
# passing the path — unflushed buffers produce an empty file → EmptyDataError.
# TRAP: if you already built summit-centered windows in your BED (e.g. summit ± 150),
# do NOT also pass summit_window=150 — the helper will re-window to the midpoint,
# which is a no-op but wastes computation. Use summit_window= only when passing
# full-width peak BED files directly.

# --- run_tomtom ---------------------------------------------------------------
# Motif-to-motif similarity comparison (faster and more rigorous than FIMO overlap).
result = run_tomtom(data_files["motif_meme"], "NFYA|jaspar|MA0060.3", "NFATC3|jaspar|MA0623.1")
# result: {p_value, e_value, q_value, overlap, query_consensus, target_consensus,
#          orientation, is_significant (q_value < 0.05), match_found (bool)}
""",
    },

    "string": {
        "hard_bans": [
            "`pd.read_csv(string_file, sep=' ')` or `sep='\\t'` for STRING files → STRING uses variable whitespace; always use `load_string_links(path)`",
        ],
        "helper_table_rows": [
            "| STRING PPI links (PREFERRED) | `load_string_links(path, min_score=400)` | `pd.read_csv(sep=' ')`, manual parsing |",
            "| STRING gene-pair lookup | `find_string_interaction(gene_a, gene_b, links_df, aliases_path)` | manual `setdefault()` ID lookup — genes map to MULTIPLE ENSP IDs and naive code picks the wrong one |",
            "| STRING shared partners (named) | `find_shared_string_partners(gene_a, gene_b, links_df, aliases_path)` | manual partner intersection + ENSP→symbol lookup — reverse-mapping `aliases[ensp] = alias` collides on PDB IDs and synonyms; output is junk like `2YRP, 10q23del, 1NF1` |",
            "| STRING API (FALLBACK ONLY) | `fetch_shared_partners([\"TF_A\", \"TF_B\"])` from `src.utils.string_client` — only if local file unavailable | custom `requests` calls |",
        ],
        "imports": [
            "load_string_links", "find_string_interaction", "find_shared_string_partners",
        ],
        "helper_notes": """\
# --- load_string_links --------------------------------------------------------
# Returns: protein1, protein2, combined_score (+ any extra score columns)
# Protein IDs are Ensembl format (9606.ENSP...), NOT gene symbols.
# DO NOT roll your own gene→ENSP or ENSP→gene mapping (see CRITICAL MISTAKE #4).
# Use find_string_interaction() and find_shared_string_partners() instead.

# --- find_string_interaction --------------------------------------------------
# Highest-scoring direct edge between two genes, handling multi-mapping.
result = find_string_interaction("SP1", "NFYA", links_df, data_files["aliases_file"])
# result: {found, score, protein_a, protein_b, all_ids_a, all_ids_b}

# --- find_shared_string_partners ----------------------------------------------
# Named shared cofactors with canonical HGNC symbols (no PDB IDs or junk).
result = find_shared_string_partners(
    "SP1", "NFYA", links_df, data_files["aliases_file"],
    min_score=400, max_partners=10,
)
# result: {shared_count, partners: [{ensp, gene_symbol, score_a, score_b}, ...]}
# Sorted by min(score_a, score_b) descending — strongest evidence on BOTH sides first.
# Real SP1/NFYA output: TBP, EP300, TBPL2, TBPL1, TP53, CREB1, NFYB, NFYC, CREBBP, JUN

# --- fetch_shared_partners (FALLBACK — STRING API, rate-limited) --------------
# Use ONLY if the local STRING file is not in the manifest. Prefer
# find_shared_string_partners() (above) which uses the local file.
from src.utils.string_client import fetch_shared_partners
partners_df, shared = fetch_shared_partners(
    ["TF_A", "TF_B"], species=9606, min_score=700, min_escore=0.2, min_dscore=0.3,
)
# shared → set of gene symbols that interact with BOTH TF_A and TF_B
""",
    },

    "rnaseq": {
        "hard_bans": [],
        "helper_table_rows": [
            "| RNA-seq loading | `load_rnaseq_with_gene_id(path)` or `load_rnaseq_expression(path)` | manual column parsing |",
            "| Nearest TSS + expression | `link_peaks_to_expression(peaks, rnaseq, gtf)` | chained TSS calls |",
        ],
        "imports": [
            "load_rnaseq_with_gene_id", "load_rnaseq_expression",
            "link_peaks_to_expression", "merge_rnaseq_with_nearest_genes",
        ],
        "helper_notes": """\
# --- load_rnaseq_with_gene_id ------------------------------------------------
# Returns a 2-tuple: (rnaseq_df, gene_id_col)
#   rnaseq_df, gene_id_col = load_rnaseq_with_gene_id(path)
# Do NOT call .columns on the raw return value — it is a tuple, not a DataFrame.

# --- load_rnaseq_expression ---------------------------------------------------
# Returns (rnaseq_df, gene_id_col, clean_col, expr_col)
# - clean_col = "gene_id_clean" (version-stripped)
# - expr_col is usually "TPM"
""",
    },

    "gencode": {
        # The gencode bundle covers gene-annotation analyses including GO/pathway
        # enrichment. GO enrichment needs a gene list (typically derived from
        # peak→nearest-gene mapping via run_bedtools_closest_to_tss), so it lives
        # next to the TSS helpers rather than in the rnaseq bundle.
        "hard_bans": [],
        "helper_table_rows": [
            "| Nearest TSS distance only | `run_bedtools_closest_to_tss(peaks_df_or_path, gtf_path=gtf)` | subprocess bedtools closest |",
            "| GO / pathway enrichment | `run_go_enrichment(gene_list)` from `src.utils.bioio` | `requests` to Enrichr/g:Profiler directly |",
        ],
        "imports": [
            "load_gencode_genes", "create_tss_bed_from_gencode", "run_bedtools_closest_to_tss",
            "run_go_enrichment",
        ],
        "helper_notes": """\
# --- run_go_enrichment --------------------------------------------------------
# GO/KEGG enrichment via g:Profiler API — no local files needed.
# Pair with run_bedtools_closest_to_tss to map peaks to nearby genes first.
enrich_df = run_go_enrichment(gene_list, organism="hsapiens", sources=["GO:BP", "GO:MF", "KEGG"])
# columns: source, name, p_value, intersection_size, term_size, query_size, native
""",
    },

    "epigenome": {
        "hard_bans": [],
        "helper_table_rows": [
            "| ChromHMM annotation | `annotate_peaks_with_chromhmm(peaks, chromhmm)` | manual intersect |",
        ],
        "imports": [
            "annotate_peaks_with_chromhmm", "parse_chromhmm_intersect",
        ],
        "helper_notes": """\
# --- annotate_peaks_with_chromhmm ---------------------------------------------
# READ THE MANIFEST FIRST. ChromHMM models have 18 states grouped into 5 categories
# (promoter, enhancer, transcription, polycomb-repressed, quiescent). The manifest
# provides the FULL state list per category — pass them all, never subset by hand.
#
# The `state` column contains the FULL state name for every peak (e.g. "1_TssA",
# "16_ReprPC", "18_Quies"). All 18 states are present regardless of what you pass.
# The `promoter_states` / `enhancer_states` arguments ONLY control the convenience
# boolean flags `is_promoter_state` / `is_enhancer_state`.
#
# CORRECT:
chrom_df = annotate_peaks_with_chromhmm(
    peaks_bed, chromhmm_bed,
    promoter_states=data_manifest["data"]["epigenome"]["chromhmm_promoter_state"],
    enhancer_states=data_manifest["data"]["epigenome"]["chromhmm_enhancer_state"],
)
# For transcription / polycomb / quiescent analyses, read the manifest categories
# and use the `state` column directly:
#   tx_states = data_manifest["data"]["epigenome"]["chromhmm_transcription_state"]
#   chrom_df["is_tx"] = chrom_df["state"].isin(tx_states)
""",
    },

    "hic": {
        "hard_bans": [],
        "helper_table_rows": [],
        "imports": [],
        "helper_notes": """\
# --- Hi-C loop / strip files (BEDPE) ------------------------------------------
# Hi-C files in the manifest are bedpe format with two anchor regions per row.
# To check if peaks fall in loop anchors, parse columns 1-3 and 4-6 separately
# and intersect each with peaks. There is no helper — load with pandas:
#   loops = pd.read_csv(data_files["hic_loops_bedpe"], sep="\\t", header=None,
#                       names=["chrom_a","start_a","end_a","chrom_b","start_b","end_b"])
""",
    },
}


def assemble_helpers_section(bundles: list[str]) -> str:
    """Assemble the REQUIRED HELPERS table + HELPER USAGE NOTES from a list of bundles.

    Args:
        bundles: List of bundle names (from select_helper_bundles())

    Returns:
        Markdown string containing the helpers table, import statement, and usage notes.
    """
    parts: list[str] = []

    # Hard bans (collected across all bundles)
    all_bans: list[str] = []
    for b in bundles:
        all_bans.extend(HELPER_BUNDLES.get(b, {}).get("hard_bans", []))
    if all_bans:
        parts.append("=== HARD BANS — every item below causes automatic rejection ===\n")
        for i, ban in enumerate(all_bans, 1):
            parts.append(f"{i}. {ban}")
        parts.append("")

    # Required helpers table (collected across all bundles)
    all_rows: list[str] = []
    for b in bundles:
        all_rows.extend(HELPER_BUNDLES.get(b, {}).get("helper_table_rows", []))
    if all_rows:
        parts.append("=== REQUIRED HELPERS — use these; hand-rolled alternatives are banned ===\n")
        parts.append("| Operation | Required helper | Never use |")
        parts.append("|-----------|----------------|-----------|")
        parts.extend(all_rows)
        parts.append("")
        parts.append("All helpers are in `src.utils.bioio` (file-based) or `src.utils.string_client` (STRING API). Only bypass a helper if it genuinely cannot produce the output shape you need, and explain why in a comment.\n")

    # Imports + helper notes
    all_imports: list[str] = []
    all_notes: list[str] = []
    for b in bundles:
        bundle = HELPER_BUNDLES.get(b, {})
        all_imports.extend(bundle.get("imports", []))
        notes = bundle.get("helper_notes", "")
        if notes:
            all_notes.append(notes)

    if all_imports or all_notes:
        parts.append("=== HELPER USAGE NOTES ===\n")
        parts.append(
            "Most helpers are self-explanatory from the table above. The notes below cover only "
            "the helpers that have non-obvious parameter names, return types, or known traps.\n"
        )
        if all_imports:
            parts.append("```python")
            # Deduplicate while preserving order
            seen = set()
            unique_imports = []
            for imp in all_imports:
                if imp not in seen:
                    seen.add(imp)
                    unique_imports.append(imp)
            parts.append("from src.utils.bioio import (")
            # Wrap imports nicely (3 per line)
            for i in range(0, len(unique_imports), 3):
                parts.append("    " + ", ".join(unique_imports[i:i+3]) + ",")
            parts.append(")")
            parts.append("")
        for note in all_notes:
            parts.append(note)
        if all_imports:
            parts.append("```")

    return "\n".join(parts)

# =============================================================================
# Static prompt blocks — composed by build_coding_base()
# =============================================================================

CODING_PROMPT_HEADER = """You are an expert bioinformatics programmer. Write Python code to verify scientific hypotheses using the data provided.

=== CRITICAL MISTAKES — read these first; they cause silently wrong results ===

These are the bug classes that have repeatedly produced confidently-wrong output. They
do not raise Python errors. The only defense is making the right thing explicit before
you compute.

1. **Define your denominator before computing any fraction.** Print the denominator and
   what it represents (`"denominator = ATF6 peaks"` vs `"denominator = REST peaks"`)
   before reporting overlap percentages. `count_overlapping_peaks(A, B)` returns
   `len(A overlapping B) / len(A)` — the FIRST argument is the denominator.

2. **State your unit of analysis before any statistic.** Print `"unit = peaks"` or
   `"unit = motif hits"` — they are NOT interchangeable. A peak with three motif hits
   is one peak and three hits; reporting "n=300 peaks have motif" when you actually
   counted 300 hits across 100 peaks is silently wrong.

3. **Never hardcode named entities (gene names, partner lists, GO terms) as string
   literals in your solution.** Every named entity must come from a variable computed
   in the same block. The summary agent runs a fabrication check against the raw
   stdout — names in your solution that do not appear in the printed output will be
   flagged and your confidence will be downgraded.

4. **STRING IDs are many-to-many in BOTH directions.** Genes map to multiple ENSP IDs;
   ENSPs map to multiple aliases (canonical HGNC, PDB structure IDs, deletion-region
   names). Hand-rolled `aliases[gene] = ensp` or `aliases[ensp] = name` lookups silently
   produce wrong results. ALWAYS use `find_string_interaction()` for pair queries and
   `find_shared_string_partners()` for named shared partners.

5. **FIMO and bigWig coordinates are ABSOLUTE genome positions.** When computing
   distances or relative offsets, print a sample row with the raw values to confirm
   units. `abs(50 - motif_center)` is meaningless if `motif_center` is a chromosome
   coordinate like 12345678.

"""


def build_coding_base(bundles: list[str] | None = None) -> str:
    """Build the coding system prompt base from a list of helper bundles.

    Combines the static CODING_PROMPT_HEADER (always shown) with bundle-specific
    hard bans, helper table rows, and helper notes assembled by
    assemble_helpers_section().

    Args:
        bundles: List of bundle names from select_helper_bundles(). If None,
                 all bundles are loaded as a safe fallback.

    Returns:
        The full base prompt as a single string.
    """
    if bundles is None:
        bundles = select_helper_bundles(None)
    helpers = assemble_helpers_section(bundles)
    return CODING_PROMPT_HEADER + helpers + "\n"


# Backwards-compat: legacy CODING_SYSTEM_PROMPT_BASE constant. Equivalent to
# loading every bundle. Kept so older imports keep working; new code should
# call build_coding_base() with explicit bundles.
CODING_SYSTEM_PROMPT_BASE = build_coding_base(None)


def build_coding_system_prompt(
    tools: list[str] | None = None,
    required_data: list[str] | None = None,
) -> str:
    """Build the complete coding system prompt with relevant tool quirks.

    Args:
        tools: List of tools being used (e.g., ['fimo', 'bedtools']).
               If None, includes default quirks (FIMO + bedtools).
        required_data: List of `category.leaf_key` strings from the hypothesis
               (e.g. ["chipseq.SP1_peaks", "string.links_file"]). Used to select
               which helper bundles to load. If None, all bundles are loaded
               (safe fallback used when bundle dispatch is not yet wired up,
               e.g. for the file inspection step).

    Returns:
        Complete system prompt with tool-specific quirks appended.
    """
    if tools is None:
        # Include default quirks for common bioinformatics workflows
        quirks = DEFAULT_QUIRKS
    else:
        quirks = get_quirks_for_tools(tools)

    bundles = select_helper_bundles(required_data)
    base = build_coding_base(bundles)

    if quirks:
        return base + "\n" + quirks
    return base


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


def _looks_like_file_path(value: Any) -> bool:
    """Heuristic: True if the value is a string that looks like a filesystem path.

    We use this to decide whether to redact the value from the coding prompt.
    Paths are redacted because showing them invites the LLM to hardcode them
    (bypassing the `data_files['key']` contract). Non-path metadata values
    (format descriptions, role annotations, id types) are kept inline because
    they carry information the LLM needs to write correct code.
    """
    if not isinstance(value, str):
        return False
    # Absolute path is the canonical pattern in our manifests
    if value.startswith("/"):
        return True
    # Known file extensions — catch any relative-path stragglers
    path_exts = (
        ".bed", ".bed.gz", ".narrowPeak", ".broadPeak",
        ".bigWig", ".bw", ".bigBed", ".bb",
        ".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fai",
        ".gtf", ".gtf.gz", ".gff", ".gff3",
        ".meme", ".tsv", ".tsv.gz", ".txt", ".txt.gz",
        ".bedpe", ".cool", ".mcool", ".hic",
        ".vcf", ".vcf.gz",
    )
    return any(value.endswith(ext) for ext in path_exts)


def _format_data_for_coding(
    data_manifest: dict[str, Any], file_summaries: dict[str, str] | None = None
) -> str:
    """Format data manifest and optionally append file summaries for coding prompts.

    File paths are redacted from the display. The LLM accesses files exclusively
    via the pre-injected `data_files[<key>]` dict — showing the absolute paths
    here only encourages the LLM to hardcode them (which corrupts kernel state
    and violates the always-bundle hard ban). Metadata values like format
    descriptions and role annotations are kept inline because the LLM needs
    them to write correct code.
    """
    parts = []

    parts.append("## How to access data")
    parts.append(
        "All file paths below are pre-injected into the kernel as `data_files[<key>]`. "
        "Use `data_files['ETS2_peaks']` (etc.) — do NOT hardcode paths or reassign "
        "`data_files`. If a key you expected is missing, run "
        "`print(sorted(data_files.keys()))` in a code block to see what's actually "
        "available, then use the real key. The flat keys below are the exact strings "
        "available in `data_files` (category-prefixed forms like "
        "`chipseq.ETS2_peaks` are also available)."
    )
    parts.append("")

    data = data_manifest.get("data", {})
    for category, items in data.items():
        parts.append(f"## {category}")
        if isinstance(items, dict):
            for name, value in items.items():
                if _looks_like_file_path(value):
                    # Redact the path — the LLM doesn't need it, and seeing
                    # it would invite hardcoding.
                    parts.append(f"- `{name}` (file)")
                else:
                    # Metadata (format description, role, etc.) — keep inline.
                    parts.append(f"- `{name}`: {value}")
        else:
            if _looks_like_file_path(items):
                parts.append("- (file)")
            else:
                parts.append(f"- {items}")
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


_REPL_SYSTEM_PROMPT_HEADER = """You are an expert bioinformatics programmer working in a persistent Python REPL.

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

## Pre-solution self-check (REQUIRED — run BEFORE writing <solution>)

Before emitting your `<solution>` block, you MUST run a final `<execute>` block that
explicitly verifies the numbers you are about to report. The summary agent only sees what
you print — it cannot read your code. If a metric is computed wrong, the summary agent will
trust the wrong number. Catching these errors here is the only safety net.

Print and verify, in plain Python, each of the following for every metric in your solution:

1. **Direction check**: For helper functions that take (query, subject) arguments like
   `count_overlapping_peaks(A, B)` — confirm which side is the denominator. Print:
   `"count_overlapping_peaks(A=tf, B=peaks): n_query={result['n_query']} (this is len(A)=tf), fraction = overlap/n_query"`
   Then state explicitly which biological question this answers ("fraction of TF peaks
   that overlap REST+ peaks" vs "fraction of REST+ peaks that have TF").

2. **Variable shadowing check**: List every variable name used in the final summary print.
   For each, confirm it holds the value computed in THIS block, not a value from an earlier
   block that was overwritten. Example anti-pattern: `mynn_pval` assigned to Fisher result,
   then later reassigned to Mann-Whitney result, then printed in the Fisher summary.
   Verify by printing the value AND a one-line description of what test produced it,
   immediately before the summary.

3. **Coordinate system check**: For any distance/position calculation, verify the units.
   FIMO returns ABSOLUTE genome coordinates (e.g. chr1:12345678), NOT relative positions
   within the peak window. If you compute `abs(50 - motif_center)`, you are mixing local
   and absolute coordinates and the result is meaningless. Print a sample row showing
   the actual coordinate values to confirm you understand the units.

4. **Unique-vs-duplicated check**: When reporting "n peaks with X", confirm whether your
   count is unique peaks or hit-level (multi-hit peaks counted multiple times). Print
   both: `len(df)` (all hits) and `df['peak_id'].nunique()` (unique peaks). If you used
   `drop_duplicates()` to collapse, state explicitly which rows were discarded.

5. **Hardcoded constants check**: Search your code for any numbers that were typed as
   literals (e.g. `412 / 1113`) instead of derived from data variables. Replace them with
   computed expressions, OR explicitly verify they match the printed counts from the
   most recent FIMO/intersect output before using them.

6. **Fisher table check**: For 2x2 contingency tables, the four cells must sum to the
   total. Print the table and verify: `sum of all cells = n_total`. A common bug is
   `n_nonoverlapping - n_overlapping` (which is `n_query - 2*n_overlap`, not the complement).

7. **Threshold direction check**: For each prediction in the hypothesis, state which
   direction the data needs to go (>= or <=) and the actual value. Don't just say "PASS" —
   show the comparison (e.g., "FC = 1.73 >= 1.2 → PASS").

8. **Sanity check on intermediate drops**: For every filter / join / set-intersection
   step, print `len(before)` and `len(after)` and the retention ratio. If ANY step drops
   more than 90% of rows (retention < 10%), STOP and investigate before trusting the
   result. The canonical failure mode: 2,312 co-bound peaks → only 67 "with both motifs"
   (2.9% retention) looks like biology but was actually a peak_id collision caused by
   writing the narrowPeak summit-offset column into BED col 4. Large unexplained drops
   are almost always a join/ID bug, not a real effect. Print the first 3 rows of both
   sides of any join and verify the join keys visually before reporting the count.

9. **Two-sample test discipline**: When comparing two groups (e.g. co-bound vs TF-only
   signal distributions), run BOTH a mean-based test (`scipy.stats.ttest_ind(a, b, equal_var=False)` — Welch's t-test)
   AND a rank-based test (`scipy.stats.mannwhitneyu(a, b, alternative='two-sided')`).
   **Flag disagreement only when the two tests give different discrete verdicts at α=0.05**
   — i.e., one says p<0.05 and the other says p≥0.05. If that happens, STOP and
   investigate — it usually means the distributions differ in *shape* (variance, skew,
   tails) not in *location* (mean/median). Do NOT compare raw p-value ratios; at large n,
   both may be under 1e-100 and the ratio becomes meaningless due to floating-point
   underflow (e.g. Welch p=3e-106 vs Mann-Whitney p=8e-276 — both are effectively zero
   and agree). Report BOTH p-values in the solution reasoning so the reviewer can see the
   agreement. Also print `mean_a`, `mean_b`, `std_a`, `std_b`, and `n_a`, `n_b` — if
   `|mean_a - mean_b| < 0.001 * mean_a`, the effect size is negligible regardless of any
   p-value, and you should report `REJECTS` for any "X is larger than Y" prediction even
   if Mann-Whitney barely crosses 0.05.

10. **Correlation test discipline**: When computing a correlation between two continuous
   variables (e.g. motif score vs ChIP-seq signal), run BOTH `scipy.stats.spearmanr`
   AND `scipy.stats.pearsonr`. Report both r values and both p-values. **If they
   disagree substantially** — one gives |r| >= 0.2 while the other gives |r| < 0.1 —
   the relationship is likely driven by outliers or non-linearity rather than a genuine
   quantitative trend. In a previous run (ILK/YY1), Spearman r=0.34 (p=2e-05) but
   Pearson r=0.06 (p=0.43) at n=152 co-bound peaks — the "correlation" was entirely
   a rank-ordering effect from a few high-signal outliers, not a real quantitative
   relationship. If only Spearman passes your prediction threshold but Pearson does
   not, note this in the reasoning and downgrade confidence accordingly.

If any check fails, fix the bug in another <execute> block BEFORE writing the solution.
If everything passes, emit the <solution>.

## Solution format

<solution>
analysis: <brief description of what you actually did — which files were used, what statistical tests were run, what comparisons were made. This helps the reviewer understand the actual analysis vs the planned verification.>
support_level: SUPPORTS|REJECTS|INCONCLUSIVE|ERROR|UNTESTABLE
confidence: 0.0-1.0
finding: <one concise sentence stating the key result>
reasoning: <explanation of the evidence and why it supports/rejects the hypothesis>
</solution>

### Confidence rubric (REQUIRED — calibrate before writing the number)

Do not default to 0.9+ just because your code ran without errors. Code running
cleanly is a baseline, not evidence. Use this rubric:

- **0.9 – 1.0**: One statistical test with p < 0.01 AND effect size (Cohen's d / OR / fold)
  comfortably above threshold AND all pre-solution checks passed AND the result is
  consistent with every sub-analysis you ran. Essentially no counter-evidence.
- **0.7 – 0.9**: p < 0.05 AND effect size above threshold, but either (a) one
  secondary check was weaker than expected, (b) sample size was modest (n<500 per group),
  or (c) only one statistical test was run.
- **0.5 – 0.7**: Mixed signals — e.g. fraction/fold threshold met but effect size small,
  OR two tests give contradictory verdicts (one significant, one not), OR the prediction
  split into multiple sub-predictions and only some met threshold.
- **0.3 – 0.5**: Primary prediction not met but unexpected secondary pattern found
  (INCONCLUSIVE leaning toward a new hypothesis).
- **0.0 – 0.3**: Technical problems, low power (n < 100 per group), or the test
  doesn't actually address the hypothesis.

**Auto-downgrade triggers** (any of these forces confidence ≤ 0.6, regardless of above):
- Welch's t-test and Mann-Whitney give different discrete verdicts (one p<0.05, other p≥0.05)
- Mean difference between groups is < 0.1% of either group's mean (near-zero effect
  despite any p < 0.05)
- Spearman and Pearson correlations disagree: one |r| >= 0.2 while other |r| < 0.1
- Any pre-solution check (#1–#10) flagged a warning you couldn't fully explain
- A sanity-check filter dropped >90% of rows for reasons you couldn't verify
- You had to reload data or re-inspect columns more than 3 times during the iteration

## Guidelines

- Run small steps: load data → inspect → compute → conclude. Do not write one huge script.
- Always print key intermediate results so you can verify them before proceeding.
- If a step fails, read the error, fix just that part, and try again.
- No matplotlib or seaborn — print statistics only.
- Max 100 permutation replicates.
- Use data_files['key'] to access files — it is pre-injected into the kernel.
- **Early stopping**: If the first prediction in the verification plan is clearly refuted
  (wrong direction, not significant, or effect far below threshold) AND there are no
  surprising secondary findings, stop immediately and emit a REJECTS <solution>.
  However, if the result contains a strong unexpected pattern (e.g., large fold enrichment
  despite low absolute percentage, effect in opposite direction, unexpected negative
  correlation), report it in the finding/reasoning even if the overall verdict is REJECTS.
  The hypothesis agent needs these secondary findings to guide the next hypothesis.
- **Co-occupancy: always report fold enrichment over background, not just absolute overlap
  percentage.** An absolute overlap of 20% can be highly significant (>100-fold over random)
  if both TFs cover a tiny fraction of the genome. When evaluating co-occupancy predictions,
  if the fold enrichment is large (≥10-fold, p < 0.05) but the absolute percentage is below
  the hypothesis threshold, report SUPPORTS based on the fold enrichment — the absolute
  percentage threshold was likely miscalibrated.

"""


def build_repl_system_prompt(required_data: list[str] | None = None) -> str:
    """System prompt for the REPL coding agent.

    Combines the REPL-specific guidance (how the REPL loop works, pre-solution
    self-check, solution format, guidelines) with a bundle-filtered version of
    CODING_PROMPT_HEADER + helpers section.

    Args:
        required_data: List of `category.leaf_key` strings from the hypothesis.
                       If None, all bundles are loaded as a safe fallback.

    Returns:
        Full REPL system prompt string.
    """
    bundles = select_helper_bundles(required_data)
    base = build_coding_base(bundles)
    return _REPL_SYSTEM_PROMPT_HEADER + base


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

    # NOTE: Helper reference lives in the system prompt now, filtered by
    # `required_data` via select_helper_bundles() / assemble_helpers_section().
    # Do NOT add an unfiltered helper block here — it would leak helpers that
    # are irrelevant to the current iteration's data and undercut bundle gating.

    parts.append(
        "\nStart by loading and inspecting the relevant data, then run your analysis "
        "step by step. When you have sufficient evidence, emit <solution>.</solution>."
    )

    return "\n".join(parts)
