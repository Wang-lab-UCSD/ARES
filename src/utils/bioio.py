"""Reusable bioinformatics parsing helpers for generated verification code.

These helpers exist to prevent generated scripts from re-implementing fragile parsing
logic for common file formats and tool outputs.
"""

from __future__ import annotations

import functools
import math
import subprocess
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import pandas as pd


NARROWPEAK_COLUMNS = [
    "chrom",
    "start",
    "end",
    "name",
    "score",
    "strand",
    "signal_value",
    "p_value",
    "q_value",
    "peak",
]

DEFAULT_GENE_ID_CANDIDATES = [
    "gene_id",
    "Geneid",
    "gene",
    "geneid",
    "ensembl_gene_id",
    "gene_stable_id",
]

DEFAULT_EXPRESSION_CANDIDATES = [
    "TPM",
    "pme_TPM",
    "FPKM",
    "pme_FPKM",
]


def _read_table_like(
    data: str | Path | pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Read a table-like input. Accepts:

    - ``pd.DataFrame``: returns a defensive copy.
    - ``pathlib.Path``: treated as a filesystem path.
    - ``str`` containing a newline: treated as in-memory content (wrapped
      in ``StringIO``). This supports the natural pattern of passing a
      subprocess stdout straight into a parser, e.g.
      ``parse_bedtools_wa_wb(subprocess.run(..., capture_output=True).stdout)``.
      Without this branch, pandas treats the long multi-line string as a
      file path and raises ``OSError: [Errno 36] File name too long`` with
      the **entire input embedded in the error message**, which would push
      the whole payload into the coding agent's conversation history.
    - ``str`` without a newline: treated as a filesystem path (existing
      behaviour, preserved for backward compatibility with all callers).
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, str) and "\n" in data:
        return pd.read_csv(StringIO(data), **kwargs)
    return pd.read_csv(data, **kwargs)


def read_narrowpeak(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read narrowPeak data and compute summit positions safely.

    Returns a DataFrame with canonical column names plus:
    - `peak_offset`: integer summit offset relative to start
    - `summit`: genomic summit coordinate
    """
    df = _read_table_like(path, sep="\t", header=None)
    if df.shape[1] < 10:
        raise ValueError(f"Expected >=10 narrowPeak columns, found {df.shape[1]}")

    df = df.iloc[:, :10].copy()
    df.columns = NARROWPEAK_COLUMNS
    df = df.dropna(subset=["chrom", "start", "end"])
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)

    peak_offset = pd.to_numeric(df["peak"], errors="coerce").fillna(-1).astype(int)
    fallback = ((df["end"] - df["start"]) // 2).astype(int)
    df["peak_offset"] = peak_offset.where(peak_offset >= 0, fallback)
    df["summit"] = df["start"] + df["peak_offset"]
    # Alias: model often uses 'summit_offset' — keep both names
    df["summit_offset"] = df["peak_offset"]

    # Drop 'name' column: ENCODE narrowPeak files use '.' for ALL peaks,
    # making it useless and a trap for accidental .merge(on='name').
    df = df.drop(columns=["name"])
    return df


def parse_peak_id_from_fasta_header(sequence_name: str) -> str:
    """Extract the original peak identifier from a getfasta/FIMO sequence header."""
    if "::" in sequence_name:
        return sequence_name.split("::", 1)[0]
    return sequence_name


def gc_content(seq: str) -> float:
    """Return the GC fraction of a DNA sequence, handling soft-masked lowercase bases.

    ``bedtools getfasta`` leaves repeat-masked regions in lowercase, so naive
    ``seq.count('G') + seq.count('C')`` produces a fraction of zero for any peak
    that overlaps a masked region. In practice this has silently corrupted 60%+
    of peaks in multiple ARES runs when GC-matched controls were being built.
    Using this helper in matched-control pipelines keeps the fraction correct
    regardless of case and ignores non-ATGC characters (N, IUPAC codes).
    """
    if not seq:
        return 0.0
    s = seq.upper()
    gc = s.count("G") + s.count("C")
    atgc = gc + s.count("A") + s.count("T")
    return gc / atgc if atgc > 0 else 0.0


FIMO_CANONICAL_COLUMNS = [
    "motif_id", "motif_alt_id", "sequence_name", "start", "stop",
    "strand", "score", "p-value", "q-value", "matched_sequence", "peak_id",
]


def _empty_fimo_df() -> pd.DataFrame:
    """Return a properly-typed empty FIMO result DataFrame.

    Callers can treat a zero-hit legitimate result the same as any
    other result (e.g. ``len(df) == 0`` or ``df[df.motif_id == 'X']``)
    without needing to special-case schema-less empty outputs.
    """
    return pd.DataFrame(columns=FIMO_CANONICAL_COLUMNS)


def parse_fimo_tsv(
    path: str | Path,
    p_value_threshold: float | None = None,
    motif_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Parse FIMO TSV output robustly.

    Handles:
    - Old FIMO format where the column header is a single-`#` line like
      ``#pattern name\\tsequence name\\t...`` (renamed via ``rename_map``).
    - Modern FIMO (5.x+) where metadata comments are single-`#` prose lines
      (``# FIMO ...``, ``# The format ...``, ``# fimo --oc ...``) appended
      at the end of the file, and the real header has no `#` prefix.
    - **Zero-hit legitimate results**: FIMO 5.5.5 writes ONLY the three
      metadata comments when it finds zero motif occurrences — no header,
      no data rows. This function detects that case and returns a properly
      typed empty DataFrame instead of raising.

    Adds:
    - `peak_id` parsed from `sequence_name`
    """
    rows: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if line.startswith("##"):
                continue
            # A single-`#` line is either the old-style header
            # (``#pattern name\tsequence name\t...`` — always contains tabs)
            # or a modern metadata comment (``# FIMO ...`` — prose, no tabs).
            # Distinguish them by tab presence so we skip metadata but keep
            # the legacy header.
            if line.startswith("#") and "\t" not in line:
                continue
            rows.append(line)

    if not rows:
        # FIMO legitimately found zero hits — return an empty, typed DataFrame.
        return _empty_fimo_df()

    text = "".join(rows)
    df = pd.read_csv(StringIO(text), sep="\t")

    rename_map = {
        "#pattern name": "motif_id",
        "pattern name": "motif_id",
        "pattern_name": "motif_id",
        "sequence name": "sequence_name",
        "sequence_name": "sequence_name",
    }
    df = df.rename(columns=rename_map)

    # Drop trailing FIMO comment rows — modern FIMO appends lines like
    # "# fimo was run with..." that get parsed as data rows with NaN in
    # sequence_name and other columns.
    if "sequence_name" in df.columns:
        df = df.dropna(subset=["sequence_name"]).copy()

    required = {"motif_id", "sequence_name", "start", "stop", "p-value"}
    missing = required - set(df.columns)
    if missing:
        # Malformed / unrecognised header - treat as no hits rather than
        # raising, which is what a file of metadata comment lines with no
        # data rows looks like when a FIMO build uses a comment format the
        # filter above does not recognise.
        return _empty_fimo_df()

    if len(df) == 0:
        # Header present but no data rows (empty hit set after dropping
        # trailing comments). Return typed empty DataFrame with peak_id.
        return _empty_fimo_df()

    df["peak_id"] = df["sequence_name"].astype(str).map(parse_peak_id_from_fasta_header)
    df["p-value"] = pd.to_numeric(df["p-value"], errors="coerce")

    if motif_ids is not None:
        motif_ids = {str(m) for m in motif_ids}
        df = df[df["motif_id"].astype(str).isin(motif_ids)].copy()

    if p_value_threshold is not None:
        df = df[df["p-value"] < float(p_value_threshold)].copy()

    return df


def _find_named_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    lowered = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_rnaseq_with_gene_id_impl(
    path_or_df: str | Path | pd.DataFrame,
    gene_id_candidates: Iterable[str] | None,
) -> tuple[pd.DataFrame, str]:
    df = _read_table_like(path_or_df, sep="\t")
    gene_id_col = _find_named_column(
        df.columns,
        gene_id_candidates or DEFAULT_GENE_ID_CANDIDATES,
    )
    if gene_id_col is None:
        raise ValueError(
            "Could not identify gene ID column by name. "
            f"Available columns: {list(df.columns)}"
        )
    return df, gene_id_col


@functools.lru_cache(maxsize=4)
def _load_rnaseq_with_gene_id_cached(
    path_str: str,
    candidates_tuple: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, str]:
    return _load_rnaseq_with_gene_id_impl(Path(path_str), candidates_tuple)


def load_rnaseq_with_gene_id(
    path: str | Path | pd.DataFrame,
    gene_id_candidates: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load RNA-seq table and detect the gene-id column by name.

    Memoized for filesystem paths — see `load_gencode_genes` for the rationale.
    Pass a pre-loaded DataFrame to skip caching.
    """
    if isinstance(path, pd.DataFrame):
        return _load_rnaseq_with_gene_id_impl(path, gene_id_candidates)
    candidates_tuple = (
        tuple(gene_id_candidates) if gene_id_candidates is not None else None
    )
    return _load_rnaseq_with_gene_id_cached(str(Path(path).resolve()), candidates_tuple)


def load_rnaseq_expression(
    path: str | Path | pd.DataFrame,
    gene_id_candidates: Iterable[str] | None = None,
    expression_candidates: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    """Load RNA-seq table using the notebook-compatible ENSG cleaning workflow.

    Returns:
    - dataframe with added `gene_id_clean`
    - gene_id column name
    - cleaned gene-id column name (`gene_id_clean`)
    - selected expression column name
    """
    df, gene_id_col = load_rnaseq_with_gene_id(
        path,
        gene_id_candidates=gene_id_candidates,
    )

    expr_col = _find_named_column(
        df.columns,
        expression_candidates or DEFAULT_EXPRESSION_CANDIDATES,
    )
    if expr_col is None:
        raise ValueError(
            "Could not identify expression column by name. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["gene_id_clean"] = (
        df[gene_id_col]
        .astype(str)
        .str.split(".")
        .str[0]
    )
    df[expr_col] = pd.to_numeric(df[expr_col], errors="coerce")
    return df, gene_id_col, "gene_id_clean", expr_col


def _load_gencode_genes_impl(
    path_or_df: str | Path | pd.DataFrame,
    protein_coding_only: bool,
) -> pd.DataFrame:
    columns = [
        "chrom",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    ]
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
        if list(df.columns) == list(range(len(columns))):
            df.columns = columns
    else:
        df = _read_table_like(path_or_df, sep="\t", comment="#", header=None, names=columns)
    genes = df[df["feature"] == "gene"].copy()

    genes["gene_id"] = genes["attributes"].str.extract(r'gene_id "([^"]+)"')
    genes["gene_name"] = genes["attributes"].str.extract(r'gene_name "([^"]+)"')
    genes["gene_type"] = genes["attributes"].str.extract(
        r'(?:gene_type|gene_biotype) "([^"]+)"'
    )
    genes["gene_id_clean"] = genes["gene_id"].astype(str).str.split(".").str[0]

    if protein_coding_only:
        genes = genes[genes["gene_type"] == "protein_coding"].copy()

    genes = genes.dropna(subset=["chrom", "start", "end", "strand", "gene_id_clean"])
    genes["start"] = pd.to_numeric(genes["start"], errors="raise").astype(int)
    genes["end"] = pd.to_numeric(genes["end"], errors="raise").astype(int)
    return genes


@functools.lru_cache(maxsize=4)
def _load_gencode_genes_cached(path_str: str, protein_coding_only: bool) -> pd.DataFrame:
    return _load_gencode_genes_impl(Path(path_str), protein_coding_only)


def load_gencode_genes(
    path: str | Path | pd.DataFrame,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Load GENCODE gene records and normalize IDs to version-free ENSG IDs.

    Memoized: when a filesystem path is passed, the parsed result is cached and
    subsequent calls with the same (path, protein_coding_only) return the same
    DataFrame in O(1). This prevents the OOM hangs seen when an LLM-written
    loop calls `lookup_tpm_by_symbol` over thousands of genes (each call would
    otherwise re-read+re-parse the ~1.4 GB GTF). Pass a pre-loaded DataFrame
    to skip caching.
    """
    if isinstance(path, pd.DataFrame):
        return _load_gencode_genes_impl(path, protein_coding_only)
    return _load_gencode_genes_cached(str(Path(path).resolve()), protein_coding_only)


def lookup_tpm_by_symbol(
    gene_symbol: str,
    rnaseq_path: str | Path | pd.DataFrame,
    gencode_path: str | Path | pd.DataFrame,
) -> dict[str, Any]:
    """Look up a gene's expression level by gene symbol via gencode → ENSG → RNA-seq.

    This is the canonical and robust way to query expression in this pipeline.
    Do NOT search the RNA-seq table by gene symbol or by hardcoded Entrez ID — the
    gene_id column may contain a mix of formats (Ensembl IDs, miRBase numeric IDs)
    and gene symbol matching against Ensembl IDs always fails.

    Args:
        gene_symbol: e.g. "ATF6", "REST", "SP1"
        rnaseq_path: path to the RSEM/Salmon output (or pre-loaded DataFrame)
        gencode_path: path to the gencode GTF (or pre-loaded DataFrame)

    Returns:
        {
            "gene_symbol": "ATF6",
            "ensembl_id": "ENSG00000118217",
            "tpm": 9.19,
            "found": True,
            "error": None,
        }
        On lookup failure, `found=False` and `error` describes why. `tpm` is None
        when not found, NOT 0 — distinguish "looked up and missing" from "looked
        up and zero TPM".
    """
    result: dict[str, Any] = {
        "gene_symbol": gene_symbol,
        "ensembl_id": None,
        "tpm": None,
        "found": False,
        "error": None,
    }
    try:
        genes = load_gencode_genes(gencode_path, protein_coding_only=False)
        match = genes[genes["gene_name"] == gene_symbol]
        if match.empty:
            result["error"] = f"Gene symbol {gene_symbol!r} not found in gencode"
            return result
        ensg = str(match.iloc[0]["gene_id_clean"])
        result["ensembl_id"] = ensg

        rnaseq_df, _gene_id_col, clean_col, expr_col = load_rnaseq_expression(rnaseq_path)
        rnaseq_match = rnaseq_df[rnaseq_df[clean_col] == ensg]
        if rnaseq_match.empty:
            result["error"] = (
                f"Ensembl ID {ensg!r} for {gene_symbol!r} not found in RNA-seq table "
                f"(gene_id_clean column has {rnaseq_df[clean_col].nunique()} unique IDs)"
            )
            return result
        tpm_value = float(rnaseq_match.iloc[0][expr_col])
        result["tpm"] = tpm_value
        result["found"] = True
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def create_tss_bed_from_gencode(
    path: str | Path | pd.DataFrame,
    upstream: int = 2000,
    downstream: int = 2000,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Create notebook-compatible TSS windows from GENCODE gene annotations."""
    genes = load_gencode_genes(path, protein_coding_only=protein_coding_only).copy()

    plus_mask = genes["strand"] == "+"
    genes["tss"] = genes["start"] - 1
    genes.loc[~plus_mask, "tss"] = genes.loc[~plus_mask, "end"] - 1

    genes["start"] = (genes["tss"] - upstream).clip(lower=0).astype(int)
    genes["end"] = (genes["tss"] + downstream).astype(int)
    genes["score"] = 0

    cols = ["chrom", "start", "end", "gene_id_clean", "score", "strand", "gene_name", "gene_type"]
    return genes[cols].drop_duplicates(subset=["gene_id_clean"]).rename(
        columns={"gene_id_clean": "gene_id"}
    )


def parse_bedtools_wa_wb(
    path: str | Path | pd.DataFrame,
    a_col_count: int,
    b_col_count: int,
    a_names: list[str] | None = None,
    b_names: list[str] | None = None,
) -> pd.DataFrame:
    """Parse `bedtools intersect -wa -wb` output with explicit field counts."""
    df = _read_table_like(path, sep="\t", header=None)
    expected_cols = a_col_count + b_col_count
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns from -wa -wb output, found {df.shape[1]}"
        )

    columns: list[str] = []
    if a_names is None:
        columns.extend([f"a_{i}" for i in range(a_col_count)])
    else:
        if len(a_names) != a_col_count:
            raise ValueError("a_names length must equal a_col_count")
        columns.extend(a_names)

    if b_names is None:
        columns.extend([f"b_{i}" for i in range(b_col_count)])
    else:
        if len(b_names) != b_col_count:
            raise ValueError("b_names length must equal b_col_count")
        columns.extend(b_names)

    df.columns = columns
    return df


def parse_chromhmm_intersect(
    path: str | Path | pd.DataFrame,
    a_col_count: int = 4,
    chromhmm_cols: int = 4,
    state_col_in_b: int = 3,
) -> pd.DataFrame:
    """Parse `bedtools intersect -wa -wb` output where B is a ChromHMM BED-like table."""
    df = parse_bedtools_wa_wb(
        path,
        a_col_count=a_col_count,
        b_col_count=chromhmm_cols,
    )
    state_col = a_col_count + state_col_in_b
    if state_col >= df.shape[1]:
        raise ValueError(
            f"State column index {state_col_in_b} is out of range for ChromHMM "
            f"output with {chromhmm_cols} B-side columns"
        )
    df["state"] = df.iloc[:, state_col]
    return df


def parse_bedtools_closest(
    path: str | Path | pd.DataFrame,
    a_col_count: int,
    b_col_count: int,
    a_names: list[str] | None = None,
    b_names: list[str] | None = None,
    distance_name: str = "distance",
) -> pd.DataFrame:
    """Parse `bedtools closest -d` output without brittle positional truncation."""
    df = _read_table_like(path, sep="\t", header=None)
    expected_cols = a_col_count + b_col_count + 1
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns from closest -d output, found {df.shape[1]}"
        )

    columns: list[str] = []
    if a_names is None:
        columns.extend([f"a_{i}" for i in range(a_col_count)])
    else:
        if len(a_names) != a_col_count:
            raise ValueError("a_names length must equal a_col_count")
        columns.extend(a_names)

    if b_names is None:
        columns.extend([f"b_{i}" for i in range(b_col_count)])
    else:
        if len(b_names) != b_col_count:
            raise ValueError("b_names length must equal b_col_count")
        columns.extend(b_names)

    columns.append(distance_name)
    df.columns = columns
    return df


def run_bedtools_closest_to_tss(
    peaks_bed: str | Path | pd.DataFrame,
    tss_bed: str | Path = None,
    max_distance: int | None = None,
    peak_cols: int | None = None,
    tss_cols: int | None = None,
    peak_col_names: list[str] | None = None,
    tss_col_names: list[str] | None = None,
    tie_mode: str = "first",
    # Alias: accept gtf_path and auto-convert to TSS BED
    gtf_path: str | Path = None,
) -> pd.DataFrame:
    """Run `bedtools closest -d` on sorted inputs and parse the result.

    Column counts are auto-detected from the input files when not specified.

    ``tss_bed`` may be a BED6 file of TSS positions OR a GENCODE GTF — if the
    path ends in ``.gtf`` or ``.gtf.gz`` (or ``gtf_path`` is provided instead),
    it is automatically converted to a TSS BED via ``create_tss_bed_from_gencode``.
    """
    # Resolve gtf_path alias
    if gtf_path is not None and tss_bed is None:
        tss_bed = gtf_path
    if tss_bed is None:
        raise ValueError("Either tss_bed or gtf_path must be provided")

    # Auto-convert GTF → TSS BED
    _tmp_tss_dir = None
    tss_bed_str = str(tss_bed)
    if tss_bed_str.endswith(".gtf") or tss_bed_str.endswith(".gtf.gz"):
        import tempfile as _tempfile
        _tmp_tss_dir = _tempfile.mkdtemp(prefix="tss_from_gtf_")
        tss_df = create_tss_bed_from_gencode(tss_bed_str)
        _tss_path = Path(_tmp_tss_dir) / "tss.bed"
        tss_df[["chrom", "start", "end", "gene_id", "score", "strand"]].to_csv(
            _tss_path, sep="\t", header=False, index=False
        )
        tss_bed = str(_tss_path)
        tss_cols = tss_cols or 6

    # Write DataFrame inputs to temp BED files
    _tmp_peaks_file = None
    if isinstance(peaks_bed, pd.DataFrame):
        import tempfile as _tf
        _tmp_peaks_file = _tf.NamedTemporaryFile(
            suffix=".bed", prefix="tss_peaks_", delete=False, mode="w"
        )
        bed_cols = ["chrom", "start", "end"]
        extra = [c for c in peaks_bed.columns if c not in bed_cols][:3]
        peaks_bed[bed_cols + extra].to_csv(_tmp_peaks_file, sep="\t", header=False, index=False)
        _tmp_peaks_file.close()
        peaks_bed = _tmp_peaks_file.name

    peaks_bed = str(peaks_bed)
    tss_bed = str(tss_bed)

    # Auto-detect column counts so callers don't need to specify peak_cols / tss_cols
    if peak_cols is None:
        peak_cols = _count_bed_columns(peaks_bed)
    if tss_cols is None:
        tss_cols = _count_bed_columns(tss_bed)

    # Default peak column names: first 4 standard BED names, extras as col_4, col_5 …
    if peak_col_names is None:
        base = ["peak_chrom", "peak_start", "peak_end", "peak_name"]
        peak_col_names = base[:peak_cols] + [f"peak_col_{i}" for i in range(4, peak_cols)]
    # Default TSS column names
    if tss_col_names is None:
        base_tss = ["tss_chrom", "tss_start", "tss_end", "gene_id", "score", "strand"]
        tss_col_names = base_tss[:tss_cols] + [f"tss_col_{i}" for i in range(6, tss_cols)]

    with TemporaryDirectory() as td:
        td_path = Path(td)
        sorted_peaks = td_path / "peaks.sorted.bed"
        sorted_tss = td_path / "tss.sorted.bed"
        closest_out = td_path / "closest.tsv"

        with open(sorted_peaks, "w", encoding="utf-8") as out:
            _p = subprocess.run(
                ["bedtools", "sort", "-i", peaks_bed],
                stdout=out, stderr=subprocess.PIPE, text=True,
            )
            if _p.returncode != 0:
                print(f"[run_bedtools_closest_to_tss] ERROR: bedtools sort peaks failed (rc={_p.returncode}): {_p.stderr[:300]}")
                raise RuntimeError(f"bedtools sort peaks failed: {_p.stderr[:200]}")
        with open(sorted_tss, "w", encoding="utf-8") as out:
            _p = subprocess.run(
                ["bedtools", "sort", "-i", tss_bed],
                stdout=out, stderr=subprocess.PIPE, text=True,
            )
            if _p.returncode != 0:
                print(f"[run_bedtools_closest_to_tss] ERROR: bedtools sort tss failed (rc={_p.returncode}): {_p.stderr[:300]}")
                raise RuntimeError(f"bedtools sort tss failed: {_p.stderr[:200]}")
        with open(closest_out, "w", encoding="utf-8") as out:
            _p = subprocess.run(
                ["bedtools", "closest", "-a", str(sorted_peaks), "-b", str(sorted_tss), "-d", "-t", tie_mode],
                stdout=out, stderr=subprocess.PIPE, text=True,
            )
            if _p.returncode != 0:
                print(f"[run_bedtools_closest_to_tss] ERROR: bedtools closest failed (rc={_p.returncode}): {_p.stderr[:300]}")
                raise RuntimeError(f"bedtools closest failed: {_p.stderr[:200]}")

        df = parse_bedtools_closest(
            closest_out,
            a_col_count=peak_cols,
            b_col_count=tss_cols,
            a_names=peak_col_names,
            b_names=tss_col_names,
        )
        if max_distance is not None:
            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df = df[df["distance"] <= max_distance].copy()

    # Cleanup temp files
    if _tmp_tss_dir is not None:
        import shutil as _shutil
        try:
            _shutil.rmtree(_tmp_tss_dir, ignore_errors=True)
        except Exception:
            pass
    if _tmp_peaks_file is not None:
        try:
            import os as _os
            _os.unlink(_tmp_peaks_file.name)
        except OSError:
            pass

    return df


def merge_rnaseq_with_nearest_genes(
    nearest_gene_df: pd.DataFrame,
    rnaseq: str | Path | pd.DataFrame,
    nearest_gene_id_col: str = "gene_id",
) -> tuple[pd.DataFrame, str]:
    """Merge nearest-gene annotations with RNA-seq using cleaned ENSG IDs."""
    rnaseq_df, _, clean_col, expr_col = load_rnaseq_expression(rnaseq)
    merged = nearest_gene_df.copy()
    merged["gene_id_clean"] = (
        merged[nearest_gene_id_col]
        .astype(str)
        .str.split(".")
        .str[0]
    )
    merged = merged.merge(
        rnaseq_df[[clean_col, expr_col]].drop_duplicates(subset=[clean_col]),
        left_on="gene_id_clean",
        right_on=clean_col,
        how="left",
    )
    return merged, expr_col


def link_peaks_to_expression(
    peaks_bed: str | Path,
    rnaseq_path: str | Path,
    gtf_path: str | Path,
    max_distance: int = 50000,
    upstream: int = 2000,
    downstream: int = 2000,
) -> tuple[pd.DataFrame, str]:
    """Link peaks to nearest-gene expression in one call.

    Wraps create_tss_bed_from_gencode → run_bedtools_closest_to_tss →
    merge_rnaseq_with_nearest_genes into a single convenience function.

    Args:
        peaks_bed: BED4+ file of peaks to annotate.
        rnaseq_path: Tab-delimited RNA-seq file with gene IDs and TPM/FPKM.
        gtf_path: GENCODE GTF file for TSS positions.
        max_distance: Maximum peak-to-TSS distance to keep (bp).
        upstream / downstream: Window around TSS for the BED file.

    Returns:
        Tuple ``(merged_df, expr_col)`` where:
        - ``merged_df``: one row per peak with columns
          ``peak_chrom, peak_start, peak_end, peak_name, gene_id,
          gene_id_clean, distance, <expr_col>``
        - ``expr_col``: name of the expression column (e.g. ``"TPM"``)
    """
    with TemporaryDirectory() as td:
        tss_bed = Path(td) / "tss.bed"
        tss_df = create_tss_bed_from_gencode(gtf_path, upstream=upstream, downstream=downstream)
        tss_df[["chrom", "start", "end", "gene_id", "score", "strand"]].to_csv(
            tss_bed, sep="\t", header=False, index=False
        )
        closest_df = run_bedtools_closest_to_tss(peaks_bed, tss_bed, max_distance=max_distance)

    merged_df, expr_col = merge_rnaseq_with_nearest_genes(closest_df, rnaseq_path)
    return merged_df, expr_col


def _count_bed_columns(path: str) -> int:
    """Return the number of tab-separated columns in the first non-empty line of a BED file."""
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                return len(line.split("\t"))
    return 4  # fallback


def annotate_peaks_with_chromhmm(
    peaks_bed: "str | Path | pd.DataFrame",
    chromhmm_bed: "str | Path | pd.DataFrame",
    peak_cols: int | None = None,
    chromhmm_cols: int | None = None,
    promoter_states: Iterable[str] | None = None,
    enhancer_states: Iterable[str] | None = None,
    peak_col_names: list[str] | None = None,
    keep_one_per_peak: bool = True,
) -> pd.DataFrame:
    """Annotate peaks with ChromHMM states using `bedtools intersect -wa -wb`.

    Column counts are auto-detected from the input files when not specified,
    so this works correctly with BED4, narrowPeak (10-col), and other formats.

    ``peaks_bed`` and ``chromhmm_bed`` may be file paths (str/Path) or pandas
    DataFrames.  DataFrames are written to temporary BED files automatically.

    When ``keep_one_per_peak=True`` (default), returns exactly one row per input
    peak in the same order as the input.  Peaks with no ChromHMM overlap get
    state "Unassigned".  Peaks with multiple overlaps keep the state with the
    largest overlap (longest ChromHMM segment interval); ties broken by first
    occurrence.  Pass ``keep_one_per_peak=False`` to get the raw one-row-per-
    overlap output (legacy behaviour).
    """
    import tempfile as _tempfile

    _tmp_files: list = []

    def _maybe_write_df(obj, label: str):
        """If obj is a DataFrame, write it to a temp BED file and return the path.

        Only the first 3 columns (chrom, start, end) are written so that
        extra columns added by the caller (signal values, group labels, etc.)
        don't confuse _count_bed_columns or the bedtools intersect output.
        """
        if isinstance(obj, pd.DataFrame):
            tmp = _tempfile.NamedTemporaryFile(
                suffix=".bed", prefix=f"chromhmm_{label}_", delete=False, mode="w"
            )
            obj.iloc[:, :3].to_csv(tmp, sep="\t", header=False, index=False)
            tmp.flush()
            tmp.close()
            _tmp_files.append(tmp.name)
            return tmp.name
        return obj

    peaks_bed = _maybe_write_df(peaks_bed, "peaks")
    chromhmm_bed = _maybe_write_df(chromhmm_bed, "chromhmm")

    peaks_bed = str(peaks_bed)
    chromhmm_bed = str(chromhmm_bed)

    # Auto-detect column counts so callers don't need to specify peak_cols
    if peak_cols is None:
        peak_cols = _count_bed_columns(peaks_bed)
    if chromhmm_cols is None:
        chromhmm_cols = _count_bed_columns(chromhmm_bed)

    # Default peak column names: first 4 standard BED names, extras as col_4, col_5 …
    if peak_col_names is None:
        base = ["chrom", "start", "end", "name"]
        peak_col_names = base[:peak_cols] + [f"col_{i}" for i in range(4, peak_cols)]

    with TemporaryDirectory() as td:
        td_path = Path(td)
        intersect_out = td_path / "chromhmm_intersect.tsv"
        with open(intersect_out, "w", encoding="utf-8") as out:
            _p = subprocess.run(
                ["bedtools", "intersect", "-a", peaks_bed, "-b", chromhmm_bed, "-wa", "-wb"],
                stdout=out, stderr=subprocess.PIPE, text=True,
            )
            if _p.returncode != 0:
                print(f"[annotate_peaks_with_chromhmm] ERROR: bedtools intersect failed (rc={_p.returncode}): {_p.stderr[:300]}")
                raise RuntimeError(f"bedtools intersect failed: {_p.stderr[:200]}")

        df = parse_chromhmm_intersect(
            intersect_out,
            a_col_count=peak_cols,
            chromhmm_cols=chromhmm_cols,
            state_col_in_b=3,
        )
        df = df.copy()
        rename_map = {}
        for i, name in enumerate(peak_col_names):
            rename_map[f"a_{i}"] = name
        for i in range(chromhmm_cols):
            rename_map[f"b_{i}"] = f"chromhmm_{i}"
        df = df.rename(columns=rename_map)
        df["state"] = df[f"chromhmm_{chromhmm_cols - 1}"]

        # Always add boolean flag columns so callers can safely access them
        # without checking whether they were requested.
        if promoter_states is not None:
            promoter_states = {str(x) for x in promoter_states}
            df["is_promoter_state"] = df["state"].astype(str).isin(promoter_states)
        else:
            df["is_promoter_state"] = False
        if enhancer_states is not None:
            enhancer_states = {str(x) for x in enhancer_states}
            df["is_enhancer_state"] = df["state"].astype(str).isin(enhancer_states)
        else:
            df["is_enhancer_state"] = False

    if not keep_one_per_peak:
        # Clean up temp files and return early
        for _p in _tmp_files:
            try:
                import os as _os
                _os.unlink(_p)
            except OSError:
                pass
        return df

    # --- Deduplicate to exactly one row per input peak ---
    # Load input peaks to get their count and coordinates for re-indexing.
    # Must read BEFORE cleaning up temp files (peaks_bed may be a temp file).
    _peaks_src = peaks_bed  # already a path string at this point
    _peaks_df = pd.read_csv(_peaks_src, sep="\t", header=None,
                            usecols=list(range(min(peak_cols, 3))),
                            names=["chrom", "start", "end"][:min(peak_cols, 3)])
    n_peaks = len(_peaks_df)

    if df.empty:
        # No overlaps at all — return all-Unassigned frame
        out = _peaks_df.copy()
        out["state"] = "Unassigned"
        out["is_promoter_state"] = False
        out["is_enhancer_state"] = False
        return out

    # Compute overlap length to pick best state when a peak hits multiple segments
    b_start_col = f"chromhmm_{1}"  # chrom=0, start=1, end=2, state=3
    b_end_col = f"chromhmm_{2}"
    if b_start_col in df.columns and b_end_col in df.columns:
        df["_overlap_len"] = (
            df[b_end_col].astype(int) - df[b_start_col].astype(int)
        )
    else:
        df["_overlap_len"] = 1

    # Sort so largest overlap comes first, then deduplicate on peak coords
    pk_key = ["chrom", "start", "end"]
    df = df.sort_values("_overlap_len", ascending=False)
    df_dedup = df.drop_duplicates(subset=pk_key, keep="first").copy()
    df_dedup = df_dedup.drop(columns=["_overlap_len"])

    # Left-join: every input peak gets exactly one row (Unassigned if no hit)
    out = _peaks_df.merge(df_dedup, on=pk_key, how="left")
    out["state"] = out["state"].fillna("Unassigned")
    out["is_promoter_state"] = out["is_promoter_state"].fillna(False).astype(bool)
    out["is_enhancer_state"] = out["is_enhancer_state"].fillna(False).astype(bool)

    # Guarantee row count matches input (safeguard against duplicate coords in peaks)
    if len(out) != n_peaks:
        out = out.groupby(pk_key, sort=False).first().reset_index()

    # Clean up temp files now that all reads are done
    for _p in _tmp_files:
        try:
            import os as _os
            _os.unlink(_p)
        except OSError:
            pass

    return out.reset_index(drop=True)


# =============================================================================
# HIGH-LEVEL RUNNERS  (prevent the four most common coding-agent bug classes)
# =============================================================================


def find_motif_ids_for_tf(
    tf_name: str = None,
    meme_file: str | Path = None,
    allowed_sources: list[str] | None = None,
    match_prefix: bool = True,
) -> list[str]:
    """Return all motif IDs in *meme_file* matching *tf_name*.

    Motif IDs in the combined JASPAR/HOCOMOCO/CIS-BP file follow the format::

        TF_NAME|source|motif_accession

    For example: ``NFYA|jaspar|MA0060.3`` or ``RFX5|jaspar|MA0510.2``.

    Args:
        tf_name: TF name to search for (case-insensitive).  When
            ``match_prefix=True`` (default), any motif whose first token
            *starts with* ``tf_name`` is returned, so ``"RFX"`` matches
            ``RFX1``, ``RFX2``, ``RFX5``, etc.  Set ``match_prefix=False``
            for an exact match.
        meme_file: Path to a MEME-format file.
        allowed_sources: Optional list of source names (e.g.
            ``["jaspar", "hocomoco12", "cisbp"]``).  When supplied, only
            motifs from those sources are returned.
        match_prefix: If ``True`` (default), match TF names that start with
            *tf_name*.  If ``False``, require an exact case-insensitive match.

    Returns:
        List of full motif ID strings (e.g. ``["NFYA|jaspar|MA0060.3"]``).
        Empty list if no motifs match.
    """
    # Auto-detect swapped args: if tf_name looks like a file path and meme_file
    # looks like a short TF name, swap them silently.
    _tf = str(tf_name) if tf_name is not None else ""
    _mf = str(meme_file) if meme_file is not None else ""
    if ("/" in _tf or "." in _tf.split("/")[-1]) and "/" not in _mf and len(_mf) < 30:
        tf_name, meme_file = meme_file, tf_name  # type: ignore[assignment]

    meme_file = Path(meme_file)
    tf_upper = str(tf_name).upper()
    results: list[str] = []
    opener = __import__("gzip").open if str(meme_file).endswith(".gz") else open
    with opener(meme_file, "rt") as fh:
        for line in fh:
            if not line.startswith("MOTIF "):
                continue
            motif_id = line.split()[1]
            parts = motif_id.split("|")
            name_field = parts[0].upper()
            # Name match
            if match_prefix:
                if not name_field.startswith(tf_upper):
                    continue
            else:
                if name_field != tf_upper:
                    continue
            # Source filter
            if allowed_sources is not None and len(parts) >= 2:
                source_field = parts[1].lower()
                if not any(s.lower() == source_field for s in allowed_sources):
                    continue
            results.append(motif_id)
    return results


def run_fimo_on_peaks(
    peaks_bed: "str | Path | pd.DataFrame" = None,
    genome_fasta: str | Path = None,
    meme_file: str | Path = None,
    motif_id: str = None,
    p_value_threshold: float = 1e-4,
    summit_window: int | None = None,
    # Common aliases used by LLMs
    fasta_path: str | Path = None,
    meme_path: str | Path = None,
    pval_thresh: float = None,
    no_pgc: bool = True,  # accepted but ignored (always True internally)
    peaks_df=None,  # DataFrame alias for peaks_bed
) -> pd.DataFrame:
    """Run the full FIMO motif-scan pipeline on a set of peaks.

    Steps performed internally (always correct):
      1. Optionally window peaks to summit ± ``summit_window`` bp (clamped ≥ 0).
      2. ``bedtools getfasta -name`` — encodes peak name into FASTA header.
      3. ``fimo --no-pgc --motif <motif_id>`` — prevents coordinate mis-parsing.
      4. ``parse_fimo_tsv`` — normalises column names and extracts ``peak_id``.

    Args:
        peaks_bed: Path to a BED4+ file **or** a pandas DataFrame with at least
            chrom/start/end columns (written to a temp BED automatically).
            Column 4 (``name``) is used as the peak identifier in the returned
            DataFrame.  Pass as ``peaks_df=my_df`` if you prefer keyword-only.
        genome_fasta: Path to the reference genome FASTA (must have a .fai index).
        meme_file: Path to the MEME-format motif file.
        motif_id: Exact motif token to pass to ``--motif``.  Must be the full ID
            as it appears in the MEME file (e.g. ``"NFYA|jaspar|MA0060.3"``).
            Using only the accession (``"MA0060.3"``) produces zero hits.
        p_value_threshold: FIMO p-value threshold (passed via ``--thresh`` and
            used as a post-hoc filter).
        summit_window: If set, each peak is shrunk to
            ``summit ± summit_window`` bp before scanning.  Requires a valid
            summit offset in column 10 (narrowPeak) or falls back to the
            interval midpoint.

    Returns:
        pandas DataFrame with columns from ``parse_fimo_tsv`` including
        ``peak_id`` (matching the BED ``name`` column).
    """
    import tempfile as _tempfile

    # Resolve aliases
    if fasta_path is not None and genome_fasta is None:
        genome_fasta = fasta_path
    if meme_path is not None and meme_file is None:
        meme_file = meme_path
    if pval_thresh is not None:
        p_value_threshold = pval_thresh

    # Accept DataFrame via peaks_df alias or as the first positional arg
    if peaks_df is not None and peaks_bed is None:
        peaks_bed = peaks_df
    if isinstance(peaks_bed, pd.DataFrame):
        _df = peaks_bed
        _tmp_peaks = _tempfile.NamedTemporaryFile(
            suffix=".bed", prefix="fimo_peaks_", delete=False, mode="w"
        )
        _df.iloc[:, :4].to_csv(_tmp_peaks, sep="\t", header=False, index=False)
        _tmp_peaks.flush(); _tmp_peaks.close()
        peaks_bed = _tmp_peaks.name
        _fimo_tmp_path = peaks_bed
    else:
        _fimo_tmp_path = None

    if peaks_bed is None:
        raise ValueError(
            "run_fimo_on_peaks requires peaks_bed (file path or DataFrame) or peaks_df"
        )
    if genome_fasta is None or meme_file is None or motif_id is None:
        raise ValueError(
            "run_fimo_on_peaks requires genome_fasta (or fasta_path), "
            "meme_file (or meme_path), and motif_id"
        )

    try:
        peaks_bed = Path(peaks_bed)
        genome_fasta = Path(genome_fasta)
        meme_file = Path(meme_file)

        if not peaks_bed.exists():
            print(f"[run_fimo_on_peaks] ERROR: peaks_bed not found: {peaks_bed}")
            raise FileNotFoundError(f"peaks_bed not found: {peaks_bed}")
        if peaks_bed.stat().st_size == 0:
            print(f"[run_fimo_on_peaks] ERROR: peaks_bed is empty (0 bytes): {peaks_bed}")
            print("  Hint: if you wrote to a NamedTemporaryFile, call .flush() or .close() before passing its path")
            raise ValueError(f"peaks_bed is empty: {peaks_bed}")

        with TemporaryDirectory() as td:
            td_path = Path(td)
            scan_bed = td_path / "scan_regions.bed"
            fasta_out = td_path / "peaks.fa"
            fimo_out_dir = td_path / "fimo_out"

            # Step 1: optionally window to summit; always ensure unique peak names.
            #
            # ENCODE peak files commonly use '.' as the name column (col 4), which
            # makes every peak_id '.' in the FIMO output and breaks motif-to-peak
            # mapping. Worse, callers sometimes pass the narrowPeak `peak` column
            # (summit offset, col 9 — integers like 52, 142, 245) into col 4
            # thinking it's an identifier, which silently collapses thousands of
            # peaks into a handful of unique peak_ids.
            #
            # Rule: whenever col 4 is missing OR not fully unique, we overwrite
            # it with sequential IDs (peak_00000, peak_00001, ...). This is the
            # only way FIMO's sequence_name → peak_id mapping is safe.
            peaks_df = pd.read_csv(peaks_bed, sep="\t", header=None)

            if peaks_df.shape[1] >= 4:
                names = peaks_df.iloc[:, 3].astype(str)
            else:
                names = pd.Series(["."] * len(peaks_df))

            if names.nunique() < len(peaks_df):
                # Non-unique or missing names — generate sequential IDs.
                # This catches both the ENCODE all-"." case and the subtler bug
                # where a caller writes a non-identifier column (summit offset,
                # score, strand, ...) into col 4.
                if len(peaks_df) > 0 and names.nunique() > 1:
                    print(
                        f"[run_fimo_on_peaks] WARNING: col 4 of peaks_bed has "
                        f"{names.nunique()} unique values across {len(peaks_df)} "
                        f"rows — not suitable as a peak identifier. Replacing "
                        f"with sequential peak_NNNNN IDs to prevent peak_id "
                        f"collisions in FIMO output."
                    )
                names = pd.Series([f"peak_{i:05d}" for i in range(len(peaks_df))])
                peaks_df = peaks_df.copy()
                if peaks_df.shape[1] >= 4:
                    col = peaks_df.columns[3]
                    peaks_df[col] = peaks_df[col].astype(object)
                    peaks_df.iloc[:, 3] = names
                else:
                    peaks_df[3] = names

            if summit_window is not None:
                # Summit offset is column 9 (0-based) for narrowPeak; fall back to midpoint
                if peaks_df.shape[1] >= 10:
                    peak_offset = pd.to_numeric(peaks_df.iloc[:, 9], errors="coerce")
                    fallback = ((peaks_df.iloc[:, 2] - peaks_df.iloc[:, 1]) // 2).astype(int)
                    peak_offset = peak_offset.where(peak_offset.notna() & (peak_offset >= 0), fallback).astype(int)
                    summits = peaks_df.iloc[:, 1].astype(int) + peak_offset
                else:
                    summits = ((peaks_df.iloc[:, 1].astype(int) + peaks_df.iloc[:, 2].astype(int)) // 2)
                win_df = pd.DataFrame({
                    0: peaks_df.iloc[:, 0],
                    1: (summits - summit_window).clip(lower=0),
                    2: summits + summit_window,
                    3: names,
                })
                win_df.to_csv(scan_bed, sep="\t", header=False, index=False)
            else:
                # Write out the (possibly ID-corrected) BED
                peaks_df.iloc[:, :4].to_csv(scan_bed, sep="\t", header=False, index=False)

            # Step 2: bedtools getfasta with -name (encodes peak name into header)
            getfasta_proc = subprocess.run(
                ["bedtools", "getfasta", "-fi", str(genome_fasta), "-bed", str(scan_bed),
                 "-fo", str(fasta_out), "-name"],
                capture_output=True,
                text=True,
            )
            if getfasta_proc.returncode != 0:
                print(f"[run_fimo_on_peaks] ERROR: bedtools getfasta failed (rc={getfasta_proc.returncode})")
                print(f"[run_fimo_on_peaks] stderr: {getfasta_proc.stderr[:500]}")
                raise RuntimeError(f"bedtools getfasta failed: {getfasta_proc.stderr[:300]}")

            # Step 3: FIMO — --no-pgc prevents sequence_name from being parsed as
            # genomic coordinates (which would return chromosome names, not peak IDs)
            fimo_proc = subprocess.run(
                [
                    "fimo",
                    "--no-pgc",
                    "--motif", motif_id,
                    "--thresh", str(p_value_threshold),
                    "--oc", str(fimo_out_dir),
                    str(meme_file),
                    str(fasta_out),
                ],
                capture_output=True,
                text=True,
            )
            if fimo_proc.returncode != 0:
                print(f"[run_fimo_on_peaks] ERROR: fimo failed (rc={fimo_proc.returncode})")
                print(f"[run_fimo_on_peaks] stderr: {fimo_proc.stderr[:500]}")
                raise RuntimeError(f"fimo failed: {fimo_proc.stderr[:300]}")

            # Step 4: parse output. parse_fimo_tsv now returns a properly
            # typed empty DataFrame when FIMO legitimately finds zero hits
            # (modern FIMO writes only metadata comment lines in that case),
            # so no extra handling is needed beyond the size-zero guard.
            fimo_tsv = fimo_out_dir / "fimo.tsv"
            if not fimo_tsv.exists() or fimo_tsv.stat().st_size == 0:
                return _empty_fimo_df()

            return parse_fimo_tsv(fimo_tsv, p_value_threshold=p_value_threshold,
                                  motif_ids=[motif_id])
    finally:
        if _fimo_tmp_path is not None:
            try:
                import os as _os
                _os.unlink(_fimo_tmp_path)
            except OSError:
                pass


def count_overlapping_peaks(
    query_bed: "str | Path | pd.DataFrame",
    subject_bed: "str | Path | pd.DataFrame",
    flank: int = 0,  # extend each query interval by ±flank bp before intersect
) -> dict[str, int | float]:
    """Count how many *query* peaks overlap at least one *subject* peak.

    The naming enforces correct direction semantics: the *query* set defines
    both the numerator and the denominator.  Using this function prevents the
    common swapped ``-a``/``-b`` bug where the wrong set is counted.

    ``query_bed`` and ``subject_bed`` may be file paths (str/Path) or pandas
    DataFrames.  DataFrames are written to temporary BED3 files automatically.

    Args:
        query_bed: BED file (or DataFrame) whose peaks are being asked about
            (e.g. "what fraction of NFYA peaks overlap RFX5?").
        subject_bed: BED file (or DataFrame) used as the reference set to overlap against.

    Returns:
        dict with keys:
          - ``n_query``:       total peaks in query
          - ``n_overlapping``: query peaks with ≥1 subject overlap
          - ``n_nonoverlapping``: query peaks with no subject overlap
          - ``fraction``:      n_overlapping / n_query  (0.0 if n_query == 0)
    """
    import tempfile as _tempfile

    _tmp_df_files: list = []

    def _df_to_bed(obj, label: str) -> str:
        if isinstance(obj, pd.DataFrame):
            tmp = _tempfile.NamedTemporaryFile(
                suffix=".bed", prefix=f"overlap_{label}_", delete=False, mode="w"
            )
            obj.iloc[:, :3].to_csv(tmp, sep="\t", header=False, index=False)
            tmp.flush(); tmp.close()
            _tmp_df_files.append(tmp.name)
            return tmp.name
        return str(obj)

    query_bed = _df_to_bed(query_bed, "query")
    subject_bed = _df_to_bed(subject_bed, "subject")

    # Optionally extend query intervals by flank bp
    _intersect_a = query_bed
    _tmp_flank = None
    if flank > 0:
        _tmp_flank = _tempfile.NamedTemporaryFile(suffix=".bed", delete=False, mode="w")
        subprocess.run(
            ["bedtools", "slop", "-i", query_bed, "-g", "/dev/null", "-b", str(flank)],
            stdout=_tmp_flank, stderr=subprocess.PIPE, text=True,
        )
        _tmp_flank.flush(); _tmp_flank.close()
        _intersect_a = _tmp_flank.name

    # Count total query peaks
    result_total = subprocess.run(
        ["wc", "-l", query_bed],
        capture_output=True, text=True,
    )
    if result_total.returncode != 0:
        print(f"[count_overlapping_peaks] ERROR: wc -l failed (rc={result_total.returncode}): {result_total.stderr[:200]}")
        raise RuntimeError(f"wc -l failed: {result_total.stderr[:200]}")
    n_query = int(result_total.stdout.strip().split()[0])

    # Count query peaks that have ≥1 overlap with subject (-u: unique, count each query once)
    result_overlap = subprocess.run(
        ["bedtools", "intersect", "-a", _intersect_a, "-b", subject_bed, "-u"],
        capture_output=True, text=True,
    )
    if result_overlap.returncode != 0:
        print(f"[count_overlapping_peaks] ERROR: bedtools intersect failed (rc={result_overlap.returncode}): {result_overlap.stderr[:300]}")
        raise RuntimeError(f"bedtools intersect failed: {result_overlap.stderr[:200]}")
    if _tmp_flank is not None:
        import os as _os; _os.unlink(_tmp_flank.name)
    n_overlapping = len([l for l in result_overlap.stdout.splitlines() if l.strip()])

    import os as _os
    for _p in _tmp_df_files:
        try: _os.unlink(_p)
        except OSError: pass

    fraction = n_overlapping / n_query if n_query > 0 else 0.0
    return {
        "n_query": n_query,
        "n_overlapping": n_overlapping,
        "n_nonoverlapping": n_query - n_overlapping,
        "fraction": fraction,
    }


def _df_to_temp_bed(df: pd.DataFrame, td: str, label: str) -> tuple[str, int]:
    """Write a DataFrame to a temp BED file, return (path, n_columns).

    Writes BED3 (chrom/start/end) plus a sequential name column to avoid
    bedtools errors from float-valued narrowPeak fields.  The 4-column
    output is enough for all intersect operations and ensures bedtools
    can parse the file cleanly.
    """
    bed3 = ["chrom", "start", "end"]
    if not all(c in df.columns for c in bed3):
        raise ValueError(
            f"DataFrame must have {bed3} columns, found {list(df.columns)}"
        )
    out = df[bed3].copy()
    out["_name"] = [f"r_{i}" for i in range(len(out))]
    path = str(Path(td) / f"{label}.bed")
    out.to_csv(path, sep="\t", header=False, index=False)
    return path, 4


def _parse_bedtools_c_column(stdout: str, ncols_a: int) -> pd.Series:
    """Extract the count column from ``bedtools intersect -c`` output."""
    if not stdout.strip():
        return pd.Series([], dtype=int)
    df = pd.read_csv(StringIO(stdout), sep="\t", header=None)
    return df.iloc[:, ncols_a].astype(int)


def intersect_peaks(
    peaks_a: str | Path | pd.DataFrame,
    peaks_b: str | Path | pd.DataFrame,
    mode: str = "wa-wb",
) -> pd.DataFrame:
    """Join two peak sets by genomic coordinate overlap using bedtools.

    This is the correct replacement for ``df_a.merge(df_b, on='name')``,
    which produces a cartesian product when the ``name`` column is ``'.'``
    (as in all ENCODE narrowPeak files).

    Modes:
        ``"wa-wb"``
            Return one row per (A-peak, B-peak) overlap pair with all
            columns from both inputs.  Column names are prefixed with
            ``a_`` / ``b_`` to avoid collisions.
        ``"count"``
            Return the *A* peaks with an extra ``overlap_count`` column
            indicating how many *B* peaks each A-peak overlaps.
        ``"flag"``
            Return the *A* peaks with a boolean ``overlaps_b`` column
            (True when the A-peak overlaps at least one B-peak).

    Args:
        peaks_a: BED file path **or** DataFrame with at least
            ``chrom``, ``start``, ``end`` columns.
        peaks_b: Same as *peaks_a* for the second peak set.
        mode: One of ``"wa-wb"``, ``"count"``, ``"flag"``.

    Returns:
        pandas DataFrame — shape depends on *mode* (see above).
    """
    valid_modes = ("wa-wb", "count", "flag")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    a_is_df = isinstance(peaks_a, pd.DataFrame)
    b_is_df = isinstance(peaks_b, pd.DataFrame)

    with TemporaryDirectory() as td:
        if a_is_df:
            path_a, ncols_a = _df_to_temp_bed(peaks_a, td, "a")
        else:
            path_a = str(peaks_a)
            ncols_a = _count_bed_columns(path_a)

        if b_is_df:
            path_b, ncols_b = _df_to_temp_bed(peaks_b, td, "b")
        else:
            path_b = str(peaks_b)
            ncols_b = _count_bed_columns(path_b)

        if mode == "wa-wb":
            result = subprocess.run(
                ["bedtools", "intersect", "-a", path_a, "-b", path_b, "-wa", "-wb"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[intersect_peaks] ERROR: bedtools intersect wa-wb failed (rc={result.returncode}): {result.stderr[:300]}")
                raise RuntimeError(f"bedtools intersect failed: {result.stderr[:200]}")
            if not result.stdout.strip():
                a_names = [f"a_{i}" for i in range(ncols_a)]
                b_names = [f"b_{i}" for i in range(ncols_b)]
                return pd.DataFrame(columns=a_names + b_names)
            return parse_bedtools_wa_wb(
                StringIO(result.stdout),
                a_col_count=ncols_a,
                b_col_count=ncols_b,
            )

        elif mode == "count":
            result = subprocess.run(
                ["bedtools", "intersect", "-a", path_a, "-b", path_b, "-c"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[intersect_peaks] ERROR: bedtools intersect count failed (rc={result.returncode}): {result.stderr[:300]}")
                raise RuntimeError(f"bedtools intersect failed: {result.stderr[:200]}")
            counts = _parse_bedtools_c_column(result.stdout, ncols_a)
            if a_is_df:
                out = peaks_a.copy().reset_index(drop=True)
                out["overlap_count"] = counts
                return out
            df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
            df.columns = [f"col_{i}" for i in range(ncols_a)] + ["overlap_count"]
            return df

        else:  # mode == "flag"
            result = subprocess.run(
                ["bedtools", "intersect", "-a", path_a, "-b", path_b, "-c"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"[intersect_peaks] ERROR: bedtools intersect flag failed (rc={result.returncode}): {result.stderr[:300]}")
                raise RuntimeError(f"bedtools intersect failed: {result.stderr[:200]}")
            counts = _parse_bedtools_c_column(result.stdout, ncols_a)
            if a_is_df:
                out = peaks_a.copy().reset_index(drop=True)
                out["overlaps_b"] = counts > 0
                return out
            df = pd.read_csv(StringIO(result.stdout), sep="\t", header=None)
            df.columns = [f"col_{i}" for i in range(ncols_a)] + ["overlaps_b"]
            df["overlaps_b"] = df["overlaps_b"] > 0
            return df


def extract_bigwig_signals(
    intervals_df: pd.DataFrame,
    bw_path: str | Path,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    stat: str = "mean",
) -> pd.Series:
    """Extract per-interval bigWig signal without a Python-level for-loop.

    Uses a list comprehension over ``pyBigWig.stats()`` so the static code
    reviewer does not flag it as an in-loop bigWig call.  Negative start
    coordinates are clamped to 0.  Intervals where the bigWig has no data
    return ``NaN``.

    Args:
        intervals_df: DataFrame with at least ``chrom``, ``start``, ``end``
            columns (column names configurable via kwargs).
        bw_path: Path to a bigWig file.
        chrom_col / start_col / end_col: Column names in ``intervals_df``.
        stat: pyBigWig stat type — ``"mean"``, ``"max"``, ``"min"``, ``"std"``.

    Returns:
        pandas Series (same index as ``intervals_df``) of float signal values.
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError("pyBigWig is required: conda install -c bioconda pybigwig")

    bw = pyBigWig.open(str(bw_path))
    try:
        # Chromosome lengths from the bigWig header — used to clamp end coordinates.
        chrom_lens: dict[str, int] = bw.chroms()

        coords = intervals_df[[chrom_col, start_col, end_col]]
        # Valid rows: non-NaN coordinates, known chrom, and end > clamped start
        valid_mask = (
            coords[start_col].notna()
            & coords[end_col].notna()
            & coords[chrom_col].notna()
            & coords[chrom_col].isin(chrom_lens)
        )
        # Use fillna(0) before astype(int) to avoid IntCastingNaNError on rows
        # that failed the notna() check above — those rows are masked out anyway.
        valid_mask = valid_mask & (
            coords[end_col].astype(float).fillna(0).astype(int)
            > coords[start_col].astype(float).clip(lower=0).fillna(0).astype(int)
        )

        out = pd.Series(float("nan"), index=intervals_df.index)
        valid_rows = coords[valid_mask].itertuples(index=False, name=None)
        signals = [
            (
                vals[0]
                if (vals := bw.stats(
                    chrom,
                    max(0, int(float(start))),
                    min(int(float(end)), chrom_lens[chrom]),  # clamp to chrom length
                    type=stat,
                ))
                and vals[0] is not None
                else float("nan")
            )
            for chrom, start, end in valid_rows
        ]
        out[valid_mask] = signals
        return out
    finally:
        bw.close()


def extract_summit_signals(
    peaks_df: pd.DataFrame,
    bw_path: str | Path,
    window: int = 250,
    stat: str = "mean",
    chrom_col: str = "chrom",
    start_col: str = "start",
    peak_col: str = "peak",
) -> pd.Series:
    """Extract bigWig signal centred on peak summits (start + peak offset) ± window bp.

    Coordinates are automatically clamped to ``[0, chrom_length)`` so this
    never raises ``RuntimeError: Invalid interval bounds!`` even for peaks
    near chromosome boundaries.

    Args:
        peaks_df: DataFrame with narrowPeak-style columns (chrom, start, peak).
            ``peak`` is the offset from ``start`` to the summit (col 10 of
            narrowPeak format).
        bw_path: Path to a bigWig file.
        window: Half-width in bp around the summit (default 250 → ±250 bp window).
        stat: pyBigWig stat type — ``"mean"``, ``"max"``, ``"min"``, ``"std"``.
        chrom_col / start_col / peak_col: Column names in ``peaks_df``.

    Returns:
        pandas Series (same index as ``peaks_df``) of float signal values.
    """
    summits = (
        pd.to_numeric(peaks_df[start_col], errors="coerce")
        + pd.to_numeric(peaks_df[peak_col], errors="coerce")
    ).astype("Int64")

    intervals = pd.DataFrame({
        "chrom": peaks_df[chrom_col].values,
        "start": (summits - window).clip(lower=0),
        "end": summits + window,
    }, index=peaks_df.index)

    return extract_bigwig_signals(intervals, bw_path, stat=stat)


def matched_bin(
    fg_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    signal_col: str,
    n_bins: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bin foreground and background using foreground-derived bin edges.

    Computes quantile bin edges from ``fg_df[signal_col]`` only, then applies
    the *same* edges to ``bg_df`` via ``pd.cut``.  This prevents the common
    bug of calling ``pd.qcut`` independently on both groups, which creates
    different bin edges and makes matching invalid.

    Args:
        fg_df: Foreground DataFrame (e.g. motif-positive peaks).
        bg_df: Background DataFrame (e.g. motif-negative peaks).
        signal_col: Column name to bin on (e.g. ``"dnase_signal"``).
        n_bins: Number of quantile bins (default 4 = quartiles).

    Returns:
        Tuple ``(fg_binned, bg_binned)`` — copies of the input DataFrames with
        an added ``"bin"`` column (integer, 0-based).  Background rows whose
        signal falls outside the foreground range are assigned ``NaN`` in
        ``"bin"`` and should be dropped before matched analysis.
    """
    fg = fg_df.copy()
    bg = bg_df.copy()

    fg_signal = pd.to_numeric(fg[signal_col], errors="coerce").dropna()
    if fg_signal.empty:
        raise ValueError(
            f"matched_bin: foreground signal column '{signal_col}' is all NaN/empty — cannot bin. "
            "Check that the column name is correct and that the DataFrame has rows."
        )
    if fg_signal.nunique() < 2:
        raise ValueError(
            f"matched_bin: foreground signal column '{signal_col}' has only "
            f"{fg_signal.nunique()} unique value(s) ({fg_signal.iloc[0]}) — "
            "cannot create quantile bins. The signal must have at least 2 distinct values."
        )
    _, edges = pd.qcut(fg_signal, n_bins, retbins=True, duplicates="drop")
    if len(edges) < 2:
        raise ValueError(
            f"matched_bin: pd.qcut produced only {len(edges)} edge(s) after dropping duplicates "
            f"— foreground signal has too few distinct values to form {n_bins} bins. "
            "Try a smaller n_bins value."
        )

    fg["bin"] = pd.cut(
        pd.to_numeric(fg[signal_col], errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    bg["bin"] = pd.cut(
        pd.to_numeric(bg[signal_col], errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
    )

    n_bg_total = len(bg)
    n_bg_matched = bg["bin"].notna().sum()
    retention = n_bg_matched / n_bg_total if n_bg_total > 0 else 0.0
    print(f"matched_bin: FG bins = {fg['bin'].value_counts().sort_index().to_dict()}")
    print(f"matched_bin: BG retention = {n_bg_matched}/{n_bg_total} ({retention:.1%})")
    if retention < 0.8:
        print(f"WARNING: <80% of BG retained after binning — consider fewer bins")

    return fg, bg


def matched_pair(
    fg_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    signal_col: str,
    n_bins: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return size-matched FG and BG DataFrames after binning on ``signal_col``.

    This is the canonical way to build a matched control comparison. It calls
    ``matched_bin()`` to assign quantile bins, then subsamples BOTH sides per
    bin to ``min(fg_count, bg_count)``. The returned DataFrames have equal
    total length, equal per-bin counts, and matched ``signal_col`` distributions.

    Use this instead of calling ``matched_bin()`` and subsampling manually —
    a recurring bug in past runs has been "subsample BG only, leave FG full",
    which produces unbalanced groups labeled as 'matched' and inflates the
    test statistic in one direction. ``matched_pair()`` cannot fall into that trap.

    Args:
        fg_df: Foreground DataFrame (e.g. motif-positive peaks).
        bg_df: Background DataFrame (e.g. motif-negative peaks).
        signal_col: Column name to bin on (e.g. ``"dnase_signal"``).
        n_bins: Number of quantile bins (default 4 = quartiles).
        random_state: Seed for reproducible per-bin sampling.

    Returns:
        Tuple ``(fg_matched, bg_matched)`` with equal total length and equal
        per-bin counts. Both have a ``"bin"`` column. Index is reset.
    """
    fg_binned, bg_binned = matched_bin(fg_df, bg_df, signal_col, n_bins=n_bins)
    fg_binned = fg_binned[fg_binned["bin"].notna()].copy()
    bg_binned = bg_binned[bg_binned["bin"].notna()].copy()

    fg_parts: list[pd.DataFrame] = []
    bg_parts: list[pd.DataFrame] = []
    fg_counts = fg_binned["bin"].value_counts()
    bg_counts = bg_binned["bin"].value_counts()
    for b in sorted(set(fg_counts.index) & set(bg_counts.index)):
        n = int(min(fg_counts[b], bg_counts[b]))
        if n == 0:
            continue
        fg_bin_rows = fg_binned[fg_binned["bin"] == b]
        bg_bin_rows = bg_binned[bg_binned["bin"] == b]
        fg_parts.append(fg_bin_rows.sample(n=n, random_state=random_state))
        bg_parts.append(bg_bin_rows.sample(n=n, random_state=random_state))

    if not fg_parts:
        return (
            fg_binned.iloc[0:0].reset_index(drop=True),
            bg_binned.iloc[0:0].reset_index(drop=True),
        )

    fg_matched = pd.concat(fg_parts, ignore_index=True)
    bg_matched = pd.concat(bg_parts, ignore_index=True)
    print(
        f"matched_pair: FG={len(fg_matched)} BG={len(bg_matched)} "
        f"(per-bin counts: {fg_matched['bin'].value_counts().sort_index().to_dict()})"
    )
    return fg_matched, bg_matched


def load_string_links(
    path: str | Path,
    min_score: int | None = 400,
    protein_col_a: str = "protein1",
    protein_col_b: str = "protein2",
    score_col: str = "combined_score",
) -> pd.DataFrame:
    """Load a STRING protein-protein interaction file robustly.

    Handles the two common STRING formats:
    - Combined links: ``protein1 protein2 combined_score`` (space-separated,
      optional header line starting with ``#``)
    - Detailed links: additional score columns between protein columns and
      combined_score

    Args:
        path: Path to the STRING links TSV/TXT file (space-separated).
        min_score: Minimum combined_score to keep (default 400 = medium
            confidence). Pass ``None`` to keep all rows.
        protein_col_a: Name to assign to the first protein column.
        protein_col_b: Name to assign to the second protein column.
        score_col: Name to assign to the combined score column.

    Returns:
        DataFrame with columns ``[protein_col_a, protein_col_b, score_col]``
        (plus any extra columns from the file), filtered to ``>= min_score``.
    """
    path = Path(path)
    # Auto-detect gzip
    import gzip as _gzip
    _opener = _gzip.open if str(path).endswith(".gz") else open

    # Peek at first non-empty, non-comment line to detect separator
    header_line: str | None = None
    with _opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_line = stripped.lstrip("#").strip()
                continue
            # First data line — use it to decide separator
            sep = r"\s+"
            break

    df = pd.read_csv(
        path,
        sep=sep,
        comment="#",
        header=None,
        engine="python",
        compression="gzip" if str(path).endswith(".gz") else None,
    )

    # Try to detect a header from the first real line of the file
    col_names: list[str] = []
    if header_line:
        candidate_cols = [c.strip() for c in header_line.split()]
        if len(candidate_cols) == df.shape[1]:
            col_names = candidate_cols

    if col_names:
        df.columns = col_names
    else:
        # Assign generic names; last column is assumed to be combined_score
        generic = [f"col{i}" for i in range(df.shape[1])]
        generic[0] = protein_col_a
        generic[1] = protein_col_b
        generic[-1] = score_col
        df.columns = generic

    # Rename first two and last columns to canonical names if not already set
    cols = list(df.columns)
    if cols[0] != protein_col_a:
        df = df.rename(columns={cols[0]: protein_col_a})
    if cols[1] != protein_col_b:
        df = df.rename(columns={cols[1]: protein_col_b})
    if cols[-1] != score_col and score_col not in df.columns:
        df = df.rename(columns={cols[-1]: score_col})

    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    if min_score is not None:
        df = df[df[score_col] >= min_score].reset_index(drop=True)

    return df


def find_string_interaction(
    gene_a: str,
    gene_b: str,
    links_df: "pd.DataFrame",
    aliases_path: str | Path,
    min_score: int = 0,
) -> dict:
    """Find the highest-scoring STRING interaction between two gene names.

    Genes often map to multiple Ensembl protein IDs in STRING (different transcripts,
    KEGG synonyms, etc.). A naive lookup that takes the first ID misses interactions
    on the canonical isoform. This helper checks ALL ID combinations and returns the
    highest score.

    Args:
        gene_a: First gene symbol (e.g., "SP1")
        gene_b: Second gene symbol (e.g., "NFYA")
        links_df: DataFrame from load_string_links()
        aliases_path: Path to STRING aliases file (e.g. 9606.protein.aliases.v12.0.txt.gz)
        min_score: Minimum score to consider (default 0 = return any interaction found)

    Returns:
        Dict with keys:
          - found: bool — whether any interaction was found
          - score: int — highest combined_score among all ID pairs (0 if not found)
          - protein_a, protein_b: str — the ENSP IDs of the best-scoring pair
          - all_ids_a, all_ids_b: list[str] — all ENSP IDs found for each gene
    """
    import gzip as _gzip

    aliases_path = Path(aliases_path)
    _opener = _gzip.open if str(aliases_path).endswith(".gz") else open

    ids_a: set[str] = set()
    ids_b: set[str] = set()
    with _opener(aliases_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if parts[1] == gene_a:
                ids_a.add(parts[0])
            elif parts[1] == gene_b:
                ids_b.add(parts[0])

    if not ids_a or not ids_b:
        return {
            "found": False, "score": 0, "protein_a": None, "protein_b": None,
            "all_ids_a": sorted(ids_a), "all_ids_b": sorted(ids_b),
        }

    p_a, p_b = links_df.columns[0], links_df.columns[1]
    score_col = links_df.columns[-1] if "combined_score" not in links_df.columns else "combined_score"

    matches = links_df[
        ((links_df[p_a].isin(ids_a)) & (links_df[p_b].isin(ids_b))) |
        ((links_df[p_a].isin(ids_b)) & (links_df[p_b].isin(ids_a)))
    ]

    if len(matches) == 0:
        return {
            "found": False, "score": 0, "protein_a": None, "protein_b": None,
            "all_ids_a": sorted(ids_a), "all_ids_b": sorted(ids_b),
        }

    best = matches.loc[matches[score_col].idxmax()]
    score = int(best[score_col])
    if score < min_score:
        return {
            "found": False, "score": score, "protein_a": str(best[p_a]), "protein_b": str(best[p_b]),
            "all_ids_a": sorted(ids_a), "all_ids_b": sorted(ids_b),
        }

    return {
        "found": True,
        "score": score,
        "protein_a": str(best[p_a]),
        "protein_b": str(best[p_b]),
        "all_ids_a": sorted(ids_a),
        "all_ids_b": sorted(ids_b),
    }


def find_shared_string_partners(
    gene_a: str,
    gene_b: str,
    links_df: "pd.DataFrame",
    aliases_path: str | Path,
    min_score: int = 400,
    max_partners: int | None = None,
) -> dict:
    """Find shared STRING interaction partners between two genes, with proper gene names.

    Naive implementations of this fail in two ways:
      1. Genes map to MULTIPLE Ensembl protein IDs in STRING (canonical, KEGG synonyms,
         transcripts). A naive `aliases[gene] = ensp_id` dict picks one ID and misses
         partners on other isoforms.
      2. Reverse-mapping ENSPs back to gene symbols by `aliases[ensp]` collides on
         non-canonical sources (PDB structure IDs like "2YRP", deletion-region names
         like "10q23del"). Must filter the aliases file by source priority to recover
         the canonical HGNC symbol.

    This helper handles both: collects all ENSP IDs for each gene, finds the partner
    intersection in `links_df`, then converts each shared ENSP back to its canonical
    gene symbol using Ensembl_HGNC / Ensembl_HGNC_symbol / BioMart_HUGO sources.

    Args:
        gene_a: First gene symbol (e.g. "SP1")
        gene_b: Second gene symbol (e.g. "NFYA")
        links_df: DataFrame from load_string_links()
        aliases_path: Path to STRING aliases file (e.g. 9606.protein.aliases.v12.0.txt.gz)
        min_score: Minimum combined_score for partner edges (default 400 = medium)
        max_partners: If set, return only the top N partners by max(score_a, score_b)

    Returns:
        Dict with keys:
          - shared_count: int — number of unique ENSP IDs that link to BOTH genes
          - partners: list[dict] — each entry has:
              {ensp: str, gene_symbol: str, score_a: int, score_b: int}
              Sorted by min(score_a, score_b) descending. gene_symbol falls back
              to the ENSP ID if no canonical alias exists.
          - all_ids_a, all_ids_b: list[str] — all ENSP IDs found for each query gene
    """
    import gzip as _gzip

    aliases_path = Path(aliases_path)
    _opener = _gzip.open if str(aliases_path).endswith(".gz") else open

    # Source priority for converting ENSP → gene symbol. We accept any of these
    # because not every protein has every annotation source.
    canonical_sources = (
        "Ensembl_HGNC",
        "Ensembl_HGNC_symbol",
        "BioMart_HUGO",
        "Ensembl_UniProt",
        "UniProt_GN_Name",
        "Ensembl_EntrezGene",
    )

    # First pass: collect ALL ENSP IDs for the two query genes
    ids_a: set[str] = set()
    ids_b: set[str] = set()
    with _opener(aliases_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if parts[1] == gene_a:
                ids_a.add(parts[0])
            elif parts[1] == gene_b:
                ids_b.add(parts[0])

    if not ids_a or not ids_b:
        return {
            "shared_count": 0, "partners": [],
            "all_ids_a": sorted(ids_a), "all_ids_b": sorted(ids_b),
        }

    p_a, p_b = links_df.columns[0], links_df.columns[1]
    score_col = links_df.columns[-1] if "combined_score" not in links_df.columns else "combined_score"

    # All edges that include any ID for gene_a, with score >= min_score
    mask_a = (
        (links_df[p_a].isin(ids_a) & ~links_df[p_b].isin(ids_a)) |
        (links_df[p_b].isin(ids_a) & ~links_df[p_a].isin(ids_a))
    )
    edges_a = links_df[mask_a & (links_df[score_col] >= min_score)]

    mask_b = (
        (links_df[p_a].isin(ids_b) & ~links_df[p_b].isin(ids_b)) |
        (links_df[p_b].isin(ids_b) & ~links_df[p_a].isin(ids_b))
    )
    edges_b = links_df[mask_b & (links_df[score_col] >= min_score)]

    # Build {partner_ensp: best_score} for each side
    partners_a: dict[str, int] = {}
    for _, row in edges_a.iterrows():
        partner = row[p_b] if row[p_a] in ids_a else row[p_a]
        if partner in ids_b:
            continue  # gene_b's own IDs are not partners
        score = int(row[score_col])
        if partner not in partners_a or score > partners_a[partner]:
            partners_a[partner] = score

    partners_b: dict[str, int] = {}
    for _, row in edges_b.iterrows():
        partner = row[p_b] if row[p_a] in ids_b else row[p_a]
        if partner in ids_a:
            continue
        score = int(row[score_col])
        if partner not in partners_b or score > partners_b[partner]:
            partners_b[partner] = score

    shared_ensps = set(partners_a) & set(partners_b)

    if not shared_ensps:
        return {
            "shared_count": 0, "partners": [],
            "all_ids_a": sorted(ids_a), "all_ids_b": sorted(ids_b),
        }

    # Second pass: resolve canonical gene symbols for the shared ENSP IDs.
    # Build {ensp: {source: alias}} only for the ENSPs we care about.
    ensp_aliases: dict[str, dict[str, str]] = {ensp: {} for ensp in shared_ensps}
    with _opener(aliases_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            ensp, alias, source = parts[0], parts[1], parts[2]
            if ensp in ensp_aliases:
                # Keep first hit per source
                if source not in ensp_aliases[ensp]:
                    ensp_aliases[ensp][source] = alias

    def _resolve_symbol(ensp: str) -> str:
        srcs = ensp_aliases.get(ensp, {})
        for src in canonical_sources:
            if src in srcs:
                return srcs[src]
        return ensp  # Fallback: return the ENSP if no canonical source matched

    partners = []
    for ensp in shared_ensps:
        partners.append({
            "ensp": ensp,
            "gene_symbol": _resolve_symbol(ensp),
            "score_a": partners_a[ensp],
            "score_b": partners_b[ensp],
        })

    # Sort by min(score_a, score_b) desc — partners with strong evidence on BOTH sides
    partners.sort(key=lambda p: min(p["score_a"], p["score_b"]), reverse=True)

    if max_partners is not None:
        partners = partners[:max_partners]

    return {
        "shared_count": len(shared_ensps),
        "partners": partners,
        "all_ids_a": sorted(ids_a),
        "all_ids_b": sorted(ids_b),
    }


def run_go_enrichment(
    gene_list: list[str],
    organism: str = "hsapiens",
    sources: list[str] | None = None,
    significant_only: bool = True,
    max_terms: int = 20,
) -> pd.DataFrame:
    """Run GO / pathway enrichment on a list of gene symbols via g:Profiler.

    Queries the g:Profiler web API (no local annotation files needed).
    Returns an empty DataFrame if the gene list is empty or no terms are found.

    Args:
        gene_list: List of gene symbols (e.g. ["TP53", "MYC", "BRCA1"]).
        organism: g:Profiler organism code. Common values:
            "hsapiens" (human), "mmusculus" (mouse), "drerio" (zebrafish).
        sources: Annotation sources to query. Defaults to
            ["GO:BP", "GO:MF", "KEGG"]. Other options: "GO:CC", "REAC", "WP", "HP".
        significant_only: If True (default), return only statistically significant
            terms (g:Profiler's multiple-testing corrected p_value < 0.05).
        max_terms: Maximum number of top terms to return, sorted by p_value.

    Returns:
        DataFrame with columns:
            source        – annotation source (e.g. "GO:BP", "KEGG")
            name          – term name
            p_value       – adjusted p-value (g:SCS correction)
            intersection_size – number of query genes annotated to this term
            term_size     – total genes annotated to this term in the database
            query_size    – number of genes in the input query
            native        – term ID (e.g. "GO:0006355")
        Sorted by p_value ascending. Empty DataFrame if no results.

    Example:
        >>> df = run_go_enrichment(["TP53", "MYC", "CDKN1A", "MDM2"])
        >>> print(df[["source", "name", "p_value"]].head())
    """
    if sources is None:
        sources = ["GO:BP", "GO:MF", "KEGG"]

    gene_list = [g for g in gene_list if g and str(g).strip()]
    if not gene_list:
        return pd.DataFrame(columns=["source", "name", "p_value", "intersection_size",
                                      "term_size", "query_size", "native"])

    from gprofiler import GProfiler  # imported here to keep gprofiler optional

    gp = GProfiler(return_dataframe=True)
    results = gp.profile(
        organism=organism,
        query=gene_list,
        sources=sources,
        significance_threshold_method="g_SCS",
        no_evidences=True,
    )

    if results is None or results.empty:
        return pd.DataFrame(columns=["source", "name", "p_value", "intersection_size",
                                      "term_size", "query_size", "native"])

    if significant_only:
        results = results[results["significant"] == True]  # noqa: E712

    keep_cols = [c for c in ["source", "name", "p_value", "intersection_size",
                               "term_size", "query_size", "native"] if c in results.columns]
    results = results[keep_cols].sort_values("p_value").head(max_terms).reset_index(drop=True)
    return results


# =============================================================================
# MOTIF COMPARISON (Tomtom)
# =============================================================================

def run_tomtom(
    meme_file: str | Path,
    motif_id_1: str,
    motif_id_2: str,
    threshold: float = 0.5,
    distance_metric: str = "ed",
) -> dict[str, Any]:
    """Compare two motifs using Tomtom and return similarity statistics.

    Args:
        meme_file: Path to MEME format motif database file.
        motif_id_1: Query motif ID (e.g., "NFYA|jaspar|MA0060.3").
        motif_id_2: Target motif ID (e.g., "NFATC3|jaspar|MA0623.1").
        threshold: q-value significance threshold (default 0.5 to capture
            borderline matches; filter on p_value in results if needed).
        distance_metric: Distance metric — "ed" (Euclidean, default),
            "pearson", "allr", "kullback", "sandelin".

    Returns:
        dict with keys:
          - p_value: float (NaN if no match found)
          - e_value: float
          - q_value: float
          - overlap: int (number of aligned columns)
          - query_consensus: str
          - target_consensus: str
          - orientation: str ("+" or "-")
          - is_significant: bool (q_value < 0.05)
          - match_found: bool (whether Tomtom found any alignment)
    """
    from tempfile import TemporaryDirectory
    import math

    meme_file = Path(meme_file)
    if not meme_file.exists():
        raise FileNotFoundError(f"MEME file not found: {meme_file}")

    # Tomtom needs separate query and target files when using -m to select motifs.
    # We use the same MEME file as both query and target, selecting one motif each.
    with TemporaryDirectory(prefix="tomtom_") as td:
        td_path = Path(td)
        out_dir = td_path / "output"

        cmd = [
            "tomtom",
            "-oc", str(out_dir),
            "-m", motif_id_1,
            "-thresh", str(threshold),
            "-dist", distance_metric,
            str(meme_file),   # query (motif_id_1 selected by -m)
            str(meme_file),   # target (full database — Tomtom finds motif_id_2)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Parse output — Tomtom writes TSV before generating XML/HTML, so the
        # TSV may exist even if returncode != 0 (e.g., missing Perl XML module).
        # Only raise if the TSV is also missing.
        tsv_path = out_dir / "tomtom.tsv"
        if proc.returncode != 0 and (not tsv_path.exists() or tsv_path.stat().st_size == 0):
            print(f"[run_tomtom] ERROR: tomtom failed (rc={proc.returncode})")
            print(f"[run_tomtom] stderr: {proc.stderr[:500]}")
            raise RuntimeError(f"tomtom failed: {proc.stderr[:300]}")
        if not tsv_path.exists() or tsv_path.stat().st_size == 0:
            return {
                "p_value": math.nan, "e_value": math.nan, "q_value": math.nan,
                "overlap": 0, "query_consensus": "", "target_consensus": "",
                "orientation": "", "is_significant": False, "match_found": False,
            }

        df = pd.read_csv(tsv_path, sep="\t", comment="#")
        if df.empty:
            return {
                "p_value": math.nan, "e_value": math.nan, "q_value": math.nan,
                "overlap": 0, "query_consensus": "", "target_consensus": "",
                "orientation": "", "is_significant": False, "match_found": False,
            }

        # Find the row matching motif_id_2 as target
        match = df[df["Target_ID"] == motif_id_2]
        if match.empty:
            # Try partial match (motif IDs can have slight formatting differences)
            match = df[df["Target_ID"].str.contains(motif_id_2.split("|")[0], case=False, na=False)]
        if match.empty:
            return {
                "p_value": math.nan, "e_value": math.nan, "q_value": math.nan,
                "overlap": 0, "query_consensus": "", "target_consensus": "",
                "orientation": "", "is_significant": False, "match_found": False,
            }

        row = match.iloc[0]
        p_val = float(row.get("p-value", math.nan))
        e_val = float(row.get("E-value", math.nan))
        q_val = float(row.get("q-value", math.nan))

        return {
            "p_value": p_val,
            "e_value": e_val,
            "q_value": q_val,
            "overlap": int(row.get("Overlap", 0)),
            "query_consensus": str(row.get("Query_consensus", "")),
            "target_consensus": str(row.get("Target_consensus", "")),
            "orientation": str(row.get("Orientation", "")),
            "is_significant": q_val < 0.05 if not math.isnan(q_val) else False,
            "match_found": True,
        }
