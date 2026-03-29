"""Reusable bioinformatics parsing helpers for generated verification code.

These helpers exist to prevent generated scripts from re-implementing fragile parsing
logic for common file formats and tool outputs.
"""

from __future__ import annotations

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
    """Read a path or return a defensive copy of a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
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


def parse_fimo_tsv(
    path: str | Path,
    p_value_threshold: float | None = None,
    motif_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Parse FIMO TSV output robustly.

    Handles the common header form `#pattern name` and preserves only tabular lines.
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
            rows.append(line)

    if not rows:
        raise ValueError("FIMO TSV is empty after removing comment lines")

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

    # Drop trailing FIMO comment rows — FIMO appends lines like
    # "# fimo was run with..." that start with a single '#' and get
    # parsed as data rows with NaN in sequence_name and other columns.
    if "sequence_name" in df.columns:
        df = df.dropna(subset=["sequence_name"]).copy()

    required = {"motif_id", "sequence_name", "start", "stop", "p-value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected FIMO columns: {sorted(missing)}")

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


def load_rnaseq_with_gene_id(
    path: str | Path | pd.DataFrame,
    gene_id_candidates: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load RNA-seq table and detect the gene-id column by name."""
    df = _read_table_like(path, sep="\t")
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


def load_gencode_genes(
    path: str | Path | pd.DataFrame,
    protein_coding_only: bool = True,
) -> pd.DataFrame:
    """Load GENCODE gene records and normalize IDs to version-free ENSG IDs."""
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
    if isinstance(path, pd.DataFrame):
        df = path.copy()
        if list(df.columns) == list(range(len(columns))):
            df.columns = columns
    else:
        df = _read_table_like(path, sep="\t", comment="#", header=None, names=columns)
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
    peaks_bed: str | Path,
    tss_bed: str | Path,
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
            subprocess.run(
                ["bedtools", "sort", "-i", peaks_bed],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        with open(sorted_tss, "w", encoding="utf-8") as out:
            subprocess.run(
                ["bedtools", "sort", "-i", tss_bed],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        with open(closest_out, "w", encoding="utf-8") as out:
            subprocess.run(
                ["bedtools", "closest", "-a", str(sorted_peaks), "-b", str(sorted_tss), "-d", "-t", tie_mode],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

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

    # Cleanup temp TSS dir if we created one
    if _tmp_tss_dir is not None:
        import shutil as _shutil
        try:
            _shutil.rmtree(_tmp_tss_dir, ignore_errors=True)
        except Exception:
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
            subprocess.run(
                ["bedtools", "intersect", "-a", peaks_bed, "-b", chromhmm_bed, "-wa", "-wb"],
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

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

    # Clean up any temp files created from DataFrame inputs
    for _p in _tmp_files:
        try:
            import os as _os
            _os.unlink(_p)
        except OSError:
            pass

    if not keep_one_per_peak:
        return df

    # --- Deduplicate to exactly one row per input peak ---
    # Load input peaks to get their count and coordinates for re-indexing.
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

        with TemporaryDirectory() as td:
            td_path = Path(td)
            scan_bed = td_path / "scan_regions.bed"
            fasta_out = td_path / "peaks.fa"
            fimo_out_dir = td_path / "fimo_out"

            # Step 1: optionally window to summit; always ensure unique peak names.
            # ENCODE peak files commonly use '.' as the name column (col 4), which makes
            # every peak_id '.' in the FIMO output and breaks motif-to-peak mapping.
            # Whenever col 4 is missing, uniform, or contains only '.' we assign
            # sequential IDs (peak_00000, peak_00001, ...) so peak_id is always unique.
            peaks_df = pd.read_csv(peaks_bed, sep="\t", header=None)

            if peaks_df.shape[1] >= 4:
                names = peaks_df.iloc[:, 3].astype(str)
            else:
                names = pd.Series(["."] * len(peaks_df))

            if names.nunique() <= 1:
                # Uniform or missing names — generate sequential IDs
                names = pd.Series([f"peak_{i:05d}" for i in range(len(peaks_df))])
                peaks_df = peaks_df.copy()
                if peaks_df.shape[1] >= 4:
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
            with open(fasta_out, "w") as fasta_fh:
                subprocess.run(
                    ["bedtools", "getfasta", "-fi", str(genome_fasta), "-bed", str(scan_bed),
                     "-fo", "/dev/stdout", "-name"],
                    stdout=fasta_fh,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                )

            # Step 3: FIMO — --no-pgc prevents sequence_name from being parsed as
            # genomic coordinates (which would return chromosome names, not peak IDs)
            subprocess.run(
                [
                    "fimo",
                    "--no-pgc",
                    "--motif", motif_id,
                    "--thresh", str(p_value_threshold),
                    "--oc", str(fimo_out_dir),
                    str(meme_file),
                    str(fasta_out),
                ],
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # Step 4: parse output
            fimo_tsv = fimo_out_dir / "fimo.tsv"
            if not fimo_tsv.exists() or fimo_tsv.stat().st_size == 0:
                return pd.DataFrame(columns=["motif_id", "sequence_name", "start", "stop",
                                              "p-value", "peak_id"])

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
        capture_output=True, text=True, check=True,
    )
    n_query = int(result_total.stdout.strip().split()[0])

    # Count query peaks that have ≥1 overlap with subject (-u: unique, count each query once)
    result_overlap = subprocess.run(
        ["bedtools", "intersect", "-a", _intersect_a, "-b", subject_bed, "-u"],
        capture_output=True, text=True, check=True,
    )
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
                capture_output=True, text=True, check=True,
            )
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
                capture_output=True, text=True, check=True,
            )
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
                capture_output=True, text=True, check=True,
            )
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
    # Peek at first non-empty, non-comment line to detect separator
    header_line: str | None = None
    with open(path, encoding="utf-8") as fh:
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
