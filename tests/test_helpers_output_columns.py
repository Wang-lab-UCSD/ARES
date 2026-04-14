"""Tests that verify the exact output column names and shapes of all helper functions.

Run with: conda run -n pipeline python -m pytest tests/test_helpers_output_columns.py -v -s

The -s flag prints the column outputs so we can update the API docs in coding.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.bioio import (
    annotate_peaks_with_chromhmm,
    count_overlapping_peaks,
    extract_bigwig_signals,
    matched_bin,
    matched_pair,
    parse_fimo_tsv,
    read_narrowpeak,
    run_fimo_on_peaks,
)


# ---------------------------------------------------------------------------
# annotate_peaks_with_chromhmm — verify exact output columns
# ---------------------------------------------------------------------------

def test_annotate_peaks_with_chromhmm_default_column_names(monkeypatch, tmp_path: Path):
    """Default peak_col_names should produce peak_chrom/peak_start/peak_end/peak_name."""
    peaks = tmp_path / "peaks.bed"
    chrom = tmp_path / "chromhmm.bed"
    peaks.write_text("chr1\t10\t20\tpeak1\n")
    chrom.write_text("chr1\t0\t100\t1_TssA\n")

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
        if cmd[:2] == ["bedtools", "intersect"]:
            stdout.write("chr1\t10\t20\tpeak1\tchr1\t0\t100\t1_TssA\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    out = annotate_peaks_with_chromhmm(
        peaks, chrom,
        promoter_states=["1_TssA"],
        enhancer_states=["9_EnhA1"],
    )

    print(f"\nannotate_peaks_with_chromhmm (default) columns: {list(out.columns)}")
    print(out.to_string())

    # Default peak column names are plain (same as BED input)
    assert "chrom" in out.columns
    assert "start" in out.columns
    assert "end" in out.columns
    assert "name" in out.columns
    # ChromHMM columns are b_0 → chromhmm_0 etc.
    assert "chromhmm_0" in out.columns
    assert "chromhmm_1" in out.columns
    assert "chromhmm_2" in out.columns
    assert "chromhmm_3" in out.columns
    # state = last chromhmm column
    assert "state" in out.columns
    assert out.loc[0, "state"] == "1_TssA"
    # Boolean flags
    assert "is_promoter_state" in out.columns
    assert "is_enhancer_state" in out.columns
    assert bool(out.loc[0, "is_promoter_state"]) is True
    assert bool(out.loc[0, "is_enhancer_state"]) is False


def test_annotate_peaks_with_chromhmm_custom_column_names(monkeypatch, tmp_path: Path):
    """When peak_col_names=['chrom','start','end','name'], columns use plain names."""
    peaks = tmp_path / "peaks.bed"
    chrom = tmp_path / "chromhmm.bed"
    peaks.write_text("chr1\t10\t20\tpeak1\n")
    chrom.write_text("chr1\t0\t100\t1_TssA\n")

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
        if cmd[:2] == ["bedtools", "intersect"]:
            stdout.write("chr1\t10\t20\tpeak1\tchr1\t0\t100\t1_TssA\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    out = annotate_peaks_with_chromhmm(
        peaks, chrom,
        peak_col_names=["chrom", "start", "end", "name"],
        promoter_states=["1_TssA"],
    )

    print(f"\nannotate_peaks_with_chromhmm (custom peak_col_names) columns: {list(out.columns)}")

    assert "chrom" in out.columns
    assert "start" in out.columns
    assert "end" in out.columns
    assert "name" in out.columns
    assert "state" in out.columns
    # is_enhancer_state always present even though enhancer_states not set
    assert "is_enhancer_state" in out.columns
    assert bool(out.loc[0, "is_enhancer_state"]) is False


# ---------------------------------------------------------------------------
# count_overlapping_peaks — verify return dict keys
# ---------------------------------------------------------------------------

def test_count_overlapping_peaks_return_keys(monkeypatch, tmp_path: Path):
    """Return dict must have n_query, n_overlapping, n_nonoverlapping, fraction."""
    query = tmp_path / "query.bed"
    subject = tmp_path / "subject.bed"
    query.write_text("chr1\t10\t20\tpeak1\nchr1\t50\t60\tpeak2\n")
    subject.write_text("chr1\t15\t25\tref1\n")

    def fake_run(cmd, capture_output=None, stdout=None, stderr=None, text=None, check=None):
        if cmd[0] == "wc":
            result = subprocess.CompletedProcess(cmd, 0, "2 /some/path\n", "")
            return result
        elif cmd[:2] == ["bedtools", "intersect"]:
            # only peak1 overlaps
            result = subprocess.CompletedProcess(cmd, 0, "chr1\t10\t20\tpeak1\n", "")
            return result
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    result = count_overlapping_peaks(query, subject)

    print(f"\ncount_overlapping_peaks return: {result}")

    assert set(result.keys()) == {"n_query", "n_overlapping", "n_nonoverlapping", "fraction"}
    assert result["n_query"] == 2
    assert result["n_overlapping"] == 1
    assert result["n_nonoverlapping"] == 1
    assert result["fraction"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# extract_bigwig_signals — verify return type and alignment
# ---------------------------------------------------------------------------

def test_extract_bigwig_signals_returns_series_aligned_to_df_index(monkeypatch):
    """Returns pd.Series with same index as input df, NaN for missing data."""

    class FakeBW:
        def chroms(self):
            return {"chr1": 1000000, "chr2": 1000000}

        def stats(self, chrom, start, end, type="mean"):
            if chrom == "chr1":
                return [1.5]
            return [None]

        def close(self):
            pass

    class FakePyBigWig:
        @staticmethod
        def open(path):
            return FakeBW()

    import sys
    import types

    fake_module = types.ModuleType("pyBigWig")
    fake_module.open = FakePyBigWig.open
    monkeypatch.setitem(sys.modules, "pyBigWig", fake_module)

    intervals = pd.DataFrame({
        "chrom": ["chr1", "chr2", "chr1"],
        "start": [100, 200, 300],
        "end": [200, 300, 400],
    }, index=[10, 20, 30])  # non-default index to verify alignment

    signals = extract_bigwig_signals(intervals, "/fake/file.bw")

    print(f"\nextract_bigwig_signals return type: {type(signals)}")
    print(f"  index: {list(signals.index)}")
    print(f"  values: {list(signals.values)}")

    assert isinstance(signals, pd.Series)
    assert list(signals.index) == [10, 20, 30]
    assert signals[10] == pytest.approx(1.5)
    assert float("nan") != float("nan")  # check NaN
    import math
    assert math.isnan(signals[20])
    assert signals[30] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# matched_bin — verify added 'bin' column, return types
# ---------------------------------------------------------------------------

def test_matched_bin_adds_bin_column_and_returns_copies():
    """matched_bin adds a 'bin' (int) column to both FG and BG copies."""
    import numpy as np

    fg = pd.DataFrame({"dnase": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    bg = pd.DataFrame({"dnase": [0.5, 2.5, 4.5, 6.5, 9.0]})  # 9.0 is out of FG range

    fg_out, bg_out = matched_bin(fg, bg, signal_col="dnase", n_bins=4)

    print(f"\nmatched_bin FG columns: {list(fg_out.columns)}")
    print(f"matched_bin BG columns: {list(bg_out.columns)}")
    print(f"FG bins:\n{fg_out[['dnase','bin']]}")
    print(f"BG bins:\n{bg_out[['dnase','bin']]}")

    # Both dfs get 'bin' column
    assert "bin" in fg_out.columns
    assert "bin" in bg_out.columns

    # FG bins are all assigned (no NaN)
    assert fg_out["bin"].notna().all()

    # BG row with value outside FG range (9.0 > max FG) gets NaN bin
    assert bg_out.loc[bg_out["dnase"] == 9.0, "bin"].isna().all()

    # Original dfs are not modified (returns copies)
    assert "bin" not in fg.columns
    assert "bin" not in bg.columns


# ---------------------------------------------------------------------------
# matched_pair — equal-size matched FG/BG (canonical wrapper around matched_bin)
# ---------------------------------------------------------------------------

def test_matched_pair_returns_equal_size_groups():
    """matched_pair returns FG and BG with equal total length and equal per-bin counts.

    This is the bug fix for the recurring 'subsample BG only, leave FG full'
    pattern. matched_pair must guarantee balanced groups so callers cannot
    accidentally compare unbalanced 'matched' sets.
    """
    # FG is small, BG is large — caller would normally subsample BG to FG size
    # but forget to also restrict FG to the bins where BG actually has rows.
    fg = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                       "value": [10, 20, 30, 40, 50, 60, 70, 80]})
    bg = pd.DataFrame({"signal": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                                   8.5, 9.5, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
                       "value": list(range(100, 116))})

    fg_m, bg_m = matched_pair(fg, bg, signal_col="signal", n_bins=4, random_state=0)

    # Equal total length — the cardinal invariant
    assert len(fg_m) == len(bg_m), (
        f"matched_pair returned unequal sizes: FG={len(fg_m)} BG={len(bg_m)}"
    )

    # Both have a 'bin' column
    assert "bin" in fg_m.columns
    assert "bin" in bg_m.columns

    # Per-bin counts must be equal
    fg_counts = fg_m["bin"].value_counts().sort_index()
    bg_counts = bg_m["bin"].value_counts().sort_index()
    assert fg_counts.equals(bg_counts), (
        f"Per-bin counts differ: FG={fg_counts.to_dict()} BG={bg_counts.to_dict()}"
    )

    # Index is reset (0..N-1) — protects against downstream index-alignment bugs
    assert list(fg_m.index) == list(range(len(fg_m)))
    assert list(bg_m.index) == list(range(len(bg_m)))

    # Original DataFrames are not mutated
    assert "bin" not in fg.columns
    assert "bin" not in bg.columns


def test_matched_pair_handles_bg_bin_with_no_overlap():
    """When a BG row falls outside FG range, matched_pair drops it cleanly.

    The dropped row must not appear in the output, and the per-bin counts
    must still be equal between FG and BG.
    """
    fg = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                       "id": list("abcdefgh")})
    # Several BG rows are far outside FG range and will get bin=NaN
    bg = pd.DataFrame({"signal": [1.5, 2.5, 3.5, 4.5, 100.0, 200.0, 300.0],
                       "id": list("ijklmno")})

    fg_m, bg_m = matched_pair(fg, bg, signal_col="signal", n_bins=4, random_state=0)

    assert len(fg_m) == len(bg_m)
    assert fg_m["bin"].notna().all()
    assert bg_m["bin"].notna().all()
    # Out-of-range BG ids must NOT appear in matched output
    assert not bg_m["id"].isin(["m", "n", "o"]).any()


# ---------------------------------------------------------------------------
# parse_fimo_tsv — verify NaN handling in sequence_name
# ---------------------------------------------------------------------------

def test_parse_fimo_tsv_drops_nan_sequence_name_rows(tmp_path: Path):
    """Trailing FIMO comment lines (single #) produce NaN in sequence_name — must be dropped."""
    fimo_tsv = tmp_path / "fimo.tsv"
    fimo_tsv.write_text(
        "## comment\n"
        "#pattern name\tsequence name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched sequence\n"
        "MA0001.1\tpeak1::chr1:10-30\t3\t8\t+\t10.0\t1e-05\t0.1\tACGT\n"
        "#\n"  # trailing single-# line that becomes NaN row
    )

    out = parse_fimo_tsv(fimo_tsv)

    print(f"\nparse_fimo_tsv columns: {list(out.columns)}")
    print(out.to_string())

    assert len(out) == 1
    assert out["sequence_name"].notna().all()
    assert "peak_id" in out.columns


def test_run_fimo_on_peaks_replaces_non_unique_names_with_sequential_ids(
    monkeypatch, tmp_path: Path
):
    """REGRESSION (SP1/NFYA K562 iter6 bug): when col 4 of the input BED has
    non-unique values (e.g. a caller mistakenly wrote the narrowPeak `peak`
    column = summit offset into col 4), the helper must replace them with
    sequential peak_NNNNN IDs before handing off to bedtools getfasta /
    FIMO. Otherwise multiple genomic peaks collapse into one peak_id and
    motif-to-peak mapping silently breaks.

    We fake subprocess.run so we don't need real bedtools/FIMO — the test
    captures the BED file that would have been handed to bedtools getfasta
    and asserts its name column is fully unique.
    """
    # BED with 5 rows but only 2 distinct values in col 4 (summit offsets 52, 142)
    peaks_bed = tmp_path / "peaks.bed"
    peaks_bed.write_text(
        "chr1\t100\t300\t52\n"
        "chr1\t500\t700\t142\n"
        "chr2\t100\t300\t52\n"
        "chr3\t500\t700\t142\n"
        "chr4\t100\t300\t52\n"
    )
    genome_fa = tmp_path / "genome.fa"
    genome_fa.write_text(">chr1\n" + "A" * 1000 + "\n")
    meme_file = tmp_path / "motif.meme"
    meme_file.write_text("MEME version 4\n")

    captured_names: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["bedtools", "getfasta"]:
            # Parse the -bed arg, read it, capture col 4 (the name column).
            bed_idx = cmd.index("-bed")
            fasta_idx = cmd.index("-fo")
            bed_path = Path(cmd[bed_idx + 1])
            fasta_path = Path(cmd[fasta_idx + 1])
            lines = bed_path.read_text().strip().split("\n")
            cols_per_row = [line.split("\t") for line in lines]
            captured_names.append([row[3] for row in cols_per_row])
            # Write a fake FASTA that bedtools -name would produce
            fasta_path.write_text(
                "\n".join(
                    f">{row[3]}::{row[0]}:{row[1]}-{row[2]}\nACGT"
                    for row in cols_per_row
                )
                + "\n"
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:1] == ["fimo"]:
            # Write a minimal empty fimo.tsv in --oc dir
            oc_idx = cmd.index("--oc")
            out_dir = Path(cmd[oc_idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fimo.tsv").write_text(
                "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\t"
                "score\tp-value\tq-value\tmatched_sequence\n"
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    run_fimo_on_peaks(
        peaks_bed=peaks_bed,
        genome_fasta=genome_fa,
        meme_file=meme_file,
        motif_id="MA0001.1",
    )

    # The helper must have rewritten col 4 to sequential IDs before writing
    # scan_regions.bed for bedtools getfasta.
    assert len(captured_names) == 1, "bedtools getfasta should have been called exactly once"
    names = captured_names[0]
    assert len(names) == 5
    assert len(set(names)) == 5, (
        f"Non-unique names in scan_regions.bed — helper did NOT replace "
        f"collided names. Got: {names}"
    )
    # Sequential IDs follow the peak_NNNNN pattern
    assert all(n.startswith("peak_") for n in names), (
        f"Expected peak_NNNNN sequential IDs, got: {names}"
    )


def test_run_fimo_on_peaks_preserves_already_unique_names(
    monkeypatch, tmp_path: Path
):
    """Sanity check: when col 4 is already fully unique, the helper must
    preserve the caller's names (not overwrite them with peak_NNNNN).
    """
    peaks_bed = tmp_path / "peaks.bed"
    peaks_bed.write_text(
        "chr1\t100\t300\tATF6_peak_001\n"
        "chr1\t500\t700\tATF6_peak_002\n"
        "chr2\t100\t300\tATF6_peak_003\n"
    )
    genome_fa = tmp_path / "genome.fa"
    genome_fa.write_text(">chr1\n" + "A" * 1000 + "\n")
    meme_file = tmp_path / "motif.meme"
    meme_file.write_text("MEME version 4\n")

    captured_names: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["bedtools", "getfasta"]:
            bed_idx = cmd.index("-bed")
            fasta_idx = cmd.index("-fo")
            bed_path = Path(cmd[bed_idx + 1])
            fasta_path = Path(cmd[fasta_idx + 1])
            lines = bed_path.read_text().strip().split("\n")
            cols_per_row = [line.split("\t") for line in lines]
            captured_names.append([row[3] for row in cols_per_row])
            fasta_path.write_text(
                "\n".join(
                    f">{row[3]}::{row[0]}:{row[1]}-{row[2]}\nACGT"
                    for row in cols_per_row
                )
                + "\n"
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:1] == ["fimo"]:
            oc_idx = cmd.index("--oc")
            out_dir = Path(cmd[oc_idx + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fimo.tsv").write_text(
                "motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\t"
                "score\tp-value\tq-value\tmatched_sequence\n"
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    run_fimo_on_peaks(
        peaks_bed=peaks_bed,
        genome_fasta=genome_fa,
        meme_file=meme_file,
        motif_id="MA0001.1",
    )

    assert len(captured_names) == 1
    names = captured_names[0]
    assert names == ["ATF6_peak_001", "ATF6_peak_002", "ATF6_peak_003"]


# ---------------------------------------------------------------------------
# read_narrowpeak — verify output columns
# ---------------------------------------------------------------------------

def test_read_narrowpeak_output_columns():
    """Output has standard narrowPeak columns plus peak_offset and summit."""
    df = pd.DataFrame([
        ["chr1", 100, 150, "peak1", 1000, ".", 5.0, 10.0, 20.0, 12],
    ])

    out = read_narrowpeak(df)

    print(f"\nread_narrowpeak columns: {list(out.columns)}")

    # 'name' is dropped (always '.' in ENCODE files, trap for .merge(on='name'))
    # 'summit_offset' is added as an alias for 'peak_offset'
    expected_cols = ["chrom", "start", "end", "score", "strand",
                     "signal_value", "p_value", "q_value", "peak",
                     "peak_offset", "summit", "summit_offset"]
    assert list(out.columns) == expected_cols
