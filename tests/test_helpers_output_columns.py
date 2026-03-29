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
