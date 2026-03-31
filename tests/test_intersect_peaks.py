"""Tests for intersect_peaks modes (flag, count, wa-wb)."""

from __future__ import annotations

import subprocess

import pandas as pd
import pytest

from src.utils.bioio import intersect_peaks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_peaks_df(rows: list[tuple]) -> pd.DataFrame:
    """Build a minimal 4-column peak DataFrame."""
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "name"])


# ---------------------------------------------------------------------------
# mode="flag"
# ---------------------------------------------------------------------------

class TestIntersectPeaksFlag:
    def test_overlapping_peak_gets_true(self, monkeypatch, tmp_path):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1"), ("chr1", 500, 600, "p2")])
        df_b = _make_peaks_df([("chr1", 150, 250, "q1")])

        # bedtools intersect -c returns original A cols + count
        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            assert "-c" in cmd
            stdout = "chr1\t100\t200\tp1\t1\nchr1\t500\t600\tp2\t0\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="flag")

        assert list(out["overlaps_b"]) == [True, False]
        assert "chrom" in out.columns
        assert len(out) == 2

    def test_no_overlaps_all_false(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1")])
        df_b = _make_peaks_df([("chr2", 100, 200, "q1")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            stdout = "chr1\t100\t200\tp1\t0\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="flag")

        assert list(out["overlaps_b"]) == [False]

    def test_flag_column_is_bool(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1")])
        df_b = _make_peaks_df([("chr1", 150, 250, "q1")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            stdout = "chr1\t100\t200\tp1\t3\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="flag")

        assert out["overlaps_b"].dtype == bool


# ---------------------------------------------------------------------------
# mode="count"
# ---------------------------------------------------------------------------

class TestIntersectPeaksCount:
    def test_overlap_count_column_added(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1"), ("chr1", 500, 600, "p2")])
        df_b = _make_peaks_df([("chr1", 150, 250, "q1"), ("chr1", 160, 210, "q2")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            stdout = "chr1\t100\t200\tp1\t2\nchr1\t500\t600\tp2\t0\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="count")

        assert list(out["overlap_count"]) == [2, 0]
        assert "chrom" in out.columns

    def test_count_column_is_integer(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1")])
        df_b = _make_peaks_df([("chr1", 150, 250, "q1")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            stdout = "chr1\t100\t200\tp1\t5\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="count")

        assert out["overlap_count"].iloc[0] == 5


# ---------------------------------------------------------------------------
# mode="wa-wb"
# ---------------------------------------------------------------------------

class TestIntersectPeaksWaWb:
    def test_joined_columns_prefixed(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1")])
        df_b = _make_peaks_df([("chr1", 150, 250, "q1")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            assert "-wa" in cmd and "-wb" in cmd
            stdout = "chr1\t100\t200\tp1\tchr1\t150\t250\tq1\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="wa-wb")

        assert any(c.startswith("a_") for c in out.columns)
        assert any(c.startswith("b_") for c in out.columns)
        assert len(out) == 1

    def test_empty_overlap_returns_empty_df(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 200, "p1")])
        df_b = _make_peaks_df([("chr2", 100, 200, "q1")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="wa-wb")

        assert len(out) == 0
        assert any(c.startswith("a_") for c in out.columns)

    def test_multiple_overlaps_per_a_peak(self, monkeypatch):
        df_a = _make_peaks_df([("chr1", 100, 300, "p1")])
        df_b = _make_peaks_df([("chr1", 110, 150, "q1"), ("chr1", 200, 250, "q2")])

        def fake_run(cmd, capture_output=None, text=None, check=None, **kw):
            stdout = (
                "chr1\t100\t300\tp1\tchr1\t110\t150\tq1\n"
                "chr1\t100\t300\tp1\tchr1\t200\t250\tq2\n"
            )
            return subprocess.CompletedProcess(cmd, 0, stdout, "")

        monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

        out = intersect_peaks(df_a, df_b, mode="wa-wb")

        assert len(out) == 2


# ---------------------------------------------------------------------------
# invalid mode
# ---------------------------------------------------------------------------

def test_invalid_mode_raises():
    df = _make_peaks_df([("chr1", 100, 200, "p1")])
    with pytest.raises(ValueError, match="mode must be one of"):
        intersect_peaks(df, df, mode="invalid")
