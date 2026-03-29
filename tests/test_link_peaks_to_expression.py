"""Tests for link_peaks_to_expression helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src.utils.bioio import link_peaks_to_expression


def test_link_peaks_to_expression_returns_merged_df_and_expr_col(monkeypatch, tmp_path: Path):
    """link_peaks_to_expression returns (merged_df, expr_col) with correct columns."""
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("chr1\t100\t200\tpeak1\n")

    rnaseq = tmp_path / "rnaseq.tsv"
    rnaseq.write_text("gene_id\tTPM\nENSG0001.5\t12.5\n")

    # Minimal GTF (gene record only)
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\tHAVANA\tgene\t1\t500\t.\t+\t.\t'
        'gene_id "ENSG0001.5"; gene_name "GENE1"; gene_type "protein_coding";\n'
    )

    # Fake subprocess.run:
    # - bedtools sort → echo back input
    # - bedtools closest → single hit: peak1 near ENSG0001
    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None, **kw):
        if cmd[:2] == ["bedtools", "sort"]:
            src = Path(cmd[-1])
            stdout.write(src.read_text())
        elif cmd[:2] == ["bedtools", "closest"]:
            stdout.write("chr1\t100\t200\tpeak1\tchr1\t1\t500\tENSG0001.5\t0\t+\t0\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    merged_df, expr_col = link_peaks_to_expression(peaks, rnaseq, gtf, max_distance=50000)

    print(f"\nlink_peaks_to_expression columns: {list(merged_df.columns)}")
    print(merged_df.to_string())
    print(f"expr_col: {expr_col}")

    assert expr_col == "TPM"
    assert "peak_name" in merged_df.columns
    assert "gene_id" in merged_df.columns
    assert "gene_id_clean" in merged_df.columns
    assert "distance" in merged_df.columns
    assert expr_col in merged_df.columns
    assert merged_df.loc[0, "peak_name"] == "peak1"
    assert merged_df.loc[0, "gene_id_clean"] == "ENSG0001"
    assert merged_df.loc[0, expr_col] == pytest.approx(12.5)
