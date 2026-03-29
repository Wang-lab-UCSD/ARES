"""Tests for reusable bioinformatics parsing helpers."""

from pathlib import Path

import pandas as pd
import subprocess

from src.utils.bioio import (
    annotate_peaks_with_chromhmm,
    create_tss_bed_from_gencode,
    load_gencode_genes,
    load_rnaseq_expression,
    load_rnaseq_with_gene_id,
    merge_rnaseq_with_nearest_genes,
    parse_bedtools_closest,
    parse_chromhmm_intersect,
    parse_fimo_tsv,
    parse_peak_id_from_fasta_header,
    read_narrowpeak,
    run_bedtools_closest_to_tss,
)


def test_read_narrowpeak_computes_summit_and_fallback():
    df = pd.DataFrame([
        ["chr1", 100, 150, "peak1", 1000, ".", 5.0, 10.0, 20.0, 12],
        ["chr1", 200, 260, "peak2", 900, ".", 4.0, 9.0, 18.0, -1],
    ])

    out = read_narrowpeak(df)

    assert list(out["summit"]) == [112, 230]
    assert list(out["peak_offset"]) == [12, 30]


def test_parse_peak_id_from_fasta_header():
    assert parse_peak_id_from_fasta_header("peak_001::chr1:10-40") == "peak_001"
    assert parse_peak_id_from_fasta_header("peak_002") == "peak_002"


def test_parse_fimo_tsv_handles_hash_header_and_filters(tmp_path: Path):
    fimo_tsv = tmp_path / "fimo.tsv"
    fimo_tsv.write_text(
        "## comment\n"
        "#pattern name\tsequence name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched sequence\n"
        "MA0001.1\tpeak1::chr1:10-30\t3\t8\t+\t10.0\t1e-05\t0.1\tACGT\n"
        "MA0002.1\tpeak2\t5\t9\t-\t8.0\t2e-03\t0.2\tTGCA\n"
    )

    out = parse_fimo_tsv(fimo_tsv, p_value_threshold=1e-4, motif_ids=["MA0001.1"])

    assert list(out["motif_id"]) == ["MA0001.1"]
    assert list(out["peak_id"]) == ["peak1"]


def test_load_rnaseq_with_gene_id_detects_named_column():
    df = pd.DataFrame({
        "Geneid": ["ENSG1", "ENSG2"],
        "TPM": [1.0, 2.0],
    })

    out, gene_col = load_rnaseq_with_gene_id(df)

    assert gene_col == "Geneid"
    assert out.equals(df)


def test_load_rnaseq_expression_adds_clean_id_and_detects_expr_col():
    df = pd.DataFrame({
        "gene_id": ["ENSG1.5", "ENSG2.1"],
        "TPM": ["1.0", "2.5"],
    })

    out, gene_col, clean_col, expr_col = load_rnaseq_expression(df)

    assert gene_col == "gene_id"
    assert clean_col == "gene_id_clean"
    assert expr_col == "TPM"
    assert list(out["gene_id_clean"]) == ["ENSG1", "ENSG2"]
    assert list(out["TPM"]) == [1.0, 2.5]


def test_load_gencode_genes_and_create_tss_bed():
    gtf = pd.DataFrame([
        ["chr1", "src", "gene", 101, 200, ".", "+", ".", 'gene_id "ENSG0001.5"; gene_name "GENE1"; gene_type "protein_coding";'],
        ["chr2", "src", "gene", 301, 400, ".", "-", ".", 'gene_id "ENSG0002.9"; gene_name "GENE2"; gene_type "protein_coding";'],
    ])

    genes = load_gencode_genes(gtf)
    tss = create_tss_bed_from_gencode(gtf, upstream=10, downstream=20)

    assert list(genes["gene_id_clean"]) == ["ENSG0001", "ENSG0002"]
    assert list(tss["gene_id"]) == ["ENSG0001", "ENSG0002"]
    assert list(tss["start"]) == [90, 389]
    assert list(tss["end"]) == [120, 419]


def test_parse_chromhmm_intersect_uses_explicit_state_column():
    df = pd.DataFrame([
        ["chr1", 10, 20, "peak1", "chr1", 0, 100, "TssA"],
    ])

    out = parse_chromhmm_intersect(df, a_col_count=4, chromhmm_cols=4, state_col_in_b=3)

    assert out.loc[0, "state"] == "TssA"


def test_parse_bedtools_closest_requires_exact_shape():
    df = pd.DataFrame([
        ["chr1", 10, 20, "peak1", "chr1", 100, 200, "gene1", 80],
    ])

    out = parse_bedtools_closest(
        df,
        a_col_count=4,
        b_col_count=4,
        a_names=["a_chrom", "a_start", "a_end", "a_name"],
        b_names=["b_chrom", "b_start", "b_end", "gene_id"],
    )

    assert list(out.columns) == [
        "a_chrom",
        "a_start",
        "a_end",
        "a_name",
        "b_chrom",
        "b_start",
        "b_end",
        "gene_id",
        "distance",
    ]
    assert out.loc[0, "distance"] == 80


def test_merge_rnaseq_with_nearest_genes_uses_clean_ids():
    nearest = pd.DataFrame({
        "gene_id": ["ENSG0001.5", "ENSG0002.7"],
        "peak_name": ["p1", "p2"],
    })
    rnaseq = pd.DataFrame({
        "gene_id": ["ENSG0001.1", "ENSG0002.9"],
        "TPM": [5.0, 7.5],
    })

    merged, expr_col = merge_rnaseq_with_nearest_genes(nearest, rnaseq)

    assert expr_col == "TPM"
    assert list(merged["gene_id_clean"]) == ["ENSG0001", "ENSG0002"]
    assert list(merged["TPM"]) == [5.0, 7.5]


def test_run_bedtools_closest_to_tss_parses_mocked_output(monkeypatch, tmp_path: Path):
    peaks = tmp_path / "peaks.bed"
    tss = tmp_path / "tss.bed"
    peaks.write_text("chr1\t10\t20\tpeak1\n")
    tss.write_text("chr1\t100\t200\tENSG1\t0\t+\n")

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
        if cmd[:2] == ["bedtools", "sort"]:
            src = Path(cmd[-1])
            stdout.write(src.read_text())
        elif cmd[:2] == ["bedtools", "closest"]:
            stdout.write("chr1\t10\t20\tpeak1\tchr1\t100\t200\tENSG1\t0\t+\t80\n")
        else:
            raise AssertionError(f"Unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    out = run_bedtools_closest_to_tss(peaks, tss, max_distance=100)

    assert list(out.columns) == [
        "peak_chrom",
        "peak_start",
        "peak_end",
        "peak_name",
        "tss_chrom",
        "tss_start",
        "tss_end",
        "gene_id",
        "score",
        "strand",
        "distance",
    ]
    assert out.loc[0, "gene_id"] == "ENSG1"
    assert out.loc[0, "distance"] == 80


def test_annotate_peaks_with_chromhmm_parses_mocked_output(monkeypatch, tmp_path: Path):
    peaks = tmp_path / "peaks.bed"
    chrom = tmp_path / "chromhmm.bed"
    peaks.write_text("chr1\t10\t20\tpeak1\n")
    chrom.write_text("chr1\t0\t100\t1_TssA\n")

    def fake_run(cmd, stdout=None, stderr=None, text=None, check=None):
        if cmd[:2] == ["bedtools", "intersect"]:
            stdout.write("chr1\t10\t20\tpeak1\tchr1\t0\t100\t1_TssA\n")
        else:
            raise AssertionError(f"Unexpected command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("src.utils.bioio.subprocess.run", fake_run)

    out = annotate_peaks_with_chromhmm(
        peaks,
        chrom,
        promoter_states=["1_TssA"],
        enhancer_states=["9_EnhA1"],
    )

    assert out.loc[0, "state"] == "1_TssA"
    assert out.loc[0, "name"] == "peak1"   # default is plain "name", not "peak_name"
    assert bool(out.loc[0, "is_promoter_state"]) is True
    assert bool(out.loc[0, "is_enhancer_state"]) is False
