"""Tests for reusable bioinformatics parsing helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import subprocess

from src.utils.bioio import (
    annotate_peaks_with_chromhmm,
    create_tss_bed_from_gencode,
    find_shared_string_partners,
    find_string_interaction,
    load_gencode_genes,
    load_rnaseq_expression,
    load_rnaseq_with_gene_id,
    load_string_links,
    merge_rnaseq_with_nearest_genes,
    parse_bedtools_closest,
    parse_chromhmm_intersect,
    parse_fimo_tsv,
    parse_peak_id_from_fasta_header,
    read_narrowpeak,
    run_bedtools_closest_to_tss,
    run_go_enrichment,
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


# ---------------------------------------------------------------------------
# run_go_enrichment
# ---------------------------------------------------------------------------

def _mock_gprofiler_result(significant_flags=None) -> pd.DataFrame:
    """Fake g:Profiler profile() response with three terms."""
    rows = [
        {
            "source": "GO:BP", "name": "transcription regulation",
            "p_value": 0.001, "significant": True,
            "intersection_size": 10, "term_size": 200,
            "query_size": 50, "native": "GO:0006355",
        },
        {
            "source": "KEGG", "name": "p53 signaling pathway",
            "p_value": 0.01, "significant": True,
            "intersection_size": 5, "term_size": 80,
            "query_size": 50, "native": "KEGG:04115",
        },
        {
            "source": "GO:MF", "name": "DNA binding",
            "p_value": 0.08, "significant": False,
            "intersection_size": 3, "term_size": 300,
            "query_size": 50, "native": "GO:0003677",
        },
    ]
    df = pd.DataFrame(rows)
    if significant_flags is not None:
        df["significant"] = significant_flags
    return df


def _make_gp_mock(return_value: pd.DataFrame) -> MagicMock:
    instance = MagicMock()
    instance.profile.return_value = return_value
    gp_cls = MagicMock(return_value=instance)
    return gp_cls, instance


class TestRunGoEnrichment:
    def test_returns_expected_columns(self):
        gp_cls, _ = _make_gp_mock(_mock_gprofiler_result())
        with patch("src.utils.bioio.GProfiler", gp_cls, create=True):
            # patch the lazy import inside the function
            with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
                df = run_go_enrichment(["TP53", "MYC"])
        expected = {"source", "name", "p_value", "intersection_size",
                    "term_size", "query_size", "native"}
        assert expected.issubset(set(df.columns))

    def test_significant_only_filters_nonsignificant(self):
        gp_cls, instance = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53", "MYC"], significant_only=True)
        # DNA binding has significant=False → should be excluded
        assert "DNA binding" not in df["name"].values
        assert len(df) == 2

    def test_significant_only_false_returns_all(self):
        gp_cls, instance = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53", "MYC"], significant_only=False)
        assert len(df) == 3

    def test_sorted_by_p_value_ascending(self):
        gp_cls, _ = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53", "MYC"], significant_only=False)
        assert list(df["p_value"]) == sorted(df["p_value"])

    def test_max_terms_limits_output(self):
        gp_cls, _ = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53", "MYC"], significant_only=False, max_terms=2)
        assert len(df) == 2

    def test_empty_gene_list_returns_empty_df_without_api_call(self):
        gp_cls, instance = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment([])
        instance.profile.assert_not_called()
        assert df.empty
        assert "p_value" in df.columns

    def test_api_returns_empty_df_gives_empty_result(self):
        gp_cls, _ = _make_gp_mock(pd.DataFrame())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53"])
        assert df.empty

    def test_api_returns_none_gives_empty_result(self):
        gp_cls, _ = _make_gp_mock(None)
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            df = run_go_enrichment(["TP53"])
        assert df.empty

    def test_custom_sources_passed_to_api(self):
        gp_cls, instance = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            run_go_enrichment(["TP53"], sources=["GO:BP"], significant_only=False)
        call_kwargs = instance.profile.call_args[1]
        assert call_kwargs["sources"] == ["GO:BP"]

    def test_custom_organism_passed_to_api(self):
        gp_cls, instance = _make_gp_mock(_mock_gprofiler_result())
        with patch.dict("sys.modules", {"gprofiler": MagicMock(GProfiler=gp_cls)}):
            run_go_enrichment(["Trp53"], organism="mmusculus", significant_only=False)
        call_kwargs = instance.profile.call_args[1]
        assert call_kwargs["organism"] == "mmusculus"


# ============================================================================
# find_string_interaction tests
# ============================================================================


def _write_string_aliases(tmp_path: Path, lines: list[tuple[str, str, str]]) -> Path:
    """Write a STRING-format aliases file (TSV: ENSP_id, alias, source)."""
    p = tmp_path / "aliases.txt"
    p.write_text("\n".join(f"{a}\t{b}\t{c}" for a, b, c in lines) + "\n")
    return p


def _make_links_df(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["protein1", "protein2", "combined_score"])


def test_find_string_interaction_canonical_pair(tmp_path):
    """Two genes with single canonical IDs and a direct interaction."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_NFYA", "NFYA", "Ensembl_HGNC"),
        ("9606.ENSP_SP1",  "SP1",  "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_NFYA", "9606.ENSP_SP1", 909),
    ])

    result = find_string_interaction("SP1", "NFYA", links, aliases)
    assert result["found"] is True
    assert result["score"] == 909
    assert set(result["all_ids_a"]) == {"9606.ENSP_SP1"}
    assert set(result["all_ids_b"]) == {"9606.ENSP_NFYA"}


def test_find_string_interaction_multimapped_gene_picks_correct_id(tmp_path):
    """Critical test: SP1 has 3 ENSP IDs (KEGG synonyms + canonical),
    only the canonical one has the NFYA interaction. Naive setdefault()
    would pick the wrong ID and miss the interaction.
    """
    aliases = _write_string_aliases(tmp_path, [
        # KEGG synonyms first (the trap)
        ("9606.ENSP_SP1_synonym1", "SP1",  "KEGG_NAME_SYNONYM"),
        ("9606.ENSP_SP1_synonym2", "SP1",  "KEGG_NAME_SYNONYM"),
        ("9606.ENSP_SP1_synonym2", "SP1",  "UniProt_GN_Synonyms"),
        # Canonical Ensembl ID (the one with the real interaction)
        ("9606.ENSP_SP1_canonical", "SP1", "Ensembl_HGNC"),
        ("9606.ENSP_SP1_canonical", "SP1", "Ensembl_HGNC_symbol"),
        # NFYA single canonical ID
        ("9606.ENSP_NFYA", "NFYA", "Ensembl_HGNC"),
    ])
    # The interaction exists ONLY with the canonical SP1 ID — synonyms have no link
    links = _make_links_df([
        ("9606.ENSP_SP1_canonical", "9606.ENSP_NFYA", 909),
    ])

    result = find_string_interaction("SP1", "NFYA", links, aliases)
    assert result["found"] is True, "Failed to find interaction with multi-mapped gene"
    assert result["score"] == 909
    assert "9606.ENSP_SP1_canonical" in (result["protein_a"], result["protein_b"])
    assert len(result["all_ids_a"]) == 3, "Should find all 3 ENSP IDs for SP1"


def test_find_string_interaction_picks_highest_score(tmp_path):
    """When multiple ID combinations link two genes, return the highest score."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A1", "GENE_A", "KEGG"),
        ("9606.ENSP_A2", "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B1", "GENE_B", "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A1", "9606.ENSP_B1", 450),  # weak link via synonym
        ("9606.ENSP_A2", "9606.ENSP_B1", 850),  # strong link via canonical
    ])

    result = find_string_interaction("GENE_A", "GENE_B", links, aliases)
    assert result["found"] is True
    assert result["score"] == 850  # not 450


def test_find_string_interaction_reverse_direction(tmp_path):
    """STRING links can be in either direction (A→B or B→A). Both must match."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A", "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B", "GENE_B", "Ensembl_HGNC"),
    ])
    # Link is stored as B→A (not A→B)
    links = _make_links_df([
        ("9606.ENSP_B", "9606.ENSP_A", 700),
    ])

    result = find_string_interaction("GENE_A", "GENE_B", links, aliases)
    assert result["found"] is True
    assert result["score"] == 700


def test_find_string_interaction_no_link_returns_not_found(tmp_path):
    """Two genes that exist in aliases but have no STRING link."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_X", "GENE_X", "Ensembl_HGNC"),
        ("9606.ENSP_Y", "GENE_Y", "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_X", "9606.ENSP_OTHER", 800),  # X links to something else
    ])

    result = find_string_interaction("GENE_X", "GENE_Y", links, aliases)
    assert result["found"] is False
    assert result["score"] == 0
    # All_ids should still be populated (the genes exist, just no link)
    assert set(result["all_ids_a"]) == {"9606.ENSP_X"}
    assert set(result["all_ids_b"]) == {"9606.ENSP_Y"}


def test_find_string_interaction_unknown_gene_returns_not_found(tmp_path):
    """Gene name not in aliases file at all."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A", "GENE_A", "Ensembl_HGNC"),
    ])
    links = _make_links_df([])

    result = find_string_interaction("GENE_A", "UNKNOWN_GENE", links, aliases)
    assert result["found"] is False
    assert result["score"] == 0
    assert result["all_ids_a"] == ["9606.ENSP_A"]
    assert result["all_ids_b"] == []


def test_find_string_interaction_min_score_filter(tmp_path):
    """min_score filters out interactions below the threshold."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A", "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B", "GENE_B", "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_B", 350),  # below 400 threshold
    ])

    result = find_string_interaction("GENE_A", "GENE_B", links, aliases, min_score=400)
    assert result["found"] is False
    assert result["score"] == 350  # still reports the score for context


def test_find_string_interaction_handles_gzip(tmp_path):
    """Aliases file in gzip format (the real STRING distribution format)."""
    import gzip as _gzip
    p = tmp_path / "aliases.txt.gz"
    with _gzip.open(p, "wt") as f:
        f.write("9606.ENSP_A\tGENE_A\tEnsembl_HGNC\n")
        f.write("9606.ENSP_B\tGENE_B\tEnsembl_HGNC\n")
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_B", 800),
    ])

    result = find_string_interaction("GENE_A", "GENE_B", links, p)
    assert result["found"] is True
    assert result["score"] == 800


# ============================================================================
# find_shared_string_partners tests
# ============================================================================


def test_find_shared_string_partners_basic(tmp_path):
    """Two genes with one shared partner that has a canonical HGNC alias."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",       "GENE_A",  "Ensembl_HGNC"),
        ("9606.ENSP_B",       "GENE_B",  "Ensembl_HGNC"),
        ("9606.ENSP_PARTNER", "EP300",   "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_PARTNER", 800),
        ("9606.ENSP_B", "9606.ENSP_PARTNER", 750),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert result["shared_count"] == 1
    assert len(result["partners"]) == 1
    assert result["partners"][0]["gene_symbol"] == "EP300"
    assert result["partners"][0]["score_a"] == 800
    assert result["partners"][0]["score_b"] == 750


def test_find_shared_string_partners_recovers_canonical_symbol_not_pdb_id(tmp_path):
    """Critical regression test: STRING aliases include PDB IDs (e.g. '2YRP'),
    deletion-region names (e.g. '10q23del'), and other non-canonical synonyms.
    A naive lookup picks one of these instead of the canonical HGNC symbol.
    The helper must filter by source priority and return the HGNC name.
    """
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",       "GENE_A",     "Ensembl_HGNC"),
        ("9606.ENSP_B",       "GENE_B",     "Ensembl_HGNC"),
        # PARTNER has many alias rows; the canonical HGNC line is NOT first
        ("9606.ENSP_PARTNER", "2YRP",       "PDB"),
        ("9606.ENSP_PARTNER", "10q23del",   "Reactome"),
        ("9606.ENSP_PARTNER", "1A02",       "PDB"),
        ("9606.ENSP_PARTNER", "PTEN",       "Ensembl_HGNC"),  # canonical, found late
        ("9606.ENSP_PARTNER", "MMAC1",      "KEGG_NAME_SYNONYM"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_PARTNER", 600),
        ("9606.ENSP_B", "9606.ENSP_PARTNER", 700),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert result["shared_count"] == 1
    assert result["partners"][0]["gene_symbol"] == "PTEN"


def test_find_shared_string_partners_multimapped_query_genes(tmp_path):
    """Critical: query genes themselves often have multiple ENSP IDs (canonical
    Ensembl plus KEGG synonyms). The helper must collect ALL of them and find
    partners across the union — naive code that grabs the first ENSP would
    miss interactions on the other isoforms.
    """
    aliases = _write_string_aliases(tmp_path, [
        # SP1 has 3 ENSP IDs; only the canonical one has the shared partner edge
        ("9606.ENSP_SP1_kegg",     "SP1",  "KEGG_NAME_SYNONYM"),
        ("9606.ENSP_SP1_uniprot",  "SP1",  "UniProt_GN_Synonyms"),
        ("9606.ENSP_SP1_canon",    "SP1",  "Ensembl_HGNC"),
        ("9606.ENSP_NFYA",         "NFYA", "Ensembl_HGNC"),
        ("9606.ENSP_PARTNER",      "TP53", "Ensembl_HGNC"),
    ])
    # Edges live on the canonical SP1 ID, not the synonyms
    links = _make_links_df([
        ("9606.ENSP_SP1_canon", "9606.ENSP_PARTNER", 850),
        ("9606.ENSP_NFYA",      "9606.ENSP_PARTNER", 800),
    ])

    result = find_shared_string_partners("SP1", "NFYA", links, aliases)
    assert result["shared_count"] == 1, "Failed to find partner via canonical SP1 ID"
    assert result["partners"][0]["gene_symbol"] == "TP53"
    # Ensure both SP1 ENSP IDs were collected (verifies multi-mapping handling)
    assert len(result["all_ids_a"]) == 3


def test_find_shared_string_partners_excludes_self(tmp_path):
    """If gene_a's own ENSP appears as a 'partner' of gene_b (because the two
    genes have an edge), it must NOT be reported as a shared partner.
    """
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A", "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B", "GENE_B", "Ensembl_HGNC"),
        ("9606.ENSP_C", "GENE_C", "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_B", 900),  # direct A-B edge
        ("9606.ENSP_A", "9606.ENSP_C", 700),
        ("9606.ENSP_B", "9606.ENSP_C", 650),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert result["shared_count"] == 1, "GENE_A and GENE_B must not be reported as their own shared partners"
    assert result["partners"][0]["gene_symbol"] == "GENE_C"


def test_find_shared_string_partners_min_score_filter(tmp_path):
    """Edges below min_score should not contribute partners."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",       "GENE_A",  "Ensembl_HGNC"),
        ("9606.ENSP_B",       "GENE_B",  "Ensembl_HGNC"),
        ("9606.ENSP_STRONG",  "STRONG",  "Ensembl_HGNC"),
        ("9606.ENSP_WEAK",    "WEAK",    "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_STRONG", 800),
        ("9606.ENSP_B", "9606.ENSP_STRONG", 700),
        ("9606.ENSP_A", "9606.ENSP_WEAK",   350),  # below 400 threshold
        ("9606.ENSP_B", "9606.ENSP_WEAK",   350),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases, min_score=400)
    assert result["shared_count"] == 1
    assert result["partners"][0]["gene_symbol"] == "STRONG"


def test_find_shared_string_partners_sorted_by_min_score(tmp_path):
    """Partners should be sorted by min(score_a, score_b) descending — the
    'evidence on both sides' criterion.
    """
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",   "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B",   "GENE_B", "Ensembl_HGNC"),
        ("9606.ENSP_P1",  "P1",     "Ensembl_HGNC"),
        ("9606.ENSP_P2",  "P2",     "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_P1", 999),  # Very strong on A side
        ("9606.ENSP_B", "9606.ENSP_P1", 410),  # Weak on B side → min = 410
        ("9606.ENSP_A", "9606.ENSP_P2", 700),
        ("9606.ENSP_B", "9606.ENSP_P2", 700),  # Balanced → min = 700
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert [p["gene_symbol"] for p in result["partners"]] == ["P2", "P1"]


def test_find_shared_string_partners_max_partners(tmp_path):
    """max_partners truncates the result list."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",   "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B",   "GENE_B", "Ensembl_HGNC"),
    ] + [
        (f"9606.ENSP_P{i}", f"P{i}", "Ensembl_HGNC") for i in range(5)
    ])
    links = _make_links_df([
        ("9606.ENSP_A", f"9606.ENSP_P{i}", 800) for i in range(5)
    ] + [
        ("9606.ENSP_B", f"9606.ENSP_P{i}", 800) for i in range(5)
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases, max_partners=3)
    assert result["shared_count"] == 5  # Total still reflects all shared
    assert len(result["partners"]) == 3  # But list is truncated


def test_find_shared_string_partners_falls_back_to_ensp_when_no_canonical_alias(tmp_path):
    """If a partner has NO canonical alias source at all, use the ENSP as the
    label rather than picking a noisy synonym.
    """
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",       "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B",       "GENE_B", "Ensembl_HGNC"),
        # PARTNER only has noisy synonyms — no canonical entry
        ("9606.ENSP_PARTNER", "junk1",  "PDB"),
        ("9606.ENSP_PARTNER", "junk2",  "Reactome"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_PARTNER", 800),
        ("9606.ENSP_B", "9606.ENSP_PARTNER", 800),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert result["shared_count"] == 1
    # Falls back to ENSP, not "junk1" or "junk2"
    assert result["partners"][0]["gene_symbol"] == "9606.ENSP_PARTNER"


def test_find_shared_string_partners_no_shared_returns_empty(tmp_path):
    """Two genes with disjoint partner sets."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A",  "GENE_A", "Ensembl_HGNC"),
        ("9606.ENSP_B",  "GENE_B", "Ensembl_HGNC"),
        ("9606.ENSP_X",  "X",      "Ensembl_HGNC"),
        ("9606.ENSP_Y",  "Y",      "Ensembl_HGNC"),
    ])
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_X", 800),
        ("9606.ENSP_B", "9606.ENSP_Y", 800),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, aliases)
    assert result["shared_count"] == 0
    assert result["partners"] == []
    assert set(result["all_ids_a"]) == {"9606.ENSP_A"}
    assert set(result["all_ids_b"]) == {"9606.ENSP_B"}


def test_find_shared_string_partners_unknown_gene_returns_empty(tmp_path):
    """Gene not in aliases file → empty result, no crash."""
    aliases = _write_string_aliases(tmp_path, [
        ("9606.ENSP_A", "GENE_A", "Ensembl_HGNC"),
    ])
    links = _make_links_df([])

    result = find_shared_string_partners("GENE_A", "UNKNOWN", links, aliases)
    assert result["shared_count"] == 0
    assert result["all_ids_b"] == []


def test_find_shared_string_partners_handles_gzip(tmp_path):
    """Real STRING aliases distribution is gzipped — must work."""
    import gzip as _gzip
    p = tmp_path / "aliases.txt.gz"
    with _gzip.open(p, "wt") as f:
        f.write("9606.ENSP_A\tGENE_A\tEnsembl_HGNC\n")
        f.write("9606.ENSP_B\tGENE_B\tEnsembl_HGNC\n")
        f.write("9606.ENSP_P\tEP300\tEnsembl_HGNC\n")
    links = _make_links_df([
        ("9606.ENSP_A", "9606.ENSP_P", 800),
        ("9606.ENSP_B", "9606.ENSP_P", 800),
    ])

    result = find_shared_string_partners("GENE_A", "GENE_B", links, p)
    assert result["shared_count"] == 1
    assert result["partners"][0]["gene_symbol"] == "EP300"
