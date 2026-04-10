"""Tests for SummaryAgent helpers — focused on the deterministic fabrication
detection that runs after the LLM evaluation in evaluate_result.
"""

from src.agents.summary_agent import (
    _detect_fabricated_entities,
    _extract_gene_symbols,
)


# ============================================================================
# _extract_gene_symbols tests
# ============================================================================


def test_extract_gene_symbols_basic():
    """Standard gene names extracted, lowercase ignored."""
    text = "EP300 and CREBBP are coactivators of TP53."
    assert _extract_gene_symbols(text) == {"EP300", "CREBBP", "TP53"}


def test_extract_gene_symbols_filters_stopwords():
    """Common scientific/statistical tokens are not gene symbols."""
    text = "FC = 1.5, P < 0.05, support level: SUPPORTS, used FIMO and BEDTOOLS"
    assert _extract_gene_symbols(text) == set()


def test_extract_gene_symbols_handles_mixed_text():
    """Real-world fragments mix prose, statistics, and gene names."""
    text = (
        "ATF6 signal at REST-bound peaks shows FC=1.30 (p<0.001). "
        "Top shared partners: EP300, CREBBP, JUN, FOS."
    )
    symbols = _extract_gene_symbols(text)
    assert "ATF6" in symbols
    assert "REST" in symbols
    assert "EP300" in symbols
    assert "CREBBP" in symbols
    assert "JUN" in symbols
    assert "FOS" in symbols
    assert "FC" not in symbols


def test_extract_gene_symbols_ignores_lowercase():
    """Lowercase or mixed-case tokens are not gene symbols (we want the
    canonical convention of all-caps).
    """
    text = "the chip-seq peak file has data"
    assert _extract_gene_symbols(text) == set()


def test_extract_gene_symbols_empty():
    """Empty input returns empty set."""
    assert _extract_gene_symbols("") == set()
    assert _extract_gene_symbols(None) == set()


def test_extract_gene_symbols_filters_chromhmm_states():
    """ChromHMM state labels (TssA, EnhG1) won't match because they're not all-caps."""
    text = "Sites in 1_TssA and 9_EnhA1 chromatin states"
    # These shouldn't show up as gene symbols
    symbols = _extract_gene_symbols(text)
    assert "TssA" not in symbols  # mixed case, regex won't match
    assert "EnhA1" not in symbols


def test_extract_gene_symbols_filters_cell_lines():
    """K562, HepG2, etc. are cell lines, not genes (in the stopword list)."""
    text = "K562 cells were used; HEPG2 was the comparison"
    assert _extract_gene_symbols(text) == set()


# ============================================================================
# _detect_fabricated_entities tests — the main contract
# ============================================================================


def test_detect_fabricated_entities_clean_case():
    """When all gene names in summary appear in raw output, nothing is fabricated."""
    summary = "ATF6 binding is enriched at REST sites with EP300 co-occupancy"
    raw = "ATF6 peaks: 5000\nREST peaks: 3000\nEP300 overlap: 2400"
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert fabricated == []


def test_detect_fabricated_entities_catches_invented_names():
    """The exact pattern from the SP1/NFYA bug — coding agent listed plausible
    cofactors that never appeared in the printed STDOUT.
    """
    summary = "Notable shared partners: NFATC4, PTEN, RUNX3, SMARCA4, EP300"
    raw = (
        "Shared partner gene symbols:\n"
        "  ENSP00000388910 -> 2YRP\n"
        "  ENSP00000361021 -> 10q23del\n"
        "  ENSP00000382800 -> 5W69\n"
    )
    fabricated = _detect_fabricated_entities(summary, [], raw)
    # All five names are in the summary but none in the raw output
    assert "NFATC4" in fabricated
    assert "PTEN" in fabricated
    assert "RUNX3" in fabricated
    assert "SMARCA4" in fabricated
    assert "EP300" in fabricated
    assert len(fabricated) == 5


def test_detect_fabricated_entities_partial_fabrication():
    """Some real names + some invented names — flag only the invented ones."""
    summary = "Top partners include EP300 (real) and FAKEGENE (invented)"
    raw = "Partner: EP300, score=900"
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert fabricated == ["FAKEGENE"]


def test_detect_fabricated_entities_checks_findings_list():
    """Findings list (separate from summary) must also be checked."""
    summary = "Mechanism is co-binding via shared cofactor"
    findings = [
        "EP300 was identified as a co-bound site (real)",
        "Top hits: FAKE1, INVENTED",
    ]
    raw = "EP300 binding score: 850"
    fabricated = _detect_fabricated_entities(summary, findings, raw)
    assert "FAKE1" in fabricated
    assert "INVENTED" in fabricated
    assert "EP300" not in fabricated


def test_detect_fabricated_entities_empty_raw_returns_empty():
    """If raw output is missing, we cannot verify — return empty rather than
    flagging everything as fabricated.
    """
    summary = "EP300 and CREBBP are shared partners"
    fabricated = _detect_fabricated_entities(summary, [], None)
    assert fabricated == []
    fabricated = _detect_fabricated_entities(summary, [], "")
    assert fabricated == []


def test_detect_fabricated_entities_ignores_stopwords_in_summary():
    """Stopwords like 'FC', 'P', 'BEDTOOLS' shouldn't be flagged as fabricated
    even though they don't appear in raw output.
    """
    summary = "FC = 1.5, P = 0.03, used BEDTOOLS for overlap"
    raw = "Done. ATF6 enrichment computed."
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert fabricated == []


def test_detect_fabricated_entities_real_world_supports_case():
    """A SUPPORTS-level summary that legitimately cites computed gene names —
    should pass cleanly.
    """
    summary = (
        "ATF6 binding is enriched at REST motif sites with strong EP300 "
        "and CREBBP co-occupancy at promoters"
    )
    raw = (
        "ATF6 peaks: 1234\n"
        "REST motif hits: 456\n"
        "EP300 overlap fraction: 0.73\n"
        "CREBBP overlap fraction: 0.65\n"
    )
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert fabricated == []


def test_detect_fabricated_entities_handles_large_token_count():
    """Stress test: many gene names, only some fabricated."""
    summary = "Partners: " + ", ".join(f"GENE{i}" for i in range(20))
    # Only the first 10 appear in raw
    raw = "\n".join(f"GENE{i}: score=500" for i in range(10))
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert set(fabricated) == {f"GENE{i}" for i in range(10, 20)}


def test_detect_fabricated_entities_case_sensitivity():
    """Gene symbols are all-caps; mixed case in summary won't be flagged."""
    summary = "We tested if Atf6 binds Rest motifs with Ep300"
    raw = "computation complete"
    # Lowercase/mixed case won't match the regex, so nothing is "fabricated"
    fabricated = _detect_fabricated_entities(summary, [], raw)
    assert fabricated == []
