"""Tests for the helper-bundle dispatch in src.prompts.coding.

The coding prompt is assembled from per-data-category bundles, gated on the
hypothesis's `required_data` field. These tests verify the dispatch is
deterministic, conservative on missing input, and produces the expected
content for each bundle combination.
"""

import warnings
from pathlib import Path

import pytest
import yaml

from src.prompts.coding import (
    HELPER_BUNDLES,
    _CATEGORY_TO_BUNDLE,
    _format_data_for_coding,
    _looks_like_file_path,
    assemble_helpers_section,
    build_coding_base,
    build_coding_system_prompt,
    build_repl_system_prompt,
    select_helper_bundles,
    validate_required_data,
)


# ============================================================================
# select_helper_bundles
# ============================================================================


def test_select_helper_bundles_none_loads_all_bundles():
    """When required_data is None (defensive default), all bundles are loaded."""
    bundles = select_helper_bundles(None)
    expected = set(HELPER_BUNDLES.keys())
    assert set(bundles) == expected
    assert bundles[0] == "always", "always bundle must be first"


def test_select_helper_bundles_empty_loads_all_bundles():
    """An empty list is treated the same as None — defensive load-all."""
    bundles = select_helper_bundles([])
    assert set(bundles) == set(HELPER_BUNDLES.keys())


def test_select_helper_bundles_chipseq_only_returns_always():
    """A hypothesis that only declares chipseq data should get just the always bundle."""
    bundles = select_helper_bundles(["chipseq.SP1_peaks", "chipseq.NFYA_peaks"])
    assert bundles == ["always"]


def test_select_helper_bundles_string_loads_string_bundle():
    """Declaring string.links_file should load the string bundle plus always."""
    bundles = select_helper_bundles(["chipseq.SP1_peaks", "string.links_file"])
    assert set(bundles) == {"always", "string"}
    assert bundles[0] == "always"


def test_select_helper_bundles_pwm_loads_fimo_bundle():
    """The pwm category maps to the fimo bundle."""
    bundles = select_helper_bundles(["chipseq.SP1_peaks", "pwm.motif_meme"])
    assert "fimo" in bundles
    assert "always" in bundles


def test_select_helper_bundles_annotations_loads_gencode_bundle():
    """The annotations category maps to the gencode bundle."""
    bundles = select_helper_bundles(["annotations.gencode"])
    assert "gencode" in bundles
    assert "always" in bundles


def test_select_helper_bundles_epigenome_loads_epigenome_bundle():
    """The epigenome category (ChromHMM, histone marks, WGBS) maps to its own bundle."""
    bundles = select_helper_bundles(["epigenome.chromhmm"])
    assert "epigenome" in bundles


def test_select_helper_bundles_rnaseq_loads_rnaseq_bundle():
    """rnaseq category maps to rnaseq bundle (expression helpers)."""
    bundles = select_helper_bundles(["rnaseq.gene_quantification"])
    assert "rnaseq" in bundles


def test_select_helper_bundles_hic_loads_hic_bundle():
    """hic category maps to hic bundle."""
    bundles = select_helper_bundles(["hic.hic_loops_bedpe"])
    assert "hic" in bundles


def test_select_helper_bundles_genome_does_not_add_extra_bundle():
    """genome.fasta is part of the always bundle (used by FIMO etc.) — no extra bundle."""
    bundles = select_helper_bundles(["genome.fasta"])
    assert bundles == ["always"]


def test_select_helper_bundles_unknown_category_silently_ignored():
    """Unknown categories don't crash; the always bundle still loads."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bundles = select_helper_bundles(["unknown_category.foo"])
    assert "always" in bundles


def test_select_helper_bundles_deduplicates():
    """Multiple keys mapping to the same bundle don't create duplicates."""
    bundles = select_helper_bundles([
        "chipseq.SP1_peaks",
        "chipseq.NFYA_peaks",
        "all_tf_chipseq.data_root",
    ])
    # All three map to "always"; result should have always exactly once
    assert bundles == ["always"]


def test_select_helper_bundles_always_first():
    """The always bundle must be the first in the returned list (it's loaded first
    so its content appears first in the assembled prompt).
    """
    bundles = select_helper_bundles(["string.links_file", "rnaseq.gene_quantification"])
    assert bundles[0] == "always"


def test_select_helper_bundles_all_categories():
    """A kitchen-sink hypothesis that uses every category loads every bundle."""
    bundles = select_helper_bundles([
        "chipseq.SP1_peaks",
        "pwm.motif_meme",
        "string.links_file",
        "rnaseq.gene_quantification",
        "annotations.gencode",
        "epigenome.chromhmm",
        "hic.hic_loops_bedpe",
    ])
    expected = {"always", "fimo", "string", "rnaseq", "gencode", "epigenome", "hic"}
    assert set(bundles) == expected


def test_select_helper_bundles_handles_keys_without_dots():
    """Some required_data entries may be bare category names without a dot.
    The dispatch should handle that without crashing.
    """
    bundles = select_helper_bundles(["string"])
    assert "string" in bundles


# ============================================================================
# assemble_helpers_section
# ============================================================================


def test_assemble_helpers_section_always_only():
    """The always bundle alone should produce a small valid section."""
    section = assemble_helpers_section(["always"])
    # Should have hard bans, helper table, helper notes
    assert "HARD BANS" in section
    assert "REQUIRED HELPERS" in section
    assert "HELPER USAGE NOTES" in section
    # Should have core helpers
    assert "count_overlapping_peaks" in section
    assert "extract_bigwig_signals" in section
    assert "intersect_peaks" in section
    # Should NOT have non-always bundle helpers
    assert "find_string_interaction" not in section
    assert "load_string_links" not in section
    assert "find_shared_string_partners" not in section
    assert "annotate_peaks_with_chromhmm" not in section
    assert "run_go_enrichment" not in section


def test_assemble_helpers_section_string_bundle_includes_string_helpers():
    """When string bundle is loaded, STRING helpers must appear."""
    section = assemble_helpers_section(["always", "string"])
    assert "find_string_interaction" in section
    assert "find_shared_string_partners" in section
    assert "load_string_links" in section
    # And should still have always-bundle content
    assert "count_overlapping_peaks" in section


def test_assemble_helpers_section_string_bundle_includes_string_hard_ban():
    """The STRING-specific pd.read_csv ban should appear when string bundle is loaded."""
    section = assemble_helpers_section(["always", "string"])
    assert "load_string_links" in section
    # The ban about pd.read_csv on STRING files should be present
    assert "STRING uses variable whitespace" in section


def test_assemble_helpers_section_no_string_bundle_no_string_hard_ban():
    """The STRING ban should NOT appear when string bundle is not loaded."""
    section = assemble_helpers_section(["always"])
    assert "STRING uses variable whitespace" not in section


def test_assemble_helpers_section_fimo_bundle_includes_fimo_helpers():
    """The fimo bundle should include FIMO-related helpers and the loop ban."""
    section = assemble_helpers_section(["always", "fimo"])
    assert "run_fimo_on_peaks" in section
    assert "find_motif_ids_for_tf" in section
    assert "parse_fimo_tsv" in section
    # FIMO loop ban
    assert "inside any `for`/`while` loop" in section


def test_assemble_helpers_section_rnaseq_includes_expression_helpers():
    """The rnaseq bundle should bring in load_rnaseq helpers (but NOT
    run_go_enrichment, which is in the gencode bundle since GO enrichment
    needs a gene list, not RNA-seq data).
    """
    section = assemble_helpers_section(["always", "rnaseq"])
    assert "load_rnaseq_with_gene_id" in section
    assert "load_rnaseq_expression" in section
    assert "run_go_enrichment" not in section


def test_assemble_helpers_section_gencode_includes_go_enrichment():
    """run_go_enrichment lives in the gencode bundle (next to TSS/closest helpers)
    because the typical GO workflow is peak → nearest gene → enrichment, not
    peak → expression → enrichment.
    """
    section = assemble_helpers_section(["always", "gencode"])
    assert "run_go_enrichment" in section
    assert "run_bedtools_closest_to_tss" in section


def test_go_enrichment_loaded_for_chipseq_plus_gencode_only():
    """REGRESSION: a GO-only iteration with chipseq + annotations + pwm
    (no rnaseq) MUST have run_go_enrichment available. Previously this
    was broken because run_go_enrichment was in the rnaseq bundle.
    """
    bundles = select_helper_bundles([
        "chipseq.SP1_peaks",
        "annotations.gencode",
        "pwm.motif_meme",
    ])
    section = assemble_helpers_section(bundles)
    assert "run_go_enrichment" in section, (
        "GO enrichment helper missing from a GO-feasible iteration "
        f"(bundles loaded: {bundles}). This was the dispatch bug "
        "where GO was incorrectly gated on rnaseq data."
    )


def test_assemble_helpers_section_epigenome_includes_chromhmm():
    """The epigenome bundle should bring in annotate_peaks_with_chromhmm and the
    manifest-aware example.
    """
    section = assemble_helpers_section(["always", "epigenome"])
    assert "annotate_peaks_with_chromhmm" in section
    # The manifest-aware example must be present
    assert "chromhmm_promoter_state" in section


def test_assemble_helpers_section_imports_deduplicated():
    """Even when multiple bundles share an import-like helper name, the import
    list should not contain duplicates.
    """
    section = assemble_helpers_section(["always", "fimo", "string"])
    # Count occurrences of an always-bundle import in the import block
    import_block_start = section.find("from src.utils.bioio import")
    import_block_end = section.find(")", import_block_start)
    import_block = section[import_block_start:import_block_end]
    assert import_block.count("intersect_peaks") == 1


def test_assemble_helpers_section_empty_list_returns_empty():
    """An empty bundle list should produce an empty (or near-empty) string,
    not crash.
    """
    section = assemble_helpers_section([])
    # No HARD BANS / REQUIRED HELPERS sections because no bundles to draw from
    assert "HARD BANS" not in section
    assert "REQUIRED HELPERS" not in section


# ============================================================================
# build_coding_base / build_repl_system_prompt / build_coding_system_prompt
# ============================================================================


def test_build_coding_base_includes_critical_mistakes():
    """The CODING_PROMPT_HEADER (critical mistakes) is unconditionally included."""
    base = build_coding_base(["always"])
    assert "CRITICAL MISTAKES" in base
    assert "Define your denominator" in base
    assert "STRING IDs are many-to-many" in base
    assert "fabrication check" in base.lower()


def test_build_coding_base_smaller_with_fewer_bundles():
    """Loading fewer bundles should produce a strictly smaller prompt."""
    full = build_coding_base(None)  # all bundles
    small = build_coding_base(["always"])
    assert len(small) < len(full)
    # The reduction should be significant — not just a few characters
    assert len(small) < 0.7 * len(full)


def test_build_repl_system_prompt_includes_self_check_and_helpers():
    """The REPL system prompt should have both the REPL-specific guidance and
    the bundle-filtered helper section.
    """
    prompt = build_repl_system_prompt(["chipseq.SP1_peaks", "string.links_file"])
    # REPL-specific content
    assert "Pre-solution self-check" in prompt
    assert "Variable shadowing check" in prompt
    # Critical mistakes
    assert "CRITICAL MISTAKES" in prompt
    # String bundle content (signature notes block, not just the name)
    assert "STRING gene-pair lookup" in prompt
    assert "Named shared cofactors with canonical HGNC symbols" in prompt
    # Should NOT have non-string bundle helper blocks
    assert "annotate_peaks_with_chromhmm" not in prompt
    assert "run_go_enrichment" not in prompt


def test_build_repl_system_prompt_none_loads_all_bundles():
    """When required_data is None, all bundles load (safe fallback)."""
    prompt = build_repl_system_prompt(None)
    # Every bundle's signature helper should be present
    assert "annotate_peaks_with_chromhmm" in prompt
    assert "find_shared_string_partners" in prompt
    assert "run_go_enrichment" in prompt
    assert "find_motif_ids_for_tf" in prompt


def test_build_coding_system_prompt_supports_required_data_arg():
    """The legacy build_coding_system_prompt should accept required_data and
    pass it to the bundle dispatch.
    """
    full = build_coding_system_prompt(tools=["bedtools"], required_data=None)
    narrow = build_coding_system_prompt(
        tools=["bedtools"], required_data=["chipseq.SP1_peaks"]
    )
    # Narrow version should be smaller
    assert len(narrow) < len(full)
    # The STRING bundle's helper notes block should be absent from narrow.
    # (The helper name itself appears in CRITICAL MISTAKES section, which is
    # always shown, so we look for the notes block instead.)
    assert "Named shared cofactors with canonical HGNC symbols" not in narrow
    assert "STRING gene-pair lookup" not in narrow  # helper table row marker
    # Full version should have the STRING helper notes
    assert "Named shared cofactors with canonical HGNC symbols" in full
    assert "STRING gene-pair lookup" in full


def test_repl_system_prompt_size_reduction():
    """Verify that bundle filtering produces a meaningful size reduction for
    a typical PPI iteration.
    """
    full = build_repl_system_prompt(None)  # all bundles
    ppi = build_repl_system_prompt(["chipseq.SP1_peaks", "string.links_file"])
    reduction_pct = (1 - len(ppi) / len(full)) * 100
    assert reduction_pct >= 15, (
        f"Expected at least 15% size reduction for PPI iteration, "
        f"got {reduction_pct:.1f}%"
    )


# ============================================================================
# Manifest coverage regression — prevents silent fallthrough on new categories
# ============================================================================


def _scan_manifest_categories() -> set[str]:
    """Scan every YAML manifest under examples/ and collect every top-level
    category name appearing under the `data:` key.
    """
    examples_dir = Path(__file__).parent.parent / "examples"
    if not examples_dir.exists():
        pytest.skip(f"examples/ directory not found at {examples_dir}")

    categories: set[str] = set()
    for yaml_path in examples_dir.rglob("*.yaml"):
        try:
            with open(yaml_path) as fh:
                manifest = yaml.safe_load(fh)
        except yaml.YAMLError:
            continue
        if not isinstance(manifest, dict):
            continue
        data = manifest.get("data") or {}
        if isinstance(data, dict):
            categories.update(data.keys())
    return categories


def test_select_helper_bundles_covers_all_manifest_categories():
    """REGRESSION GUARD: every category that appears in any manifest YAML must
    be mapped in _CATEGORY_TO_BUNDLE. This test catches silent dispatch gaps
    when someone adds a new manifest type without updating coding.py.

    If this test fails, add the missing category to _CATEGORY_TO_BUNDLE in
    src/prompts/coding.py.
    """
    manifest_categories = _scan_manifest_categories()
    assert manifest_categories, "Found no manifest categories — examples/ may be empty"

    mapped_categories = set(_CATEGORY_TO_BUNDLE.keys())
    unmapped = manifest_categories - mapped_categories
    assert not unmapped, (
        f"Manifest categories {sorted(unmapped)} are not mapped in "
        f"_CATEGORY_TO_BUNDLE. Add them to src/prompts/coding.py so the "
        f"coding agent gets the right helper bundles for hypotheses that "
        f"declare these data types."
    )


def test_select_helper_bundles_warns_on_unknown_category():
    """When required_data contains an unknown category, the dispatch should
    fall back to 'always' AND emit a RuntimeWarning. The warning is the
    visibility mechanism for silent fallthrough — without it, future manifest
    additions would quietly drop their bundles.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundles = select_helper_bundles(["totally_made_up_category.foo"])

    assert bundles == ["always"]
    # Should have emitted exactly one warning about the unknown category
    relevant = [w for w in caught if "unknown manifest categories" in str(w.message)]
    assert len(relevant) == 1
    assert "totally_made_up_category" in str(relevant[0].message)
    assert relevant[0].category is RuntimeWarning


def test_select_helper_bundles_no_warning_on_known_categories():
    """Known categories should not produce any warnings — only unknown ones."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_helper_bundles([
            "chipseq.SP1_peaks",
            "string.links_file",
            "epigenome.chromhmm",
            "phyloP.phyloP_score",
        ])
    relevant = [w for w in caught if "unknown manifest categories" in str(w.message)]
    assert len(relevant) == 0


def test_select_helper_bundles_phyloP_maps_to_always():
    """phyloP is a bigWig conservation track; it should map to the always
    bundle (which has extract_bigwig_signals) without loading a separate bundle.
    """
    bundles = select_helper_bundles(["phyloP.phyloP_score"])
    assert bundles == ["always"]


def test_select_helper_bundles_warning_lists_all_unknown_categories():
    """If multiple unknown categories appear, the warning should list them all
    (not just the first one), so a single warning gives the maintainer the
    full picture.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        select_helper_bundles([
            "totally_fake.x",
            "another_fake.y",
            "chipseq.SP1_peaks",  # this one IS known, should not appear in warning
        ])
    relevant = [w for w in caught if "unknown manifest categories" in str(w.message)]
    assert len(relevant) == 1
    msg = str(relevant[0].message)
    assert "totally_fake" in msg
    assert "another_fake" in msg
    assert "chipseq" not in msg


# ============================================================================
# validate_required_data — keyword scan for missing manifest categories
#
# This is the log-only validator (Step 1 of the rollout). It scans the
# hypothesis's name + prediction + verification_plan for unambiguous keywords
# (FIMO, ChromHMM, GO enrichment, etc.) and returns categories implied by the
# text but missing from `required_data`. Conservative on purpose — false
# negatives are better than noisy false positives.
# ============================================================================


def _hyp(name="t", prediction="p", plan=None, required_data=None):
    """Build a minimal hypothesis dict for the validator."""
    return {
        "name": name,
        "prediction": prediction,
        "verification_plan": plan or [],
        "required_data": required_data or [],
    }


def test_validate_returns_empty_for_clean_hypothesis():
    """No keyword hits → empty list."""
    h = _hyp(
        name="REST and ATF6 co-bound peaks",
        prediction="Co-bound peaks are enriched compared to single-TF peaks",
        plan=["count overlaps", "compute fold enrichment"],
        required_data=["chipseq.atf6_peaks", "chipseq.rest_peaks"],
    )
    assert validate_required_data(h) == []


def test_validate_returns_empty_when_categories_already_declared():
    """Keyword fires but the category is already in required_data — no warning."""
    h = _hyp(
        prediction="Run FIMO with the JASPAR REST motif on ATF6 peaks",
        plan=["scan motifs with FIMO", "report hit fraction"],
        required_data=["chipseq.atf6_peaks", "pwm.jaspar_rest"],
    )
    assert validate_required_data(h) == []


def test_validate_detects_missing_pwm():
    """FIMO mentioned but pwm.* not declared."""
    h = _hyp(
        prediction="Use FIMO to scan motifs in co-bound peaks",
        plan=["run FIMO on the peak FASTA"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["pwm"]


def test_validate_detects_missing_string():
    """STRING db / PPI mentioned but string.* not declared."""
    h = _hyp(
        prediction="ATF6 and REST share interaction partners in STRING",
        plan=["query STRING for shared partners"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["string"]


def test_validate_detects_missing_annotations_for_go_enrichment():
    """GO enrichment requires gencode (annotations bundle) for gene mapping."""
    h = _hyp(
        prediction="Genes near co-bound peaks are enriched for stress-response GO terms",
        plan=["assign nearest gene", "run GO enrichment via g:Profiler"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["annotations"]


def test_validate_detects_missing_chromhmm():
    """ChromHMM / chromatin state mentioned but epigenome.* not declared."""
    h = _hyp(
        prediction="Co-bound sites are enriched in active enhancer chromatin states",
        plan=["intersect peaks with ChromHMM annotations"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["epigenome"]


def test_validate_detects_missing_epigenome_for_histone_marks():
    """Specific histone marks (H3K27ac, H3K4me3) imply epigenome category."""
    h = _hyp(
        prediction="Co-bound sites have higher H3K27ac signal than single-TF sites",
        plan=["extract H3K27ac signal at peak centers"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["epigenome"]


def test_validate_detects_missing_hic():
    """Hi-C / loop / TAD mentioned but hic.* not declared."""
    h = _hyp(
        prediction="Co-bound sites cluster at loop anchors in Hi-C data",
        plan=["intersect peaks with chromatin loops"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["hic"]


def test_validate_detects_missing_rnaseq():
    """Expression / TPM / RNA-seq mentioned but rnaseq.* not declared."""
    h = _hyp(
        prediction="Genes near co-bound peaks have higher TPM in K562 RNA-seq",
        plan=["link peaks to nearest gene", "compare TPM distributions"],
        required_data=["chipseq.atf6_peaks", "annotations.gencode"],
    )
    # nearest gene → annotations is already declared, but RNA-seq is not
    assert validate_required_data(h) == ["rnaseq"]


def test_validate_detects_missing_phylop():
    """phyloP / conservation mentioned but phyloP.* not declared."""
    h = _hyp(
        prediction="Co-bound peaks have elevated phyloP conservation",
        plan=["extract phyloP signal at peak centers"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["phyloP"]


def test_validate_detects_multiple_missing_categories():
    """A complex hypothesis touching motifs + GO + ChromHMM should return all three."""
    h = _hyp(
        prediction=(
            "REST motifs in co-bound peaks lie in ChromHMM enhancer states "
            "and their nearest genes are enriched for stress GO terms"
        ),
        plan=[
            "FIMO scan REST motif",
            "intersect with ChromHMM",
            "GO enrichment via g:Profiler on nearest genes",
        ],
        required_data=["chipseq.atf6_peaks"],
    )
    missing = validate_required_data(h)
    assert "pwm" in missing
    assert "epigenome" in missing
    assert "annotations" in missing
    assert missing == sorted(missing)  # output is sorted


def test_validate_handles_none_required_data():
    """If required_data is None (not yet set), every detected category is missing."""
    h = {
        "name": "FIMO scan",
        "prediction": "scan motifs with FIMO",
        "verification_plan": ["run FIMO"],
        "required_data": None,
    }
    assert validate_required_data(h) == ["pwm"]


def test_validate_handles_missing_text_fields():
    """Empty / missing fields should not crash; return empty."""
    assert validate_required_data({}) == []
    assert validate_required_data({"name": "", "prediction": "", "verification_plan": []}) == []


def test_validate_ignores_passing_mentions():
    """Conservative matching: 'gene' alone shouldn't trigger any category."""
    h = _hyp(
        name="The gene set is conserved",
        prediction="Gene-level signal is similar between conditions",
        plan=["compute per-gene aggregate signal"],
        required_data=["chipseq.atf6_peaks"],
    )
    # No tool/data names — should not flag annotations or rnaseq
    assert validate_required_data(h) == []


def test_validate_detects_chromhmm_state_labels():
    """Bare ChromHMM state labels (TssA, EnhG1, ReprPC) should trigger epigenome."""
    h = _hyp(
        prediction="Co-bound peaks fall in TssA and EnhG1 states more often",
        plan=["count peaks per state"],
        required_data=["chipseq.atf6_peaks"],
    )
    assert validate_required_data(h) == ["epigenome"]


def test_validate_case_insensitive():
    """Keywords should match regardless of case (FIMO, fimo, Fimo)."""
    for variant in ["FIMO", "fimo", "Fimo"]:
        h = _hyp(prediction=f"Use {variant} to scan motifs")
        assert validate_required_data(h) == ["pwm"]


def test_validate_does_not_flag_categories_in_other_dotted_form():
    """Declaring 'pwm.foo' satisfies a 'pwm' detection."""
    h = _hyp(
        prediction="Run FIMO on JASPAR motifs",
        plan=["scan with FIMO"],
        required_data=["pwm.jaspar_2024"],
    )
    assert validate_required_data(h) == []


def test_validate_handles_string_verification_plan():
    """verification_plan can be a string instead of a list — should still scan it."""
    h = {
        "name": "t",
        "prediction": "p",
        "verification_plan": "run FIMO on the peaks",
        "required_data": [],
    }
    assert validate_required_data(h) == ["pwm"]


def test_validate_handles_non_string_required_data_entries():
    """Malformed entries (e.g. None in the list) should be ignored without crashing."""
    h = _hyp(
        prediction="Run FIMO",
        required_data=[None, "pwm.foo", 123],  # type: ignore[list-item]
    )
    assert validate_required_data(h) == []  # pwm is satisfied by pwm.foo


# ============================================================================
# _looks_like_file_path — heuristic used to decide whether to redact values
# from the data display in the coding prompt
# ============================================================================


def test_looks_like_file_path_absolute_paths():
    """Any string starting with '/' is treated as a path."""
    assert _looks_like_file_path("/new-stg/data/foo.bed") is True
    assert _looks_like_file_path("/home/user/data") is True
    assert _looks_like_file_path("/") is True


def test_looks_like_file_path_known_extensions():
    """Relative paths with known bio-file extensions are still paths."""
    assert _looks_like_file_path("data/foo.bed") is True
    assert _looks_like_file_path("ENCFF123.bigWig") is True
    assert _looks_like_file_path("genome.fa") is True
    assert _looks_like_file_path("motifs.meme") is True
    assert _looks_like_file_path("annotations.gtf.gz") is True
    assert _looks_like_file_path("loops.bedpe") is True


def test_looks_like_file_path_metadata_strings():
    """Human-readable metadata values are NOT paths."""
    assert _looks_like_file_path("narrowPeak (hg38); col 10 = summit offset") is False
    assert _looks_like_file_path("fold change over control") is False
    assert _looks_like_file_path("ensembl_gene_id") is False
    assert _looks_like_file_path("TPM") is False


def test_looks_like_file_path_non_strings():
    """Non-string values are never paths."""
    assert _looks_like_file_path(None) is False
    assert _looks_like_file_path(123) is False
    assert _looks_like_file_path(["a", "b"]) is False
    assert _looks_like_file_path({"key": "value"}) is False


# ============================================================================
# _format_data_for_coding — redacts file paths but keeps metadata inline
#
# The critical regression we're guarding against: a previous run (ETS2/YY1,
# 2026-04-09) had the LLM hardcode `/new-stg/home/hanbei/data/.../ENCFF*.bed`
# paths into iter 3+ code and clobber the pre-injected `data_files` dict.
# Root cause: absolute paths were visible in the prompt, so the LLM copied
# them instead of using `data_files['key']`. Redacting removes the temptation.
# ============================================================================


def test_format_data_redacts_absolute_paths():
    """File paths must NOT appear in the rendered output — only keys."""
    manifest = {
        "data": {
            "chipseq": {
                "ETS2_peaks": "/new-stg/home/hanbei/data/TF_ENCODE4/K562/ETS2_human/ENCFF772QLT.bed",
                "ETS2_bigwig_fold_change_over_control": "/new-stg/home/hanbei/data/TF_ENCODE4/K562/ETS2_human/ENCFF505DZN.bigWig",
            },
            "genome": {
                "fasta": "/new-stg/home/hanbei/data/hg38.fa",
            },
        }
    }
    out = _format_data_for_coding(manifest)

    # Keys are visible
    assert "ETS2_peaks" in out
    assert "ETS2_bigwig_fold_change_over_control" in out
    assert "fasta" in out

    # Paths are NOT visible — this is the whole point
    assert "/new-stg/home/hanbei/data" not in out
    assert "ENCFF772QLT" not in out
    assert "ENCFF505DZN" not in out
    assert "hg38.fa" not in out


def test_format_data_keeps_metadata_inline():
    """Non-path values (format descriptions, role annotations) are kept inline
    because the LLM needs them to write correct code."""
    manifest = {
        "data": {
            "chipseq": {
                "ETS2_peaks": "/data/foo.bed",
                "ETS2_peaks_format": "narrowPeak (hg38); col 10 = summit offset from start",
            },
            "rnaseq": {
                "gene_quantification": "/data/rnaseq.tsv",
                "gene_quantification_id_type": "ensembl_gene_id",
            },
        }
    }
    out = _format_data_for_coding(manifest)

    # Metadata kept verbatim
    assert "narrowPeak (hg38); col 10 = summit offset from start" in out
    assert "ensembl_gene_id" in out

    # Paths still redacted
    assert "/data/foo.bed" not in out
    assert "/data/rnaseq.tsv" not in out


def test_format_data_includes_access_instructions():
    """The rendered output must tell the LLM how to access files."""
    manifest = {"data": {"chipseq": {"ETS2_peaks": "/data/foo.bed"}}}
    out = _format_data_for_coding(manifest)

    # Explicit instruction on the access method
    assert "data_files[" in out
    # Explicit recovery procedure for wrong keys
    assert "print(sorted(data_files.keys()))" in out
    # Explicit ban on reassignment (defense-in-depth with the hard ban)
    assert "do NOT" in out or "NEVER" in out or "not" in out.lower()


def test_format_data_preserves_category_headers():
    """Category headers (chipseq, rnaseq, etc.) stay visible."""
    manifest = {
        "data": {
            "chipseq": {"ETS2_peaks": "/data/a.bed"},
            "rnaseq": {"gene_quantification": "/data/b.tsv"},
            "epigenome": {"h3k27ac": "/data/c.bigWig"},
        }
    }
    out = _format_data_for_coding(manifest)
    assert "## chipseq" in out
    assert "## rnaseq" in out
    assert "## epigenome" in out


def test_format_data_no_paths_even_with_file_summaries():
    """File summaries from the pre-run inspection may legitimately mention
    paths (the inspection code prints them). That's OK — file_summaries is
    a separate, trusted pre-run output block. But the primary manifest
    display must still redact."""
    manifest = {"data": {"chipseq": {"ETS2_peaks": "/data/foo.bed"}}}
    summaries = {"all_files": "ETS2_peaks: 905 lines, 13 columns"}
    out = _format_data_for_coding(manifest, file_summaries=summaries)
    # The manifest section still redacts
    assert "/data/foo.bed" not in out
    # But the file summary is preserved
    assert "905 lines" in out
