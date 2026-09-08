# MPRA validation

Does disrupting the partner motif change reporter activity? Tests each atlas dependency against the
PARM massively parallel reporter assay, producing the input to Fig. 3a.

## Pipeline

    data/ares_atlas.tsv
        |  make_pairs_from_atlas.py          one <CELL>_pairs.tsv per cell line
        v
    pairs_by_cellline/
        |  run_validation_combined.py        per cell line: FIMO-scan the PARM fragments for the
        |                                    partner motif, pair motif-positive against
        v                                    motif-negative fragments sharing a parent locus,
    <CELL>_mpra_validation.tsv               Wilcoxon per locus, Benjamini-Hochberg within the
        |                                    cell line
        |  postprocess_fixes.py -> finalize_tsvs.py
        v
    data/fig3/mpra_locus_results.csv         the status == "ok" rows
        |  figures/fig3.py :: mpra()
        v
    figures/output/fig3/fig3_mpra.pdf

`status` records why a pair could not be tested rather than dropping it silently:
`underpowered_locus_clustering` (too few shared parent loci), `motif_too_narrow_for_thresh`,
`too_few_for_test` (fewer than five non-zero paired differences, so Wilcoxon is undefined) and
`insufficient_coverage`. Only `ok` pairs carry a q-value and reach the figure.

## Running it

Needs the PARM library and the ENCODE ChIP peaks, neither of which ships with this repository; the
paths are at the top of `run_validation_combined.py`. FIMO (MEME Suite) must be on the system.

    python make_pairs_from_atlas.py
    python run_validation_combined.py --cellline HepG2 \
        --pairs pairs_by_cellline/HepG2_pairs.tsv --out-dir out/HepG2

## Other files

`run_validation.py` is the earlier per-fold variant, superseded by the combined-mode script that
concatenates all six PARM splits; `combine_folds.py` and `aggregate.py` belong to that route.
`select_pilot.py` and `verify_pair.sh` are the pilot-selection and single-pair sanity checks.
`investigate_propensity.py`, `investigate_ipw_null.py` and `investigate_b_c.py` are diagnostics for
cohort confounding and inverse-probability weighting; they informed the choice of the locus-paired
test and are kept for the record.
