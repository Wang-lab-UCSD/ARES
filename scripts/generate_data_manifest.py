#!/usr/bin/env python3
"""
Generate a data_manifest.yaml like examples/sp1_nfya_K562/data_manifest.yaml.

You only provide:
  - target TF (TF_A): whose binding signal the model predicts
  - regulator motif TF (TF_B): motif that is predictive for TF_A

The script tries to find ENCODE-style ChIP-seq peaks + bigWig under:
  /new-stg/home/hanbei/data/TF_ENCODE4/<CELL_LINE>/<TF>_human/

If regulator ChIP-seq is missing, the manifest will still be written, but the
regulator chipseq entries will be omitted.

Usage:
  python scripts/generate_data_manifest.py --target-tf ATF6 --regulator-tf REST --cell-line K562
  python scripts/generate_data_manifest.py --target-tf SP1 --regulator-tf NFYA --out examples/sp1_nfya_K562/data_manifest.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/new-stg/home/hanbei/data")
DEFAULT_TF_ENCODE4 = DEFAULT_DATA_ROOT / "TF_ENCODE4"


def _first_or_none(paths: list[Path]) -> Path | None:
    return sorted(paths)[0] if paths else None


def find_tf_chipseq(tf_encode4_root: Path, cell_line: str, tf: str) -> tuple[Path | None, Path | None]:
    """
    Return (bed_path, bigwig_path) for TF in a cell line, or (None, None) if not found.
    """
    tf_dir = tf_encode4_root / cell_line / f"{tf}_human"
    if not tf_dir.exists():
        return None, None
    bed = _first_or_none(list(tf_dir.glob("*.bed")))
    bw = _first_or_none(list(tf_dir.glob("*.bigWig"))) or _first_or_none(list(tf_dir.glob("*.bw")))
    return bed, bw


def yaml_quote(p: Path | str) -> str:
    return f"\"{str(p)}\""


def render_manifest(
    target_tf: str,
    regulator_tf: str,
    cell_line: str,
    target_bed: Path,
    target_bw: Path | None,
    regulator_bed: Path | None,
    regulator_bw: Path | None,
    data_root: Path,
) -> str:
    """
    Render a YAML manifest matching the example structure.
    """
    # Keep most paths identical to the example, just swap TF-specific chipseq entries.
    pwm_meme = data_root / "combined_TF_motifs_jaspar_hocomoco12_cisbp.meme"
    genome_fa = data_root / "hg38.fa"
    rnaseq = data_root / "RNAseq_ENCODE" / cell_line / "K562.tsv"  # placeholder; fixed below
    # The example uses /RNAseq_ENCODE/K562.tsv (no cell subdir)
    rnaseq = data_root / "RNAseq_ENCODE" / f"{cell_line}.tsv"

    gencode = data_root / "gencode.v38.annotation.gtf"
    dnase_bw = data_root / "DNaseseq_ENCODE" / cell_line / "ENCFF655HFU.bigWig"
    h3k27ac_bw = data_root / "histone" / cell_line / "H3K27ac" / "ENCFF381NDD.bigWig"
    h3k4me3_bw = data_root / "histone" / cell_line / "H3K4me3" / "ENCFF253TOF.bigWig"
    chromhmm = data_root / "chromHMM" / "E123_18_core_K27ac_hg38lift_mnemonics.bed"
    phylop_bw = data_root / "phyloP" / "hg38.phyloP100way.bw"
    hic_loops = data_root / "HiC" / cell_line / "ENCFF134HIZ.bedpe"
    hic_strips = data_root / "HiC" / cell_line / "ENCFF448GZL.bedpe"
    wgbs_minus = data_root / "WGBS" / cell_line / "ENCFF430PNX.bigWig"
    wgbs_plus = data_root / "WGBS" / cell_line / "ENCFF459XNY.bigWig"

    # Fallback strings if some optional bigWigs are missing
    target_bw_line = ""
    if target_bw is not None:
        target_bw_line = f"    {target_tf}_bigwig_fold_change_over_control: {yaml_quote(target_bw)}\n"

    regulator_block = ""
    if regulator_bed is not None:
        regulator_block += f"\n    {regulator_tf}_peaks: {yaml_quote(regulator_bed)}\n"
        regulator_block += f"    {regulator_tf}_peaks_format: \"narrowPeak (hg38)\"\n"
        if regulator_bw is not None:
            regulator_block += f"    {regulator_tf}_bigwig_fold_change_over_control: {yaml_quote(regulator_bw)}\n"

    # Write with the same high-signal comments as the example (shortened only slightly).
    return f"""# {target_tf}/{regulator_tf} Data Manifest ({cell_line})
# Investigation: Why does {regulator_tf} motif predict {target_tf} binding signal?

finding: >
  The {regulator_tf} motif is the most important feature for predicting {target_tf} binding signal in {cell_line}.
  The goal is to explain the biomolecular mechanism behind this relationship.

context: |
  I want to understand the mechanism of how different transcription factor corporate
  with other transcription factor to regulate gene expression. I currently download the ChIP-seq
  data from ENCODE for various transcription factors in diverse cell lines, also I
  download RNA-seq, Hi-C data, ChromHMM, H3K27ac, H3K4me3 ChIP-seq, DNase-seq, phyloP score,
  WGBS. It is possible to download other public available data to verify if needed.

  My ML model currently suggests that for {target_tf} binding affinity prediction task,
  {regulator_tf} motif plays the most significant role in {cell_line}.

  STRING protein–protein interaction network is available and can be used to test
  whether {regulator_tf}/{target_tf} interact directly or via shared cofactors.

investigation_objective: >
  Characterize the mechanism linking TF_B (motif) to TF_A binding; synthesize a coherent
  causal narrative, possibly identifying a new mechanism not in the predefined taxonomy.

execution_requirements_path: "../../config/requirements_execution.txt"

data:
  # COORDINATE SYSTEM: All files below use hg38 with chr-prefixed contigs (chr1, chr2, ..., chrX, chrY).
  chipseq:
    {target_tf}_peaks: {yaml_quote(target_bed)}
    {target_tf}_peaks_format: "narrowPeak (hg38); col 10 = summit offset from start"
{target_bw_line.rstrip()}
{regulator_block.rstrip()}

  pwm:
    motif_meme: {yaml_quote(pwm_meme)}
    meme_motif_line_example: "MOTIF NFYA|jaspar|MA0060.3"

  genome:
    fasta: {yaml_quote(genome_fa)}

  rnaseq:
    gene_quantification: {yaml_quote(rnaseq)}
    gene_quantification_id_type: "Mostly Ensembl gene IDs (ENSG with version suffix, e.g. ENSG00000000003.14); TPM column available."

  annotations:
    gencode: {yaml_quote(gencode)}

  epigenome:
    dnase_bigwig_read_depth_normalized_signal: {yaml_quote(dnase_bw)}
    dnase_signal_role: "Chromatin accessibility; higher = more open"
    h3k27ac_bigwig_fold_change_over_control: {yaml_quote(h3k27ac_bw)}
    h3k27ac_signal_role: "Active enhancer/promoter mark"
    h3k4me3_bigwig_fold_change_over_control: {yaml_quote(h3k4me3_bw)}
    h3k4me3_signal_role: "Active promoter mark; enriched at TSS"
    chromhmm: {yaml_quote(chromhmm)}
    chromhmm_format: "BED, 4 columns: chrom, start, end, state_mnemonic (e.g. 9_EnhA1). State is 4th column."
    chromhmm_promoter_state: ["1_TssA", "2_TssFlnk", "3_TssFlnkU", "4_TssFlnkD", "14_TssBiv"]
    chromhmm_enhancer_state: ["7_EnhG1", "8_EnhG2", "9_EnhA1", "10_EnhA2", "11_EnhWk", "15_EnhBiv"]
    chromhmm_transcription_state: ["5_Tx", "6_TxWk"]
    chromhmm_polycomb_repressed_state: ["16_ReprPC", "17_ReprPCWk"]
    chromhmm_quiescent_state: ["18_Quies"]
    WGBS_minus_strand_bigwig: {yaml_quote(wgbs_minus)}
    WGBS_minus_strand_signal_role: "CpG methylation level (0–1)"
    WGBS_plus_strand_bigwig: {yaml_quote(wgbs_plus)}
    WGBS_plus_strand_signal_role: "CpG methylation level (0–1)"

  phyloP:
    phyloP_score: {yaml_quote(phylop_bw)}
    phyloP_score_role: "Vertebrate conservation per base (100-way)"

  hic:
    hic_loops_bedpe: {yaml_quote(hic_loops)}
    hic_loops_role: "Chromatin loop calls; treat endpoints as loop anchors"
    hic_chromatin_strips_bedpe: {yaml_quote(hic_strips)}

  string:
    species: 9606
    id_type: "symbol"
    min_score: 700
    links_file: {yaml_quote(data_root / "string" / "9606.protein.links.v12.0.txt.gz")}
    aliases_file: {yaml_quote(data_root / "string" / "9606.protein.aliases.v12.0.txt.gz")}

tools:
  - bedtools
  - samtools
  - meme
  - fimo
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tf", required=True, help="TF_A (binding signal target), e.g. SP1")
    ap.add_argument("--regulator-tf", required=True, help="TF_B (motif/regulator), e.g. NFYA")
    ap.add_argument("--cell-line", default="K562")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--tf-encode4-root", default=str(DEFAULT_TF_ENCODE4))
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for data_manifest.yaml. Default: examples/<target>_<regulator>_<cell>/data_manifest.yaml",
    )
    args = ap.parse_args()

    target_tf = args.target_tf.strip()
    regulator_tf = args.regulator_tf.strip()
    cell_line = args.cell_line.strip()

    data_root = Path(args.data_root)
    tf_encode4_root = Path(args.tf_encode4_root)

    target_bed, target_bw = find_tf_chipseq(tf_encode4_root, cell_line, target_tf)
    if target_bed is None:
        raise FileNotFoundError(f"Could not find target TF peaks for {target_tf} in {tf_encode4_root}/{cell_line}/")

    regulator_bed, regulator_bw = find_tf_chipseq(tf_encode4_root, cell_line, regulator_tf)

    if args.out is None:
        out_path = Path(__file__).resolve().parents[1] / "examples" / f"{target_tf.lower()}_{regulator_tf.lower()}_{cell_line}" / "data_manifest.yaml"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = render_manifest(
        target_tf=target_tf,
        regulator_tf=regulator_tf,
        cell_line=cell_line,
        target_bed=target_bed,
        target_bw=target_bw,
        regulator_bed=regulator_bed,
        regulator_bw=regulator_bw,
        data_root=data_root,
    )
    out_path.write_text(text)

    print(f"Wrote: {out_path}")
    print(f"Target peaks: {target_bed}")
    print(f"Target bigWig: {target_bw if target_bw else 'MISSING'}")
    if regulator_bed is None:
        print(f"Regulator peaks for {regulator_tf}: MISSING (ok; omitted from manifest)")
    else:
        print(f"Regulator peaks: {regulator_bed}")
        print(f"Regulator bigWig: {regulator_bw if regulator_bw else 'MISSING'}")


if __name__ == "__main__":
    main()

