"""Pick 10 pilot pairs from K562_all.tsv across the 4 buckets in Hanbei's PLAN.md.

Buckets (with PLAN.md → TSV category mapping):
  - 3 × DIRECT_SEQUENCE_RECOGNITION
  - 3 × ARTIFACT / PARTNER_MOTIF_UNSUPPORTED (PARTNER_MOTIF_UNSUPPORTED, TECHNICAL_ARTIFACT, MODEL_ARTIFACT)
  - 2 × PROTEIN_TETHERING (DIRECT_PPI_TETHERING, INDIRECT_TETHERING, COFACTOR_TETHERING, 3D_LOOP_TETHERING)
  - 2 × CONTEXTUAL_PROXY (CHROMATIN_ACCESSIBILITY_PROXY, CHROMATIN_STATE_GATING, CPG_DENSITY_SEQUENCE_PROXY)

Selection rules:
  - converged == True
  - target TF must have a K562 ChIP-seq directory
  - prefer promoter-binding target TFs (a curated whitelist) so PARM has decent
    fragment coverage; if not enough such pairs, fall back to any target with
    a ChIP dir
  - partner must have a motif in the combined .meme
  - within a bucket, prefer the highest n_supports first
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

K562_TSV = Path("/new-stg/home/hanbei/ARES/analysis/extractor_outputs/K562_all.tsv")
CHIP_ROOT = Path("/new-stg/home/hanbei/data/TF_ENCODE4/K562")
MEME = Path("/new-stg/home/hanbei/data/combined_TF_motifs_jaspar_hocomoco12_cisbp.meme")
OUT = Path("/new-stg/home/jieyuan/mpra_validation/pilot_pairs.tsv")

# Classic promoter-proximal binders — preferred targets for a promoter-tiled MPRA
PROMOTER_BINDERS = {
    "YY1", "NRF1", "SP1", "SP2", "SP3", "SP4", "MAX", "MYC", "MXI1",
    "ELK1", "ELK3", "ELK4", "GABPA", "GABPB1", "ELF1", "ELF2", "ELF4",
    "E2F1", "E2F4", "E2F6", "E2F7", "E2F8",
    "NFYA", "NFYB", "NFYC",
    "TBP", "TAF1", "TAF7",
    "CTCF",
    "REST", "ZFX", "ZNF143", "ZNF263", "ZNF274",
    "KLF1", "KLF6", "KLF13", "KLF16",
    "USF1", "USF2",
    "ATF1", "ATF2", "ATF3", "ATF4", "ATF7",
    "CREB1", "CREB3",
    "CEBPB", "CEBPG", "CEBPZ",
    "HNF4A", "HNF4G",
    "IRF1", "IRF2", "IRF3",
    "ETV1", "ETV4", "ETV5", "ETV6", "ETS1", "ETS2",
    "BHLHE40",
}

# final_mechanism_category values (high-level buckets) — match PLAN.md
BUCKETS = {
    "DIRECT_DNA_RECOGNITION": (3, {"DIRECT_DNA_RECOGNITION"}),
    "ARTIFACT": (3, {"ARTIFACT"}),
    "PROTEIN_TETHERING": (2, {"PROTEIN_TETHERING"}),
    "CONTEXTUAL_PROXY": (2, {"CONTEXTUAL_PROXY"}),
}
# When the ARTIFACT bucket has subtype info, prefer PARTNER_MOTIF_UNSUPPORTED
# (per PLAN.md "ARTIFACT / PARTNER_MOTIF_UNSUPPORTED")
ARTIFACT_PREFERRED_SUBTYPE = "PARTNER_MOTIF_UNSUPPORTED"


def has_chip(tf: str) -> bool:
    return (CHIP_ROOT / f"{tf}_human").is_dir()


def load_meme_motifs() -> set[str]:
    """Return set of TF names that have a motif entry."""
    names = set()
    with MEME.open() as f:
        for line in f:
            if line.startswith("MOTIF "):
                rest = line.split(None, 1)[1].strip()
                names.add(rest.split("|", 1)[0])
    return names


def main():
    df = pd.read_csv(K562_TSV, sep="\t")
    motif_tfs = load_meme_motifs()
    print(f"loaded {len(df)} pairs, {len(motif_tfs)} TFs in motif file", file=sys.stderr)

    df = df[df["converged"] == True].copy()
    df["has_target_chip"] = df["target_tf"].map(has_chip)
    df["has_partner_motif"] = df["partner_tf"].isin(motif_tfs)
    df["target_promoter"] = df["target_tf"].isin(PROMOTER_BINDERS)

    df = df[df["has_target_chip"] & df["has_partner_motif"]].copy()
    print(f"after chip+motif filter: {len(df)}", file=sys.stderr)

    picks = []
    for bucket_name, (need, cats) in BUCKETS.items():
        sub = df[df["final_mechanism_category"].isin(cats)].copy()
        if sub.empty:
            print(f"  WARN: bucket {bucket_name} empty — skip", file=sys.stderr)
            continue
        # rank: promoter-binding target first, then preferred subtype (for ARTIFACT), then highest n_supports
        sub["pref_subtype"] = sub["final_mechanism"] == ARTIFACT_PREFERRED_SUBTYPE
        sub = sub.sort_values(
            ["target_promoter", "pref_subtype", "n_supports"],
            ascending=[False, False, False],
        )
        # diversify on target_tf (don't pick the same target multiple times in one bucket)
        seen_targets: set[str] = set()
        bucket_picks = []
        for _, row in sub.iterrows():
            if row["target_tf"] in seen_targets:
                continue
            bucket_picks.append(row)
            seen_targets.add(row["target_tf"])
            if len(bucket_picks) >= need:
                break
        # if not enough after diversification, fall back without dedup
        if len(bucket_picks) < need:
            for _, row in sub.iterrows():
                if row.name in [p.name for p in bucket_picks]:
                    continue
                bucket_picks.append(row)
                if len(bucket_picks) >= need:
                    break
        for p in bucket_picks:
            picks.append({
                "bucket": bucket_name,
                "pair_id": p["pair_id"],
                "target_tf": p["target_tf"],
                "partner_tf": p["partner_tf"],
                "final_mechanism_category": p["final_mechanism_category"],
                "final_mechanism": p["final_mechanism"],
                "dichotomy_class": p["dichotomy_class"],
                "n_supports": p["n_supports"],
                "target_promoter": p["target_promoter"],
            })

    out_df = pd.DataFrame(picks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, sep="\t", index=False)
    print(f"\nwrote {OUT} ({len(out_df)} pairs)", file=sys.stderr)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
