"""Aggregate plots from per-pair MPRA validation results (PLAN.md section 4).

Inputs: K562_mpra_validation.tsv (output of run_validation.py)
Outputs:
  - effect_by_pair.png   : effect_size by dichotomy_class
  - category_concordance.png: % of pairs with q<0.05 by final_mechanism_category
  - outliers.tsv            : pairs that disagree with their category's prediction
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Predictions per PLAN.md section 3
NULL_CATEGORIES = {
    "ARTIFACT", "PROTEIN_TETHERING", "PROTEIN_COOPERATIVE_BINDING",
    "PROTEIN_ASSOCIATED_CHROMATIN_PRIMING", "COFACTOR_ASSOCIATED_CO_OCCUPANCY",
    "CONTEXTUAL_PROXY",
}
NONZERO_CATEGORIES = {
    "DIRECT_DNA_RECOGNITION", "MOTIF_GRAMMAR", "MOTIF_MARKED_THIRD_FACTOR_RECRUITMENT",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--qcut", type=float, default=0.05)
    ap.add_argument("--effect-floor", type=float, default=0.1,
                    help="abs(effect) floor for 'significant'")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results, sep="\t")
    ok = df[df["status"] == "ok"].copy()
    # Primary test = locus-level paired Wilcoxon (q_paired_locus, effect_size_locus)
    ok["sig"] = (ok["q_paired_locus"] < args.qcut) & (ok["effect_size_locus"].abs() > args.effect_floor)
    # Diagnostic: paired vs unpaired locus tests should roughly agree; >0.5 OOM
    # divergence suggests high between-locus variance that the paired test absorbs
    # but the unpaired doesn't (or vice versa). NaN for rows where either p is missing.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_diff = np.log10(ok["p_paired_locus"]) - np.log10(ok["p_unpaired_locus"])
    evaluable = log_diff.notna()
    ok["inspect_locus_variance"] = pd.NA
    ok.loc[evaluable, "inspect_locus_variance"] = log_diff[evaluable].abs() > 0.5
    print(f"loaded {len(df)} pairs, {len(ok)} testable (n_shared_parents>=10)")

    # (A) Per-pair effect size, sorted, colored by ARES dichotomy class.
    # Horizontal bars are much clearer than box/strip plots at n=10.
    dich_color = {"SEQUENCE": "#1f77b4", "PROTEIN": "#d62728",
                  "ARTIFACT": "#7f7f7f", "UNRESOLVED": "#bcbd22"}
    plot_df = ok.sort_values("effect_size_locus", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(plot_df) + 2)))
    y_pos = np.arange(len(plot_df))
    colors = [dich_color.get(d, "#999999") for d in plot_df["dichotomy_class"]]
    ax.barh(y_pos, plot_df["effect_size_locus"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    # value annotations at bar end
    for i, (val, sig) in enumerate(zip(plot_df["effect_size_locus"], plot_df["sig"])):
        marker = " *" if sig else ""
        ha = "left" if val >= 0 else "right"
        pad = 0.03 if val >= 0 else -0.03
        ax.text(val + pad, i, f"{val:+.2f}{marker}", va="center", ha=ha, fontsize=9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace("K562_", "") for p in plot_df["pair_id"]])
    ax.set_xlabel("locus-level effect size  (median per-parent: motif+ − motif−)")
    ax.set_title(f"Per-pair MPRA effect (paired by parent locus) — * = q<{args.qcut} & |Δ|>{args.effect_floor}")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor="black", linewidth=0.5)
               for c in dich_color.values()]
    ax.legend(handles, list(dich_color.keys()), title="ARES dichotomy",
              loc="lower right", frameon=True)
    ax.grid(axis="x", alpha=0.3)
    # widen x to fit annotations
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin - 0.3, xmax + 0.4)
    fig.tight_layout()
    fig.savefig(args.out_dir / "effect_by_pair.png", dpi=140)
    plt.close(fig)

    # (B) Category concordance — fraction of pairs significant by category
    cat_summary = ok.groupby("final_mechanism_category").agg(
        n=("pair_id", "size"),
        n_sig=("sig", "sum"),
        median_effect=("effect_size_locus", "median"),
    ).reset_index()
    cat_summary["pct_sig"] = 100 * cat_summary["n_sig"] / cat_summary["n"]
    cat_summary["expected_null"] = cat_summary["final_mechanism_category"].isin(NULL_CATEGORIES)
    cat_summary = cat_summary.sort_values("pct_sig", ascending=False)
    cat_summary.to_csv(args.out_dir / "category_summary.tsv", sep="\t", index=False)
    print("\n=== Category summary ===")
    print(cat_summary.to_string(index=False))

    # Skip the category plot when too few categories survive — the figure misleads
    # readers (a single 100% bar implies a trend that doesn't exist).
    if len(cat_summary) >= 2 and len(ok) >= 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#d62728" if e else "#1f77b4" for e in cat_summary["expected_null"]]
        ax.barh(cat_summary["final_mechanism_category"], cat_summary["pct_sig"], color=colors)
        for i, (cat, pct, n) in enumerate(zip(cat_summary["final_mechanism_category"],
                                              cat_summary["pct_sig"],
                                              cat_summary["n"])):
            ax.text(pct + 1, i, f"{pct:.0f}% ({n})", va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel(f"% pairs significant (q<{args.qcut}, |Δ|>{args.effect_floor})")
        ax.set_title("Category concordance: motif+ ≠ motif− by ARES category\n"
                     "red = predicted null, blue = predicted non-null")
        fig.tight_layout()
        fig.savefig(args.out_dir / "category_concordance.png", dpi=140)
        plt.close(fig)
    else:
        print(f"\n(skipped category_concordance.png: only {len(cat_summary)} category / "
              f"{len(ok)} ok rows — too few for meaningful comparison)")

    # (C) Outliers — flag both null-category-but-significant AND
    # nonzero-category-but-null/wrong-direction. When partner function is known,
    # require direction_match for "concordant" — wrong-direction sig results are
    # also outliers.
    def flag(row) -> str:
        cat = row["final_mechanism_category"]
        sig = row["sig"]
        dm = row.get("direction_match")
        flags = []
        if cat in NULL_CATEGORIES and sig:
            flags.append("NULL_CATEGORY_BUT_SIGNIFICANT")
        if cat in NONZERO_CATEGORIES and not sig:
            flags.append("NONZERO_CATEGORY_BUT_NULL")
        if cat in NONZERO_CATEGORIES and sig and dm is False:
            flags.append("WRONG_DIRECTION")
        if cat in NULL_CATEGORIES and sig and dm is True:
            flags.append("SIG_AND_DIRECTION_MATCHES")  # candidate misclassification
        return ";".join(flags)

    ok["outlier_flag"] = ok.apply(flag, axis=1)
    outliers = ok[ok["outlier_flag"] != ""].copy()
    outliers_out = outliers[[
        "pair_id", "target_tf", "partner_tf",
        "final_mechanism_category", "final_mechanism", "dichotomy_class",
        "partner_tf_function", "expected_sign", "direction_match",
        "n_shared_parents", "effect_size_locus",
        "p_paired_locus", "p_unpaired_locus", "q_paired_locus",
        "inspect_locus_variance", "outlier_flag",
    ]]
    outliers_out.to_csv(args.out_dir / "outliers.tsv", sep="\t", index=False)
    print("\n=== Outliers ===")
    if outliers.empty:
        print("  (none)")
    else:
        print(outliers_out.to_string(index=False))

    cat_plot = args.out_dir / "category_concordance.png"
    extras = ", category_concordance.png" if cat_plot.exists() else ""
    print(f"\nWrote: {args.out_dir}/effect_by_pair.png{extras}, "
          f"category_summary.tsv, outliers.tsv")


if __name__ == "__main__":
    main()
