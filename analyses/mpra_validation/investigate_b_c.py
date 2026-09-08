"""Investigate Problems B and C with stronger diagnostics.

B (cohort confound): For each "ok" pair, fit Y ~ motif_pos + length + gc + tss_distance
on the full peak-overlapping fragment cohort (NOT the 1:1 matched subset). Compare the
sign distribution of the motif coefficient to the matched-Wilcoxon direction. If the
positive asymmetry persists under regression, the signal is robust to analysis choice
(not an artifact of 1:1 matching). Also report direction asymmetry of fragment-level
Mann-Whitney (unmatched) already in the TSV as a sanity check.

C (K562 scale): Compute standardized effect = effect_size_locus_nonzero / std(Y_subset)
per ok pair. Compare distributions across cell lines. If K562 falls in line with other
CLs after standardization, the 10x magnitude gap is normalization, not biology.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT_BASE = Path("/new-stg/home/jieyuan/mpra_validation/all_celllines_out")
CELLLINES = ["HCT116", "MCF7", "HEK293", "K562", "HepG2"]


def regression_coef_sign(pair_dir: Path) -> tuple[float, float, float] | None:
    """Returns (coef, p_value, std_Y) from Y ~ motif + length + gc + tss_distance."""
    sub_path = pair_dir / "subset.csv"
    if not sub_path.exists():
        return None
    sub = pd.read_csv(sub_path)
    if len(sub) < 30 or sub["motif_pos"].nunique() < 2:
        return None
    X = sub[["motif_pos", "length", "gc", "tss_distance"]].values.astype(float)
    y = sub["activity"].values.astype(float)
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    # OLS via lstsq
    try:
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    # Compute SE for motif_pos coefficient (col 1)
    n, p = X.shape
    y_pred = X @ beta
    rss = float(np.sum((y - y_pred) ** 2))
    if n - p <= 0:
        return None
    sigma2 = rss / (n - p)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    se_motif = float(np.sqrt(cov[1, 1]))
    coef_motif = float(beta[1])
    if se_motif <= 0:
        return None
    t = coef_motif / se_motif
    # two-sided p from t distribution
    pval = float(2 * stats.t.sf(abs(t), df=n - p))
    std_y = float(np.std(y, ddof=1))
    return coef_motif, pval, std_y


def main():
    rows = []
    for cl in CELLLINES:
        df = pd.read_csv(OUT_BASE / cl / f"{cl}_mpra_validation.tsv", sep="\t")
        ok = df[df["status"] == "ok"].copy()
        coefs, pvals, std_ys = [], [], []
        for pid in ok["pair_id"]:
            res = regression_coef_sign(OUT_BASE / cl / "per_pair" / pid)
            if res is None:
                coefs.append(np.nan); pvals.append(np.nan); std_ys.append(np.nan)
            else:
                coefs.append(res[0]); pvals.append(res[1]); std_ys.append(res[2])
        ok["regr_motif_coef"] = coefs
        ok["regr_motif_p"] = pvals
        ok["std_y_subset"] = std_ys
        ok["standardized_effect"] = ok["effect_size_locus_nonzero"] / ok["std_y_subset"]

        # --- B: direction asymmetry comparisons ---
        sig_paired = ok[ok["q_paired_locus"] < 0.05]
        # Fragment-level Mann-Whitney from TSV (use raw p, not BH-adjusted)
        sig_frag = ok[ok["p_fragment"] < 0.05]
        # Regression: BH on regr_motif_p among ok rows
        from scipy.stats import false_discovery_control
        valid_regr = ok["regr_motif_p"].notna()
        q_regr = np.full(len(ok), np.nan)
        if valid_regr.sum() > 0:
            q_regr[valid_regr.values] = false_discovery_control(ok.loc[valid_regr, "regr_motif_p"].values)
        ok["q_regr"] = q_regr
        sig_regr = ok[ok["q_regr"] < 0.05]

        def pos_neg(s: pd.DataFrame, col: str) -> tuple[int, int]:
            v = s[col].dropna()
            return int((v > 0).sum()), int((v < 0).sum())

        p_paired_pos, p_paired_neg = pos_neg(sig_paired, "effect_size_locus_nonzero")
        p_frag_pos, p_frag_neg = pos_neg(sig_frag, "effect_size_fragment")
        p_regr_pos, p_regr_neg = pos_neg(sig_regr, "regr_motif_coef")

        rows.append({
            "cellline": cl,
            "ok_n": len(ok),
            # Paired locus (matched 1:1 Wilcoxon)
            "paired_sig": len(sig_paired),
            "paired_pos:neg": f"{p_paired_pos}:{p_paired_neg}",
            "paired_pos_rate": p_paired_pos / max(1, p_paired_pos + p_paired_neg),
            # Fragment-level Mann-Whitney (no matching, no length/GC/tss control)
            "frag_sig": len(sig_frag),
            "frag_pos:neg": f"{p_frag_pos}:{p_frag_neg}",
            "frag_pos_rate": p_frag_pos / max(1, p_frag_pos + p_frag_neg),
            # Regression (full cohort, controls length/GC/tss)
            "regr_sig": len(sig_regr),
            "regr_pos:neg": f"{p_regr_pos}:{p_regr_neg}",
            "regr_pos_rate": p_regr_pos / max(1, p_regr_pos + p_regr_neg),
            # C: standardized effect distribution
            "med_abs_eff_raw": float(sig_paired["effect_size_locus_nonzero"].abs().median()) if len(sig_paired) else np.nan,
            "med_abs_eff_std": float(sig_paired["standardized_effect"].abs().median()) if len(sig_paired) else np.nan,
            "med_std_y": float(ok["std_y_subset"].median()),
        })

        # Write augmented per-CL TSV (adds regr columns + standardized effect — non-destructive)
        out_path = OUT_BASE / cl / f"{cl}_mpra_validation_augmented.tsv"
        ok_full = df.merge(
            ok[["pair_id", "regr_motif_coef", "regr_motif_p", "q_regr", "std_y_subset", "standardized_effect"]],
            on="pair_id", how="left"
        )
        ok_full.to_csv(out_path, sep="\t", index=False)
        print(f"{cl}: wrote {out_path}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_BASE / "investigation_b_c_summary.tsv", sep="\t", index=False)
    print()
    print("=" * 100)
    print("PROBLEM B — direction asymmetry across analysis methods")
    print("=" * 100)
    print(summary[[
        "cellline", "ok_n",
        "paired_sig", "paired_pos:neg", "paired_pos_rate",
        "frag_sig", "frag_pos:neg", "frag_pos_rate",
        "regr_sig", "regr_pos:neg", "regr_pos_rate",
    ]].to_string(index=False))
    print()
    print("=" * 100)
    print("PROBLEM C — effect magnitude scale across cell lines")
    print("=" * 100)
    print(summary[["cellline", "med_abs_eff_raw", "med_std_y", "med_abs_eff_std"]].to_string(index=False))
    print()
    print(f"Wrote: {OUT_BASE / 'investigation_b_c_summary.tsv'}")


if __name__ == "__main__":
    main()
