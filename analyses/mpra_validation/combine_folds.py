"""Combine per-fold MPRA validation results into one cross-fold summary.

For each pair, the 5 folds give 5 independent estimates of effect size and p-value
(since folds are disjoint partitions of the PARM library). Combine by:
  - Stouffer's Z (weighted by sqrt(n_shared_parents)) for the locus paired p
  - Median of per-fold effect_size_locus
  - Sum of n_shared_parents (effective coverage across folds)

A pair is "ok_combined" if at least 2 folds have status=="ok" (n_shared >= 10).

Inputs: --fold-tsv arg N times (e.g. all_folds_out/fold0/K562_mpra_validation.tsv ...).
Output: combined.tsv with one row per pair.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control


def stouffer(ps: np.ndarray, weights: np.ndarray) -> float:
    """Stouffer's Z combination, weighted. Returns combined two-sided p."""
    ps = np.asarray(ps, dtype=float)
    ws = np.asarray(weights, dtype=float)
    keep = np.isfinite(ps) & (ps > 0) & (ps < 1) & np.isfinite(ws) & (ws > 0)
    if keep.sum() == 0:
        return float("nan")
    ps, ws = ps[keep], ws[keep]
    zs = stats.norm.isf(ps / 2)  # two-sided -> one-sided positive z; we lose direction
    z = np.sum(ws * zs) / np.sqrt(np.sum(ws ** 2))
    return float(2 * stats.norm.sf(abs(z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-tsv", action="append", required=True,
                    help="One per fold; pass --fold-tsv path multiple times")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    frames = []
    for i, path in enumerate(args.fold_tsv):
        df = pd.read_csv(path, sep="\t")
        df["fold"] = i
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    print(f"loaded {len(long)} rows across {long.fold.nunique()} folds")

    # group by pair_id
    rows = []
    for pid, g in long.groupby("pair_id"):
        meta = {
            "pair_id": pid,
            "target_tf": g.target_tf.iloc[0],
            "partner_tf": g.partner_tf.iloc[0],
            "partner_tf_function": g.partner_tf_function.iloc[0],
            "final_mechanism_category": g.final_mechanism_category.iloc[0] if "final_mechanism_category" in g.columns else None,
            "final_mechanism": g.final_mechanism.iloc[0] if "final_mechanism" in g.columns else None,
            "dichotomy_class": g.dichotomy_class.iloc[0] if "dichotomy_class" in g.columns else None,
        }
        n_folds_ok = int((g.status == "ok").sum())
        n_folds_underpowered = int((g.status == "underpowered_locus_clustering").sum())
        n_folds_other = int(((g.status != "ok") & (g.status != "underpowered_locus_clustering")).sum())

        # combine across folds (ok rows only for the headline test)
        ok = g[g.status == "ok"].copy()
        ws = np.sqrt(ok.n_shared_parents.values.astype(float)) if len(ok) else np.array([])
        combined_p = stouffer(ok.p_paired_locus.values, ws) if len(ok) >= 2 else float("nan")
        combined_effect = float(ok.effect_size_locus.median()) if len(ok) else float("nan")
        total_shared = int(g.n_shared_parents.fillna(0).sum())

        # combined status
        if n_folds_ok >= 2:
            combined_status = "ok"
        elif n_folds_ok == 1:
            combined_status = "single_fold_only"
        else:
            combined_status = "underpowered"

        rows.append({
            **meta,
            "n_folds_ok": n_folds_ok,
            "n_folds_underpowered": n_folds_underpowered,
            "n_folds_other": n_folds_other,
            "total_n_shared_parents": total_shared,
            "combined_effect_locus": combined_effect,
            "combined_p_stouffer": combined_p,
            "combined_status": combined_status,
            # per-fold detail
            "per_fold_p_paired": ";".join(f"{p:.2e}" if not np.isnan(p) else "NA"
                                         for p in g.sort_values("fold").p_paired_locus.values),
            "per_fold_effect": ";".join(f"{e:+.3f}" if not np.isnan(e) else "NA"
                                       for e in g.sort_values("fold").effect_size_locus.values),
            "per_fold_n_shared": ";".join(str(int(n)) if not np.isnan(n) else "NA"
                                         for n in g.sort_values("fold").n_shared_parents.values),
            "per_fold_status": ";".join(g.sort_values("fold").status.values),
        })

    out_df = pd.DataFrame(rows)
    ok_mask = out_df.combined_status == "ok"
    if ok_mask.sum() > 1:
        out_df.loc[ok_mask, "combined_q_stouffer"] = false_discovery_control(
            out_df.loc[ok_mask, "combined_p_stouffer"].values)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep="\t", index=False)
    print(f"wrote {args.out}  ({len(out_df)} pairs)")
    print()
    print(out_df.combined_status.value_counts().to_string())
    print()
    sig = out_df[(out_df.combined_status == "ok") &
                 (out_df.combined_q_stouffer < 0.05) &
                 (out_df.combined_effect_locus.abs() > 0.1)]
    print(f"sig at q<0.05 & |effect|>0.1: {len(sig)}")
    if len(sig):
        print(sig[["pair_id", "final_mechanism_category", "combined_effect_locus",
                   "combined_p_stouffer", "combined_q_stouffer", "n_folds_ok"]].to_string(index=False))


if __name__ == "__main__":
    main()
