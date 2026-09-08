"""Finalize the 5 per-cellline TSVs:
  1. Add IPW columns (ipw_coef, ipw_p, ipw_q) by re-running IPW per pair.
  2. Merge in regression columns from the augmented TSV (regr_motif_coef, regr_motif_p,
     q_regr, std_y_subset, standardized_effect).
  3. Relabel pairs with status=too_few_for_test AND n_motif_pos=0 as
     "motif_too_narrow_for_thresh".
  4. Write back to the main TSV (in place).
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from investigate_propensity import propensity_weighted_regression

OUT_BASE = Path("/new-stg/home/jieyuan/mpra_validation/all_celllines_out")
CELLLINES = ["HCT116", "MCF7", "HEK293", "K562", "HepG2"]


def main():
    for cl in CELLLINES:
        main_path = OUT_BASE / cl / f"{cl}_mpra_validation.tsv"
        aug_path = OUT_BASE / cl / f"{cl}_mpra_validation_augmented.tsv"
        df = pd.read_csv(main_path, sep="\t")

        # 1. IPW per pair (only for ok rows; others get NaN)
        ipw_coef = np.full(len(df), np.nan)
        ipw_p = np.full(len(df), np.nan)
        for i, row in df.iterrows():
            if row["status"] != "ok":
                continue
            sub_path = OUT_BASE / cl / "per_pair" / row["pair_id"] / "subset.csv"
            if not sub_path.exists():
                continue
            sub = pd.read_csv(sub_path)
            res = propensity_weighted_regression(sub)
            if res is None:
                continue
            ipw_coef[i] = res[0]
            ipw_p[i] = res[1]
        df["ipw_coef"] = ipw_coef
        df["ipw_p"] = ipw_p
        ipw_q = np.full(len(df), np.nan)
        valid = ~np.isnan(ipw_p)
        if valid.sum() > 0:
            ipw_q[valid] = false_discovery_control(ipw_p[valid])
        df["ipw_q"] = ipw_q

        # 2. Merge regression columns from augmented TSV
        if aug_path.exists():
            aug = pd.read_csv(aug_path, sep="\t")
            regr_cols = ["regr_motif_coef", "regr_motif_p", "q_regr",
                         "std_y_subset", "standardized_effect"]
            available = [c for c in regr_cols if c in aug.columns]
            df = df.merge(aug[["pair_id"] + available], on="pair_id", how="left")

        # 3. Relabel narrow-PWM pairs
        narrow_mask = (df["status"] == "too_few_for_test") & (df["n_motif_pos"].fillna(0) == 0)
        n_relabel = int(narrow_mask.sum())
        df.loc[narrow_mask, "status"] = "motif_too_narrow_for_thresh"

        # 4. Write back (in place)
        df.to_csv(main_path, sep="\t", index=False)

        # Summary
        ok = df[df["status"] == "ok"]
        ipw_sig = ok[ok["ipw_q"] < 0.05]
        pos = int((ipw_sig["ipw_coef"] > 0).sum())
        neg = int((ipw_sig["ipw_coef"] < 0).sum())
        narrow = int((df["status"] == "motif_too_narrow_for_thresh").sum())
        print(f"{cl}: rows={len(df)}  ok={len(ok)}  "
              f"narrow_relabeled={n_relabel}  total_narrow={narrow}  "
              f"ipw_sig={len(ipw_sig)} ({pos}:{neg})")


if __name__ == "__main__":
    main()
