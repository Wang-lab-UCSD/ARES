"""Post-process fixes for the MPRA validation TSVs.

Fix 1 (Problem 1): add effect_size_locus_nonzero column = median of non-zero
locus diffs (matches what the Wilcoxon paired test actually evaluates).

Fix 2 (Problem 2): within-pair label-permutation null on ok rows. For each
matched-pair row, swap pos/neg labels with prob 0.5 (preserving locus structure),
recompute the non-zero effect direction per pair, aggregate the pos:neg ratio
across pairs that are significant in the OBSERVED data. Compare observed
pos-rate to null distribution.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT_BASE = Path("/new-stg/home/jieyuan/mpra_validation/all_celllines_out")
CELLLINES = ["HCT116", "MCF7", "HEK293", "K562", "HepG2"]
N_PERMS = 200
RNG = np.random.default_rng(42)


def locus_diffs(mp: pd.DataFrame, pos_act: np.ndarray, neg_act: np.ndarray) -> np.ndarray:
    """Per-locus diff matching the pipeline: median(pos_activity) grouped by pos_parent,
    median(neg_activity) grouped by neg_parent, then intersect parent indices."""
    tmp = mp.assign(_pos=pos_act, _neg=neg_act)
    pos_locus = tmp.groupby("pos_parent")["_pos"].median()
    neg_locus = tmp.groupby("neg_parent")["_neg"].median()
    shared = sorted(set(pos_locus.index) & set(neg_locus.index))
    if not shared:
        return np.array([])
    return np.array([pos_locus.loc[par] - neg_locus.loc[par] for par in shared])


def median_nonzero(diffs: np.ndarray) -> float:
    if len(diffs) == 0:
        return np.nan
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    return float(np.median(nz))


def process_cellline(cl: str) -> dict:
    df_path = OUT_BASE / cl / f"{cl}_mpra_validation.tsv"
    df = pd.read_csv(df_path, sep="\t")
    pair_dir = OUT_BASE / cl / "per_pair"

    # Cache per-pair matched data once (used for both fixes)
    cached = {}
    for pid in df["pair_id"]:
        mp_path = pair_dir / pid / "matched_pairs.csv"
        if not mp_path.exists():
            cached[pid] = None
            continue
        mp = pd.read_csv(mp_path)
        if len(mp) == 0:
            cached[pid] = None
            continue
        cached[pid] = mp

    # --- Fix 1: effect_size_locus_nonzero observed ---
    obs_eff_nz = []
    for pid in df["pair_id"]:
        mp = cached.get(pid)
        if mp is None:
            obs_eff_nz.append(np.nan)
            continue
        diffs = locus_diffs(mp, mp["pos_activity"].values, mp["neg_activity"].values)
        obs_eff_nz.append(median_nonzero(diffs))
    df["effect_size_locus_nonzero"] = obs_eff_nz
    df.to_csv(df_path, sep="\t", index=False)

    # --- Fix 2: permutation null on ok rows ---
    ok_mask = df["status"].values == "ok"
    sig_mask = ok_mask & (df["q_paired_locus"].values < 0.05)
    obs_signs = np.array([np.sign(e) if pd.notna(e) else 0 for e in df["effect_size_locus_nonzero"]])

    obs_pos = int(((obs_signs > 0) & sig_mask).sum())
    obs_neg = int(((obs_signs < 0) & sig_mask).sum())
    obs_zero = int(((obs_signs == 0) & sig_mask).sum())
    obs_pos_rate = obs_pos / max(1, obs_pos + obs_neg)

    null_pos_rates = []
    null_pos_counts = []
    null_neg_counts = []
    ok_pair_ids = df.loc[sig_mask, "pair_id"].tolist()

    for perm in range(N_PERMS):
        null_signs = []
        for pid in ok_pair_ids:
            mp = cached.get(pid)
            if mp is None:
                null_signs.append(0)
                continue
            swap = RNG.random(len(mp)) < 0.5
            pa = np.where(swap, mp["neg_activity"].values, mp["pos_activity"].values)
            na = np.where(swap, mp["pos_activity"].values, mp["neg_activity"].values)
            diffs = locus_diffs(mp, pa, na)
            eff = median_nonzero(diffs)
            null_signs.append(np.sign(eff) if not np.isnan(eff) else 0)
        s = np.array(null_signs)
        npos = int((s > 0).sum())
        nneg = int((s < 0).sum())
        null_pos_counts.append(npos)
        null_neg_counts.append(nneg)
        null_pos_rates.append(npos / max(1, npos + nneg))

    return {
        "cellline": cl,
        "n_sig": obs_pos + obs_neg + obs_zero,
        "obs_pos": obs_pos,
        "obs_neg": obs_neg,
        "obs_zero": obs_zero,
        "obs_pos_rate": obs_pos_rate,
        "null_pos_rate_mean": float(np.mean(null_pos_rates)),
        "null_pos_rate_sd": float(np.std(null_pos_rates)),
        "null_pos_rate_min": float(np.min(null_pos_rates)),
        "null_pos_rate_max": float(np.max(null_pos_rates)),
    }


if __name__ == "__main__":
    results = []
    for cl in CELLLINES:
        print(f"=== {cl} ===", flush=True)
        r = process_cellline(cl)
        print(
            f"  observed pos:neg = {r['obs_pos']}:{r['obs_neg']} "
            f"(rate={r['obs_pos_rate']:.3f}, zero-effect sig={r['obs_zero']})"
        )
        print(
            f"  null pos-rate (N={N_PERMS} perms): "
            f"{r['null_pos_rate_mean']:.3f} ± {r['null_pos_rate_sd']:.3f} "
            f"[min={r['null_pos_rate_min']:.3f}, max={r['null_pos_rate_max']:.3f}]"
        )
        results.append(r)

    print("\n=== Summary ===")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    summary.to_csv(OUT_BASE / "permutation_null_summary.tsv", sep="\t", index=False)
    print(f"\nWrote summary: {OUT_BASE / 'permutation_null_summary.tsv'}")
