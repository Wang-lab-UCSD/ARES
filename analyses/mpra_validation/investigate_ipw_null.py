"""IPW permutation null and propensity-overlap diagnostics.

(1) IPW null: for each ok pair, permute motif_pos labels at the fragment level
    (preserving length/GC/tss/activity), refit propensity, refit IPW regression,
    record sign of motif coefficient. Repeat N=100 times. Aggregate null direction
    asymmetry across pairs per perm, compare to observed IPW pos-rate.

(2) Propensity diagnostics per pair: effective sample size after weighting,
    max-to-median weight ratio, fraction of weights at the truncation bound.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from investigate_propensity import fit_logistic, propensity_weighted_regression

OUT_BASE = Path("/new-stg/home/jieyuan/mpra_validation/all_celllines_out")
CELLLINES = ["HCT116", "MCF7", "HEK293", "K562", "HepG2"]
N_PERMS = 100
RNG = np.random.default_rng(13)


def ipw_coef_and_diagnostics(sub: pd.DataFrame) -> tuple[float, float, float, float, float] | None:
    """Returns (coef, p_value, ess, max_over_med_w, frac_at_bound). None on failure."""
    if len(sub) < 30 or sub["motif_pos"].nunique() < 2:
        return None
    Xcov = sub[["length", "gc", "tss_distance"]].values.astype(float)
    mu = Xcov.mean(axis=0)
    sd = Xcov.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xstd = (Xcov - mu) / sd
    Xlogit = np.column_stack([np.ones(len(Xstd)), Xstd])
    y_motif = sub["motif_pos"].values.astype(float)
    if (y_motif.sum() < 5) or (y_motif.sum() > len(y_motif) - 5):
        return None
    beta_ps = fit_logistic(Xlogit, y_motif)
    if beta_ps is None:
        return None
    z = np.clip(Xlogit @ beta_ps, -30, 30)
    p_score = 1.0 / (1.0 + np.exp(-z))
    p_raw = p_score.copy()
    p_score = np.clip(p_score, 1e-3, 1 - 1e-3)
    frac_at_bound = float(((p_raw < 1e-3) | (p_raw > 1 - 1e-3)).mean())
    w = np.where(y_motif == 1, 1.0 / p_score, 1.0 / (1.0 - p_score))
    w = w * (len(w) / w.sum())  # normalize
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    max_over_med = float(w.max() / max(np.median(w), 1e-12))
    X = np.column_stack([np.ones(len(sub)), y_motif, Xcov])
    y = sub["activity"].values.astype(float)
    try:
        XtWX = X.T @ (w[:, None] * X)
        XtWy = X.T @ (w * y)
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return None
    resid = y - X @ beta
    n, k = X.shape
    if n - k <= 0:
        return None
    sigma2 = float(np.sum(w * resid ** 2) / (n - k))
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None
    se_motif = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    if not np.isfinite(se_motif) or se_motif == 0:
        return None
    from scipy import stats
    t = float(beta[1]) / se_motif
    pval = float(2 * stats.t.sf(abs(t), df=n - k))
    return float(beta[1]), pval, ess, max_over_med, frac_at_bound


def ipw_coef_only(sub: pd.DataFrame, motif_perm: np.ndarray) -> float | None:
    """Fast version for permutations: returns coef sign only (well, the coef value)."""
    if motif_perm.sum() < 5 or motif_perm.sum() > len(motif_perm) - 5:
        return None
    Xcov = sub[["length", "gc", "tss_distance"]].values.astype(float)
    mu = Xcov.mean(axis=0)
    sd = Xcov.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xstd = (Xcov - mu) / sd
    Xlogit = np.column_stack([np.ones(len(Xstd)), Xstd])
    beta_ps = fit_logistic(Xlogit, motif_perm.astype(float))
    if beta_ps is None:
        return None
    z = np.clip(Xlogit @ beta_ps, -30, 30)
    p_score = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-3, 1 - 1e-3)
    w = np.where(motif_perm == 1, 1.0 / p_score, 1.0 / (1.0 - p_score))
    w = w * (len(w) / w.sum())
    X = np.column_stack([np.ones(len(motif_perm)), motif_perm.astype(float), Xcov])
    y = sub["activity"].values.astype(float)
    try:
        XtWX = X.T @ (w[:, None] * X)
        XtWy = X.T @ (w * y)
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return None
    return float(beta[1])


def main():
    summary_rows = []
    per_pair_diag = []
    for cl in CELLLINES:
        df = pd.read_csv(OUT_BASE / cl / f"{cl}_mpra_validation.tsv", sep="\t")
        ok = df[df["status"] == "ok"].copy().reset_index(drop=True)
        print(f"=== {cl}: ok_n={len(ok)} ===", flush=True)

        # --- Observed IPW direction asymmetry (with diagnostics) ---
        cached = {}
        obs_signs = []
        ess_list = []
        max_w_list = []
        frac_bound_list = []
        for pid in ok["pair_id"]:
            sub_path = OUT_BASE / cl / "per_pair" / pid / "subset.csv"
            if not sub_path.exists():
                cached[pid] = None
                obs_signs.append(0)
                continue
            sub = pd.read_csv(sub_path)
            cached[pid] = sub
            res = ipw_coef_and_diagnostics(sub)
            if res is None:
                obs_signs.append(0)
                continue
            coef, pval, ess, max_w, frac_b = res
            obs_signs.append(np.sign(coef))
            ess_list.append(ess); max_w_list.append(max_w); frac_bound_list.append(frac_b)
            per_pair_diag.append({"cellline": cl, "pair_id": pid, "ess": ess,
                                  "max_over_med_w": max_w, "frac_at_bound": frac_b})

        obs_signs = np.array(obs_signs)
        obs_pos = int((obs_signs > 0).sum())
        obs_neg = int((obs_signs < 0).sum())
        obs_rate = obs_pos / max(1, obs_pos + obs_neg)

        # --- IPW null: permute motif labels per pair, recompute IPW coef ---
        null_rates = []
        n_perm_done = 0
        for perm in range(N_PERMS):
            null_signs = []
            for pid in ok["pair_id"]:
                sub = cached.get(pid)
                if sub is None:
                    null_signs.append(0); continue
                motif_perm = RNG.permutation(sub["motif_pos"].values.astype(int))
                coef = ipw_coef_only(sub, motif_perm)
                null_signs.append(np.sign(coef) if coef is not None else 0)
            s = np.array(null_signs)
            npos = int((s > 0).sum()); nneg = int((s < 0).sum())
            null_rates.append(npos / max(1, npos + nneg))
            n_perm_done += 1
            if (perm + 1) % 10 == 0:
                print(f"  perm {perm+1}/{N_PERMS}: null_rate={null_rates[-1]:.3f}", flush=True)

        null_rates = np.array(null_rates)
        z_score = (obs_rate - null_rates.mean()) / max(null_rates.std(), 1e-9)

        summary_rows.append({
            "cellline": cl,
            "ok_n": len(ok),
            "obs_pos": obs_pos,
            "obs_neg": obs_neg,
            "obs_rate": obs_rate,
            "null_rate_mean": float(null_rates.mean()),
            "null_rate_sd": float(null_rates.std()),
            "null_rate_min": float(null_rates.min()),
            "null_rate_max": float(null_rates.max()),
            "Z_score_vs_null": float(z_score),
            "median_ess": float(np.median(ess_list)) if ess_list else np.nan,
            "median_max_w": float(np.median(max_w_list)) if max_w_list else np.nan,
            "median_frac_at_bound": float(np.median(frac_bound_list)) if frac_bound_list else np.nan,
        })
        print(f"  observed pos-rate = {obs_rate:.3f} ({obs_pos}:{obs_neg})")
        print(f"  IPW null pos-rate = {null_rates.mean():.3f} ± {null_rates.std():.3f}  [Z={z_score:.2f}]")
        print()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_BASE / "investigation_ipw_null_summary.tsv", sep="\t", index=False)
    pd.DataFrame(per_pair_diag).to_csv(OUT_BASE / "investigation_ipw_diagnostics.tsv", sep="\t", index=False)
    print("=" * 90)
    print("IPW PERMUTATION NULL + PROPENSITY DIAGNOSTICS")
    print("=" * 90)
    print(summary.to_string(index=False))
    print()
    print(f"Wrote: {OUT_BASE / 'investigation_ipw_null_summary.tsv'}")
    print(f"Wrote: {OUT_BASE / 'investigation_ipw_diagnostics.tsv'}")


if __name__ == "__main__":
    main()
