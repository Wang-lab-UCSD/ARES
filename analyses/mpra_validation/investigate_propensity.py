"""Propensity-score-weighted regression diagnostic.

For each ok pair:
  1. Estimate propensity p_i = P(motif_pos=1 | length, gc, tss_distance) via logistic
     regression on subset.csv (peak-overlapping fragments).
  2. IPW weights: w_i = 1/p_i if motif+; 1/(1-p_i) if motif-.
  3. Weighted OLS: Y ~ motif + length + gc + tss_distance with these weights.
  4. Compare motif coefficient sign distribution against (a) unweighted regression
     and (b) the matched paired test from the TSV.

If propensity-weighted matches paired-Wilcoxon's positive rate, matching was doing
real bias control. If it matches the unweighted regression instead, matching was
amplifying bias.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

OUT_BASE = Path("/new-stg/home/jieyuan/mpra_validation/all_celllines_out")
CELLLINES = ["HCT116", "MCF7", "HEK293", "K562", "HepG2"]


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-6,
                 ridge: float = 1e-4) -> np.ndarray | None:
    """Logistic regression by IRLS with ridge regularization and step damping.
    X is (n, k), y in {0,1}. Returns beta of length k, or None on failure."""
    n, k = X.shape
    beta = np.zeros(k)
    reg = ridge * np.eye(k)
    for it in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        grad = X.T @ (y - p) - ridge * beta
        H_neg = (X.T * W) @ X + reg  # Fisher info (positive definite)
        try:
            step = np.linalg.solve(H_neg, grad)
        except np.linalg.LinAlgError:
            return None
        # Damp step if too large to avoid overshoot
        step_norm = np.max(np.abs(step))
        if step_norm > 5.0:
            step = step * (5.0 / step_norm)
        beta_new = beta + step
        if not np.all(np.isfinite(beta_new)):
            return None
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new
        beta = beta_new
    return beta  # return last iterate even if not converged


def propensity_weighted_regression(sub: pd.DataFrame) -> tuple[float, float] | None:
    """Returns (coef_motif, p_value) from IPW-weighted OLS. None on failure."""
    if len(sub) < 30 or sub["motif_pos"].nunique() < 2:
        return None
    Xcov = sub[["length", "gc", "tss_distance"]].values.astype(float)
    # Standardize covariates for numerical stability of logistic fit
    mu = Xcov.mean(axis=0)
    sd = Xcov.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Xstd = (Xcov - mu) / sd
    Xlogit = np.column_stack([np.ones(len(Xstd)), Xstd])
    y_motif = sub["motif_pos"].values.astype(float)
    if (y_motif.sum() < 5) or (y_motif.sum() > len(y_motif) - 5):
        return None  # not enough variation
    beta_ps = fit_logistic(Xlogit, y_motif)
    if beta_ps is None:
        return None
    z = Xlogit @ beta_ps
    p_score = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    p_score = np.clip(p_score, 1e-3, 1 - 1e-3)  # truncate extreme scores
    # IPW weights
    w = np.where(y_motif == 1, 1.0 / p_score, 1.0 / (1.0 - p_score))
    # Normalize weights (stabilizes large weights without changing coef estimates)
    w = w * (len(w) / w.sum())
    # Weighted OLS: Y ~ intercept + motif + length + gc + tss_distance
    X = np.column_stack([np.ones(len(sub)), sub["motif_pos"].values.astype(float), Xcov])
    y = sub["activity"].values.astype(float)
    try:
        XtWX = X.T @ (w[:, None] * X)
        XtWy = X.T @ (w * y)
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return None
    y_pred = X @ beta
    resid = y - y_pred
    n, k = X.shape
    if n - k <= 0:
        return None
    # Sandwich estimator for robust SE (simple version with weights)
    sigma2 = float(np.sum(w * resid ** 2) / (n - k))
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None
    se_motif = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    coef_motif = float(beta[1])
    if not np.isfinite(se_motif) or se_motif == 0:
        return None
    t = coef_motif / se_motif
    pval = float(2 * stats.t.sf(abs(t), df=n - k))
    return coef_motif, pval


def main():
    rows = []
    for cl in CELLLINES:
        df = pd.read_csv(OUT_BASE / cl / f"{cl}_mpra_validation.tsv", sep="\t")
        ok = df[df["status"] == "ok"].copy()
        coefs, pvals = [], []
        for pid in ok["pair_id"]:
            sub_path = OUT_BASE / cl / "per_pair" / pid / "subset.csv"
            if not sub_path.exists():
                coefs.append(np.nan); pvals.append(np.nan); continue
            sub = pd.read_csv(sub_path)
            res = propensity_weighted_regression(sub)
            if res is None:
                coefs.append(np.nan); pvals.append(np.nan)
            else:
                coefs.append(res[0]); pvals.append(res[1])
        ok["ipw_coef"] = coefs
        ok["ipw_p"] = pvals
        # BH on ok rows
        from scipy.stats import false_discovery_control
        valid = ok["ipw_p"].notna()
        q = np.full(len(ok), np.nan)
        if valid.sum() > 0:
            q[valid.values] = false_discovery_control(ok.loc[valid, "ipw_p"].values)
        ok["ipw_q"] = q
        sig_ipw = ok[ok["ipw_q"] < 0.05]
        pos = int((sig_ipw["ipw_coef"] > 0).sum())
        neg = int((sig_ipw["ipw_coef"] < 0).sum())
        rate = pos / max(1, pos + neg)
        # Pull existing rates from earlier investigation summary
        existing = pd.read_csv(OUT_BASE / "investigation_b_c_summary.tsv", sep="\t")
        ex = existing[existing["cellline"] == cl].iloc[0]
        rows.append({
            "cellline": cl,
            "ok_n": len(ok),
            "paired_pos:neg": ex["paired_pos:neg"],
            "paired_rate": ex["paired_pos_rate"],
            "regr_pos:neg": ex["regr_pos:neg"],
            "regr_rate": ex["regr_pos_rate"],
            "ipw_sig": len(sig_ipw),
            "ipw_pos:neg": f"{pos}:{neg}",
            "ipw_rate": rate,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_BASE / "investigation_propensity_summary.tsv", sep="\t", index=False)
    print("=" * 100)
    print("PROPENSITY-WEIGHTED REGRESSION vs PAIRED-WILCOXON vs UNWEIGHTED REGRESSION")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"Wrote: {OUT_BASE / 'investigation_propensity_summary.tsv'}")


if __name__ == "__main__":
    main()
