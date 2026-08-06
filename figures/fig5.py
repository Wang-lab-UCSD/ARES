#!/usr/bin/env python
"""Figure 5 — the route predicts how genetic variants act on the partner motif.

  onmotif_frequency   fraction of pairs carrying an allele-specific-binding QTL on the partner motif,
                      by route, adjusted for motif width and bQTL count
  motif_effect        across on-motif bQTLs, how the allele-specific target-binding effect scales
                      with how much the variant changes the partner-motif score

Both panels use ADASTRA allele-specific-binding QTLs mapped onto the ARES partner motifs. The route
label on every input row is already the atlas dichotomy.

Run as:  python fig5.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

from _common import with_route, CC, INK, data, save, spines

SUB = 'fig5'
ORDER = ['SEQUENCE', 'PROTEIN', 'CONTEXT']
LAB = {'SEQUENCE': 'Sequence', 'PROTEIN': 'Protein', 'CONTEXT': 'Context'}


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def _logit(X, y, iters=200):
    """Newton-Raphson logistic fit, returning coefficients and Wald p-values."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H + 1e-9 * np.eye(X.shape[1]), X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H + 1e-9 * np.eye(X.shape[1]))))
    return beta, 2 * (1 - norm.cdf(np.abs(beta / se)))


def _p10(p):
    """m x 10^e with a mathtext exponent (font-independent)."""
    e = int(np.floor(np.log10(max(p, 1e-300))))
    return rf'{p / 10 ** e:.1f}$\times$10$^{{{e}}}$'


def onmotif_frequency():
    """Fraction of pairs with a bQTL on the partner motif, by route.

    One outcome per (cell, target, partner) triple that has target ADASTRA data and a partner motif:
    does at least one of the target's allele-specific-binding QTLs fall inside the partner motif?
    A logistic regression on route + motif width + log(bQTL count) adjusts for the two obvious
    confounders (longer motifs and more QTLs both raise the hit rate), with SEQUENCE as reference.
    Bars are the raw proportions with Wilson 95% intervals; the span reports the adjusted ordinal
    trend across the three routes, with the unadjusted Cochran-Armitage trend beneath it.
    """
    cov = with_route(pd.read_csv(data('fig5', 'bqtl_pair_coverage.csv')))
    wmap = pd.read_csv(data('fig5', 'bqtl_motif_width.csv')).groupby('partner').width.first().to_dict()
    d = cov[(cov.target_has_adastra) & (cov.partner_has_motif) & (cov.n_asb_snps >= 1)
            & (cov.consensus.isin(ORDER))].copy()
    d['width'] = d.partner.map(wmap)
    d = d.dropna(subset=['width'])
    d['on'] = (d.n_in_partner_motif > 0).astype(int)

    wc = (d.width - d.width.mean()).values
    ln = (np.log(d.n_asb_snps) - np.log(d.n_asb_snps).mean()).values
    y = d.on.values.astype(float)
    od = d.consensus.map({'SEQUENCE': 0, 'PROTEIN': 1, 'CONTEXT': 2}).values.astype(float)
    _, pt = _logit(np.column_stack([np.ones(len(d)), od, wc, ln]), y)
    p_trend = pt[1]

    # unadjusted Cochran-Armitage linear trend across the ordered routes
    nn = np.array([(d.consensus == c).sum() for c in ORDER], float)
    xx = np.array([d[d.consensus == c].on.sum() for c in ORDER], float)
    sc = np.array([0, 1, 2], float)
    N, Rr = nn.sum(), xx.sum()
    pb = Rr / N
    num = (np.sum(xx * sc) - Rr * np.sum(nn * sc) / N) ** 2
    den = pb * (1 - pb) * (np.sum(nn * sc * sc) - (np.sum(nn * sc)) ** 2 / N)
    p_ca = float(chi2.sf(num / den, 1))

    ct = {c: (int(d[d.consensus == c].on.sum()), int((d.consensus == c).sum())) for c in ORDER}
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    his = []
    for i, c in enumerate(ORDER):
        k, n = ct[c]
        p = 100 * k / n
        lo, hi = [100 * v for v in _wilson(k, n)]
        his.append(hi)
        ax.bar(i, p, width=0.58, color=CC[c], alpha=0.88, lw=0, zorder=2)
        ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt='none', ecolor=INK, elinewidth=0.8,
                    capsize=2.5, capthick=0.8, zorder=3)
        ax.text(i, -2.4, f'{LAB[c]}\nn = {n}', ha='center', va='top', fontsize=6.6, color=INK,
                linespacing=1.45)
    yb = max(his) + 5
    ax.plot([0, 2], [yb, yb], color=INK, lw=0.7, clip_on=False)
    ax.text(1, yb + 1.4, f'trend  $P$ = {_p10(p_trend)}', ha='center', va='bottom',
            fontsize=6.6, color=INK)
    ax.text(1, yb + 0.2, f'Cochran–Armitage  {_p10(p_ca)}', ha='center', va='top',
            fontsize=5.4, color='0.45')
    ax.set_xticks([])
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(0, 72)
    ax.set_yticks([0, 20, 40, 60])
    spines(ax)
    ax.tick_params(colors=INK, labelsize=6.6, width=0.7, length=3)
    ax.set_ylabel('Pairs with bQTL on partner motif (%)', fontsize=7.2, color=INK)
    fig.tight_layout()
    save(fig, 'fig5_onmotif_frequency', SUB)
    print('  on-motif frequency: ' +
          '  '.join(f'{LAB[c]} {100 * ct[c][0] / ct[c][1]:.1f}% ({ct[c][0]}/{ct[c][1]})'
                    for c in ORDER) +
          f'  |  trend P={p_trend:.1e}, C-A P={p_ca:.1e}')


def motif_effect():
    """Allele-specific target binding scales with the variant's effect on the partner-motif score.

    Each of the 283 on-motif bQTLs contributes a motif-score change between alleles (x) and a signed
    allele-specific target-binding effect (y), both oriented reference-minus-alternative. Points are
    binned by motif-score change; each shows the mean binding effect with its SEM. A variant that
    lowers the motif score (moving right) is associated with weaker binding on the same allele, so
    the trend runs from upper-left to lower-right.
    """
    S = with_route(pd.read_csv(data('fig5', 'bqtl_snp_coords.csv')))
    S['bind'] = np.where(S.fdr_ref < S.fdr_alt, S.es_ref, -S.es_alt)
    BINS = [-12, -2, -0.5, 0.5, 2, 8, 40]
    TL = ['≤ −2', '−2 to\n−0.5', '−0.5 to\n0.5', '0.5 to\n2', '2 to\n8', '> 8']
    S['bin'] = pd.cut(S.motif_delta, BINS, labels=range(len(TL)))
    g = (S.groupby('bin', observed=True)
           .agg(n=('bind', 'size'), mean=('bind', 'mean'), sem=('bind', 'sem'))
           .reset_index())

    FS = 7
    fig, ax = plt.subplots(figsize=(3.9, 3.3))
    ax.axhline(0, color='0.80', lw=0.8, zorder=1)
    x = np.arange(len(TL))
    ax.plot(x, g['mean'], '-', color='black', lw=1.6, zorder=3)
    ax.errorbar(x, g['mean'], yerr=g['sem'], fmt='o', ms=6, color='black', ecolor='black',
                elinewidth=1.0, capsize=2.5, mfc='white', mec='black', mew=1.4, zorder=4)
    for xi, r in zip(x, g.itertuples()):
        ax.text(xi, r.mean + r.sem + 0.09, f'n={r.n}', ha='center', va='bottom',
                fontsize=FS, color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(TL, fontsize=FS)
    ax.set_xlim(-0.5, len(TL) - 0.5)
    ax.set_ylim(-1.7, 1.7)
    spines(ax)
    ax.tick_params(colors='black', labelsize=FS, width=0.7, length=3)
    ax.set_xlabel('motif-score change  (ref − alt allele)', fontsize=FS, color='black')
    ax.set_ylabel('mean allele-specific target binding\n(+ = alt allele decreases binding)',
                  fontsize=FS, color='black', linespacing=1.3)
    ax.text(len(TL) - 0.55, -1.62, 'alt allele lowers the motif score →', fontsize=FS,
            color='black', ha='right', va='bottom', clip_on=False)
    fig.tight_layout()
    save(fig, 'fig5_motif_effect', SUB)
    print(f'  motif effect: {len(S)} on-motif bQTLs, {len(g)} bins '
          f'({", ".join(f"n={int(r.n)}" for r in g.itertuples())})')


PANELS = [onmotif_frequency, motif_effect]


def main():
    print('Figure 5')
    for fn in PANELS:
        fn()


if __name__ == '__main__':
    main()
