#!/usr/bin/env python
"""Figure 1 — cooperation mode.

Three panels, each reading one processed table from the data archive:

  route distribution         how the 1,552 dependencies split across cooperation modes
  partner-dependence         distribution of the motif x partner-protein interaction coefficient
  motif SHAP rank            where the ML-selected partner motif ranks, against the target's own
                                motif and its closest paralog's

Run as:  python fig1.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from _common import CC, INK, data, dk, save, spines

SUB = 'fig1'


def route_distribution():
    """Composition of the atlas: three cooperation modes, then the two non-call categories.

    Unresolved and Artifact are drawn in neutral greys rather than a mode colour, because they are
    not modes; keeping them on the same axis makes the denominator explicit instead of hiding the
    calls the pipeline declined to make.
    """
    a = pd.read_csv(data('ares_atlas.tsv'), sep='\t')
    order = ['CONTEXT', 'PROTEIN', 'SEQUENCE', 'UNRESOLVED', 'ARTIFACT']
    vc = a.dichotomy.value_counts()
    COL = dict(CC, UNRESOLVED='#a9a8ab', ARTIFACT='#dcd8d4')
    AL = {'CONTEXT': 0.82, 'PROTEIN': 0.82, 'SEQUENCE': 0.82, 'UNRESOLVED': 1.0, 'ARTIFACT': 1.0}

    fig, ax = plt.subplots(figsize=(2.9, 2.5))
    y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        ax.barh(yi, vc[c], color=COL[c], height=0.68, alpha=AL[c], edgecolor='none')
        ax.text(vc[c] + 8, yi, f'{vc[c]} ({100 * vc[c] / len(a):.0f}%)',
                va='center', fontsize=6.1, color='0.32')
    ax.set_yticks(y)
    ax.set_yticklabels([c.capitalize() for c in order], fontsize=6.6, color=INK)
    ax.set_xlim(0, 615)
    ax.set_xticks([0, 200, 400])
    ax.set_xlabel('Dependencies (n)', fontsize=6.6, color=INK)
    ax.tick_params(axis='x', labelsize=6.2, width=0.6, color='#8a8a8a', labelcolor=INK)
    ax.tick_params(axis='y', length=0)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#8a8a8a')
        ax.spines[s].set_linewidth(0.6)
    fig.tight_layout()
    save(fig, 'fig1_route_distribution', SUB)
    print(f'  route distribution: n={len(a)}  ' +
          '  '.join(f'{c}={vc[c]}' for c in order))


def partner_dependence():
    """Distribution of the partner-dependence coefficient across every partner-ChIP-covered pair.

    The coefficient is the motif x partner-protein interaction term in the standardized model
    z(target occupancy) ~ z(motif score) + protein_bound + z(motif):protein_bound. Negative means
    the partner weakens the motif's effect on target binding, positive that it amplifies it.

    Bars are a single neutral grey: the sign is already carried by position relative to the dashed
    zero line and by the two directional labels, so colouring the bars as well encoded the same
    thing three times and made a continuous spectrum look like two discrete classes.
    """
    d = pd.read_csv(data('partner_dependence_coefficient.csv'))
    v = d.bMP_std.values.astype(float)
    n = len(v)

    BARc, KDEc = '#C9C9C9', '#707070'
    LO, HI, binw = -0.65, 0.65, 0.04
    bins = np.arange(LO, HI + binw, binw)
    cnt, edges = np.histogram(np.clip(v, LO, HI), bins=bins)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar(edges[:-1], cnt, width=binw, align='edge',
           color=BARc, edgecolor='white', linewidth=0.5, zorder=3)
    xs = np.linspace(LO, HI, 400)
    ax.plot(xs, stats.gaussian_kde(v)(xs) * n * binw, color=KDEc, lw=1.0, alpha=0.9, zorder=5)
    ax.axvline(0, color='#c6c6c6', lw=0.6, ls=(0, (3, 2)), zorder=4)

    ax.set_xlim(LO, HI)
    ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.margins(y=0)
    ax.set_ylim(0, cnt.max() * 1.16)
    ax.set_xlabel('Partner-dependence coefficient', fontsize=8, color=INK, labelpad=2)
    ax.set_ylabel('Dependencies', fontsize=8, color=INK, labelpad=2)
    ax.text(0.015, 0.975, 'partner weakens target binding', transform=ax.transAxes,
            ha='left', va='top', fontsize=5.5, color='#bb928d')
    ax.text(0.985, 0.975, 'partner amplifies target binding', transform=ax.transAxes,
            ha='right', va='top', fontsize=5.5, color='#88a784')
    ax.text(0.015, 0.86, f'$n$ = {n}', transform=ax.transAxes,
            ha='left', va='top', fontsize=5.8, color='#4D4D4D')
    spines(ax)
    ax.tick_params(colors='#4D4D4D', labelsize=6.5, width=0.7, length=2.8)
    fig.tight_layout()
    save(fig, 'fig1_partner_dependence', SUB)
    print(f'  partner dependence: n={n}  median={np.median(v):+.3f}  '
          f'%weaken={(v < 0).mean():.0%}  range[{v.min():.2f},{v.max():.2f}]')


def shap_rank():
    """Where the ML-selected partner motif ranks, against the target's own motif and its paralog's.

    Restricted to targets that have a JASPAR motif of their own, since a self rank cannot be
    computed otherwise; that is what sets the denominator well below the full atlas.
    """
    d = pd.read_csv(data('fig1', 'rf_shap_full_summary.csv'))
    d = d[d.self_rank.notna()].copy()
    d['homolog'] = d[['self_rank', 'best_paralog_rank']].min(axis=1)
    n = len(d)

    C_SELF, C_PART = '#4a6f86', '#d7a25c'
    BINS = ['1', '2–10', '11–50', '51–100', '101–250', '251–841']

    def rbin(r):
        r = int(r)
        return ('1' if r == 1 else '2–10' if r <= 10 else '11–50' if r <= 50 else
                '51–100' if r <= 100 else '101–250' if r <= 250 else '251–841')

    hf = d.homolog.apply(rbin).value_counts().reindex(BINS).fillna(0) / n * 100
    pf = d.regulator_rank.apply(rbin).value_counts().reindex(BINS).fillna(0) / n * 100

    def round100(vals):
        """Largest-remainder rounding so the printed labels sum to 100."""
        vals = np.asarray(vals, float)
        fl = np.floor(vals).astype(int)
        deficit = int(round(vals.sum())) - int(fl.sum())
        out = fl.copy()
        for i in np.argsort(-(vals - fl))[:deficit]:
            out[i] += 1
        return out

    hf_lab, pf_lab = round100(hf.values), round100(pf.values)
    x = np.arange(len(BINS))
    w = 0.4

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    b1 = ax.bar(x - w / 2, hf.values, w, color=C_SELF, label='Target self / paralog',
                edgecolor='white', linewidth=0.4)
    b2 = ax.bar(x + w / 2, pf.values, w, color=C_PART, label='Selected partner',
                edgecolor='white', linewidth=0.4)
    for bars, vals, labs in [(b1, hf.values, hf_lab), (b2, pf.values, pf_lab)]:
        for bb, v, lab in zip(bars, vals, labs):
            if lab >= 1:
                ax.text(bb.get_x() + bb.get_width() / 2, v + 0.9, f'{lab}',
                        ha='center', va='bottom', fontsize=6.0, color='0.4')
    ax.set_xticks(x)
    ax.set_xticklabels(BINS, ha='center')
    ax.set_xlabel('Motif SHAP rank', fontsize=7)
    ax.set_ylabel('ChIP-seq experiments (%)', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0, 64)
    ax.set_yticks([0, 20, 40, 60])
    ax.legend(loc='upper right', handlelength=1.0, handletextpad=0.4,
              labelspacing=0.3, fontsize=6.2)
    spines(ax)
    fig.tight_layout(pad=0.4)
    save(fig, 'fig1_shap_rank', SUB)
    print(f'  SHAP rank: n={n}  self/paralog={dict(zip(BINS, hf_lab))}  '
          f'partner={dict(zip(BINS, pf_lab))}')


PANELS = [route_distribution, partner_dependence, shap_rank]


def main():
    print('Figure 1')
    for fn in PANELS:
        fn()


if __name__ == '__main__':
    main()
