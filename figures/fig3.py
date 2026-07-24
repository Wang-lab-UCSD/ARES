#!/usr/bin/env python
"""Figure 3 — is the assigned route supported by orthogonal measurements?

Four independent lines of evidence, none of which ARES sees when it makes the call:

  mpra                MPRA: does the partner motif change reporter activity?
  phylop              evolutionary conservation of the partner-motif site
  ag_target_binding   AlphaGenome: scrambling the partner motif in silico, how much target binding
                      is lost, split by route
  capselex            CAP-SELEX: do the two proteins bind DNA cooperatively in vitro?

Every table in data/fig3/ is pre-filtered to exactly the rows the panels use, and its route labels
are already synchronised with the atlas, so the code here reads and plots without further filtering.

Run as:  python fig3.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

from _common import CC, INK, MODES, data, pbody, save, spines

SUB = 'fig3'
ORDER = ['SEQUENCE', 'CONTEXT', 'PROTEIN']


def _raincloud(ax, groups, col, xlabel, zeroline=False, seed=3, sfs=7.5):
    """Half-violin + jittered points + median bar, one row per route.

    A raincloud rather than a box plot because the routes differ in spread as well as in centre,
    and n varies several-fold between them; showing every point keeps that visible.
    """
    rng = np.random.default_rng(seed)
    vw = 0.40
    for i, c in enumerate(ORDER):
        v = groups[c]
        colr = CC[c]
        xs = np.linspace(v.min() - 0.2, v.max() + 0.2, 200)
        dens = stats.gaussian_kde(v)(xs)
        dens = dens / dens.max() * vw
        ax.fill_between(xs, i, i + dens, color=colr, alpha=0.55, lw=0, zorder=2)
        ax.plot(xs, i + dens, color=colr, lw=0.8, zorder=2)
        ax.scatter(v, i - 0.07 - rng.uniform(0, 0.30, len(v)), s=4, color=colr,
                   alpha=0.32, edgecolors='none', zorder=3)
        mm = np.median(v)
        ax.plot([mm, mm], [i - 0.40, i + dens[np.argmin(abs(xs - mm))]],
                color='0.15', lw=1.0, zorder=4)
        ax.text(mm, i + vw + 0.05, f'{mm:+.2f}' if zeroline else f'{mm:.2f}',
                ha='center', va='bottom', fontsize=6.5, color=colr, fontweight='bold')
    if zeroline:
        ax.axvline(0, color='0.6', lw=0.7, ls='--', zorder=0)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f'{c.capitalize()}\n(n={len(groups[c])})' for c in ORDER],
                       fontsize=sfs, color=INK)
    ax.set_ylim(-0.55, 2.8)
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    spines(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)


def _brackets(ax, groups, key='PROTEIN'):
    """Nested right-hand brackets comparing one route against the other two."""
    others = [c for c in ORDER if c != key]
    ps = [stats.mannwhitneyu(groups[key], groups[o])[1] for o in others]
    idx = {c: i for i, c in enumerate(ORDER)}
    x0, x1 = ax.get_xlim()
    xr = x1 - x0
    ax.set_xlim(x0, x1 + 0.34 * xr)
    spans = sorted(zip(others, ps), key=lambda t: -abs(idx[t[0]] - idx[key]))
    for (o, p), off in zip(spans, (0.19, 0.07)):
        y0, y1 = sorted((idx[o], idx[key]))
        xb = x1 + off * xr
        ax.plot([xb, xb], [y0 + 0.12, y1 - 0.12], color=INK, lw=0.8, clip_on=False)
        for yy in (y0 + 0.12, y1 - 0.12):
            ax.plot([xb - 0.025 * xr, xb], [yy, yy], color=INK, lw=0.8, clip_on=False)
        ax.text(xb + 0.015 * xr, (y0 + y1) / 2, f'$P$ = {pbody(p)}', ha='center', va='center',
                fontsize=6, color=INK, rotation=270, clip_on=False)
    return dict(zip(others, ps))


def mpra():
    """Fraction of dependencies where disrupting the partner motif changes reporter activity.

    The test is a locus-level paired Wilcoxon per parent promoter: PARM tiles each locus into about
    124 correlated fragments, so a fragment-level shuffle null is anti-conservative and was
    abandoned. The dashed line is the 5% expected under the null.
    """
    TEAL, REF = '#8a9ca1', '#d3b3af'
    CELLS = ['HepG2', 'K562', 'HEK293', 'MCF.7']
    d = pd.read_csv(data('fig3', 'mpra_locus_results.csv'))
    d['sig'] = d.q_paired_locus < 0.05
    rows = [dict(cell=c, n=int((d.cell == c).sum()), pct=100 * d[d.cell == c].sig.mean())
            for c in CELLS]
    nsig, psig = int(d.sig.sum()), 100 * d.sig.mean()
    pdir = 100 * d[d.sig].direction_match.mean()

    FS = 7.5
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    x = np.arange(len(rows))
    tx = ax.get_xaxis_transform()
    ax.bar(x, [r['pct'] for r in rows], width=0.62, color=TEAL, alpha=0.9,
           edgecolor='none', zorder=3)
    for xi, r in zip(x, rows):
        ax.text(xi, r['pct'] + 1.0, f"{r['pct']:.0f}%", ha='center', va='bottom',
                fontsize=FS, color=INK)
        ax.text(xi, -0.155, f"n = {r['n']}", transform=tx, ha='center', va='top',
                fontsize=6.0, color='0.5')
    ax.axhline(5, color=REF, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(len(rows) - 0.45, 5.8, '5% expected', ha='right', va='bottom',
            fontsize=6.0, color='#a98f8b')
    ax.set_xticks(x)
    ax.set_xticklabels([r['cell'] for r in rows], fontsize=FS, color=INK)
    ax.set_ylabel('Significant motif effect (%)', fontsize=FS, color=INK)
    ax.set_ylim(0, 50)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    spines(ax)
    ax.tick_params(axis='y', labelsize=FS)
    ax.text(0.015, 0.985, f'{nsig:,}/{len(d):,} significant pairs ({psig:.0f}%)',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.0, color='0.55')
    fig.tight_layout()
    save(fig, 'fig3_mpra', SUB)
    print(f'  MPRA: {nsig}/{len(d)} significant ({psig:.0f}%), '
          f'direction-concordant among significant {pdir:.0f}%')


def phylop():
    """Evolutionary conservation of the partner-motif site, by route.

    phyloP is measured at the partner-motif positions inside the target's peaks. It is an entirely
    external measurement: nothing about conservation enters the ARES call.
    """
    c = pd.read_csv(data('fig3', 'conservation_by_mechanism_final.csv'))
    c['consensus'] = c.dichotomy if 'dichotomy' in c.columns else c['consensus']
    groups = {m: c[c.consensus == m].phylop_partner.dropna().values for m in ORDER}

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    _raincloud(ax, groups, 'phylop_partner', 'mean phyloP at partner-motif sites')
    ps = _brackets(ax, groups)
    fig.tight_layout()
    save(fig, 'fig3_phylop', SUB)
    print(f'  phyloP: ' + '  '.join(f'{m} n={len(groups[m])} med={np.median(groups[m]):.2f}'
                                    for m in ORDER) +
          f'  |  Prot vs Seq P={ps["SEQUENCE"]:.2g}  Prot vs Ctx P={ps["CONTEXT"]:.2g}')


def ag_target_binding():
    """AlphaGenome: how much target binding is lost when the partner motif is scrambled.

    Drawn per cell line for the two best-covered cell lines and then pooled, because a single
    pooled panel could be carried by whichever cell line contributes most pairs. One-sided
    Mann-Whitney tests Protein above each other route within each facet.
    """
    g = pd.read_csv(data('fig3', 'ag_effect_by_mechanism_allcells.csv'))
    SH = {'SEQUENCE': 'Seq', 'PROTEIN': 'Prot', 'CONTEXT': 'Ctx'}
    facets = ['HepG2', 'K562', 'All cells']
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(facets), figsize=(2.1 * len(facets), 2.6), sharey=True)
    for ax, fc in zip(np.atleast_1d(axes), facets):
        sub = g if fc == 'All cells' else g[g.cell == fc]
        for i, c in enumerate(MODES):
            v = sub[sub.consensus == c].abs_paired.dropna().values
            if not len(v):
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.add_patch(plt.Rectangle((i - 0.28, q1), 0.56, q3 - q1, fc=CC[c], ec='none',
                                       alpha=0.30, zorder=1))
            ax.hlines(med, i - 0.30, i + 0.30, color=CC[c], lw=2, zorder=4)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=CC[c],
                       alpha=0.55, lw=0, zorder=3)
        P = sub[sub.consensus == 'PROTEIN'].abs_paired
        S = sub[sub.consensus == 'SEQUENCE'].abs_paired
        C = sub[sub.consensus == 'CONTEXT'].abs_paired

        def bar(x0, x1, y, p):
            ax.plot([x0, x0, x1, x1], [y * 0.88, y, y, y * 0.88], color=INK, lw=0.8,
                    clip_on=False)
            ax.text((x0 + x1) / 2, y * 1.04, f'$P$ = {pbody(p)}', ha='center', va='bottom',
                    fontsize=6.0, color=INK, clip_on=False)

        if len(P) >= 4 and len(S) >= 4:
            bar(0, 1, 4.6, stats.mannwhitneyu(P, S, alternative='greater')[1])
        if len(P) >= 4 and len(C) >= 4:
            bar(1, 2, 11.0, stats.mannwhitneyu(P, C, alternative='greater')[1])
        ax.set_yscale('log')
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(2e-2, 16)
        spines(ax)
        ax.set_xticks(range(3))
        ax.set_xticklabels([SH[c] for c in MODES], fontsize=8.4, color=INK)
        ax.tick_params(axis='y', labelsize=8.4)
        ax.set_title(fc, fontsize=8.4, color=INK, pad=12)
        ax.text(0.96, 0.04, f'n={len(sub)}', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.4, color=INK)
    np.atleast_1d(axes)[0].set_ylabel('|Δ target binding|\n(AlphaGenome, log)',
                                      fontsize=8.4, color=INK)
    fig.tight_layout()
    save(fig, 'fig3_ag_target_binding', SUB)
    for fc in facets:
        sub = g if fc == 'All cells' else g[g.cell == fc]
        med = {c: np.median(sub[sub.consensus == c].abs_paired.dropna()) for c in MODES}
        print(f'  AG {fc}: n={len(sub)}  ' + '  '.join(f'{c}={med[c]:.3f}' for c in MODES))


def capselex():
    """CAP-SELEX: fraction of pairs that bind DNA cooperatively in vitro.

    In vitro cooperativity is measured on naked DNA with purified protein, so it is independent of
    anything ARES observes in cells. Broken out by mechanism leaf within Protein, with Sequence and
    Context as single rows. Bars are Wilson 95% intervals, which stay inside [0, 1] at these small
    denominators where a normal approximation would not.
    """
    iv = pd.read_csv(data('fig3', 'invitro_inscope_pairs_annotated.tsv'), sep='\t')
    iv['co'] = iv.coop.astype(str).str.lower().isin(['true', '1', 'yes'])

    def shade(base, f):
        b = mpl.colors.to_rgb(base)
        return tuple(b[i] + (1 - b[i]) * f for i in range(3))

    def wilson(k, n, z=1.96):
        if n == 0:
            return 0., 0., 0.
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)

    rows = [('Cooperative co-binding', ('mechanism', 'COOPERATIVE_COBINDING'), CC['PROTEIN']),
            ('Tethering', ('mechanism', 'TETHERING'), shade(CC['PROTEIN'], 0.5)),
            ('Co-occupancy', ('mechanism', 'CO_OCCUPANCY'), shade(CC['PROTEIN'], 0.60)),
            ('Sequence', ('dichotomy', 'SEQUENCE'), CC['SEQUENCE']),
            ('Context', ('dichotomy', 'CONTEXT'), CC['CONTEXT'])]
    g = lambda col, val: (int(iv[iv[col] == val].co.sum()), len(iv[iv[col] == val]))

    CFS = 8.4
    fig, ax = plt.subplots(figsize=(4.7, 3.1))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, (col, val), c) in zip(y, rows):
        k, n = g(col, val)
        p, lo, hi = wilson(k, n)
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.7, solid_capstyle='round', zorder=2)
        ax.scatter(p, yi, s=34, color=c, zorder=3, ec='white', lw=0.7)
        ax.text(103, yi, f'{p:.0f}%', va='center', ha='left', fontsize=CFS, color=c,
                fontweight='bold')
        ax.text(118, yi, f'{k}/{n}', va='center', ha='left', fontsize=CFS, color='0.5')
    ax.axhline(y[2] - 0.5, color='0.85', lw=0.7, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=CFS, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Cooperative CAP-SELEX pairs (%)', fontsize=CFS, color=INK)
    ax.tick_params(axis='x', labelsize=CFS)
    spines(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)

    P = iv[iv.dichotomy == 'PROTEIN'].co
    SC = iv[iv.dichotomy.isin(['SEQUENCE', 'CONTEXT'])].co
    pf = stats.fisher_exact([[int(P.sum()), len(P) - int(P.sum())],
                             [int(SC.sum()), len(SC) - int(SC.sum())]])[1]

    def bracket(y0, y1, t, col):
        x = 131
        ax.plot([x, x + 2, x + 2, x], [y0, y0, y1, y1], color=col, lw=0.9, clip_on=False)
        ax.text(x + 5, (y0 + y1) / 2, t, va='center', ha='left', fontsize=CFS,
                fontweight='bold', color=col, rotation=270, clip_on=False)

    bracket(y[0] + 0.4, y[2] - 0.4, f'PROTEIN  {100 * P.mean():.0f}%', CC['PROTEIN'])
    bracket(y[3] + 0.4, y[4] - 0.4, f'Seq+Ctx  {100 * SC.mean():.0f}%', '0.45')
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.text(0, 1.0, f'Protein vs Seq+Ctx: $P$ = {pbody(pf)}', transform=ax.transAxes,
            ha='left', fontsize=6.8, color='0.45')
    fig.tight_layout()
    save(fig, 'fig3_capselex', SUB)
    print(f'  CAP-SELEX: ' +
          '  '.join(f'{lab.split()[0]} {g(col, val)[0]}/{g(col, val)[1]}'
                    for lab, (col, val), _ in rows) +
          f'  |  Protein vs Seq+Ctx P={pf:.1e}')


PANELS = [mpra, phylop, ag_target_binding, capselex]


def main():
    print('Figure 3')
    for fn in PANELS:
        fn()


if __name__ == '__main__':
    main()
