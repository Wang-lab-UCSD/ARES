#!/usr/bin/env python
"""Figure 4 — the reader.

Nine panels. The claim of the figure is that the factor which reads a partner motif is often not the
transcription factor the motif is named after, and that this ARES-inferred reader is the functional
unit: it is the factor present in the cell, centred on the motif, tracking the target where the
namesake is absent, and the one whose perturbation moves the target.

  namesake_expression      is the motif's namesake TF even expressed in that cell type?
  reader_assignments       who reads the motif across the 1,146 route-resolved dependencies
  reader_vs_namesake_tpm   reader stays expressed across the whole TPM sweep; the namesake does not
  centering                the reader is centred on the partner motif, against matched random TFs
  reader_tracks_target     the hidden reader tracks the target where the namesake does not bind
  gata1_dose               knocking down the reader GATA1 costs the target, dose-dependently
  scramble_binding         AlphaGenome: scrambling the motif removes the reader, not control TFs
  scramble_cage            AlphaGenome: transcription then follows the reader's valence
  reader_odds              logistic odds ratios: outcome is reader-specific, region is not
  target_reader_concordance  Perturb-seq: of the factors on the same peaks, only the named reader's
                           knockdown reproduces the target's own

Run as:  python fig4.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from _common import AXG, CC, INK, MODES, atlas, data, desat, dk, pbody, save, spines

SUB = 'fig4'


def _pf(p):
    """Scientific mantissa x 10^e for small P. Keeps one decimal on the mantissa unless it rounds
    cleanly to an integer, so 8.68e-33 shows as 8.7, not 9.

    The switch to scientific notation is at 0.01 rather than 0.001, because three decimal places
    leave only one significant figure just below 0.01: a permutation P of 0.0044 would print as
    0.004, which reads as 4e-3 and no longer matches the value in the source table.
    """
    if p >= 1e-2:
        return f'{p:.3f}'
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    mant = f'{m:.0f}' if abs(m - round(m)) < 0.05 else f'{m:.1f}'
    return rf'{mant} $\times$ 10$^{{{e}}}$'


def namesake_expression():
    """Is the motif's namesake TF even present in the cell where the motif was identified?

    Rank-ordered dot plot over the unique (cell, namesake motif) units. Collapsed to (cell, motif)
    because single motifs are heavily reused across dependencies, so per-dependency counting would
    be pseudo-replicated. The 0.5 TPM threshold is ARES's own artifact-check cut; the shaded run
    below it has a length the reader can measure off the x-axis.
    """
    d = pd.read_csv(data('fig4', 'namesake_expression_full_atlas.csv')).dropna(subset=['ns_tpm'])
    C = (d.groupby(['cell', 'namesake']).agg(tpm=('ns_tpm', 'first'), n_dep=('target', 'size'))
           .reset_index().sort_values(['tpm', 'namesake'], kind='mergesort').reset_index(drop=True))
    C['rank'] = np.arange(1, len(C) + 1)
    THR, PC = 0.5, 0.01
    n, nun = len(C), int((C.tpm < THR).sum())
    pct = 100 * nun / n
    y = C.tpm.values + PC
    below = C.tpm.values < THR

    cLo, cHi = desat('#c0392b', 0.55), '#6b7b8c'
    fig, ax = plt.subplots(figsize=(3.94, 2.58))
    ax.set_yscale('log')
    ax.axhspan(1e-3, THR + PC, color=cLo, alpha=0.055, lw=0, zorder=0)
    ax.plot(C['rank'], y, color=AXG, lw=0.5, alpha=0.55, zorder=1, solid_capstyle='round')
    ax.scatter(C['rank'][below], y[below], s=4.2, color=cLo, lw=0, zorder=3)
    ax.scatter(C['rank'][~below], y[~below], s=4.2, color=cHi, lw=0, zorder=3)
    ax.axhline(THR + PC, color=cLo, lw=0.8, ls=(0, (3.2, 2.2)), zorder=4)
    ax.text(n * 0.995, THR + PC, f'TPM = {THR}', ha='right', va='bottom', fontsize=6.0,
            color=cLo, zorder=5)
    ax.text(n * 0.615, 0.200, f'{pct:.1f}%', ha='center', va='center', fontsize=12.5,
            color=cLo, fontweight='bold', zorder=6)
    ax.text(n * 0.615, 0.105, f'{nun}/{n} partner-motif TFs\nnot expressed in that cell line',
            ha='center', va='top', fontsize=6.1, color=cLo, zorder=6, linespacing=1.3)
    yb = 10 ** (np.log10(PC * 0.62))
    ax.plot([1, 1, nun, nun], [yb * 1.30, yb, yb, yb * 1.30], color=cLo, lw=0.7, clip_on=False,
            zorder=6)
    ax.set_xlim(-n * 0.02, n * 1.02)
    ax.set_ylim(PC * 0.5, C.tpm.max() * 2)
    ax.set_xlabel('Partner-motif TF, ranked by expression', fontsize=7, color=INK)
    ax.set_ylabel('Expression (TPM)', fontsize=7, color=INK)
    spines(ax)
    ax.tick_params(labelsize=6.2, colors=INK)
    fig.tight_layout()
    save(fig, 'fig4_namesake_expression', SUB)
    print(f'  namesake expression: {nun}/{n} ({pct:.1f}%) below {THR} TPM')


def reader_assignments():
    """Who reads the motif across the 1,146 route-resolved dependencies.

    A single stacked bar of the four reader categories from the curated atlas: the motif's namesake
    (the naive call), the target reading its own site, a hidden third factor, or unassigned. The
    bracket sums the two categories where the reader is NOT the namesake.
    """
    r = atlas()
    r = r[r.consensus.isin(MODES)].copy()
    n_res = len(r)
    CATMAP = {'partner': 'motif owner (naive call)', 'self': 'target reads it directly',
              'hidden': 'hidden 3rd factor', 'none': 'unassigned'}
    r['rdcat'] = r.reader_cat.map(CATMAP)
    order = ['motif owner (naive call)', 'target reads it directly', 'hidden 3rd factor',
             'unassigned']
    COLR = {'motif owner (naive call)': '0.62', 'target reads it directly': CC['SEQUENCE'],
            'hidden 3rd factor': desat('#e67e22'), 'unassigned': '0.87'}
    LAB = {'motif owner (naive call)': 'Motif\nnamesake', 'target reads it directly': 'Target\nfactor',
           'hidden 3rd factor': 'Third-factor\nreader', 'unassigned': 'No assigned\nreader'}

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    left = 0
    segs = {}
    for c in order:
        v = 100 * (r.rdcat == c).mean()
        segs[c] = (left, v)
        ax.barh(0, v, left=left, color=COLR[c], edgecolor='white', height=0.40)
        ax.text(left + v / 2, 0, f'{v:.0f}%', ha='center', va='center', fontsize=10.5,
                color='white' if c != 'unassigned' else '0.35', fontweight='bold')
        ax.text(left + v / 2, -0.32, LAB[c], ha='center', va='top', fontsize=9.0, color=INK,
                linespacing=1.1)
        left += v
    a0 = segs['target reads it directly'][0]
    b0 = segs['hidden 3rd factor'][0] + segs['hidden 3rd factor'][1]
    yb = 0.32
    ax.plot([a0, b0], [yb, yb], color='0.5', lw=0.8)
    ax.plot([a0, a0], [yb - 0.06, yb], color='0.5', lw=0.8)
    ax.plot([b0, b0], [yb - 0.06, yb], color='0.5', lw=0.8)
    ax.text((a0 + b0) / 2, yb + 0.07, f'Non-namesake reader: {b0 - a0:.0f}%', ha='center',
            va='bottom', fontsize=9.5, fontweight='bold', color='0.15')
    ax.text(0, -0.66, f'$n$ = {n_res:,} route-resolved dependencies', ha='left', va='top',
            fontsize=7.5, color='0.45', fontstyle='italic')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.80, 0.56)
    ax.axis('off')
    fig.tight_layout()
    save(fig, 'fig4_reader_assignments', SUB)
    print(f'  reader assignments (n={n_res}): ' +
          '  '.join(f'{c.split()[0]}={100 * (r.rdcat == c).mean():.0f}%' for c in order))


def reader_vs_namesake_tpm():
    """Fraction expressed across the TPM sweep: assigned reader vs namesake.

    Counted per (cell, TF) unit, not per dependency, so a factor recurring across many dependencies
    is counted once per cell line. Brown, the assigned reader pooled across the three reader
    categories; grey, the namesake in dependencies where a different factor reads the motif.
    """
    D = pd.read_csv(data('fig4', 'reader_vs_namesake_expression.csv'))
    RDc, NSc = desat('#e67e22'), '#8f8f8f'
    A = D[D.reader_name.notna()].dropna(subset=['reader_tpm']).drop_duplicates(['cell', 'reader_name'])
    B = D[D.cat.isin(['self', 'hidden'])].dropna(subset=['ns_tpm']).drop_duplicates(['cell', 'namesake'])
    THR = np.logspace(np.log10(0.1), np.log10(50), 60)
    pa = np.array([100 * (A.reader_tpm >= t).mean() for t in THR])
    pb = np.array([100 * (B.ns_tpm >= t).mean() for t in THR])

    fig, ax = plt.subplots(figsize=(3.7, 3.0))
    ax.set_xscale('log')
    ax.plot(THR, pa, color=dk(RDc), lw=2.2, zorder=4, solid_capstyle='round')
    ax.plot(THR, pb, color=dk(NSc), lw=2.2, zorder=4, solid_capstyle='round')
    ax.axvline(0.5, color='0.85', lw=0.7, ls=(0, (3, 3)), zorder=1)
    for col, series in [(RDc, A.reader_tpm), (NSc, B.ns_tpm)]:
        ax.scatter([0.5], [100 * (series >= 0.5).mean()], s=16, color=dk(col), zorder=5)
    ax.annotate('TPM 0.5', (0.5, 2), xytext=(0.56, 6), fontsize=5.6, color='0.45', ha='left')
    ax.annotate(f'assigned reader ($n$ = {len(A)})', (2.0, np.interp(2.0, THR, pa)),
                xytext=(0, 12), textcoords='offset points', fontsize=6.6, color=dk(RDc),
                ha='center', va='bottom', fontweight='semibold')
    ax.annotate(f'namesake, where something else\nreads its motif ($n$ = {len(B)})',
                (1.4, np.interp(1.4, THR, pb)), xytext=(0, -30), textcoords='offset points',
                fontsize=6.6, color=dk(NSc), ha='center', va='top', fontweight='semibold',
                linespacing=1.3,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=dk(NSc), shrinkA=2, shrinkB=2))
    ax.set_xlim(0.1, 50)
    ax.set_ylim(0, 103)
    ax.set_xticks([0.1, 0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(['0.1', '0.5', '1', '2', '5', '10', '20', '50'])
    ax.set_xlabel('Expression threshold (TPM)', fontsize=7, color='black')
    ax.set_ylabel('TF $\\times$ cell-line units expressed (%)', fontsize=7, color='black')
    spines(ax)
    ax.tick_params(colors='black', labelsize=6.4)
    fig.tight_layout()
    save(fig, 'fig4_reader_vs_namesake_tpm', SUB)
    print(f'  reader vs namesake TPM: reader n={len(A)}, namesake n={len(B)}; '
          f'at 0.5 TPM {100 * (A.reader_tpm >= 0.5).mean():.0f}% vs '
          f'{100 * (B.ns_tpm >= 0.5).mean():.0f}%')


def centering():
    """The reader is centred on the partner motif, against matched random TFs.

    Reads pre-computed profiles: reader and matched-null median with IQR, every profile normalised
    to its own flank so the y-axis is fold-over-flank (scale-free, immune to antibody and depth
    differences). Statistics (median centre-to-flank ratio, P) are pre-computed too.
    """
    prof = pd.read_csv(data('fig4', 'centering_profiles.csv'))
    st = pd.read_csv(data('fig4', 'centering_stats.csv')).iloc[0]
    x = prof.pos_bp.values
    RED, GREY, AX = '#c0392b', '#5f6266', '#b0b3b6'
    FS = 6.6

    fig, ax = plt.subplots(figsize=(3.9, 3.15))
    ax.axvspan(-75, 75, color=RED, alpha=0.040, lw=0, zorder=0)
    for a, b in [(-500, -400), (400, 500)]:
        ax.axvspan(a, b, color='0.5', alpha=0.030, lw=0, zorder=0)
    ax.axhline(1, color=AX, lw=0.55, ls=(0, (2.5, 2.5)), zorder=1)
    ax.axvline(0, color=AX, lw=0.55, ls=(0, (2.5, 2.5)), zorder=1)
    ax.fill_between(x, prof.reader_p25, prof.reader_p75, color=RED, alpha=0.10, lw=0, zorder=3)
    ax.fill_between(x, prof.null_p25, prof.null_p75, color=GREY, alpha=0.30, lw=0, zorder=4)
    ax.plot(x, prof.null_median, color=GREY, lw=3.0, zorder=5, solid_capstyle='round')
    ax.plot(x, prof.reader_median, color=RED, lw=4.0, zorder=6, solid_capstyle='round')

    ax.set_xlim(-500, 500)
    ax.set_ylim(0, prof.reader_p75.max() * 1.14)
    ax.set_xticks([-500, -250, 0, 250, 500])
    spines(ax)
    ax.tick_params(labelsize=FS, colors=INK, width=0.8, length=2.8, pad=2)
    ax.set_xlabel('Distance from partner-motif centre (bp)', fontsize=FS, color=INK)
    ax.set_ylabel('Motif-centred ChIP enrichment\n(fold over flank)', fontsize=FS, color=INK,
                  linespacing=1.35)

    ax.text(0.035, 0.965, 'Median centre-to-flank ratio', transform=ax.transAxes, ha='left',
            va='top', fontsize=FS, color=INK, fontweight='semibold')
    lines = [('Reader', f'{st.reader_cf:.1f}'), ('Matched null', f'{st.null_cf:.1f}'),
             ('Reader > null', f'{int(st.reader_gt_null)}/{int(st.n)}')]
    for i, (lab, val) in enumerate(lines):
        yy = 0.885 - i * 0.072
        ax.text(0.035, yy, lab, transform=ax.transAxes, ha='left', va='top', fontsize=FS, color=INK)
        ax.text(0.335, yy, val, transform=ax.transAxes, ha='right', va='top', fontsize=FS, color=INK)
    ax.text(0.035, 0.885 - 3 * 0.072, f'$P$ = {_pf(float(st.wilcoxon_p))}',
            transform=ax.transAxes, ha='left', va='top', fontsize=FS, color=INK)
    ax.legend(handles=[Line2D([0], [0], color=RED, lw=3.4, label='ARES-inferred reader'),
                       Line2D([0], [0], color=GREY, lw=2.8,
                              label='Matched random TFs\n(4 per dependency)')],
              loc='upper right', fontsize=FS, frameon=False, handlelength=1.6, labelspacing=0.75,
              borderaxespad=0.4, handletextpad=0.6)
    fig.tight_layout()
    save(fig, 'fig4_centering', SUB)
    print(f'  centering: reader CF {st.reader_cf:.1f} vs null {st.null_cf:.1f}, '
          f'{int(st.reader_gt_null)}/{int(st.n)}, P={st.wilcoxon_p:.1e}')


def reader_tracks_target():
    """The hidden reader tracks the target at partner-motif loci where the namesake does not bind.

    x = median Spearman r of matched random TFs with the target at those loci; y = the ARES reader.
    A point above the diagonal means the reader tracks the target better than an arbitrary expressed
    TF does at the same places. Marker area is the reader's ChIP-peak overlap with the target
    (10/50/90% reference sizes in the key); colour is the per-point significance class.

    The paired Wilcoxon across the 42 dependencies is the evidence. The per-point q (a t-approx to
    Spearman r, BH-corrected) only classifies the markers; loci within a dependency are not
    independent, so that q is optimistic and is not the test.
    """
    d = pd.read_csv(data('fig4', 'reader_corr_namesake_unbound_hidden.csv'))
    pres = pd.read_csv(data('fig4', 'reader_presence_merged.csv'))
    d = d.merge(pres[['cell', 'target', 'namesake', 'frac_reader_bound']],
                on=['cell', 'target', 'namesake'], how='left')
    fb = d.frac_reader_bound.fillna(d.frac_reader_bound.median()).values
    x, y = d.r_null_median.values, d.r_reader.values
    above = int((y > x).sum())
    p = stats.wilcoxon(y, x, alternative='greater').pvalue
    q = d.q_reader.values if 'q_reader' in d else np.ones(len(d))
    sig_pos, sig_neg = (q < 0.05) & (y > 0), (q < 0.05) & (y < 0)
    ns = ~sig_pos & ~sig_neg
    cR, cNeg = desat('#e67e22'), CC['SEQUENCE']

    def _sz(f):
        return 7 + 40 * np.asarray(f)

    fig, ax = plt.subplots(figsize=(3.15, 3.15))
    fig.subplots_adjust(bottom=0.22)
    lo, hi = -0.42, 0.98
    ax.plot([lo, hi], [lo, hi], color=AXG, lw=0.8, ls=(0, (3.5, 2.5)), zorder=1)
    ax.axhline(0, color=AXG, lw=0.45, alpha=0.45, zorder=0)
    ax.axvline(0, color=AXG, lw=0.45, alpha=0.45, zorder=0)
    ax.scatter(x[sig_pos], y[sig_pos], s=_sz(fb[sig_pos]), facecolor=cR, edgecolor=dk(cR),
               lw=0.5, zorder=4)
    ax.scatter(x[ns], y[ns], s=_sz(fb[ns]), facecolor='white', edgecolor='0.58', lw=0.85, zorder=3)
    ax.scatter(x[sig_neg], y[sig_neg], s=_sz(fb[sig_neg]), facecolor=cNeg, edgecolor=dk(cNeg),
               lw=0.5, zorder=5)

    # three named points: the hero (carried through other panels) and the two inversions the text
    # must account for. Partner is written as a motif, M_{TF}.
    hq = d[(d.cell == 'HepG2') & (d.target == 'KDM6A') & (d.namesake == 'HNF4G')]
    if len(hq):
        hr = hq.iloc[0]
        ax.scatter([hr.r_null_median], [hr.r_reader], s=_sz(hr.frac_reader_bound) + 46,
                   facecolor='none', edgecolor=dk(cR, 0.5), lw=0.9, zorder=6)
        ax.annotate(r'KDM6A $\leftarrow$ $M_{\mathrm{HNF4G}}$', (hr.r_null_median, hr.r_reader),
                    xytext=(0.505, 0.875), textcoords='data', fontsize=5.8, ha='left',
                    color=dk(cR, 0.5), va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.45, color=dk(cR, 0.5),
                                    shrinkA=0, shrinkB=4))
    for r in d[sig_neg].itertuples():
        ax.annotate(rf'{r.target} $\leftarrow$ $M_{{\mathrm{{{r.namesake}}}}}$',
                    (r.r_null_median, r.r_reader), xytext=(7, -1), textcoords='offset points',
                    fontsize=5.6, color=dk(cNeg), ha='left', va='center')

    ax.text(0.03, 0.975, f'{above}/{len(d)} above the diagonal\n$P$ = {_pf(p)}',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.2, color=INK, linespacing=1.5)

    for i, (fc, ec, lab) in enumerate([
            (cR, dk(cR), f'correlated ({int(sig_pos.sum())})'),
            ('white', '0.58', f'not significant ({int(ns.sum())})'),
            (cNeg, dk(cNeg), f'anti-correlated ({int(sig_neg.sum())})')]):
        yy = 0.215 - i * 0.070
        ax.scatter([0.615], [yy], s=20, facecolor=fc, edgecolor=ec,
                   lw=0.5 if fc != 'white' else 0.85, transform=ax.transAxes, zorder=6,
                   clip_on=False)
        ax.text(0.660, yy, lab, transform=ax.transAxes, fontsize=5.9, color=INK,
                ha='left', va='center')

    for i, f in enumerate([0.1, 0.5, 0.9]):
        fig.add_artist(plt.Line2D([0.500 + i * 0.105], [0.062], marker='o', ls='',
                                  ms=np.sqrt(_sz(f)), mfc='0.82', mec='0.5', mew=0.5,
                                  transform=fig.transFigure))
        fig.text(0.500 + i * 0.105, 0.022, f'{f:.0%}', fontsize=5.2, color=AXG,
                 ha='center', va='center')
    fig.text(0.462, 0.062, 'Reader ChIP peak overlap (%)', fontsize=5.5, color=AXG,
             ha='right', va='center')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Random-TF null (median Spearman $r$)', fontsize=7, color='black')
    ax.set_ylabel('ARES-assigned reader (Spearman $r$ with target)', fontsize=7, color='black')
    spines(ax)
    ax.tick_params(colors='black', labelsize=6.2, width=0.7, length=2.6)
    save(fig, 'fig4_reader_tracks_target', SUB)
    print(f'  reader tracks target: {above}/{len(d)} above diagonal, P={p:.1e}, '
          f'sig+ {int(sig_pos.sum())} ns {int(ns.sum())} sig- {int(sig_neg.sum())}')


def gata1_dose():
    """Knocking down the reader GATA1 costs the target, dose-dependently in GATA1 occupancy.

    Reads the pre-computed dose curve (median log2FC per GATA1-occupancy quintile, with 95%
    bootstrap intervals) for the target TEAD4, the positive control TAL1 (a GATA1 complex member),
    and two negative controls (CTCF, H3K27ac). All scored on the 8,094 GATA5-motif+ TEAD4 peaks.
    """
    c = pd.read_csv(data('fig4', 'kd_dose_curve.csv'))
    SER = [('TEAD4', 'the target', desat('#e67e22'), 1.6, '-', (6, -2)),
           ('TAL1', 'GATA1 complex member', desat('#c0392b'), 1.3, '-', (6, 5)),
           ('CTCF', 'GATA1-independent', '#6b7280', 1.3, '-', (6, 8)),
           ('H3K27ac', 'regional activity', '#a8adb7', 1.0, (0, (4, 2)), (6, -10))]
    xq = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(3.9, 3.2))
    ax.axhline(0, color='0.88', lw=0.6, zorder=0)
    q5 = {}
    for name, role, col, lw, ls, off in SER:
        t = c[c.track == name].sort_values('quintile')
        ax.fill_between(xq, t.ci_lo, t.ci_hi, color=col, alpha=0.16, lw=0, zorder=1)
        ax.plot(xq, t['median'], color=col, lw=lw, ls=ls, zorder=3, marker='o', ms=3.4,
                mec=dk(col), mew=0.4)
        ax.annotate(f'{name}\n{role}', (5, t['median'].iloc[-1]), xytext=off,
                    textcoords='offset points', ha='left', va='center', fontsize=5.8,
                    color=dk(col), linespacing=1.25)
        q5[name] = t['median'].iloc[-1]
    ax.set_xticks(xq)
    ax.set_xticklabels([f'Q{i}' for i in xq])
    ax.set_xlim(0.85, 5.15)
    ax.set_ylim(-1.16, 0.12)
    ax.set_xlabel('GATA1 occupancy before knockdown (quintile)', fontsize=7, color='black')
    ax.set_ylabel('ChIP signal after GATA1 loss\n(median $\\log_2$FC, depleted / control)',
                  fontsize=7, color='black', linespacing=1.35)
    ax.annotate('$n$ = 8,094 GATA5-motif$^{+}$ peaks\n\n'
                'Q5 median $\\log_2$FC\n'
                f'TEAD4 $-${abs(q5["TEAD4"]):.2f}, TAL1 $-${abs(q5["TAL1"]):.2f}\n'
                f'CTCF {q5["CTCF"]:+.2f}, H3K27ac $-${abs(q5["H3K27ac"]):.2f}',
                (0.03, 0.05), xycoords='axes fraction', ha='left', va='bottom', fontsize=6.0,
                color='0.25', linespacing=1.5)
    spines(ax)
    ax.tick_params(colors='black', labelsize=6.2, width=0.7, length=2.6)
    fig.tight_layout()
    save(fig, 'fig4_gata1_dose', SUB)
    print('  GATA1 dose: Q5  ' + '  '.join(f'{k} {v:+.2f}' for k, v in q5.items()))


def scramble_binding():
    """AlphaGenome: scrambling the partner motif removes the reader more than matched control TFs.

    Each point is a mode-resolved hidden-reader dependency. y is the reader's paired log2FC (motif
    window minus a control window in the same interval), x is the median of 12 non-reader,
    non-same-family control TFs read from the identical prediction. Fill is the reader's own drop
    significance (one-sided Wilcoxon, BH), not its margin over the control.
    """
    D = pd.read_csv(data('fig4', 'per_dependency.csv'))
    sig = pd.read_csv(data('fig4', 'per_dependency_readerdrop_sig.csv'))
    RDc = desat('#e67e22')
    m1 = D.dropna(subset=['reader_x', 'ctrl_x']).merge(
        sig[['cell', 'target', 'partner', 'q']], on=['cell', 'target', 'partner'], how='left')
    m1['below'] = m1.reader_x < m1.ctrl_x
    m1['sig'] = m1.q < 0.05
    p1 = stats.wilcoxon(m1.reader_x - m1.ctrl_x).pvalue
    nb, ns = int(m1.below.sum()), int(m1.sig.sum())

    lo, hi = -6.6, 0.7
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.axhline(0, color='0.88', lw=0.6, zorder=0)
    ax.axvline(0, color='0.88', lw=0.6, zorder=0)
    ax.plot([lo, hi], [lo, hi], color='0.6', lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.scatter(m1[m1.sig].ctrl_x, m1[m1.sig].reader_x, s=18, color=RDc, edgecolor=dk(RDc),
               linewidth=0.35, alpha=0.8, zorder=4, rasterized=True)
    ax.scatter(m1[~m1.sig].ctrl_x, m1[~m1.sig].reader_x, s=16, color='white', edgecolor=dk(RDc),
               linewidth=0.7, alpha=0.8, zorder=5)
    ax.annotate(f'$n$ = {len(m1)}\n\n{nb}/{len(m1)} below diagonal\n{ns} FDR < 0.05\n\n'
                f'$P$ = {_pf(p1)}',
                (0.035, 0.975), xycoords='axes fraction', ha='left', va='top', fontsize=6.0,
                color=dk(RDc), linespacing=1.55)
    h = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=RDc, markeredgecolor=dk(RDc),
                markersize=4.0, markeredgewidth=0.35, label='FDR < 0.05'),
         Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white',
                markeredgecolor=dk(RDc), markersize=4.0, markeredgewidth=0.7, label='n.s.')]
    ax.legend(handles=h, fontsize=5.8, frameon=False, handletextpad=0.3, labelspacing=0.32,
              loc='upper left', bbox_to_anchor=(0.03, 0.60))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('$\\Delta$ predicted control TFs binding\nafter motif scrambling ($\\log_2$FC)',
                  fontsize=7, color='black', linespacing=1.35)
    ax.set_ylabel('$\\Delta$ predicted reader binding ($\\log_2$FC)', fontsize=7, color='black')
    spines(ax)
    ax.tick_params(colors='black', labelsize=6.2, width=0.7, length=2.6)
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_yticks([-6, -4, -2, 0])
    save(fig, 'fig4_scramble_binding', SUB)
    print(f'  scramble binding: n={len(m1)}, {nb}/{len(m1)} below diagonal, '
          f'{ns} FDR<0.05, P={p1:.1e}')


def scramble_cage():
    """AlphaGenome: once the reader is displaced, transcription follows the reader's GO valence.

    x = predicted reader binding change, y = predicted CAGE change (both median paired log2FC). Fill
    is whether the CAGE change itself is significant. MAX is labelled because GO calls it a repressor
    from the MAX-MXD/MNT complexes, but it is a bHLH hub whose MYC:MAX arm activates.
    """
    D = pd.read_csv(data('fig4', 'per_dependency.csv'))
    cs = pd.read_csv(data('fig4', 'per_dependency_cage_sig.csv'))
    ACTc, REPc = desat('#c0392b'), desat('#2e8b57')
    m = D.dropna(subset=['cage_y', 'reader_x']).query("rv in ['ACT','REP']").merge(
        cs[['cell', 'target', 'partner', 'q']], on=['cell', 'target', 'partner'], how='left')
    m['csig'] = m.q < 0.05
    Ac, Rc = m[m.rv == 'ACT'].cage_y, m[m.rv == 'REP'].cage_y
    p2 = stats.mannwhitneyu(Ac, Rc, alternative='less').pvalue
    rd = m.groupby(['reader', 'rv']).cage_y.median().reset_index()
    p2r = stats.mannwhitneyu(rd[rd.rv == 'ACT'].cage_y, rd[rd.rv == 'REP'].cage_y,
                             alternative='less').pvalue

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.set_xscale('symlog', linthresh=0.3, linscale=0.9)
    ax.axhline(0, color='0.88', lw=0.6, zorder=0)
    ax.axvline(0, color='0.88', lw=0.6, zorder=0)
    for rv, col in [('ACT', ACTc), ('REP', REPc)]:
        q = m[(m.rv == rv) & m.csig]
        ax.scatter(q.reader_x, q.cage_y, s=20, color=col, edgecolor=dk(col), linewidth=0.4,
                   alpha=0.8, zorder=3)
        q = m[(m.rv == rv) & ~m.csig]
        ax.scatter(q.reader_x, q.cage_y, s=18, color='white', edgecolor=dk(col), linewidth=0.7,
                   alpha=0.9, zorder=4)
    mx = m[(m.reader == 'MAX') & (m.reader_x < -0.5)]
    LX, LY = -0.47, -0.33
    for x, y in mx[['reader_x', 'cage_y']].values:
        ax.annotate('', (x, y), xytext=(LX, LY), zorder=5,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color='0.72', shrinkA=7.5, shrinkB=2.5))
    ax.text(LX, LY, 'MAX', fontsize=5.8, color='0.35', ha='center', va='center', zorder=6)
    for (x, y, rdr), off in zip(
            m[m.rv == 'ACT'].nsmallest(2, 'cage_y')[['reader_x', 'cage_y', 'reader']].values,
            [(14, -1), (15, 8)]):
        ax.annotate(rdr, (x, y), xytext=off, textcoords='offset points', fontsize=5.8,
                    color='0.35', ha='left', va='center', zorder=6,
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='0.6', shrinkA=0.5, shrinkB=2.5))
    ax.set_xlim(-7.5, 0.6)
    ax.set_xticks([-6, -3, -1, -0.3, 0])
    ax.set_xticklabels(['−6', '−3', '−1', '−0.3', '0'])
    ax.set_xlabel('$\\Delta$ predicted reader binding\nafter motif scrambling ($\\log_2$FC)',
                  fontsize=7, color='black', linespacing=1.35)
    ax.set_ylabel('$\\Delta$ predicted CAGE expression ($\\log_2$FC)', fontsize=7, color='black')
    ax.annotate(f'$n$ = {len(m)} ({len(Ac)} act / {len(Rc)} rep)\n\n'
                f'median $\\Delta$CAGE $-${abs(Ac.median()):.2f} vs $-${abs(Rc.median()):.2f}\n\n'
                f'$P$ = {_pf(p2)} ({len(m)} dependencies)\n'
                f'$P$ = {_pf(p2r)} ({len(rd)} readers)',
                (0.985, 0.02), xycoords='axes fraction', ha='right', va='bottom', fontsize=6.0,
                color='0.25', linespacing=1.55)
    h2 = [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor=c, markeredgecolor=dk(c),
                 markersize=4.0, markeredgewidth=0.4, label=lb)
          for c, lb in [(ACTc, 'activator reader'), (REPc, 'repressor reader')]]
    h2 += [Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='white',
                  markeredgecolor='0.45', markersize=4.0, markeredgewidth=0.7,
                  label='$\\Delta$CAGE n.s.')]
    ax.legend(handles=h2, fontsize=5.8, frameon=False, handletextpad=0.3, labelspacing=0.32,
              loc='lower right', bbox_to_anchor=(1.0, 0.44))
    spines(ax)
    ax.tick_params(colors='black', labelsize=6.2, width=0.7, length=2.6)
    save(fig, 'fig4_scramble_cage', SUB)
    print(f'  scramble CAGE: n={len(m)} ({len(Ac)}/{len(Rc)}), '
          f'P_dep={p2:.4f} P_reader={p2r:.4f}')


def reader_odds():
    """Logistic odds ratios: the outcome is reader-specific, the region is not.

    Each annotation (same reader / same partner motif / same cell) is fit as a separate logistic
    predictor of two endpoints: the region of regulation and the transcriptional outcome. Solid
    markers are significant by the permutation test; the dashed null line is OR = 1.
    """
    R = json.load(open(data('fig4', 'reader_function_results.json')))
    G = pd.read_csv(data('fig4', 'reader_odds_grid.csv'))
    SEQc, READERc, CELLc = '#3f6fa3', '#b5564a', '#8f8f8f'

    ann = [('same_reader', 'Reader', READERc),
           ('same_cell', 'Cell', CELLc),
           ('same_motif', 'Motif', SEQc)]
    endpoints = [('region', 'region_of_regulation', 'Regulatory\nregion'),
                 ('outcome', 'transcriptional_outcome', 'Transcriptional\noutcome')]

    def pval(col, foc):
        return float(G[(G.col == col) & (G.focus == foc)].perm_p.iloc[0])

    def plabel(p):
        return 'n.s.' if p >= 0.05 else f'$P$ = {_pf(p)}'

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.axhline(1, color='#cfcfcf', lw=0.7, ls=(0, (4, 3)), zorder=1)
    xpos = {'region': 0, 'outcome': 1}
    for foc, lab, col in ann:
        ys = [R[ek][foc] for ek, _, _ in endpoints]
        xs = [xpos[ek] for ek, _, _ in endpoints]
        ps = [pval(gcol, foc) for _, gcol, _ in endpoints]
        ax.plot(xs, ys, color=col, lw=1.1, zorder=3)                 # solid line for every series
        for xi, yi, pi in zip(xs, ys, ps):
            ax.scatter(xi, yi, s=42, color=col if pi < 0.05 else 'white', edgecolor=col,
                       linewidth=1.1, zorder=4)
            # P (or n.s.) beside each point: region on the left, outcome on the right
            dx, ha = (-7, 'right') if xi == 0 else (7, 'left')
            ax.annotate(plabel(pi), (xi, yi), xytext=(dx, 0), textcoords='offset points',
                        ha=ha, va='center', fontsize=5.6, color=col)
        ax.annotate(lab, (xs[-1], ys[-1]), xytext=(46, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=6.8, color=col, fontweight='semibold')
    # log axis with explicit ticks (0.7, 1, 2, 3): the default log formatter would clutter the
    # 1-3 range with x10^0 minor labels, so ticks are fixed and given a plain formatter. OR = 1
    # (no association) is the dashed reference.
    import matplotlib.ticker as mticker
    ax.set_yscale('log')
    ax.set_xlim(-0.55, 2.35)
    ax.set_ylim(0.65, 3.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([e[2] for e in endpoints], fontsize=6.6, color=INK)
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.7, 1, 2, 3]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(['0.7', '1', '2', '3']))
    ax.set_ylabel('Odds ratio (shared annotation)', fontsize=7, color=INK)
    spines(ax)
    ax.tick_params(colors=INK, labelsize=6.4)
    fig.tight_layout()
    save(fig, 'fig4_reader_odds', SUB)
    print('  reader odds: ' + '  '.join(
        f'{lab} region OR={R["region"][foc]:.1f}(P={pval(endpoints[0][1], foc):.3f}) '
        f'outcome OR={R["outcome"][foc]:.1f}(P={pval(endpoints[1][1], foc):.3f})'
        for foc, lab, _ in ann))


def target_reader_concordance():
    """Among the factors that occupy a target's partner-motif+ peaks, the one ARES names as the
    reader is the one whose knockdown best mirrors the target's own.

    Genes at the target's motif+ peaks are kept if the TARGET's knockdown moves them, and scored by
    the fraction the READER's knockdown moves the same way. The controls are the 10 factors whose
    occupancy of those same peaks is closest to the reader's, so they bind the same loci to the same
    extent and cannot lose merely by being absent. Direction agreement over independently perturbed
    genes puts an unrelated factor at 50%, which is where the control cloud in fact sits.

    Both axes are percentages of target-moved genes, so the summary is a difference in percentage
    points. It is the median of the WITHIN-dependency differences, which is not the difference of
    the two medians; the paired form is the one the test uses. Point area is the number of
    target-moved genes.
    """
    D = pd.read_csv(data('fig4', 'target_reader_concordance.csv'))
    for c in ('conc_Mp', 'ctrl_med'):
        D[c] = D[c] * 100                          # fraction -> % of target-moved genes
    D['excess'] = D.conc_Mp - D.ctrl_med           # percentage points
    m = D.excess.median()
    p = stats.wilcoxon(D.conc_Mp, D.ctrl_med, alternative='greater').pvalue

    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    spines(ax)
    lim = (min(D.ctrl_med.min(), D.conc_Mp.min()) - 3,
           max(D.ctrl_med.max(), D.conc_Mp.max()) + 3)
    ax.plot(lim, lim, color=AXG, lw=.8, ls=(0, (3, 2)), zorder=1)
    sz = lambda n: 5 + 62 * np.sqrt(np.asarray(n) / D.n_affected_Mp.max())
    for cat, col, lab in [('partner', CC['SEQUENCE'], 'reader = motif namesake'),
                          ('hidden', desat('#e67e22'), 'hidden reader')]:
        q = D[D.cat == cat]
        ax.scatter(q.ctrl_med, q.conc_Mp, s=sz(q.n_affected_Mp), facecolor=col,
                   edgecolor='white', linewidths=.35, alpha=.78, zorder=3,
                   label=f'{lab}  ($n$ = {len(q)})')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect('equal')
    tk = np.arange(30, 90, 10.)                    # bare numbers; the unit lives in the axis label
    tk = tk[(tk >= lim[0]) & (tk <= lim[1])]
    ax.set_xticks(tk)
    ax.set_yticks(tk)
    ax.set_xlabel('10 TFs matched on occupancy of the same peaks\n'
                  '(median % of those genes moved the same way)',
                  fontsize=6.8, color=INK, linespacing=1.4)
    ax.set_ylabel('ARES-inferred reader\n(% of target-moved genes moved the same way)',
                  fontsize=6.8, color=INK, linespacing=1.4)
    ax.text(.035, .975,
            f'{int((D.excess > 0).sum())}/{len(D)} above the diagonal\n'
            f'median $\\Delta$ = {m:+.1f} pp\n$P$ = {_pf(p)}',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.3, color=INK, linespacing=1.6)
    # both legends sit in the empty lower-right wedge: an identity-line panel must share x and y
    # limits, so the region to the right of the data is dead space rather than something to crop
    lg1 = ax.legend(loc='lower right', bbox_to_anchor=(1.03, -.01), fontsize=6.0,
                    handletextpad=.4, borderpad=.3, labelspacing=.35, markerscale=.8)
    ax.add_artist(lg1)
    h = [Line2D([], [], ls='', marker='o', mfc='0.62', mec='white', mew=.35, alpha=.78,
                ms=np.sqrt(sz(n)), label=f'{n}') for n in (10, 60, 300)]
    lg = ax.legend(handles=h, loc='center right', bbox_to_anchor=(1.02, .42), fontsize=6.0,
                   title='target-moved\ngenes', title_fontsize=6.0, handletextpad=.6,
                   borderpad=.3, labelspacing=.8)
    lg._legend_box.align = 'left'
    fig.tight_layout()
    save(fig, 'fig4_target_reader_concordance', SUB)


PANELS = [namesake_expression, reader_assignments, reader_vs_namesake_tpm, centering,
          reader_tracks_target, gata1_dose, scramble_binding, scramble_cage, reader_odds,
          target_reader_concordance]


def main():
    print('Figure 4')
    for fn in PANELS:
        fn()


if __name__ == '__main__':
    main()
