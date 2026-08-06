#!/usr/bin/env python
"""Figure 2 — how ARES infers a route, and whether the call is reproducible.

  partner-dependence by route     the interaction coefficient separates the routes from data alone
  cross-cell reproducibility      two cells agree on the route far above a label-shuffled null
  route assignment, recurrent     pairs resolved in three or more cell types, cell by cell
  route composition of hubs       recurrent partner motifs keep their route across cell types

Run as:  python fig2.py
"""
import collections
import itertools

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from _common import CC, INK, LBL, LEAVES, MODES, atlas, data, pbody, save, spines

SUB = 'fig2'
RED = '#b03a2e'


def partner_dependence_by_route():
    """Partner-dependence coefficient split by the route ARES assigned.

    The coefficient is computed per dependency from ChIP and motif data alone, with no knowledge of
    the route, so a separation between the routes is evidence that the call tracks something
    measurable rather than being a relabelling of the same distribution.

    Restricted to dependencies with partner ChIP-seq, since the interaction term needs the partner
    protein's occupancy; that is what sets n well below the full atlas.
    """
    d = (pd.read_csv(data('partner_dependence_coefficient.csv'))
         .merge(atlas()[['cell', 'tf_a', 'tf_b', 'consensus']],
                on=['cell', 'tf_a', 'tf_b'], how='left'))
    d = d[d.consensus.isin(MODES)]
    order = ['SEQUENCE', 'CONTEXT', 'PROTEIN']
    rngx = d.bMP_std.max() - d.bMP_std.min()

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    rng = np.random.default_rng(0)
    vw = 0.40
    ax.axvline(0, color='#bdbdbd', lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i, c in enumerate(order):
        v = d[d.consensus == c].bMP_std.values
        col = CC[c]
        xs = np.linspace(v.min() - 0.03 * rngx, v.max() + 0.03 * rngx, 200)
        dens = stats.gaussian_kde(v)(xs)
        dens = dens / dens.max() * vw
        ax.fill_between(xs, i, i + dens, color=col, alpha=0.55, lw=0, zorder=2)
        ax.plot(xs, i + dens, color=col, lw=0.8, zorder=2)
        ax.scatter(v, i - 0.07 - rng.uniform(0, 0.30, len(v)), s=5, color=col,
                   alpha=0.40, edgecolors='none', zorder=3)
        med = np.median(v)
        ax.plot([med, med], [i - 0.40, i + dens[np.argmin(abs(xs - med))]],
                color='0.15', lw=1.0, zorder=4)
        ax.text(med, i + vw + 0.06, f'{med:+.2f}', ha='center', va='bottom',
                fontsize=6, color=col, fontweight='bold')

    ax.set_yticks(range(3))
    ax.set_yticklabels([f'{c.capitalize()}\n(n={int((d.consensus == c).sum())})' for c in order],
                       fontsize=8, color=INK)
    ax.set_ylim(-0.55, 2.75)
    ax.set_xlim(d.bMP_std.min() - 0.05 * rngx, d.bMP_std.max() + 0.06 * rngx)
    ax.set_xlabel('Partner-dependence coefficient', fontsize=8, color=INK)
    spines(ax)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, labelsize=7)

    # nested brackets on the right: Protein against each of the other two routes
    mwu = lambda a, b: stats.mannwhitneyu(d[d.consensus == a].bMP_std,
                                          d[d.consensus == b].bMP_std)[1]
    pps, ppc = mwu('PROTEIN', 'SEQUENCE'), mwu('PROTEIN', 'CONTEXT')
    x0, x1 = ax.get_xlim()
    xr = x1 - x0
    ax.set_xlim(x0, x1 + 0.34 * xr)
    for y0, y1, p, xb in [(0, 2, pps, x1 + 0.19 * xr), (1, 2, ppc, x1 + 0.07 * xr)]:
        ax.plot([xb, xb], [y0 + 0.12, y1 - 0.12], color=INK, lw=0.8, clip_on=False)
        for yy in (y0 + 0.12, y1 - 0.12):
            ax.plot([xb - 0.025 * xr, xb], [yy, yy], color=INK, lw=0.8, clip_on=False)
        ax.text(xb + 0.015 * xr, (y0 + y1) / 2, f'$P$ = {pbody(p)}', ha='center', va='center',
                fontsize=6, color=INK, rotation=270, clip_on=False)

    fig.tight_layout()
    save(fig, 'fig2_partner_dependence_by_route', SUB)
    print(f'  partner dependence by route: n={len(d)}  '
          f'Prot vs Seq P={pps:.2g}  Prot vs Ctx P={ppc:.2g}')


def cross_cell_reproducibility():
    """Cell-cell route agreement against a label-shuffled null.

    Dependencies are grouped by shared target TF and partner motif; groups seen in more than one
    cell type contribute every within-group cell-cell comparison. The null permutes the route
    labels across the same observations while holding group membership fixed, which preserves the
    route marginals; without that control a protein-heavy atlas would look reproducible by
    construction. The reported P is the add-one estimator over 10^6 permutations, so it is bounded
    below by 1 / (10^6 + 1) and no parametric tail approximation is used.
    """
    r = atlas()
    r = r[r.consensus.isin(MODES)].copy()
    r['pair'] = r.tf_a + '<-' + r.tf_b
    pc = r.groupby('pair').cell.nunique()
    sub = r[r.pair.isin(set(pc[pc >= 2].index))]
    grp = list(sub.groupby('pair').indices.values())
    modes = sub.consensus.values

    # every within-group cell-cell comparison, as index pairs into the observations
    codes = pd.Series(modes).map({m: i for i, m in enumerate(MODES)}).values.astype(np.int8)
    I = np.array([x for g in grp for x, _ in itertools.combinations(g, 2)])
    J = np.array([y for g in grp for _, y in itertools.combinations(g, 2)])
    obs = float((codes[I] == codes[J]).mean())

    # Route labels are permuted across the 219 observations while the assignment of observations
    # to groups is held fixed, so the null preserves the route marginals and the group structure
    # and only destroys the correspondence between them. The P value is the add-one estimator
    # (b + 1) / (m + 1) over m permutations, which is bounded below by 1 / (m + 1); no parametric
    # approximation to the tail is used.
    NPERM, BATCH = 10 ** 6, 20_000
    rng = np.random.default_rng(0)
    n = len(codes)
    null = np.empty(NPERM)
    for start in range(0, NPERM, BATCH):
        b = min(BATCH, NPERM - start)
        lab = codes[np.argsort(rng.random((b, n)), axis=1)]
        null[start:start + b] = (lab[:, I] == lab[:, J]).mean(axis=1)
    ge = int((null >= obs).sum())
    z = (obs - null.mean()) / null.std()
    p = (ge + 1) / (NPERM + 1)
    npair, ncell = len(grp), len(I)

    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    ax.hist(null * 100, bins=22, color='#d3d7dc', edgecolor='white', lw=0.3, zorder=2)
    ax.axvline(null.mean() * 100, color='0.55', lw=1.2, ls=':', zorder=4)
    ax.axvline(obs * 100, color=INK, lw=1.8, ls=':', zorder=5)
    ytop = ax.get_ylim()[1]
    ax.text(obs * 100 - 2, ytop * 0.95, f'ARES\n{obs * 100:.0f}%', ha='right', va='top',
            fontsize=7, color=INK, fontweight='bold', linespacing=1.05)
    ax.text(null.mean() * 100 + 1.4, ytop * 0.92, f'chance {null.mean() * 100:.0f}%',
            ha='left', va='top', fontsize=6.4, color='0.45')
    ptxt = (f'$P$ < {pbody(1 / (NPERM + 1))}' if ge == 0 else f'$P$ = {pbody(p)}')
    ax.text(obs * 100 - 1.6, ytop * 0.08, ptxt, ha='right', va='bottom',
            fontsize=6.2, color=INK)
    ax.set_xlabel('Cell–cell route agreement (%)', fontsize=6.8, color=INK)
    ax.set_ylabel('shuffles', fontsize=6.8, color=INK)
    spines(ax)
    fig.tight_layout()
    save(fig, 'fig2_cross_cell_reproducibility', SUB)
    print(f'  cross-cell: obs={100 * obs:.1f}% null={100 * null.mean():.1f}% z={z:.1f} '
          f'(groups={npair}, observations={n}, comparisons={ncell}); '
          f'{ge}/{NPERM} permutations >= observed, add-one P={p:.3g}')


def recurrent_pairs():
    """Route per cell type for every pair resolved in three or more cell types.

    Tiles are the assigned route; a ringed tile is a cell where the call differs from that pair's
    dominant route, so the exceptions are marked individually rather than summarised away. Rows are
    banded by dominant route.
    """
    r = atlas()
    r = r[r.consensus.isin(MODES)].copy()
    r['pair'] = r.tf_a + '<-' + r.tf_b
    pc = r.groupby('pair').cell.nunique()
    keep = set(pc[pc >= 3].index)
    sub = r[r.pair.isin(keep)]
    m = {(p, c): cls for p, c, cls in zip(sub.pair, sub.cell, sub.consensus)}
    cells = list(sub.cell.value_counts().index)
    classord = {'SEQUENCE': 0, 'CONTEXT': 1, 'PROTEIN': 2}

    def info(p):
        cs = [m[(p, c)] for c in cells if (p, c) in m]
        cc = collections.Counter(cs)
        mc, mn = cc.most_common(1)[0]
        return mc, mn, len(cs), len(cc)

    pairs = sorted(keep, key=lambda p: (classord[info(p)[0]], info(p)[3] > 1,
                                        -info(p)[1] / info(p)[2], p))
    bottom = 'PKNOX1<-NFYC'
    if bottom in pairs:
        pairs = [p for p in pairs if p != bottom] + [bottom]
    n_dom = sum(info(p)[1] / info(p)[2] >= 0.66 for p in pairs)
    agree = sum(info(p)[1] for p in pairs) / sum(info(p)[2] for p in pairs)

    NT = '#eceef0'
    GAP, depth, d, prev = 0.45, [], 0.0, None
    for p in pairs:
        mode = info(p)[0]
        if prev is not None and mode != prev:
            d += GAP
        depth.append(len(depth) + d)
        prev = mode
    span = max(depth)
    Y = [span - x for x in depth]
    ncol = len(cells)

    fig, ax = plt.subplots(figsize=(0.34 * ncol + 3.2, 0.285 * (span + 1) + 1.1))
    for i, p in enumerate(pairs):
        y = Y[i]
        mc = info(p)[0]
        for j, c in enumerate(cells):
            if (p, c) in m:
                ax.add_patch(plt.Rectangle((j - 0.5, y - 0.5), 1, 1, facecolor=CC[m[(p, c)]],
                                           edgecolor='white', lw=0.6, zorder=2))
                if m[(p, c)] != mc:
                    ax.add_patch(plt.Rectangle((j - 0.5, y - 0.5), 1, 1, facecolor='none',
                                               edgecolor=INK, lw=1.1, zorder=4))
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, y - 0.5), 1, 1, facecolor=NT,
                                           edgecolor='white', lw=0.6, zorder=1))
    x0 = ncol - 0.5 + 0.34
    for mode in ['SEQUENCE', 'CONTEXT', 'PROTEIN']:
        yy = [Y[i] for i, p in enumerate(pairs) if info(p)[0] == mode and p != bottom]
        if not yy:
            continue
        lo, hi = min(yy) - 0.37, max(yy) + 0.37
        ax.add_patch(plt.Rectangle((x0, lo), 0.12, hi - lo, facecolor=CC[mode],
                                   edgecolor='none', clip_on=False, zorder=2))
        ax.text(x0 + 0.30, (lo + hi) / 2, mode.capitalize(), rotation=270, ha='left',
                va='center', fontsize=7, fontweight='bold', color=CC[mode], clip_on=False)
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([c.replace('.', '-') for c in cells], rotation=45, ha='center',
                       fontsize=7, color=INK)
    ax.set_yticks(Y)
    ax.set_yticklabels([rf'{p.split("<-")[0]} ← $M_{{\mathrm{{{p.split("<-")[1]}}}}}$'
                        for p in pairs], fontsize=6.5, color=INK)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(min(Y) - 0.5, max(Y) + 0.5)
    ax.set_aspect('equal', adjustable='box')
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.legend(handles=[Patch(fc='none', ec=INK, lw=1.1, label='differs from dominant'),
                       Patch(fc=NT, ec='white', lw=0.5, label='not tested')],
              loc='upper left', bbox_to_anchor=(0.0, -0.085), fontsize=6.5, frameon=False,
              ncol=2, handlelength=1.1, handleheight=1.1, columnspacing=1.4)
    fig.tight_layout()
    save(fig, 'fig2_recurrent_pairs', SUB)
    print(f'  recurrent pairs: {len(pairs)} in >=3 cells | dominant>=2/3: {n_dom} | '
          f'observed agreement {agree:.0%}')


def hub_route_composition():
    """Route composition of the recurrent partner motifs (>= 15 resolved pairs in a cell type).

    Each bar is one partner motif in one cell type, split by mechanism leaf and shaded within its
    parent route, so both the coarse route and the finer mechanism are readable at once. REST is
    highlighted because it recurs in three cell types and keeps the same route in each.
    """
    res = atlas()
    res = res[res.consensus.isin(MODES)].copy()
    g = res.groupby(['tf_b', 'cell']).size()
    hubs = [(t, c) for (t, c), n in g.items() if n >= 15]
    psf = lambda tf: (res[res.tf_b == tf].consensus == 'SEQUENCE').mean()
    nfun = lambda tc: len(res[(res.tf_b == tc[0]) & (res.cell == tc[1])])
    hubs = sorted(hubs, key=lambda tc: (-psf(tc[0]), tc[0], -nfun(tc)))
    ORDER = LEAVES['SEQUENCE'] + LEAVES['PROTEIN'] + LEAVES['CONTEXT']

    def shade_map(cls):
        base = np.array(mpl.colors.to_rgb(CC[cls]))
        ls = LEAVES[cls]
        return {l: tuple(base + (1 - base) * 0.55 * (i / max(1, len(ls) - 1)))
                for i, l in enumerate(ls)}

    COL2 = {}
    for cls in MODES:
        COL2.update(shade_map(cls))

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(hubs) + 1.1))
    for k, (tf, cell) in enumerate(hubs):
        y = len(hubs) - 1 - k
        s = res[(res.tf_b == tf) & (res.cell == cell)]
        n = len(s)
        pct = {l: 100 * (s.mechanism == l).sum() / n for l in ORDER}
        dom = max(pct, key=pct.get)
        left = 0
        for l in ORDER:
            v = pct[l]
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=COL2.get(l, '#ccc'), edgecolor='white',
                    lw=0.4, height=0.72)
            if l == dom and v >= 13:
                ax.text(left + v / 2, y, f'{LBL.get(l, l)} {v:.0f}%', ha='center', va='center',
                        fontsize=6.8, color=INK, fontweight='bold')
            elif v >= 7:
                ax.text(left + v / 2, y, f'{v:.0f}', ha='center', va='center',
                        fontsize=6.3, color=INK)
            left += v
    for k in range(len(hubs) - 1):
        if hubs[k][0] != hubs[k + 1][0]:
            ax.axhline(len(hubs) - 1 - k - 0.5, color='0.85', lw=0.6, zorder=0)

    ax.set_yticks(range(len(hubs)))
    ax.set_yticklabels([rf'$M_{{\mathrm{{{tf}}}}}$ / {cell}' for tf, cell in reversed(hubs)],
                       fontsize=8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(hubs) - 0.1)
    ax.set_xlabel('Resolved ARES pairs (%)', fontsize=8, color=INK)
    ax.set_xticks([0, 25, 50, 75, 100])
    spines(ax)
    ax.tick_params(labelsize=7.4)
    XN = 107
    ax.text(XN, len(hubs) - 0.45, 'Pair count', ha='center', va='bottom', fontsize=7.3,
            fontweight='bold', color=INK, clip_on=False)
    for k, (tf, cell) in enumerate(hubs):
        y = len(hubs) - 1 - k
        n = len(res[(res.tf_b == tf) & (res.cell == cell)])
        ax.text(XN, y, str(n), ha='center', va='center', fontsize=7.6,
                color=RED if tf == 'REST' else INK, clip_on=False)
    for lab, (tf, cell) in zip(ax.get_yticklabels(), reversed(hubs)):
        lab.set_color(RED if tf == 'REST' else INK)
    ax.legend(handles=[Patch(fc=CC[c], label=c.capitalize()) for c in MODES],
              loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 'fig2_hub_route_composition', SUB)
    print(f'  hubs: {[f"{t}/{c}" for t, c in hubs]}')


PANELS = [partner_dependence_by_route, cross_cell_reproducibility,
          recurrent_pairs, hub_route_composition]


def main():
    print('Figure 2')
    for fn in PANELS:
        fn()


if __name__ == '__main__':
    main()
