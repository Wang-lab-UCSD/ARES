#!/usr/bin/env python
"""The full 4x2 annotation-sharing grid, with each cell's null built from its own label.

Run as:  python analysis/fig3c_qap_fullgrid.py [--nperm 5000] [--nboot 2000]
"""
import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Same convention as figures/_common.py: the data archive sits next to the repo unless ARES_DATA
# says otherwise, so this runs from a fresh clone plus the Zenodo download and nothing else.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ARES_DATA', os.path.join(HERE, os.pardir, os.pardir, 'data'))


def data(*parts):
    p = os.path.normpath(os.path.join(DATA, *parts))
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'missing data file: {p}\n'
            f'Unpack the Zenodo archive into {os.path.normpath(DATA)}, '
            f'or set ARES_DATA to its location.')
    return p
FOCI = ['same_reader', 'same_motif', 'same_cell', 'same_target']
ENDPOINTS = ['region_of_regulation', 'transcriptional_outcome']
DROP = ['NOT_TESTED', 'NOT_REPORTED', 'MIXED', 'nan']


def decoupled_subset():
    """PROTEIN/CONTEXT dependencies whose reader is not the partner-motif namesake."""
    a = pd.read_csv(data('ares_atlas.tsv'), sep='\t').rename(
        columns={'target': 'tf_a', 'partner': 'tf_b'})
    d = a[a.dichotomy.isin(['PROTEIN', 'CONTEXT'])].copy()
    d['reader'] = d.reader.fillna('').astype(str)
    d['rset'] = d.reader.apply(lambda s: frozenset(t.strip().upper() for t in s.split(';') if t.strip()))
    d = d[d.rset.map(len) > 0].reset_index(drop=True)
    part = lambda p: set(str(p).upper().replace(' ', '').replace('::', ';').split(';'))
    return d[~d.apply(lambda r: bool(r.rset & part(r.tf_b)), axis=1)].reset_index(drop=True)


def motif_sim():
    """Tomtom q<0.05 similarity as an unordered label-pair set."""
    tt = pd.read_csv(data('fig4', 'partner_motif_tomtom.tsv'), sep='\t', comment='#').dropna(
        subset=['Query_ID', 'Target_ID'])
    s = set()
    for _, r in tt[tt['q-value'] < 0.05].iterrows():
        x, y = str(r.Query_ID).upper(), str(r.Target_ID).upper()
        s.add((x, y)); s.add((y, x))
    return s


class Dyads:
    """Dyadic design for one endpoint, precomputed so a permutation is just an index shuffle.

    Rebuilding the motif-similarity matrix inside the loop is what makes the direct version too
    slow to run 5,000 times per grid cell; here it is built once over the unique partner labels and
    every permutation reindexes it.
    """

    def __init__(self, dd, col, sim):
        self.n = len(dd)
        self.yraw = np.asarray(dd[col].values, dtype=object)
        prots = sorted(set().union(*dd['rset'].values))
        pidx = {p: i for i, p in enumerate(prots)}
        self.OH = np.zeros((self.n, max(1, len(prots))), bool)
        for i, s in enumerate(dd['rset'].values):
            for p in s:
                self.OH[i, pidx[p]] = True
        lab = [str(s).upper() for s in dd['tf_b'].values]
        uq = sorted(set(lab))
        ui = {u: i for i, u in enumerate(uq)}
        S = np.zeros((len(uq), len(uq)), bool)
        for i, x in enumerate(uq):
            for j, y in enumerate(uq):
                S[i, j] = (x == y) or ((x, y) in sim)
        self.S = S
        self.midx = np.array([ui[x] for x in lab])
        self.cell = np.asarray(dd['cell'].values, dtype=object)
        self.targ = np.asarray(dd['tf_a'].values, dtype=object)

    def build(self, perm=None, rows=None):
        """Design matrix and outcome. `perm` maps one focus to a shuffled row order."""
        base = np.arange(self.n) if rows is None else rows
        p = perm or {}
        take = lambda k: p.get(k, base)
        iu = np.triu_indices(len(base), 1)
        eq = lambda v: (v[:, None] == v[None, :])[iu].astype(float)
        oh = self.OH[take('same_reader')]
        mi = self.midx[take('same_motif')]
        X = np.column_stack([
            ((oh @ oh.T) > 0).astype(float)[iu],
            self.S[mi[:, None], mi[None, :]][iu].astype(float),
            eq(self.cell[take('same_cell')]),
            eq(self.targ[take('same_target')]),
        ])
        return X, eq(self.yraw[base])


def fit(X, y):
    if len(np.unique(y)) < 2:
        return None
    try:
        m = sm.Logit(y, sm.add_constant(X)).fit(disp=0, method='lbfgs', maxiter=80)
    except Exception:
        return None
    return {FOCI[i]: float(np.exp(m.params[i + 1])) for i in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nperm', type=int, default=5000)
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--out', default=os.path.join(DATA, 'fig4', 'reader_odds_grid.csv'))
    args = ap.parse_args()

    d, sim = decoupled_subset(), motif_sim()
    print(f'decoupled subset: {len(d)} dependencies', flush=True)
    ref = json.load(open(data('fig4', 'reader_function_results.json')))
    rows = []
    for col in ENDPOINTS:
        dd = d[~d[col].isin(DROP)].dropna(subset=[col]).reset_index(drop=True)
        D = Dyads(dd, col, sim)
        X, y = D.build()
        obs = fit(X, y)
        key = 'region' if col.startswith('region') else 'outcome'
        print(f'\n{col}: n={len(dd)} dyads={len(y)} cells={dd.cell.nunique()}', flush=True)
        for f in FOCI:                       # the published ORs must come back unchanged
            got, want = obs[f], ref[key][f]
            flag = 'ok' if abs(got - want) < 1e-6 else f'MISMATCH (published {want:.4f})'
            print(f'   OR {f:12s} {got:8.4f}   {flag}', flush=True)

        rng = np.random.default_rng(0)
        boot = []
        while len(boot) < args.nboot:
            r = fit(*D.build(rows=rng.choice(len(dd), len(dd), replace=True)))
            if r:
                boot.append(r)
        for f in FOCI:
            # crc32, not hash(): Python salts string hashing per process, so hash() here
            # would reseed differently on every run and the P-values would not reproduce.
            rng = np.random.default_rng(zlib.crc32(f'{col}|{f}'.encode()))
            null = []
            while len(null) < args.nperm:
                r = fit(*D.build(perm={f: rng.permutation(len(dd))}))
                if r:
                    null.append(r[f])
            null = np.array(null)
            c = null.mean()
            p = (np.sum(np.abs(null - c) >= abs(obs[f] - c)) + 1) / (len(null) + 1)
            lo, hi = np.percentile([b[f] for b in boot], [2.5, 97.5])
            rows.append(dict(col=col, focus=f, perm_field=f.replace('same_', ''), OR=obs[f],
                             lo=lo, hi=hi, perm_p=p, null_med=float(np.median(null)),
                             n=len(dd), cells=int(dd.cell.nunique())))
            print(f'   {f:12s} OR={obs[f]:7.3f} CI[{lo:7.3f},{hi:10.3f}] '
                  f'null_med={np.median(null):.3f} perm_p={p:.5f}', flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
