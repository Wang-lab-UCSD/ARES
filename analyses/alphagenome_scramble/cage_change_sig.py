#!/usr/bin/env python
"""Per-dependency significance of the CAGE change, the companion of per_dependency_readerdrop_sig.

For each dependency the 50 scrambled loci give 50 paired CAGE values (motif window minus control
window). The question is whether that distribution is shifted away from zero at all, in either
direction, because activators and repressors are expected to move CAGE opposite ways. So this is a
two-sided Wilcoxon signed-rank against zero, Benjamini-Hochberg across dependencies.
"""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

S = pd.read_csv('reader_track_all/reader_track_all_signal.csv')
S = S.dropna(subset=['paired_CAGE'])
rows = []
for k, g in S.groupby(['cell', 'target', 'partner'], sort=False):
    v = g.paired_CAGE.values
    v = v[np.isfinite(v)]
    if len(v) < 5 or np.allclose(v, 0):
        p = 1.0
    else:
        p = wilcoxon(v).pvalue                       # two-sided: direction is set by valence
    rows.append(dict(cell=k[0], target=k[1], partner=k[2], n=len(v), med=np.median(v), p=p))
D = pd.DataFrame(rows)
D['q'] = multipletests(D.p, method='fdr_bh')[1]
D.to_csv('reader_track_all/per_dependency_cage_sig.csv', index=False)
print(f'{len(D)} dependencies, q<0.05 in {int((D.q < 0.05).sum())}')
