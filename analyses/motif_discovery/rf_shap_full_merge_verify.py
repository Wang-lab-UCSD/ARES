#!/usr/bin/env python3
"""Merge per-pair full SHAP rankings and verify the regulator vs the stored table.

Reads rf_shap_full_rank/pair_*.csv (full per-motif ranking with self/paralog/other),
derives per pair:
  - regulator       = first (lowest-rank) motif with category 'other'
  - regulator_rank
  - self_rank       = lowest rank among 'self' motifs
  - best_paralog    = lowest rank among 'paralog' motifs
Compares the derived regulator to rf_shap_top_regulator.csv (SHAP_Regulator_mean)
and reports how many pairs FLIPPED regulator.

Outputs:
  rf_shap_full_summary.csv   (one row per pair, incl. self_rank / best_paralog_rank)
  rf_shap_regulator_flips.csv (pairs whose regulator changed vs stored)
"""
import glob, os
import numpy as np
import pandas as pd

FULL_DIR = '/new-stg/home/hanbei/HanX/result/rf_shap_full_rank'
STORED   = '/new-stg/home/hanbei/HanX/result/rf_shap_top_regulator.csv'
OUT_SUM  = '/new-stg/home/hanbei/HanX/result/rf_shap_full_summary.csv'
OUT_FLIP = '/new-stg/home/hanbei/HanX/result/rf_shap_regulator_flips.csv'

files = sorted(glob.glob(f'{FULL_DIR}/pair_*.csv'))
print(f'merging {len(files)} per-pair ranking files...', flush=True)

rows = []
for f in files:
    d = pd.read_csv(f)
    if len(d) == 0:
        continue
    reg = d[d.category == 'other'].head(1)
    selfd = d[d.category == 'self']
    par = d[d.category == 'paralog']
    rows.append({
        'pair_idx': int(d.pair_idx.iloc[0]),
        'Cellline': d.Cellline.iloc[0],
        'Experiment': d.Experiment.iloc[0],
        'top_motif': d.iloc[0].motif_tf,
        'top_motif_category': d.iloc[0].category,
        'regulator': reg.motif_tf.iloc[0] if len(reg) else None,
        'regulator_rank': int(reg['rank'].iloc[0]) if len(reg) else None,
        'self_rank': int(selfd['rank'].min()) if len(selfd) else None,
        'best_paralog_rank': int(par['rank'].min()) if len(par) else None,
    })
summary = pd.DataFrame(rows).sort_values('pair_idx')
summary.to_csv(OUT_SUM, index=False)
print(f'wrote {OUT_SUM}  ({len(summary)} pairs)', flush=True)

# ---- verify vs stored ----
stored = pd.read_csv(STORED).reset_index().rename(columns={'index': 'pair_idx'})
m = summary.merge(stored[['pair_idx', 'SHAP_Regulator_mean', 'SHAP_Regulator_Rank_mean']], on='pair_idx', how='left')
m['stored_reg'] = m['SHAP_Regulator_mean'].astype(str).str.upper()
m['new_reg'] = m['regulator'].astype(str).str.upper()
both = m[m.regulator.notna() & m.SHAP_Regulator_mean.notna()].copy()
flip = both[both.stored_reg != both.new_reg]

print(f'\n=== Regulator verification ===')
print(f'pairs comparable:        {len(both)}')
print(f'regulator UNCHANGED:     {len(both) - len(flip)}  ({100*(len(both)-len(flip))/len(both):.1f}%)')
print(f'regulator FLIPPED:       {len(flip)}  ({100*len(flip)/len(both):.1f}%)')
flip[['pair_idx', 'Cellline', 'Experiment', 'stored_reg', 'new_reg',
      'SHAP_Regulator_Rank_mean', 'regulator_rank']].to_csv(OUT_FLIP, index=False)
if len(flip):
    print(f'\nFlipped pairs (first 20) -> {OUT_FLIP}:')
    print(flip[['Cellline', 'Experiment', 'stored_reg', 'new_reg']].head(20).to_string(index=False))
