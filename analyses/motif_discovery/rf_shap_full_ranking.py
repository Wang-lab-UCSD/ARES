#!/usr/bin/env python3
"""Re-run TreeSHAP (APPROXIMATE) per TF pair and save the FULL per-motif ranking.

Unlike rf_shap_regulator.py (which kept only the top regulator), this saves every
motif's mean(|SHAP|) importance, its rank, and a self / paralog / other label, so
downstream analyses (e.g. rank CDFs of self vs paralog vs other) are possible.

The top regulator / self_rank / best_paralog_rank all derive from this full file:
  - regulator      = first motif (lowest rank) with category 'other'
  - self_rank      = lowest rank among 'self' motifs
  - best_paralog   = lowest rank among 'paralog' motifs

Uses the SAME seeded subsample (seed=42, MAX_FULL=1000) and the SAME family logic
as rf_shap_regulator.py, so the 'mean' regulator is directly comparable to the
stored rf_shap_top_regulator.csv (for the verify step).

SLURM array: one pair per call (--pair_idx). Always uses --approximate behaviour.
"""
import sys
from motif_scan import parse_meme_file, motif_id_to_tf_name

import os
import argparse
import time
import re

import joblib
import numpy as np
import pandas as pd
import shap

JASPAR_MEME = '/new-stg/home/hanbei/HanX/data/JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_v2.meme'
FAMILY_CSV  = '/new-stg/home/hanbei/HanX/data/JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_v2_family_and_class_updated.csv'
RF_SUMMARY  = '/new-stg/home/hanbei/HanX/result/rf_top_motif_summary.csv'
BASE_DIR    = '/new-stg/home/hanbei/data/TF_ENCODE4_processed'
FULL_DIR    = '/new-stg/home/hanbei/HanX/result/rf_shap_full_rank'
MAX_FULL    = 1000


def _norm(name):
    """Factor name reduced to letters and digits, so a lookup does not turn on punctuation.

    Target names reach this module as ENCODE ChIP directory names and motif names as JASPAR
    symbols, and the two disagree on punctuation for at least one factor: the directory is
    NKX3_1_human while JASPAR calls the motif NKX3-1. Matched literally, such a target never
    recognises its own motif, so neither the self motif nor its paralogs are excluded from the
    partner-motif candidates. Composite separators are kept; only punctuation inside a name goes.
    """
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def make_get_families(tf_to_family):
    fam_norm = {_norm(k): v for k, v in tf_to_family.items()}

    def get_families(name):
        name = str(name).lower()
        fams = set()
        direct = tf_to_family.get(name, '') or fam_norm.get(_norm(name), '')
        if direct:
            fams |= {f.strip() for f in direct.split(',')}
        if '::' in name:
            for comp in name.split('::'):
                s = tf_to_family.get(comp.strip(), '') or fam_norm.get(_norm(comp), '')
                if s:
                    fams |= {f.strip() for f in s.split(',')}
        return fams
    return get_families


def categorize(motif_tf, target_tf, get_families):
    """self / paralog / other — mirrors the skip logic in rf_shap_regulator.find_regulator."""
    target_lower = target_tf.lower()
    comps = {c.strip() for c in str(motif_tf).split('::')}
    if target_lower in comps or _norm(target_tf) in {_norm(c) for c in comps}:
        return 'self'
    target_fams = get_families(target_lower)
    motif_fams = get_families(motif_tf)
    if target_fams and motif_fams and (target_fams & motif_fams):
        return 'paralog'
    return 'other'


def process_pair(pair_idx, cl, tf, motif_tfs, get_families, approximate=True):
    out_path = f"{FULL_DIR}/pair_{pair_idx:05d}.csv"
    tf_dir = f"{BASE_DIR}/{cl}/{tf}_human"
    model_path  = f"{tf_dir}/HanX_JASPAR_sklearn/RandomForest.joblib"
    scores_path = f"{tf_dir}/HanX_JASPAR_sklearn/motif_scores_all.npy"
    if not (os.path.exists(model_path) and os.path.exists(scores_path)):
        print(f"  [skip] {pair_idx} ({cl}/{tf}): missing files", flush=True)
        return

    t0 = time.time()
    model = joblib.load(model_path)
    X = np.load(scores_path)
    n_full = X.shape[0]
    if n_full > MAX_FULL:
        np.random.seed(42)                        # same seed as original -> same subsample
        idx = np.random.choice(n_full, MAX_FULL, replace=False)
        X = X[idx]
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X, check_additivity=False, approximate=approximate)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    imp_mean = np.abs(shap_values).mean(axis=0)   # canonical metric

    order = np.argsort(imp_mean)[::-1]             # descending importance
    rows = []
    for rank_idx, i in enumerate(order, 1):
        mt = motif_tfs[i]
        rows.append({
            'rank': rank_idx,
            'motif_tf': mt.upper(),
            'imp_mean': float(imp_mean[i]),
            'category': categorize(mt, tf, get_families),
        })
    df = pd.DataFrame(rows)
    df.insert(0, 'pair_idx', pair_idx)
    df.insert(1, 'Cellline', cl)
    df.insert(2, 'Experiment', tf.upper())
    df.to_csv(out_path, index=False)
    reg = df[df.category == 'other'].head(1)
    print(f"  [done] {pair_idx} {cl}/{tf}: regulator={reg.motif_tf.iloc[0] if len(reg) else 'NA'} "
          f"rank={reg['rank'].iloc[0] if len(reg) else 'NA'}  ({time.time()-t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair_idx', type=int, default=None)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--exact', action='store_true', help='Use exact TreeSHAP (default: approximate)')
    args = ap.parse_args()
    approximate = not args.exact
    os.makedirs(FULL_DIR, exist_ok=True)

    motifs_d  = parse_meme_file(JASPAR_MEME)
    motif_ids = list(motifs_d.keys())
    motif_tfs = [motif_id_to_tf_name(m, JASPAR_MEME).lower() for m in motif_ids]

    fam = pd.read_csv(FAMILY_CSV).drop_duplicates()
    fam['family'] = fam['family'].fillna('').str.strip("[]'")
    get_families = make_get_families(fam.set_index('tf')['family'].to_dict())

    rf = pd.read_csv(RF_SUMMARY)
    rf['Experiment'] = rf['Experiment'].astype(str).str.upper()

    if args.pair_idx is not None:
        r = rf.iloc[args.pair_idx]
        process_pair(args.pair_idx, r['Cellline'], r['Experiment'], motif_tfs, get_families, approximate)
    else:
        end = args.end if args.end is not None else len(rf)
        for i in range(args.start, end):
            r = rf.iloc[i]
            process_pair(i, r['Cellline'], r['Experiment'], motif_tfs, get_families, approximate)


if __name__ == '__main__':
    main()
