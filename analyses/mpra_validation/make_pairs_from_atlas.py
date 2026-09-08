#!/usr/bin/env python
"""Per-cell-line pair lists for the MPRA validation, taken from the atlas.

The pairs to test are the atlas dependencies themselves, so nothing has to be excluded downstream:
cell lines ARES never ran on never appear, and paralog pairs -- where the "partner motif" is really
the target's own motif -- are already absent from the atlas.

Emits <CELL>_pairs.tsv with the columns run_validation_combined.py expects (pair_id, target_tf,
partner_tf), one file per cell line that both the atlas and the PARM library cover.

Run as:  python make_pairs_from_atlas.py [--out-dir pairs_by_cellline]
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('ARES_DATA', os.path.join(HERE, os.pardir, os.pardir, 'data'))

# The PARM library covers these cell lines; the atlas spells two of them differently.
# HCT116 is in the library but not in the atlas, so it is simply not listed here.
CELLS = {'HepG2': 'HepG2', 'K562': 'K562', 'HEK293': 'HEK293', 'MCF7': 'MCF.7'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', default=os.path.join(DATA, 'ares_atlas.tsv'))
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'pairs_by_cellline'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    atlas = pd.read_csv(args.atlas, sep='\t')
    total = 0
    for mpra_name, atlas_name in CELLS.items():
        sub = atlas[atlas.cell == atlas_name]
        pairs = (pd.DataFrame({
            'pair_id': mpra_name + '_' + sub.target.astype(str) + '_' + sub.partner.astype(str),
            'target_tf': sub.target.values,
            'partner_tf': sub.partner.values,
        }).drop_duplicates(subset='pair_id').reset_index(drop=True))
        path = os.path.join(args.out_dir, f'{mpra_name}_pairs.tsv')
        pairs.to_csv(path, sep='\t', index=False)
        total += len(pairs)
        print(f'{mpra_name}: {len(pairs)} pairs -> {path}')
    print(f'{total} pairs over {len(CELLS)} cell lines')


if __name__ == '__main__':
    main()
