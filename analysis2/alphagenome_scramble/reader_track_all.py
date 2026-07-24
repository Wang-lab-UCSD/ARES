#!/usr/bin/env python
"""Scramble the partner motif for EVERY mode-resolved hidden reader AlphaGenome can measure.

Supersedes reader_track_ctrl, which only scored dependencies that happened to be in an earlier
AlphaGenome batch and additionally required an unambiguous GO valence. Neither restriction belongs in
the binding-level question:

  Figure 1 (binding)   does the nominated reader leave the partner motif, more than control TFs at
                       the same locus? Needs only an AlphaGenome track for the reader.
                       Population: 180 of the 208 mode-resolved hidden readers (28 have no track).

  Figure 2 (function)  does transcription follow the reader's valence? A strict subset of Figure 1,
                       additionally needing CAGE in that cell line and an unambiguous GO valence.
                       Population: 80 (60 activator / 20 repressor).

Valence is recorded but never used as a filter here, so one run serves both figures. Everything else
matches reader_track_ctrl: same peak selection, same motif-vs-control-window difference-in-
differences, same 12 non-reader non-same-family control TFs read from the identical prediction.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_v3_all as V
from reader_track_full import toks, valence, reader_tracks
from reader_track_ctrl import controls_for, predict, REQ
from alphagenome.models import dna_client

AN = Path('/new-stg/home/hanbei/ARES/analysis')
OUT = HERE / 'reader_track_all'
OUT.mkdir(exist_ok=True)
SIG = OUT / 'reader_track_all_signal.csv'
CAND = OUT / 'candidates_all.csv'
NPEAKS = 50


def build_all():
    """Every mode-resolved hidden reader with an AlphaGenome track. No valence filter, no
    requirement that the pair was in a previous batch."""
    a = pd.read_csv(AN / 'ares_atlas_final/ares_atlas.tsv', sep='\t')
    a['partner'] = a.partner.astype(str).str.upper()
    h = a[(a.reader_category == 'hidden') &
          (a.dichotomy.isin(['SEQUENCE', 'PROTEIN', 'CONTEXT']))][
        ['cell', 'target', 'partner', 'reader', 'dichotomy']].drop_duplicates().reset_index(drop=True)
    h['rv'] = h.reader.map(valence)                      # recorded, NOT filtered on
    h['tracks'] = [';'.join(reader_tracks(r.cell, r.reader)) for r in h.itertuples()]
    m = h[h.tracks != ''].reset_index(drop=True)
    m.to_csv(CAND, index=False)
    print(f'mode-resolved hidden readers: {len(h)};  AG has a reader track: {len(m)}', flush=True)
    print(f'  with unambiguous valence (fig 2 eligible): {int(m.rv.notna().sum())}', flush=True)
    return m


def main():
    C = build_all()
    rows, done = [], set()
    if SIG.exists():
        prev = pd.read_csv(SIG)
        rows = prev.to_dict('records')
        done = set(map(tuple, prev[['cell', 'target', 'partner']].drop_duplicates().values))
        print(f'resuming: {len(done)} dependencies already scored', flush=True)
    client = dna_client.create(V.load_api_key(0))
    onto = pd.read_csv(V.ONTOLOGY_TSV, sep='\t').set_index('cell_line').efo_id.to_dict()

    for i, P in enumerate(C.itertuples(), 1):
        if (P.cell, P.target, P.partner) in done:
            continue
        if P.cell not in onto:
            print(f'[skip {i}] {P.cell}: no EFO', flush=True)
            continue
        pm = V.find_partner_motif_id(P.partner)
        if pm is None:
            print(f'[skip {i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell}): no JASPAR motif',
                  flush=True)
            continue
        mid = pm[0] if isinstance(pm, tuple) else pm
        root = Path(f'/new-stg/home/hanbei/data/TF_ENCODE4/{P.cell}')
        try:
            peaks = V.top_peaks_with_partner_motif(P.target, mid, n=NPEAKS, tf_data_root=root)
        except Exception as e:
            print(f'[skip {i}/{len(C)}] {P.target}: peaks failed {type(e).__name__}', flush=True)
            continue
        if not peaks:
            print(f'[skip {i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell}): no motif+ peaks',
                  flush=True)
            continue
        rds = P.tracks.split(';')
        ctrls = controls_for(P.cell, [P.target] + toks(P.partner) + toks(P.reader))
        t0, n_ok = time.time(), 0
        for pi, pk in enumerate(peaks):
            try:
                m_ = predict(client, pk, 'motif', onto[P.cell], P.target, P.partner, rds, ctrls)
                c_ = predict(client, pk, 'ctrl', onto[P.cell], P.target, P.partner, rds, ctrls)
            except Exception as e:
                print(f'    peak {pi} failed: {type(e).__name__}: {str(e)[:50]}', flush=True)
                continue
            n_ok += 1
            rows.append(dict(cell=P.cell, target=P.target, partner=P.partner, reader=P.reader,
                             reader_tracks=P.tracks, ctrl_tracks=';'.join(ctrls),
                             n_ctrl=m_['n_ctrl'], rv=P.rv, dichotomy=P.dichotomy, peak=pi,
                             fimo=pk['fimo_score'],
                             **{f'paired_{k}': m_[f'{k}_log2fc'] - c_[f'{k}_log2fc']
                                for k in ('target', 'partner', 'reader', 'ctrl', 'CAGE')}))
        pd.DataFrame(rows).to_csv(SIG, index=False)
        print(f'[{i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell})  reader {P.tracks} '
              f'rv={P.rv}  {n_ok}/{len(peaks)} peaks  {time.time() - t0:.0f}s', flush=True)

    D = pd.DataFrame(rows)
    D['spec'] = D.paired_reader - D.paired_ctrl
    D.to_csv(SIG, index=False)
    dep = D.groupby(['cell', 'target', 'partner', 'reader', 'rv', 'dichotomy'], dropna=False).agg(
        reader_x=('paired_reader', 'median'), ctrl_x=('paired_ctrl', 'median'),
        spec_x=('spec', 'median'), cage_y=('paired_CAGE', 'median'),
        target_x=('paired_target', 'median'), n=('peak', 'size')).reset_index()
    dep.to_csv(OUT / 'per_dependency.csv', index=False)
    from scipy.stats import wilcoxon
    s = dep.spec_x.dropna()
    print(f'\nscored dependencies: {len(dep)}')
    print(f'  FIG 1  reader minus same-locus controls: median {s.median():+.4f}, '
          f'below zero in {int((s < 0).sum())}/{len(s)}, Wilcoxon P={wilcoxon(s).pvalue:.2g}')
    f2 = dep.dropna(subset=['cage_y']).query("rv in ['ACT','REP']")
    print(f'  FIG 2  with CAGE and valence: {len(f2)}  ({f2.rv.value_counts().to_dict()})')


if __name__ == '__main__':
    main()
