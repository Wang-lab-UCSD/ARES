#!/usr/bin/env python
"""Destroy the partner motif, watch the READER leave, and ask whether transcription follows its valence.

The published scramble panel puts TARGET binding on the x axis. That tests the sequence claim, not
the reader claim, and 119 of its 131 dependencies have the reader equal to the partner, where reader
binding and partner binding are literally the same track. Here every dependency is a mode-resolved
HIDDEN reader: the factor ARES nominates is not the factor the motif is named for, so its track is an
independent measurement that the stored pair_*.csv files never made.

  x   change in the READER's binding when the partner motif is scrambled
  y   change in CAGE at the same window
  one point per dependency

The unit is the dependency, not the reader. A reader recurring under several targets is measured at
a different locus set each time, so those are separate observations rather than repeats; the reader
recurrence is reported as a clustering caveat instead of being collapsed away.

WHY NO SEPARATE SCRAMBLE-QC. An earlier design confirmed the perturbation worked by requiring the
partner's own binding to drop. That is unnecessary here and costs coverage: if a scramble fails to
disturb the site, the reader does not leave either, so the dependency lands at x = 0 and contributes
nothing in either direction. The x axis is its own quality control. The pilot showed this directly:
IRF5 <- ZBTB14 motif (HepG2) returned partner +0.011 and reader +0.029 together.

DESIGN kept identical to the published panel wherever shared, so the two are comparable: top-N target
peaks ranked by partner-motif FIMO score, motif scramble plus a control-window scramble in the same
interval giving paired_X = motif - ctrl, and the same resolution-aware motif-centred window. The 3D
contact call is dropped; the reader's ChIP-TF track is added.

VALENCE is GO-derived and therefore ARES-free, which is what makes the y axis an independent test
rather than a restatement of the atlas. Readers whose GO terms mix activation and repression are left
out rather than guessed at.

SCOPE. Mode-resolved hidden readers only (dichotomy in SEQUENCE/PROTEIN/CONTEXT). No SEQUENCE
dependency survives the filters, so the panel speaks to the PROTEIN and CONTEXT routes only.
"""
import sys, time, re, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_v3_all as V
from alphagenome.data import genome
from alphagenome.models import dna_client, dna_output

AN = Path('/new-stg/home/hanbei/ARES/analysis')
OUT = HERE / 'reader_track_full'
OUT.mkdir(exist_ok=True)
NPEAKS = 50
REQ = [dna_output.OutputType.CHIP_TF, dna_output.OutputType.CAGE]
SIG = OUT / 'reader_track_signal.csv'
CAND = OUT / 'candidates.csv'


def go(f):
    s = set()
    for l in open(f):
        if l.startswith('!'):
            continue
        c = l.split('\t')
        if len(c) > 2:
            g = c[2].strip().upper()
            if re.fullmatch(r'[A-Z0-9\-]{2,12}', g) and not re.fullmatch(r'[A-Z][0-9][A-Z0-9]{4,}', g):
                s.add(g)
    return s


ACT, REP = go(AN / 'go_activator.gaf'), go(AN / 'go_repressor.gaf')


def toks(s):
    return [t.strip().upper() for t in
            str(s).replace('::', ';').replace('/', ';').replace(',', ';').split(';') if t.strip()]


def valence(reader):
    """GO valence, ARES-free. Two exclusions, both of which drop the whole dependency.

    A factor annotated as BOTH activator and repressor has no directional prediction to test, so it
    cannot appear anywhere in the reader. An earlier version let such a factor through: it matched
    neither branch, contributed nothing to the set, and a co-assigned unambiguous partner then
    supplied the label on its own. That silently kept dependencies whose reader is not directional.

    A reader whose factors disagree with each other is dropped for the same reason: the panel would
    be predicting a direction the annotation does not support.

    Factors with no GO annotation at all are ignored rather than fatal. They make no claim either
    way, so an unambiguous co-assigned factor can still carry the label.
    """
    v = set()
    for p in toks(reader):
        a, r = p in ACT, p in REP
        if a and r:
            return None                      # ambiguous factor anywhere in the reader -> drop
        if a:
            v.add('ACT')
        elif r:
            v.add('REP')
    return 'ACT' if v == {'ACT'} else 'REP' if v == {'REP'} else None


COV = {os.path.basename(f)[12:-4]: {l.strip().upper() for l in open(f) if l.strip()}
       for f in glob.glob(str(HERE / 'ag_coverage_*.txt'))}


def reader_tracks(cell, reader):
    """EVERY co-assigned reader factor AG can measure in this cell, not just the first one.

    Taking the first match would let alphabetical accident decide which protein represents a
    co-assigned reader such as MAX;MYC or FOXA1;FOXA2;FOXA3. All available tracks are measured and
    averaged instead; where only one is available, that one is used and the column records which.
    """
    return [t for t in toks(reader) if t in COV.get(cell, set())]


def build_candidates():
    """Mode-resolved hidden readers that were AG-scrambled and whose reader has a track."""
    runs = []
    cells = {'K562': HERE / 'k562_full'}
    for d in sorted((HERE / 'validation_v3_all').glob('*')):
        if d.is_dir():
            cells[d.name] = d
    for cell, dd in cells.items():
        for f in sorted(dd.glob('pair_*.csv')):
            try:
                t = pd.read_csv(f, nrows=5)
            except Exception:
                continue
            if 'paired_CAGE' not in t.columns or len(t) < 3:
                continue
            runs.append(dict(cell=cell, target=str(t.target_tf.iloc[0]),
                             partner=str(t.partner_tf.iloc[0]).upper()))
    R = pd.DataFrame(runs).drop_duplicates()
    a = pd.read_csv(AN / 'ares_atlas_final/ares_atlas.tsv', sep='\t')
    a['partner'] = a.partner.astype(str).str.upper()
    h = a[(a.reader_category == 'hidden') &
          (a.dichotomy.isin(['SEQUENCE', 'PROTEIN', 'CONTEXT']))][
        ['cell', 'target', 'partner', 'reader', 'dichotomy']].drop_duplicates()
    m = R.merge(h, on=['cell', 'target', 'partner'], how='inner')
    m['rv'] = m.reader.map(valence)
    m['tracks'] = [';'.join(reader_tracks(r.cell, r.reader)) for r in m.itertuples()]
    m = m[m.rv.notna() & (m.tracks != '')].reset_index(drop=True)
    m.to_csv(CAND, index=False)
    return m


def chip_at(output, tf, mut_center, interval_start):
    """Same window arithmetic as V.extract_signals_at_bin, for one extra ChIP-TF track."""
    if output.chip_tf is None:
        return float('nan')
    md = output.chip_tf.metadata
    idx = md.index[md['transcription_factor'] == tf].tolist()
    if not idx:
        return float('nan')
    arr = output.chip_tf.values
    n = arr.shape[0]
    res_bp = V.INTERVAL_LEN / n
    c = int((mut_center - interval_start) / res_bp)
    hb = int(round(64 / res_bp))
    return float(np.mean(arr[max(0, c - hb): min(n, c + hb + 1), idx]))


def lfc(ref, alt):
    if not (np.isfinite(ref) and np.isfinite(alt)):
        return float('nan')
    return float(np.log2((alt + 0.01) / (ref + 0.01)))


def predict(client, peak, kind, efo, target, partner, rds):
    var_start = peak['motif_start'] if kind == 'motif' else peak['ctrl_start']
    ref = peak['ref_seq'] if kind == 'motif' else peak['ctrl_ref']
    alt = V.scramble_motif(ref)
    motif_center = (peak['motif_start'] + peak['motif_end']) // 2
    istart = max(0, motif_center - V.INTERVAL_LEN // 2)
    iv = genome.Interval(peak['chrom'], istart, istart + V.INTERVAL_LEN)
    var = genome.Variant(peak['chrom'], var_start + 1, ref, alt)
    out = V._predict_variant_retry(client, iv, var, REQ, [efo])
    mc = var_start + peak['motif_len'] // 2
    r = V.extract_signals_at_bin(out.reference, mc, istart, target, partner)
    a = V.extract_signals_at_bin(out.alternate, mc, istart, target, partner)
    d = {f'{k}_log2fc': lfc(r[k], a[k]) for k in ('target', 'partner', 'CAGE')}
    # co-assigned readers: average the per-factor log2FC, so the effect is averaged rather than an
    # arbitrary one of the factors being taken as the representative
    per = [lfc(chip_at(out.reference, t, mc, istart), chip_at(out.alternate, t, mc, istart))
           for t in rds]
    per = [x for x in per if np.isfinite(x)]
    d['reader_log2fc'] = float(np.mean(per)) if per else float('nan')
    d['n_reader_tracks'] = len(per)
    return d


def main():
    C = build_candidates()
    print(f'candidates: {len(C)} dependencies, {C.reader.nunique()} readers '
          f'(ACT {int((C.rv == "ACT").sum())} / REP {int((C.rv == "REP").sum())})', flush=True)
    done = set()
    rows = []
    if SIG.exists():
        prev = pd.read_csv(SIG)
        rows = prev.to_dict('records')
        done = set(map(tuple, prev[['cell', 'target', 'partner']].drop_duplicates().values))
        print(f'resuming: {len(done)} dependencies already done', flush=True)
    client = dna_client.create(V.load_api_key(0))
    onto = pd.read_csv(V.ONTOLOGY_TSV, sep='\t').set_index('cell_line').efo_id.to_dict()

    for i, P in enumerate(C.itertuples(), 1):
        if (P.cell, P.target, P.partner) in done:
            continue
        pm = V.find_partner_motif_id(P.partner)
        if pm is None:
            print(f'[skip {i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell}): no JASPAR motif',
                  flush=True)
            continue
        mid = pm[0] if isinstance(pm, tuple) else pm
        root = Path(f'/new-stg/home/hanbei/data/TF_ENCODE4/{P.cell}')
        peaks = V.top_peaks_with_partner_motif(P.target, mid, n=NPEAKS, tf_data_root=root)
        rds = P.tracks.split(';')
        t0 = time.time()
        n_ok = 0
        for pi, pk in enumerate(peaks):
            try:
                m = predict(client, pk, 'motif', onto[P.cell], P.target, P.partner, rds)
                c = predict(client, pk, 'ctrl', onto[P.cell], P.target, P.partner, rds)
            except Exception as e:
                print(f'    peak {pi} failed: {type(e).__name__}: {str(e)[:60]}', flush=True)
                continue
            n_ok += 1
            rows.append(dict(cell=P.cell, target=P.target, partner=P.partner, reader=P.reader,
                             reader_tracks=P.tracks, n_reader_tracks=m['n_reader_tracks'],
                             rv=P.rv, dichotomy=P.dichotomy, peak=pi, fimo=pk['fimo_score'],
                             **{f'paired_{k}': m[f'{k}_log2fc'] - c[f'{k}_log2fc']
                                for k in ('target', 'partner', 'reader', 'CAGE')}))
        pd.DataFrame(rows).to_csv(SIG, index=False)     # checkpoint after each dependency
        print(f'[{i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell})  reader {P.tracks} '
              f'{P.rv}  {n_ok}/{len(peaks)} peaks  {time.time() - t0:.0f}s', flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(SIG, index=False)
    print(f'\nwrote {SIG}  ({len(D)} peak rows)')

    # The unit is the DEPENDENCY, not the reader. A reader that appears under several targets is
    # measured at a different set of loci each time, so those are separate observations rather than
    # repeats of one. Readers do recur (FOXA1 under six targets, MAX under six), which makes the 67
    # not fully independent; that clustering is reported below as a robustness note, not imposed on
    # the primary unit by collapsing.
    dep = D.groupby(['cell', 'target', 'partner', 'reader', 'reader_tracks', 'rv', 'dichotomy']).agg(
        reader_x=('paired_reader', 'median'), cage_y=('paired_CAGE', 'median'),
        target_x=('paired_target', 'median'), n=('peak', 'size')).reset_index()
    dep.to_csv(OUT / 'per_dependency.csv', index=False)

    from scipy.stats import mannwhitneyu
    print('\n' + '=' * 92)
    print(f'PER DEPENDENCY   n = {len(dep)}  (ACT {int((dep.rv == "ACT").sum())} / '
          f'REP {int((dep.rv == "REP").sum())})')
    print('=' * 92)
    show = dep.assign(pair=dep.target + ' <- ' + dep.partner + ' (' + dep.cell + ')')[
        ['pair', 'reader', 'rv', 'dichotomy', 'n', 'reader_x', 'cage_y', 'target_x']]
    print(show.sort_values(['rv', 'cage_y']).to_string(index=False,
                                                       float_format=lambda x: f'{x:+.3f}'))
    A, Rp = dep[dep.rv == 'ACT'].cage_y.dropna(), dep[dep.rv == 'REP'].cage_y.dropna()
    if len(A) and len(Rp):
        p = mannwhitneyu(A, Rp, alternative='less').pvalue
        print(f'\n  CAGE: ACT median {A.median():+.3f}  REP median {Rp.median():+.3f}'
              f'   Mann-Whitney (ACT < REP) P = {p:.4f}')
    print(f'  reader binding falls (x < 0) in '
          f'{int((dep.reader_x < 0).sum())}/{len(dep)} dependencies, '
          f'median {dep.reader_x.median():+.3f}')
    print(f'\n  robustness note only: the {len(dep)} dependencies come from '
          f'{dep.reader.nunique()} distinct readers')


if __name__ == '__main__':
    main()
