#!/usr/bin/env python
"""Same scramble, but the reader's response is now measured against non-reader TFs at the same site.

The uncontrolled version of this analysis (reader_track_full.py) is not interpretable. Across its
dependencies the reader's predicted binding change correlated with the target's at rho = +0.90, and
within a dependency, peak by peak, at rho +0.42 to +0.98. AlphaGenome drops every ChIP-TF track at a
window together when the local sequence is destroyed, because they share the same local sequence and
accessibility drivers. "The reader left" was therefore indistinguishable from "everything left", and
a TF with no relationship to the motif would very likely have scored the same.

THE CONTROL. At every locus, the same prediction output is also read out for NCTRL non-reader TFs
drawn from that cell's AlphaGenome coverage. No extra API calls are needed: the expensive step is the
prediction, and these are additional columns of an array already in hand. The reported quantity is

    reader_specificity = reader log2FC  -  median( control TF log2FC )   at the same locus

so a site where everything collapses contributes zero, and only a reader that leaves MORE than the
local background counts as reading the motif.

Controls exclude the target, the partner, the reader, and any factor sharing a JASPAR family with
them, so a paralog of the reader cannot be used to argue the reader is unremarkable. They are drawn
once per cell with a fixed seed so every dependency in that cell is scored against the same
background, and the draw is recorded.

The y axis is unchanged: CAGE at the same window, tested against GO valence, which is ARES-free.
The unit remains the dependency.
"""
import sys, time, re, glob, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_v3_all as V
from reader_track_full import (go, toks, valence, COV, reader_tracks, build_candidates,
                               chip_at, lfc, ACT, REP, AN)
from alphagenome.data import genome
from alphagenome.models import dna_client, dna_output

OUT = HERE / 'reader_track_ctrl'
OUT.mkdir(exist_ok=True)
NPEAKS, NCTRL, SEED = 50, 12, 0
REQ = [dna_output.OutputType.CHIP_TF, dna_output.OutputType.CAGE]
SIG = OUT / 'reader_track_ctrl_signal.csv'

# JASPAR family lookup, so a control TF cannot be a paralog of the reader it is meant to contrast
# with. Same table ARES itself uses to tag motifs self / paralog / other during partner selection,
# which keeps the exclusion consistent with how the atlas was built. Heterodimer rows ('a::b') are
# expanded so each component carries the family.
FAMTAB = Path('/new-stg/home/hanbei/HanX/data/JASPAR/'
              'JASPAR2022_CORE_vertebrates_non-redundant_v2_family_and_class_updated.csv')
FAM = {}
try:
    fam = pd.read_csv(FAMTAB)
    for _, r in fam.iterrows():
        for t in toks(r['tf']):
            FAM.setdefault(t.upper(), str(r['family']))
    print(f'TF family table: {len(FAM)} factors', flush=True)
except Exception as e:
    raise SystemExit(f'cannot read the TF family table ({type(e).__name__}: {e}). Controls would '
                     f'then be filtered by exact name only, which would let a paralog of the reader '
                     f'sit in the control set and flatten the very contrast being measured.')


def families(names):
    return {FAM[n] for n in names if n in FAM}


def controls_for(cell, exclude_names):
    """NCTRL TFs from this cell's AG coverage, excluding the players and their families."""
    pool = sorted(COV.get(cell, set()))
    bad_names = {n.upper() for n in exclude_names}
    bad_fams = families(bad_names)
    pool = [t for t in pool if t not in bad_names and FAM.get(t) not in bad_fams]
    rng = np.random.default_rng(SEED)
    return list(rng.choice(pool, size=min(NCTRL, len(pool)), replace=False)) if pool else []


def predict(client, peak, kind, efo, target, partner, rds, ctrls):
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

    def track_lfc(tf):
        return lfc(chip_at(out.reference, tf, mc, istart), chip_at(out.alternate, tf, mc, istart))

    per = [x for x in (track_lfc(t) for t in rds) if np.isfinite(x)]
    d['reader_log2fc'] = float(np.mean(per)) if per else float('nan')
    cv = [x for x in (track_lfc(t) for t in ctrls) if np.isfinite(x)]
    d['ctrl_log2fc'] = float(np.median(cv)) if cv else float('nan')
    d['n_ctrl'] = len(cv)
    return d


def main():
    C = build_candidates()
    print(f'candidates: {len(C)} dependencies, {C.reader.nunique()} readers '
          f'(ACT {int((C.rv == "ACT").sum())} / REP {int((C.rv == "REP").sum())})', flush=True)
    rows, done = [], set()
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
        ctrls = controls_for(P.cell, [P.target] + toks(P.partner) + toks(P.reader))
        t0, n_ok = time.time(), 0
        for pi, pk in enumerate(peaks):
            try:
                m = predict(client, pk, 'motif', onto[P.cell], P.target, P.partner, rds, ctrls)
                c = predict(client, pk, 'ctrl', onto[P.cell], P.target, P.partner, rds, ctrls)
            except Exception as e:
                print(f'    peak {pi} failed: {type(e).__name__}: {str(e)[:60]}', flush=True)
                continue
            n_ok += 1
            rows.append(dict(cell=P.cell, target=P.target, partner=P.partner, reader=P.reader,
                             reader_tracks=P.tracks, ctrl_tracks=';'.join(ctrls),
                             n_ctrl=m['n_ctrl'], rv=P.rv, dichotomy=P.dichotomy, peak=pi,
                             fimo=pk['fimo_score'],
                             **{f'paired_{k}': m[f'{k}_log2fc'] - c[f'{k}_log2fc']
                                for k in ('target', 'partner', 'reader', 'ctrl', 'CAGE')}))
        pd.DataFrame(rows).to_csv(SIG, index=False)
        print(f'[{i}/{len(C)}] {P.target} <- {P.partner} motif ({P.cell})  reader {P.tracks} '
              f'{P.rv}  {n_ok}/{len(peaks)} peaks  {len(ctrls)} controls  '
              f'{time.time() - t0:.0f}s', flush=True)

    D = pd.DataFrame(rows)
    D['spec'] = D.paired_reader - D.paired_ctrl
    D.to_csv(SIG, index=False)
    dep = D.groupby(['cell', 'target', 'partner', 'reader', 'rv', 'dichotomy']).agg(
        reader_x=('paired_reader', 'median'), ctrl_x=('paired_ctrl', 'median'),
        spec_x=('spec', 'median'), cage_y=('paired_CAGE', 'median'),
        target_x=('paired_target', 'median'), n=('peak', 'size')).reset_index()
    dep.to_csv(OUT / 'per_dependency.csv', index=False)

    from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
    print('\n' + '=' * 100)
    print(f'PER DEPENDENCY   n = {len(dep)}   '
          f'(ACT {int((dep.rv == "ACT").sum())} / REP {int((dep.rv == "REP").sum())})')
    print('=' * 100)
    show = dep.assign(pair=dep.target + ' <- ' + dep.partner + ' (' + dep.cell + ')')[
        ['pair', 'reader', 'rv', 'n', 'reader_x', 'ctrl_x', 'spec_x', 'cage_y']]
    print(show.sort_values(['rv', 'spec_x']).to_string(index=False,
                                                       float_format=lambda x: f'{x:+.3f}'))
    s = dep.spec_x.dropna()
    print(f'\n  SPECIFICITY  reader minus same-locus control TFs')
    print(f'    median {s.median():+.4f}   below zero in {int((s < 0).sum())}/{len(s)}'
          f'   Wilcoxon vs 0 P = {wilcoxon(s).pvalue:.4f}' if len(s) >= 5 else '')
    print(f'    reader alone median {dep.reader_x.median():+.3f}   '
          f'controls median {dep.ctrl_x.median():+.3f}')
    m = dep.dropna(subset=['reader_x', 'ctrl_x'])
    if len(m) > 3:
        print(f'    Spearman(reader, controls) = {spearmanr(m.reader_x, m.ctrl_x).statistic:+.3f}'
              f'   (high means the site moves as a block)')
    A, Rp = dep[dep.rv == 'ACT'].cage_y.dropna(), dep[dep.rv == 'REP'].cage_y.dropna()
    if len(A) and len(Rp):
        print(f'\n  CAGE by valence: ACT median {A.median():+.3f}  REP median {Rp.median():+.3f}'
              f'   Mann-Whitney (ACT < REP) P = '
              f'{mannwhitneyu(A, Rp, alternative="less").pvalue:.4f}')
    print(f'\n  {len(dep)} dependencies from {dep.reader.nunique()} distinct readers '
          f'(clustering caveat, not collapsed)')


if __name__ == '__main__':
    main()
