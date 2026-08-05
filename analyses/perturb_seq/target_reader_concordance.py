#!/usr/bin/env python
"""Does the factor ARES names as the reader behave like the reader in a perturbation screen?

A partner motif predicts a target's binding, and ARES nominates one factor as the reader of that
motif. The obvious objection is that a motif-positive peak is occupied by a dozen factors at once,
so naming any one of them costs nothing. This script turns that objection into a test.

For each dependency we take the genes at the target's partner-motif-positive peaks, keep the ones
the TARGET's own knockdown moves, and ask what fraction of them a second knockdown moves in the
SAME direction. That fraction is scored for the ARES reader and for N_CTRL control factors chosen
to occupy the same peaks of the same target to the same extent. Matching on occupancy rather than
on knockdown strength is what makes the comparison sharp: a control that does not bind these loci
would lose trivially and would prove nothing.

Because direction agreement is measured over independently perturbed genes, an unrelated factor is
expected near 50%, so the control cloud doubles as a null anchor.

Per dependency
  1  peaks of the target are partitioned by the presence of the partner motif (FIMO, see below)
  2  genes whose promoter window overlaps a motif-positive peak, kept if |z_target| > Z_CUT
  3  reader occupancy = fraction of those motif-positive peaks overlapped by the reader's own peaks
  4  controls = the N_CTRL factors whose occupancy of the same peaks is closest to the reader's,
     drawn from annotated TFs that are in the CRISPRi library, pass knockdown QC, and have their
     own ChIP-seq in the same cell type
  5  concordance with target-KD, reader vs its controls

Inputs
  BULK      Replogle et al. (Cell 2022) genome-wide CRISPRi Perturb-seq in K562, authors' normalized
            pseudobulk. Cells were FACS-sorted on GFP three days after sgRNA transduction and
            harvested for scRNA-seq eight days after transduction.
  PEAKS     ENCODE4 narrowPeak per TF, complete peak set, no top-N cap
  COORDS    data/fig4/gene_coords.csv, gene coordinates with strand, restricted to the 8,444 genes
            quantified in the screen; gene assignment is therefore limited to that set
  MEME      combined JASPAR / HOCOMOCO v12 / CIS-BP motif database; only JASPAR entries are used
  ATLAS     ARES atlas, for the (target, partner, reader) dependencies to test

Output
  data/fig4/target_reader_concordance.csv, one row per dependency; this is what figures/fig4.py
  reads. Concordances are stored as fractions, not percentages.

Run as:  python target_reader_concordance.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- configuration
CELL = 'K562'
Z_CUT = 1.96        # |z| above which the target's knockdown is taken to have moved a gene
FOLD_MAX = 0.80     # knockdown QC: residual expression of the targeted transcript (>20% depletion)
MIN_EXPR = 0.05     # knockdown QC: expression in control cells
N_CTRL = 10         # occupancy-matched control factors per reader
MIN_GENES = 10      # minimum target-moved genes for a dependency to be scored
MIN_PEAKS = 20      # minimum motif-positive peaks for a dependency to be scored
FIMO_P = 1e-4       # FIMO significance threshold
HALF_WIN = 150      # FIMO is run on +/- HALF_WIN bp around the peak CENTRE, not the narrowPeak summit
PROMOTER = (-2000, 500)   # gene window relative to the TSS, strand-aware

ROOT = Path(os.environ.get('ARES_ROOT', '/new-stg/home/hanbei/ARES'))
DATA = Path(os.environ.get('ARES_DATA', ROOT / 'data'))
BULK = Path(os.environ.get(
    'PERTURBSEQ_BULK',
    '/new-stg/home/alex/projects/grn_inference/causalbench/data_cache/'
    'K562_gwps_normalized_bulk_01.h5ad'))
PEAKS = Path(os.environ.get('ENCODE_PEAKS', f'/new-stg/home/hanbei/data/TF_ENCODE4/{CELL}'))
GENOME = Path(os.environ.get('GENOME_FASTA', '/new-stg/home/hanbei/data/hg38.fa'))
MEME = Path(os.environ.get(
    'MOTIF_MEME', '/new-stg/home/hanbei/data/combined_TF_motifs_jaspar_hocomoco12_cisbp.meme'))
ATLAS = DATA / 'ares_atlas.tsv'
COORDS = DATA / 'fig4' / 'gene_coords.csv'
# FIMO partitions are an expensive intermediate, so they are cached next to the data rather than
# recomputed; delete the directory to force a rescan
CACHE = Path(os.environ.get('MOTIF_PARTITION_CACHE', DATA / 'fig4' / 'motif_partition'))
OUT = DATA / 'fig4' / 'target_reader_concordance.csv'

sys.path.insert(0, str(ROOT))
from src.utils.bioio import find_motif_ids_for_tf, parse_fimo_tsv  # noqa: E402


# ---------------------------------------------------------------- peaks and motif partition
ENCODE_BED = re.compile(r'^ENCFF[0-9A-Z]{6}\.bed$')


def find_bed(tf):
    """The narrowPeak file for a TF in this cell type, or None if it was not assayed.

    Matched on the bare ENCODE accession: some directories also hold derived files written there
    by earlier analyses (*_shuffled.bed, *_decomp.bed, *_for_fimo.bed), and picking one of those
    would silently swap the peak set.
    """
    d = PEAKS / f'{tf}_human'
    beds = sorted(f for f in d.glob('*.bed') if ENCODE_BED.match(f.name)) if d.exists() else []
    return beds[0] if beds else None


def load_peaks(tf):
    """Complete peak set. No top-N cap: the motif partition and the occupancy statistic are both
    defined over every peak the factor has, so truncating would change both."""
    bed = find_bed(tf)
    if bed is None:
        return None
    df = pd.read_csv(bed, sep='\t', header=None, usecols=[0, 1, 2, 6],
                     names=['chrom', 'start', 'end', 'signal'])
    if not str(df.chrom.iloc[0]).startswith('chr'):
        df.chrom = 'chr' + df.chrom.astype(str)
    df = df.sort_values('signal', ascending=False).reset_index(drop=True)
    df['name'] = [f'{tf}_peak_{i:05d}' for i in range(len(df))]
    return df


def _fai(fasta):
    idx = {}
    with open(str(fasta) + '.fai') as fh:
        for line in fh:
            name, length, offset, line_bases, line_width = line.split()[:5]
            idx[name] = (int(length), int(offset), int(line_bases), int(line_width))
    return idx


def _fetch(handle, fai, chrom, start, end):
    if chrom not in fai:
        return ''
    length, offset, line_bases, line_width = fai[chrom]
    start, end = max(0, start), min(end, length)
    handle.seek(offset + start // line_bases * line_width + start % line_bases)
    raw = handle.read((end - start) + (end - start) // line_bases + 2)
    return raw.replace('\n', '')[:end - start]


def motif_partition(target, partner):
    """Label every peak of the target by whether it carries the partner's JASPAR motif.

    FIMO is run on a fixed 2 * HALF_WIN window around the peak centre rather than on the whole
    peak, so that a wide peak does not acquire a motif call simply by being wide.
    """
    CACHE.mkdir(exist_ok=True)
    cache = CACHE / f'{target}_{partner}.csv'
    if cache.exists():
        return pd.read_csv(cache)

    peaks = load_peaks(target)
    if peaks is None:
        return None
    ids = find_motif_ids_for_tf(tf_name=partner, meme_file=MEME,
                                allowed_sources=['jaspar'], match_prefix=False)
    if not ids:
        raise RuntimeError(f'no JASPAR motif for {partner}')
    motif_id = ids[0]

    fai = _fai(GENOME)
    with TemporaryDirectory() as td, open(GENOME) as fh:
        fa, oc = Path(td) / 'peaks.fa', Path(td) / 'fimo_out'
        with fa.open('w') as out:
            for r in peaks.itertuples():
                mid = (int(r.start) + int(r.end)) // 2
                seq = _fetch(fh, fai, str(r.chrom), max(0, mid - HALF_WIN), mid + HALF_WIN)
                out.write(f'>{r.name}\n')
                for j in range(0, len(seq), 80):
                    out.write(seq[j:j + 80] + '\n')
        proc = subprocess.run(['fimo', '--no-pgc', '--motif', motif_id, '--thresh', str(FIMO_P),
                               '--oc', str(oc), str(MEME), str(fa)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'fimo failed: {proc.stderr[:300]}')
        tsv = oc / 'fimo.tsv'
        hits = (parse_fimo_tsv(tsv, p_value_threshold=FIMO_P, motif_ids=[motif_id])
                if tsv.exists() and tsv.stat().st_size else None)

    hit_ids = set(hits['peak_id'].unique()) if hits is not None and len(hits) else set()
    peaks['has_motif'] = peaks.name.isin(hit_ids)
    peaks.to_csv(cache, index=False)
    print(f'  [fimo] {target}: {int(peaks.has_motif.sum())}/{len(peaks)} peaks carry the '
          f'{partner} motif ({motif_id})', flush=True)
    return peaks


def peak_index(tf):
    """Per-chromosome (sorted starts, running max of ends) for constant-time overlap queries."""
    bed = find_bed(tf)
    if bed is None:
        return None
    df = pd.read_csv(bed, sep='\t', header=None, usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
    if not str(df.chrom.iloc[0]).startswith('chr'):
        df.chrom = 'chr' + df.chrom.astype(str)
    out = {}
    for c, s in df.groupby('chrom'):
        o = np.argsort(s.start.values)
        out[c] = (s.start.values[o], np.maximum.accumulate(s.end.values[o]))
    return out


def occupied(idx, chrom, start, end):
    """Vectorised: does each query interval overlap any interval of this factor?"""
    m = np.zeros(len(start), bool)
    for c in np.unique(chrom):
        if c not in idx:
            continue
        bs, be = idx[c]
        w = np.where(chrom == c)[0]
        j = np.searchsorted(bs, end[w], side='right')
        ok = j > 0
        m[w[ok]] = be[np.maximum(j[ok] - 1, 0)] > start[w][ok]
    return m


# ---------------------------------------------------------------- genes
def promoter_windows():
    g = pd.read_csv(COORDS)
    g = g.drop(columns=[c for c in g.columns if c.startswith('Unnamed')], errors='ignore')
    up, down = PROMOTER
    plus = g.strand == 1
    g['w_start'] = np.where(plus, np.maximum(0, g.start + up), np.maximum(0, g.end - down))
    g['w_end'] = np.where(plus, g.start + down, g.end - up)
    return g


def genes_at(win, peaks):
    """Gene symbols whose promoter window overlaps at least one of these peaks."""
    out = set()
    for chrom in set(win.chrom) & set(peaks.chrom):
        gs = win[win.chrom == chrom]
        ps = peaks[peaks.chrom == chrom]
        if ps.empty:
            continue
        for g in gs.itertuples():
            if ((ps.start < g.w_end) & (ps.end > g.w_start)).any():
                out.add(g.gene_name)
    return out


# ---------------------------------------------------------------- perturbation screen
def load_screen():
    """Knockdown effect z per (perturbation, gene), plus per-perturbation QC.

    The pseudobulk value is multiplied by sqrt(number of cells) so that perturbations measured in
    few cells are not given the same weight as those measured in many. Where a factor appears more
    than once, the row with the strongest depletion of its own transcript is kept.
    """
    a = ad.read_h5ad(BULK)
    X = np.nan_to_num(np.asarray(a.X), nan=0.)
    bad = (np.abs(X) > 100).any(0)                    # runaway values, a handful of genes
    gmean = np.asarray(a.var['mean']).astype(float)
    keep = (~bad) & (gmean > np.quantile(gmean[gmean > 0], .10))
    sym = (a.var['gene_name'] if 'gene_name' in a.var
           else pd.Series(a.var.index)).astype(str).str.upper().values

    obs = a.obs.copy()
    obs['sym'] = obs.index.str.split('_').str[1].str.upper()
    obs = obs.reset_index(drop=True)
    ncells = obs['num_cells_filtered'].values.astype(float)
    best = obs.sort_values('fold_expr').drop_duplicates('sym')
    row = {r.sym: int(r.Index) for r in best.itertuples()}
    best = best.set_index('sym')
    qc = {s: bool(best.loc[s, 'control_expr'] >= MIN_EXPR and best.loc[s, 'fold_expr'] < FOLD_MAX)
          for s in best.index}
    return X, keep, {g: i for i, g in enumerate(sym)}, row, ncells, qc


# ---------------------------------------------------------------- inference
BOOT_N = 10_000
BOOT_SEED = 0


def cluster_bootstrap(excess, reader, n=BOOT_N, seed=BOOT_SEED):
    """Resample READERS with replacement, not dependencies.

    The 50 dependencies share far fewer distinct readers, so a factor that happens to look good in
    one dependency tends to look good in its others; treating the dependencies as independent
    overstates the evidence. Each resample draws whole readers and pools all of their dependencies,
    so the effective sample size is the number of readers.

    The reported P is ONE-SIDED, matching the directional Wilcoxon it accompanies: the hypothesis is
    that readers exceed their controls, not that they differ from them. Doubling it gives the
    two-sided value; do not mix the two conventions in the same sentence.
    """
    rng = np.random.default_rng(seed)
    excess = np.asarray(excess, float)
    uniq, idx = np.unique(reader, return_inverse=True)
    where = [np.flatnonzero(idx == i) for i in range(len(uniq))]
    draws = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[b] = np.median(excess[np.concatenate([where[i] for i in pick])])
    return (float(np.median(excess)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float((draws <= 0).mean()), len(uniq))


# ---------------------------------------------------------------- main
def main():
    X, KEEP, gi, row, ncells, QC = load_screen()
    _z = {}

    def z(g):
        if g not in _z:
            _z[g] = X[row[g]] * np.sqrt(ncells[row[g]])
        return _z[g]

    def concordance(zt, zc, idx):
        return float((np.sign(zt[idx]) == np.sign(zc[idx])).mean())

    tf_ann = pd.read_csv(DATA / 'tf_annotation' / 'tf_dna_binding.csv')
    tfs = set(tf_ann.loc[tf_ann.is_tf.eq('Yes'), 'tf'].astype(str).str.upper())
    pool = sorted(g for g in row if g in tfs and QC.get(g, False) and find_bed(g) is not None)
    print(f'control pool: {len(pool)} factors '
          f'(annotated TF, in the library, passes knockdown QC, has {CELL} ChIP-seq)', flush=True)
    IDX = {g: peak_index(g) for g in pool}

    win = promoter_windows()
    A = pd.read_csv(ATLAS, sep='\t')
    dep = A[A.cell.eq(CELL) & A.dichotomy.ne('UNRESOLVED')].dropna(subset=['reader']).copy()
    dep['cands'] = dep.reader.astype(str).str.split(';').apply(
        lambda L: [x.strip().upper() for x in L])
    # a target nominated as its own reader is not a test of anything
    dep = dep[dep.apply(lambda r: r.target.upper() not in r.cands, axis=1)]
    dep['ok'] = dep.cands.apply(
        lambda L: [c for c in L if c in row and QC.get(c, False) and find_bed(c) is not None])
    dep = dep[(dep.ok.str.len() > 0) & dep.target.map(lambda t: t in row and QC.get(t, False))]
    print(f'{len(dep)} dependencies with a testable target and at least one testable reader',
          flush=True)

    rows = []
    for r in dep.itertuples():
        pk = motif_partition(r.target, r.partner)
        if pk is None:
            continue
        mp = pk[pk.has_motif]
        if len(mp) < MIN_PEAKS:
            continue
        zt = z(r.target)

        # genes are assigned to exactly one partition; those reachable from both are dropped
        gp, gm = genes_at(win, mp), genes_at(win, pk[~pk.has_motif])
        both = gp & gm
        gp, gm = gp - both, gm - both
        ip = np.array([gi[x] for x in gp if x in gi and KEEP[gi[x]]], int)
        im = np.array([gi[x] for x in gm if x in gi and KEEP[gi[x]]], int)
        moved_p = ip[np.abs(zt[ip]) > Z_CUT]
        moved_m = im[np.abs(zt[im]) > Z_CUT]
        if len(moved_p) < MIN_GENES:
            continue

        ch, st, en = mp.chrom.values.astype(str), mp.start.values, mp.end.values
        occ = {g: occupied(IDX[g], ch, st, en).mean() for g in pool if IDX[g] is not None}

        for reader in r.ok:
            o_read = occupied(IDX.get(reader) or peak_index(reader), ch, st, en).mean()
            # the reader, the target and any co-nominated reader cannot serve as their own control
            excluded = {reader, r.target, *r.cands}
            near = sorted((abs(occ[g] - o_read), g) for g in occ if g.upper() not in excluded)
            ctrls = [g for _, g in near[:N_CTRL]]
            cc = [concordance(zt, z(g), moved_p) for g in ctrls]
            c_read = concordance(zt, z(reader), moved_p)
            rows.append(dict(
                target=r.target, partner=r.partner, reader=reader, cat=r.reader_category,
                n_Mp_peaks=len(mp), n_affected_Mp=len(moved_p), n_affected_Mm=len(moved_m),
                occ_reader=100 * o_read,
                occ_ctrl_med=100 * float(np.median([occ[g] for g in ctrls])),
                occ_gap=100 * float(np.median([abs(occ[g] - o_read) for g in ctrls])),
                conc_Mp=c_read, ctrl_mean=float(np.mean(cc)), ctrl_med=float(np.median(cc)),
                ctrl_sd=float(np.std(cc)), n_beat=int(sum(c_read > c for c in cc)),
                ctrls='|'.join(ctrls),
                conc_Mm=concordance(zt, z(reader), moved_m) if len(moved_m) >= MIN_GENES else np.nan,
                ctrl_med_Mm=float(np.median([concordance(zt, z(g), moved_m) for g in ctrls]))
                if len(moved_m) >= MIN_GENES else np.nan))

    R = pd.DataFrame(rows)
    # Two distinct quantities, kept apart deliberately.
    #   select_score - against the MEAN of the ten controls. Its only job is to choose between
    #     co-nominated candidate readers below. It is the criterion the published selection was
    #     made with; switching it to the median would silently retain a different candidate for
    #     some of the multi-candidate dependencies and change the released table.
    #   excess       - against the MEDIAN of the ten controls. This is the reported effect and the
    #     quantity the figure plots on both axes; nothing downstream should use select_score.
    R['select_score'] = R.conc_Mp - R.ctrl_mean
    R['excess'] = R.conc_Mp - R.ctrl_med
    # where ARES nominates several candidate readers, the best-scoring one is kept; this is a
    # selection step and is reported as such in the Methods
    R = (R.sort_values('select_score', ascending=False)
          .drop_duplicates(['target', 'partner'])
          .drop(columns=['select_score']).reset_index(drop=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False)

    ex = R.conc_Mp - R.ctrl_med
    p = wilcoxon(R.conc_Mp, R.ctrl_med, alternative='greater').pvalue
    m_boot, lo, hi, p_boot, n_read = cluster_bootstrap(ex.values, R.reader.values)
    print(f'\n{len(R)} dependencies / {R.reader.nunique()} distinct readers')
    print(f'  motif+ peaks per dependency, median {R.n_Mp_peaks.median():.0f}; '
          f'target-moved genes, median {R.n_affected_Mp.median():.0f}')
    print(f'  occupancy  reader {R.occ_reader.median():.1f}%  controls {R.occ_ctrl_med.median():.1f}%'
          f'  (median matching error {R.occ_gap.median():.2f} points)')
    print(f'  concordance  reader {100 * R.conc_Mp.median():.1f}%  '
          f'controls {100 * R.ctrl_med.median():.1f}%')
    print(f'  paired median difference {100 * ex.median():+.1f} points, '
          f'{int((ex > 0).sum())}/{len(R)} above the diagonal, Wilcoxon P = {p:.3g}')
    print(f'  clustered on {n_read} readers: {100 * m_boot:+.1f} points '
          f'[{100 * lo:+.1f}, {100 * hi:+.1f}], one-sided P = {p_boot:.3g}')
    print(f'  reader beats a median of {R.n_beat.median():.0f}/{N_CTRL} of its own controls')
    v = R.dropna(subset=['conc_Mm'])
    if len(v) >= 5:
        d = v.conc_Mm - v.ctrl_med_Mm
        print(f'  same contrast at motif-negative peaks: {100 * d.median():+.1f} points, '
              f'P = {wilcoxon(d, alternative="greater").pvalue:.3g}')
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
