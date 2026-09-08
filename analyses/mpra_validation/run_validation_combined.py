"""Combined-mode MPRA validation pipeline.

Concatenates all 6 PARM splits (fold0..fold4 + test) into ONE dataset and runs
the per-pair pipeline against a single cell-line's Log2Norm column.

Per Hanbei's instructions: no cross-fold combination, no filtering of pairs.
Every pair that can be tested is reported with its raw stats.

Parent-id check (verified pre-run): all 6 splits are disjoint at the parent
locus level, so plain concatenation introduces no double-counting.

Usage:
  python run_validation_combined.py \
      --cellline HepG2 \
      --pairs pairs_by_cellline/HepG2_pairs.tsv \
      --out-dir all_celllines_out/HepG2
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from tf_function import lookup as tf_function_lookup, expected_sign

MPRA_DIR = Path("/stg3/data1/sam/enhancer_prediction/multi_agent/universal_regulators/validation/MPRA_data")
FASTA_DIR = Path("/stg3/data1/sam/enhancer_prediction/multi_agent/universal_regulators/validation")
CHIP_BASE = Path("/new-stg/home/hanbei/data/TF_ENCODE4")
MEME = Path("/new-stg/home/hanbei/data/combined_TF_motifs_jaspar_hocomoco12_cisbp.meme")
FIMO = "/new-stg/home/hanbei/meme/bin/fimo"

# MPRA score-column name (= Log2Norm_<CL>) → ChIP subdir under CHIP_BASE
CHIP_DIRNAME = {
    "K562": "K562", "HepG2": "HepG2", "HEK293": "HEK293",
    "MCF7": "MCF.7", "HCT116": "HCT116",
}

# (bed_suffix, fasta_basename, fasta_header_prefix)
SPLITS = [
    ("fold0", "fold0",      "fold0"),
    ("fold1", "fold1",      "fold1"),
    ("fold2", "fold2",      "fold2"),
    ("fold3", "fold3",      "fold3"),
    ("fold4", "fold4",      "fold4"),
    ("test",  "foldtest",   "foldtest"),
]

FIMO_THRESH = 1e-4
FIMO_THRESH_TIGHT = 1e-5
MOTIF_POS_RATE_CAP = 0.5
MIN_FRAG = 30
SOURCE_PREF = {"jaspar": 0, "hocomoco12": 1, "cisbp": 2}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_fragments_combined(score_col: str):
    """Concatenate all 6 splits' bed.gz, return (frags, activity, tss_distance,
    parent_id, split_offsets). Global row_idx = 0..N-1 in concat order."""
    cols = ["chr", "start", "end", "FEATtype", "FEATstart", "FEATend", "FEATsense", score_col]
    frames = []
    offsets = {}
    current = 0
    for bed_suf, _, _ in SPLITS:
        bed_gz = MPRA_DIR / f"GSE301246_PARM_Training_Library_v2_MH4_normalized_{bed_suf}.bed.gz"
        log(f"loading {bed_gz.name}")
        df = pd.read_csv(bed_gz, sep="\t", usecols=cols)
        offsets[bed_suf] = (current, current + len(df))
        current += len(df)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["row_idx"] = np.arange(len(all_df), dtype=np.int64)
    sense = all_df["FEATsense"].astype(bool).values
    tss = np.where(sense, all_df["FEATstart"].values, all_df["FEATend"].values)
    frag_center = (all_df["start"].values + all_df["end"].values) / 2
    tss_distance = np.abs(frag_center - tss).astype(np.float32)
    activity = all_df[score_col].values.astype(np.float32)
    parent_id = (
        all_df["FEATtype"].astype(str).values + ":" +
        all_df["FEATstart"].astype(str).values + "-" +
        all_df["FEATend"].astype(str).values
    )
    log(f"combined dataset: {len(all_df):,} rows across {len(SPLITS)} splits")
    return all_df[["chr", "start", "end", "FEATtype", "row_idx"]], activity, tss_distance, parent_id, offsets


def load_fasta_combined(n_seqs: int, offsets: dict[str, tuple[int, int]]) -> list[str]:
    """Load all 6 fasta files into a list of size n_seqs, indexed by global row_idx.

    Each fasta has headers `<prefix>_<local_i>`; global_idx = split_offset + local_i.
    """
    seqs: list[str] = [""] * n_seqs
    for bed_suf, fasta_base, header_pref in SPLITS:
        fasta_path = FASTA_DIR / f"{fasta_base}.fasta"
        off_start, off_end = offsets[bed_suf]
        rx = re.compile(rf"^{re.escape(header_pref)}_(\d+)$")
        log(f"loading {fasta_path.name} (offset={off_start}, n={off_end - off_start})")
        cur_local = None
        chunks: list[str] = []
        with fasta_path.open() as f:
            for line in f:
                if line.startswith(">"):
                    if cur_local is not None:
                        seqs[off_start + cur_local] = "".join(chunks)
                    m = rx.match(line[1:].split()[0])
                    cur_local = int(m.group(1)) if m else None
                    chunks = []
                else:
                    if cur_local is not None:
                        chunks.append(line.strip())
            if cur_local is not None:
                seqs[off_start + cur_local] = "".join(chunks)
    log(f"  {sum(1 for s in seqs if s):,} sequences loaded total")
    return seqs


def load_meme_motifs():
    motifs: dict[str, tuple[str, str, str, int]] = {}
    cur_h: str | None = None
    cur_body: list[str] = []
    cur_name: str | None = None
    cur_src: str = ""

    def flush():
        nonlocal cur_h, cur_body, cur_name, cur_src
        if cur_h is None or cur_name is None:
            return
        pref = SOURCE_PREF.get(cur_src, 3)
        if cur_name not in motifs or pref < motifs[cur_name][3]:
            motifs[cur_name] = (cur_h, "".join(cur_body), cur_src, pref)

    with MEME.open() as f:
        meme_header_lines: list[str] = []
        for line in f:
            if line.startswith("MOTIF "):
                rest = line.split(None, 1)[1].strip()
                parts = rest.split("|")
                cur_name = parts[0]
                cur_src = parts[1] if len(parts) > 1 else ""
                cur_h = line
                cur_body = []
                break
            meme_header_lines.append(line)
        for line in f:
            if line.startswith("MOTIF "):
                flush()
                rest = line.split(None, 1)[1].strip()
                parts = rest.split("|")
                cur_name = parts[0]
                cur_src = parts[1] if len(parts) > 1 else ""
                cur_h = line
                cur_body = []
            else:
                cur_body.append(line)
        flush()
    return {k: (v[0], v[1], v[2]) for k, v in motifs.items()}, "".join(meme_header_lines)


def load_chip_peaks(tf: str, chip_root: Path) -> pd.DataFrame:
    files = sorted((chip_root / f"{tf}_human").glob("*.bed"))
    if not files:
        raise FileNotFoundError(f"no .bed for {tf} in {chip_root}")
    frames = []
    for fp in files:
        d = pd.read_csv(fp, sep="\t", header=None, usecols=[0, 1, 2],
                        names=["chr", "start", "end"])
        frames.append(d)
    peaks = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return peaks


def intersect_frags_peaks(frags: pd.DataFrame, peaks: pd.DataFrame) -> np.ndarray:
    hit_rows: list[int] = []
    peak_grouped = {c: g[["start", "end"]].sort_values("start").values
                    for c, g in peaks.groupby("chr")}
    for chrom, fg in frags.groupby("chr"):
        if chrom not in peak_grouped:
            continue
        parr = peak_grouped[chrom]
        pstarts = parr[:, 0]
        pends = parr[:, 1]
        if len(pends) == 0:
            continue
        M = np.maximum.accumulate(pends)
        fs = fg["start"].values
        fe = fg["end"].values
        ridx = fg["row_idx"].values
        idx = np.searchsorted(pstarts, fe, side="left")
        keep = np.zeros(len(fs), dtype=bool)
        mask = idx > 0
        keep[mask] = M[idx[mask] - 1] > fs[mask]
        hit_rows.extend(ridx[keep].tolist())
    return np.array(sorted(hit_rows), dtype=np.int64)


def write_partner_meme(meme_header: str, name: str, header_line: str, body: str, out: Path):
    with out.open("w") as f:
        f.write(meme_header)
        if not meme_header.endswith("\n"):
            f.write("\n")
        f.write(header_line)
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")


def write_subset_fasta(seqs: list[str], row_indices: np.ndarray, out: Path):
    with out.open("w") as f:
        for ri in row_indices:
            s = seqs[ri]
            if not s:
                continue
            f.write(f">row_{ri}\n{s}\n")


def run_fimo(meme_path: Path, fa_path: Path, out_path: Path, thresh: float = FIMO_THRESH) -> int:
    with out_path.open("w") as f:
        proc = subprocess.run(
            [FIMO, "--no-qvalue", "--text", "--thresh", str(thresh),
             str(meme_path), str(fa_path)],
            stdout=f, stderr=subprocess.PIPE, text=True,
        )
    return proc.returncode


def parse_fimo_hits(fimo_path: Path, partner_name: str) -> set[int]:
    hits: set[int] = set()
    with fimo_path.open() as f:
        f.readline()
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            mname = parts[0].split("|", 1)[0]
            if mname != partner_name:
                continue
            sn = parts[2]
            if sn.startswith("row_"):
                try:
                    hits.add(int(sn[4:]))
                except ValueError:
                    continue
    return hits


def gc_content(s: str) -> float:
    if not s:
        return 0.0
    gc = sum(1 for c in s if c in "GCgc")
    return gc / len(s)


def matched_pairs(motif_pos: pd.DataFrame, motif_neg: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if motif_pos.empty or motif_neg.empty:
        return np.array([]), np.array([])
    feat_cols = ["length", "gc", "tss_distance"]
    pool = pd.concat([motif_pos[feat_cols], motif_neg[feat_cols]], ignore_index=True)
    mu = pool.mean()
    sd = pool.std().replace(0, 1.0)
    pos_xy = ((motif_pos[feat_cols] - mu) / sd).values
    neg_xy = ((motif_neg[feat_cols] - mu) / sd).values
    from scipy.spatial import cKDTree
    tree = cKDTree(neg_xy)
    used_neg: set[int] = set()
    pos_pick: list[int] = []
    neg_pick: list[int] = []
    k_init = min(len(neg_xy), 8)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pos_xy))
    for i in order:
        k = min(k_init, len(neg_xy))
        while True:
            dists, idxs = tree.query(pos_xy[i], k=k)
            idxs = np.atleast_1d(idxs)
            chosen = None
            for j in idxs:
                if j not in used_neg:
                    chosen = int(j)
                    break
            if chosen is not None:
                used_neg.add(chosen)
                pos_pick.append(int(i))
                neg_pick.append(chosen)
                break
            if k >= len(neg_xy):
                chosen = -1
                break
            k = min(k * 2, len(neg_xy))
        if chosen == -1:
            break
    pos_row = motif_pos["row_idx"].values[pos_pick]
    neg_row = motif_neg["row_idx"].values[neg_pick]
    return pos_row, neg_row


def process_pair(target: str, partner: str, pair_id: str,
                 frags: pd.DataFrame, seqs: list[str],
                 activity: np.ndarray, tss_distance: np.ndarray,
                 parent_id: np.ndarray,
                 motif_db: dict, meme_header: str,
                 chip_root: Path, out_dir: Path) -> dict:
    log(f"=== {pair_id}  ({target} → {partner}) ===")
    pair_out = out_dir / pair_id
    pair_out.mkdir(parents=True, exist_ok=True)
    rec: dict = {
        "pair_id": pair_id, "target_tf": target, "partner_tf": partner,
        "partner_tf_function": tf_function_lookup(partner),
    }
    try:
        peaks = load_chip_peaks(target, chip_root)
    except FileNotFoundError as e:
        rec["status"] = f"no_chip:{e}"
        return rec
    rec["n_peaks"] = int(len(peaks))
    log(f"  peaks: {len(peaks)}")

    if partner not in motif_db:
        rec["status"] = "no_partner_motif"
        return rec
    p_header, p_body, p_source = motif_db[partner]
    rec["partner_motif_source"] = p_source

    t0 = time.time()
    hit_rows = intersect_frags_peaks(frags, peaks)
    rec["n_overlap_fragments"] = int(len(hit_rows))
    log(f"  overlap fragments: {len(hit_rows)} ({time.time()-t0:.1f}s)")

    if len(hit_rows) < MIN_FRAG:
        rec["status"] = "insufficient_coverage"
        return rec

    sub_fa = pair_out / "subset.fa"
    write_subset_fasta(seqs, hit_rows, sub_fa)
    p_meme = pair_out / "partner.meme"
    write_partner_meme(meme_header, partner, p_header, p_body, p_meme)

    fimo_out = pair_out / "fimo.tsv"
    t0 = time.time()
    rc = run_fimo(p_meme, sub_fa, fimo_out, thresh=FIMO_THRESH)
    if rc != 0:
        rec["status"] = f"fimo_failed:{rc}"
        return rec
    motif_pos_set = parse_fimo_hits(fimo_out, partner)
    pos_rate = len(motif_pos_set) / len(hit_rows) if hit_rows.size else 0.0
    used_thresh = FIMO_THRESH
    if pos_rate > MOTIF_POS_RATE_CAP:
        log(f"  motif+ rate {pos_rate:.2f} > {MOTIF_POS_RATE_CAP}; tightening to {FIMO_THRESH_TIGHT:g}")
        rc = run_fimo(p_meme, sub_fa, fimo_out, thresh=FIMO_THRESH_TIGHT)
        if rc != 0:
            rec["status"] = f"fimo_failed_tight:{rc}"
            return rec
        motif_pos_set = parse_fimo_hits(fimo_out, partner)
        used_thresh = FIMO_THRESH_TIGHT
    rec["fimo_thresh"] = used_thresh
    log(f"  fimo: {len(motif_pos_set)} motif+ / {len(hit_rows)} fragments ({time.time()-t0:.1f}s, thresh={used_thresh:g})")

    rows = []
    for ri in hit_rows:
        s = seqs[ri]
        if not s:
            continue
        rows.append({
            "row_idx": int(ri),
            "length": len(s),
            "gc": gc_content(s),
            "tss_distance": float(tss_distance[ri]),
            "motif_pos": ri in motif_pos_set,
            "activity": float(activity[ri]),
        })
    sub_df = pd.DataFrame(rows)

    motif_pos = sub_df[sub_df["motif_pos"]].reset_index(drop=True)
    motif_neg = sub_df[~sub_df["motif_pos"]].reset_index(drop=True)
    rec["n_motif_pos"] = int(len(motif_pos))
    rec["n_motif_neg"] = int(len(motif_neg))
    rec["motif_pos_rate"] = float(len(motif_pos) / len(sub_df)) if len(sub_df) else None

    if len(motif_pos) < 5 or len(motif_neg) < 5:
        rec["status"] = "too_few_for_test"
        sub_df.to_csv(pair_out / "subset.csv", index=False)
        return rec

    pos_rows, neg_rows = matched_pairs(motif_pos, motif_neg)
    rec["n_matched"] = int(len(pos_rows))
    if len(pos_rows) < 5:
        rec["status"] = "matching_failed"
        sub_df.to_csv(pair_out / "subset.csv", index=False)
        return rec

    a_pos = activity[pos_rows]
    a_neg = activity[neg_rows]
    p_pos = parent_id[pos_rows]
    p_neg = parent_id[neg_rows]

    matched_df = pd.DataFrame({
        "pos_parent": p_pos, "neg_parent": p_neg,
        "pos_activity": a_pos, "neg_activity": a_neg,
    })
    pos_locus = matched_df.groupby("pos_parent")["pos_activity"].median()
    neg_locus = matched_df.groupby("neg_parent")["neg_activity"].median()
    shared = sorted(set(pos_locus.index) & set(neg_locus.index))
    n_pos_parents = int(pos_locus.size)
    n_neg_parents = int(neg_locus.size)
    n_shared = len(shared)

    if n_shared >= 5:
        diffs = np.array([pos_locus.loc[par] - neg_locus.loc[par] for par in shared])
        effect_locus = float(np.median(diffs))
        nz = diffs != 0
        p_paired_locus = float(stats.wilcoxon(diffs[nz]).pvalue) if nz.sum() >= 5 else float("nan")
    else:
        effect_locus = float("nan")
        p_paired_locus = float("nan")

    if n_pos_parents >= 2 and n_neg_parents >= 2:
        p_unpaired_locus = float(stats.mannwhitneyu(
            pos_locus.values, neg_locus.values, alternative="two-sided").pvalue)
    else:
        p_unpaired_locus = float("nan")

    mw = stats.mannwhitneyu(a_pos, a_neg, alternative="two-sided")
    median_pos = float(np.median(a_pos))
    median_neg = float(np.median(a_neg))
    effect_frag = median_pos - median_neg

    # Status: keep informational labels but DO NOT use them to drop rows.
    if n_shared < 10:
        status = "underpowered_locus_clustering"
    elif np.isnan(p_paired_locus):
        status = "degenerate_paired_diffs"
    else:
        status = "ok"

    sign = expected_sign(partner)
    if sign == 0 or status != "ok":
        direction_match: bool | None = None
    else:
        direction_match = (np.sign(effect_locus) == sign) if effect_locus != 0 else False

    rec.update({
        "n_pos_parents": n_pos_parents,
        "n_neg_parents": n_neg_parents,
        "n_shared_parents": n_shared,
        "effect_size_locus": effect_locus,
        "p_paired_locus": p_paired_locus,
        "p_unpaired_locus": p_unpaired_locus,
        "median_motif_pos": median_pos,
        "median_motif_neg": median_neg,
        "mean_motif_pos": float(np.mean(a_pos)),
        "mean_motif_neg": float(np.mean(a_neg)),
        "effect_size_fragment": effect_frag,
        "p_fragment": float(mw.pvalue),
        "mw_U_fragment": float(mw.statistic),
        "rank_biserial_fragment": float((2 * mw.statistic) / (len(a_pos) * len(a_neg)) - 1),
        "expected_sign": sign,
        "direction_match": direction_match,
        "status": status,
    })

    matched = pd.DataFrame({
        "pos_row": pos_rows, "neg_row": neg_rows,
        "pos_parent": p_pos, "neg_parent": p_neg,
        "pos_activity": a_pos, "neg_activity": a_neg,
        "pos_length": [len(seqs[r]) for r in pos_rows],
        "neg_length": [len(seqs[r]) for r in neg_rows],
        "pos_gc": [gc_content(seqs[r]) for r in pos_rows],
        "neg_gc": [gc_content(seqs[r]) for r in neg_rows],
        "pos_tss_dist": [float(tss_distance[r]) for r in pos_rows],
        "neg_tss_dist": [float(tss_distance[r]) for r in neg_rows],
    })
    matched.to_csv(pair_out / "matched_pairs.csv", index=False)
    sub_df.to_csv(pair_out / "subset.csv", index=False)
    log(f"  locus: n_shared={n_shared} effect={effect_locus:+.3f} p_paired={p_paired_locus:.2e}"
        f"  | frag: effect={effect_frag:+.3f} p={mw.pvalue:.2e} n={len(pos_rows)}  [{status}]")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellline", required=True,
                    help="MPRA cell line: K562, HepG2, HEK293, MCF7, HCT116")
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    if args.cellline not in CHIP_DIRNAME:
        raise SystemExit(f"--cellline must be one of {sorted(CHIP_DIRNAME)}")
    score_col = f"Log2Norm_{args.cellline}"
    chip_root = CHIP_BASE / CHIP_DIRNAME[args.cellline]
    log(f"cellline={args.cellline}  score_col={score_col}  chip_root={chip_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "per_pair").mkdir(parents=True, exist_ok=True)

    pairs_df = pd.read_csv(args.pairs, sep="\t")
    log(f"will process {len(pairs_df)} pairs")

    frags, activity, tss_distance, parent_id, offsets = load_fragments_combined(score_col)
    seqs = load_fasta_combined(len(frags), offsets)
    motif_db, meme_header = load_meme_motifs()
    log(f"motif db: {len(motif_db)} TFs")

    passthrough_cols = [c for c in ["bucket", "final_mechanism_category",
                                    "final_mechanism", "dichotomy_class"]
                        if c in pairs_df.columns]

    results: list[dict] = []
    for _, row in pairs_df.iterrows():
        rec = process_pair(
            target=row["target_tf"], partner=row["partner_tf"], pair_id=row["pair_id"],
            frags=frags, seqs=seqs, activity=activity, tss_distance=tss_distance,
            parent_id=parent_id, motif_db=motif_db, meme_header=meme_header,
            chip_root=chip_root, out_dir=args.out_dir / "per_pair",
        )
        for c in passthrough_cols:
            rec[c] = row[c]
        results.append(rec)

    out_df = pd.DataFrame(results)
    ok = out_df["status"] == "ok"
    if ok.sum() > 1:
        from scipy.stats import false_discovery_control
        out_df.loc[ok, "q_paired_locus"] = false_discovery_control(
            out_df.loc[ok, "p_paired_locus"].values)
    out_tsv = args.out_dir / f"{args.cellline}_mpra_validation.tsv"
    out_df.to_csv(out_tsv, sep="\t", index=False)
    log(f"wrote {out_tsv}")


if __name__ == "__main__":
    main()
