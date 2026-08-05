#!/usr/bin/env python3
"""AlphaGenome multi-readout validation across all (cell, TF_A, TF_B) pairs.

Generalization of validate_v3.py (K562-only). Adds:
  --cell <CELL>            : choose cell line (must be in cell_line_ontology.tsv)
  --api_key_idx {0,1}      : choose .env (key 0) or .env.2 (key 1) — for parallel runs
  --min_peaks <N>          : minimum motif-positive peaks before skip (default 5)
  --pair_idx / --chunk_idx : same as before, plus --start / --end for explicit ranges

For each pair:
  - Take top-50 TF_A peaks (by FIMO score for the partner motif)
  - For each peak, do two predict_variant calls:
      (a) motif scramble at peak's partner-motif hit
      (b) within-peak control scramble (60 bp offset from motif)
    Each call requests multiple OutputTypes filtered to the cell's ontology.
  - Record REF and ALT signals per readout
  - Write per-pair CSV under validation_v3_all/<CELL>/pair_<idx>.csv

Usage:
  python validate_v3_all.py --cell HepG2 --api_key_idx 0 --pair_idx 0
  python validate_v3_all.py --cell HepG2 --api_key_idx 0 --start 0 --end 276
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome.models import dna_client, dna_output  # noqa: E402
from alphagenome.data import genome  # noqa: E402

# ----- paths and constants -----
ANALYSIS_DIR  = Path(__file__).resolve().parent
ALL_ROOT      = ANALYSIS_DIR / "validation_v3_all"
ALL_ROOT.mkdir(parents=True, exist_ok=True)

TF_DATA_BASE = Path("/new-stg/home/hanbei/data/TF_ENCODE4")
GENOME_FASTA = Path("/new-stg/home/hanbei/data/hg38.fa")
MEME_FILE    = Path("/new-stg/home/hanbei/data/combined_TF_motifs_jaspar_hocomoco12_cisbp.meme")
JASPAR_MEME  = Path("/new-stg/home/hanbei/HanX/data/JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_v2.meme")
PAIR_TABLE   = Path("/new-stg/home/hanbei/HanX/result/rf_shap_top_regulator.csv")
ONTOLOGY_TSV = ANALYSIS_DIR / "cell_line_ontology.tsv"

INTERVAL_LEN = 16_384
BIN_SIZE     = 128
PEAKS_PER_PAIR    = 50
# Scan ALL TF_A peaks with FIMO for the partner motif; among peaks with a hit,
# keep the top PEAKS_PER_PAIR by TF_A ChIP signal (signalValue, column 7 of narrowPeak).
CONTROL_OFFSET_BP = 60
SLEEP_BETWEEN_CALLS = 0.3

BEDTOOLS = "/new-stg/home/hanbei/miniconda3/envs/pipeline/bin/bedtools"
FIMO     = "/new-stg/home/hanbei/meme/bin/fimo"

CHUNKS_TOTAL = 16
QUOTA_RETRY_BACKOFF = [30, 60, 120, 240, 480]   # seconds; Google quota resets per-minute, so ≥30s often clears it
MAX_RETRY_ATTEMPTS = len(QUOTA_RETRY_BACKOFF)

ACTIVE_MARKS     = {"H3K27ac", "H3K4me1", "H3K4me3"}
REPRESSIVE_MARKS = {"H3K27me3"}  # H3K9me3 not available in K562 AlphaGenome tracks
ALL_MARKS        = ACTIVE_MARKS | REPRESSIVE_MARKS

REQUESTED_OUTPUTS = [
    dna_output.OutputType.CHIP_TF,
    dna_output.OutputType.CHIP_HISTONE,
    dna_output.OutputType.DNASE,
    dna_output.OutputType.ATAC,
    dna_output.OutputType.RNA_SEQ,
    dna_output.OutputType.CAGE,
    dna_output.OutputType.PROCAP,
]

# 3D contact maps: SEPARATE no-ontology call (no per-cell Hi-C track), generic 28-track 4DN consensus,
# at a wider window so loop-scale contacts are in range. CONTACT = magnitude of the contact perturbation
# the scramble causes at the motif locus (mean |Δ| of the motif bin's contacts, ref vs alt).
CONTACT_OUTPUTS      = [dna_output.OutputType.CONTACT_MAPS]
CONTACT_INTERVAL_LEN = 131_072   # 128 kb -> 64x64 contact map


# ============================================================
# JASPAR motif lookup
# ============================================================
def parse_combined_motif_index(meme_path: Path) -> dict[str, str]:
    """Return {tf_name_lower: motif_id_string} for the COMBINED meme file
    (which is what FIMO actually reads). Motif IDs look like
    'YY1|jaspar|MA0095.3', and we want the JASPAR version that exists here.
    Heterodimer names like 'MAFG::NFE2L1' are stored as-is (lowercased).
    Prefer JASPAR-source motifs; fall back to others if not available."""
    by_tf_jaspar: dict[str, str] = {}
    by_tf_other:  dict[str, str] = {}
    with open(meme_path) as f:
        for line in f:
            m = re.match(r"^MOTIF\s+(\S+)", line)
            if not m: continue
            mid = m.group(1)
            parts = mid.split('|')
            if len(parts) < 2: continue
            tf, source = parts[0], parts[1]
            key = tf.lower()
            if source == 'jaspar':
                # keep first JASPAR hit; the combined file's order is the one FIMO will find
                by_tf_jaspar.setdefault(key, mid)
            else:
                by_tf_other.setdefault(key, mid)
    # JASPAR preferred, then other sources
    out = dict(by_tf_other)
    out.update(by_tf_jaspar)   # JASPAR overrides
    return out

MOTIF_INDEX = parse_combined_motif_index(MEME_FILE)


def find_partner_motif_id(partner_tf: str) -> tuple[str, str] | None:
    """Return (motif_id_in_combined_meme, partner_tf_resolved_name) or None."""
    name = partner_tf.lower()
    if name in MOTIF_INDEX:
        return MOTIF_INDEX[name], partner_tf
    for c in partner_tf.split('::'):
        cn = c.strip().lower()
        if cn in MOTIF_INDEX:
            return MOTIF_INDEX[cn], c.strip()
    return None


# ============================================================
# Peak selection + FIMO motif filtering
# ============================================================
def find_peak_bed(tf: str, tf_data_root: Path) -> Path | None:
    cands = sorted((tf_data_root / f"{tf}_human").glob("*.bed"))
    return cands[0] if cands else None


def top_peaks_with_partner_motif(tf_a: str, partner_motif_id: str,
                                  n: int = PEAKS_PER_PAIR,
                                  tf_data_root: Path | None = None) -> list[dict]:
    """FIMO-scan ALL tf_a peaks for the partner motif, then return the top-n
    motif-positive peaks ranked by partner-motif PWM score (FIMO score).

    Rationale: AlphaGenome's response to motif scrambling depends on motif
    quality; ranking by FIMO score selects the cleanest perturbations."""
    bed = find_peak_bed(tf_a, tf_data_root)
    if bed is None or bed.stat().st_size == 0:
        return []
    try:
        df = pd.read_csv(bed, sep="\t", header=None)
    except pd.errors.EmptyDataError:
        return []
    if len(df) == 0:
        return []
    cols = ["chrom","start","end","name","score","strand",
            "signal_value","p_value","q_value","peak_offset"]
    df.columns = cols[:df.shape[1]]
    if "peak_offset" not in df.columns:
        df["peak_offset"] = ((df["end"] - df["start"]) // 2).astype(int)
    df["summit"] = df["start"].astype(int) + df["peak_offset"].astype(int)
    # NO pre-filter: scan ALL peaks
    df = df.reset_index(drop=True)
    df["name"] = [f"peak_{i:05d}" for i in range(len(df))]
    with tempfile.TemporaryDirectory() as td:
        bed_path = Path(td) / "peaks.bed"
        fa_path  = Path(td) / "peaks.fa"
        with open(bed_path, "w") as f:
            for _, row in df.iterrows():
                mid = int(row["summit"])
                f.write(f"{row['chrom']}\t{max(0, mid-500)}\t{mid+500}\t{row['name']}\n")
        subprocess.run([BEDTOOLS, "getfasta", "-fi", str(GENOME_FASTA),
                        "-bed", str(bed_path), "-fo", str(fa_path), "-name"],
                       check=True, capture_output=True)
        fimo_out = Path(td) / "fimo_out"
        subprocess.run([FIMO, "--no-pgc", "--motif", partner_motif_id,
                        "--thresh", "1e-4", "--oc", str(fimo_out),
                        str(MEME_FILE), str(fa_path)],
                       check=True, capture_output=True)
        fimo_tsv = fimo_out / "fimo.tsv"
        rows_out = []
        with open(fimo_tsv) as f:
            header = None
            for line in f:
                if not line.strip() or line.startswith("##"): continue
                if line.startswith("#") and "\t" not in line: continue
                if header is None:
                    header = [h.strip("# ").replace(" ", "_") for h in line.rstrip("\n").split("\t")]
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header): continue
                rows_out.append(dict(zip(header, parts)))
        if not rows_out:
            return []
        fimo_df = pd.DataFrame(rows_out)
        fimo_df["start"] = pd.to_numeric(fimo_df["start"], errors="coerce")
        fimo_df["stop"]  = pd.to_numeric(fimo_df["stop"],  errors="coerce")
        fimo_df["score"] = pd.to_numeric(fimo_df["score"], errors="coerce")
        fimo_df["peak_id"] = fimo_df["sequence_name"].str.split("::").str[0]
        # Keep the BEST motif hit per peak (highest FIMO score)
        fimo_df = fimo_df.sort_values("score", ascending=False).drop_duplicates(subset="peak_id", keep="first")
    # Rank motif-positive peaks by FIMO score (partner-motif PWM strength)
    peak_lookup = df.set_index("name")
    fimo_df = fimo_df[fimo_df["peak_id"].isin(peak_lookup.index)].copy()
    fimo_df = fimo_df.sort_values("score", ascending=False)   # ← motif score, not signal
    out = []
    for _, f in fimo_df.iterrows():
        pid = f["peak_id"]
        row = peak_lookup.loc[pid]
        chrom = row["chrom"]
        summit = int(row["summit"])
        motif_start_in_peak = int(f["start"])  # 1-based, within the 1000bp peak window
        motif_stop_in_peak  = int(f["stop"])
        motif_len = motif_stop_in_peak - motif_start_in_peak + 1
        motif_start = (summit - 500) + (motif_start_in_peak - 1)
        motif_end   = motif_start + motif_len
        ctrl_start = motif_start + CONTROL_OFFSET_BP
        try:
            ref_seq  = fetch_genomic_sequence(chrom, motif_start, motif_len)
            ctrl_ref = fetch_genomic_sequence(chrom, ctrl_start,  motif_len)
        except Exception:
            continue
        if len(ref_seq) != motif_len or len(ctrl_ref) != motif_len: continue
        out.append({
            "chrom": chrom, "summit": summit,
            "motif_start": motif_start, "motif_end": motif_end,
            "motif_len": motif_len, "ref_seq": ref_seq,
            "ctrl_start": ctrl_start, "ctrl_ref": ctrl_ref,
            "fimo_score": float(f["score"]),
        })
        if len(out) >= n: break
    return out


def fetch_genomic_sequence(chrom, start, length=12):
    p = subprocess.run([BEDTOOLS, "getfasta", "-fi", str(GENOME_FASTA), "-bed", "-"],
                       input=f"{chrom}\t{start}\t{start+length}\n",
                       capture_output=True, text=True, check=True)
    lines = p.stdout.strip().splitlines()
    return lines[1].upper() if len(lines) >= 2 else ""


def scramble_motif(ref):
    swap = {"A":"C","C":"A","G":"T","T":"G","N":"N",
            "a":"C","c":"A","g":"T","t":"G","n":"N"}
    return "".join(swap.get(b, "N") for b in ref)


# ============================================================
# AlphaGenome multi-readout extraction
# ============================================================
def extract_signals_at_bin(output, mut_center, interval_start, target_tf, partner_tf,
                           interval_len=INTERVAL_LEN, half_bp=64):
    """Pull mean signal in a motif-centered window for each readout.

    Resolution-aware: AlphaGenome returns ATAC/DNase (and RNA/CAGE/PROCAP) at 1 bp
    (16384 bins) but CHIP-TF/histone at 128 bp (128 bins). A single shared bin index is
    therefore WRONG for the fine tracks — it read the far-left edge of the interval, so
    every pair got the same ~background value. Here the window is recomputed per array
    from its own length: a 128 bp track collapses to the single motif bin (preserving
    prior CHIP/histone results exactly), a 1 bp track averages +/-half_bp around the motif.
    """
    res = {}
    def safe_mean(arr, idxs):
        if not idxs: return float("nan")
        n = arr.shape[0]
        res_bp = interval_len / n                       # bp per position-bin
        c = int((mut_center - interval_start) / res_bp)
        hb = int(round(half_bp / res_bp))               # 0 for 128bp grid, half_bp for 1bp grid
        lo = max(0, c - hb); hi = min(n, c + hb + 1)
        return float(np.mean(arr[lo:hi, idxs]))
    # ChIP-TF: target and partner
    if output.chip_tf is not None:
        md = output.chip_tf.metadata
        tf_a_idx = md.index[md["transcription_factor"]==target_tf].tolist()
        # heterodimer partner: try direct then components
        partner_candidates = [partner_tf] + ([c.strip() for c in partner_tf.split('::')] if '::' in partner_tf else [])
        tf_b_idx = []
        for c in partner_candidates:
            tf_b_idx = md.index[md["transcription_factor"]==c].tolist()
            if tf_b_idx: break
        arr = output.chip_tf.values
        res["target"] = safe_mean(arr, tf_a_idx)
        res["partner"] = safe_mean(arr, tf_b_idx)
    else:
        res["target"] = res["partner"] = float("nan")
    # Histone marks
    if output.chip_histone is not None:
        md = output.chip_histone.metadata
        arr = output.chip_histone.values
        for mark in ALL_MARKS:
            idx = md.index[md["histone_mark"]==mark].tolist() if "histone_mark" in md.columns else []
            res[mark] = safe_mean(arr, idx)
    else:
        for mark in ALL_MARKS: res[mark] = float("nan")
    # Accessibility
    for name, attr in [("DNase","dnase"), ("ATAC","atac")]:
        track = getattr(output, attr, None)
        if track is None:
            res[name] = float("nan")
        else:
            arr = track.values
            n_tracks = arr.shape[1]
            res[name] = safe_mean(arr, list(range(n_tracks))) if n_tracks else float("nan")
    # Transcription
    for name, attr in [("RNA","rna_seq"), ("CAGE","cage"), ("PROCAP","procap")]:
        track = getattr(output, attr, None)
        if track is None:
            res[name] = float("nan")
        else:
            arr = track.values
            n_tracks = arr.shape[1]
            res[name] = safe_mean(arr, list(range(n_tracks))) if n_tracks else float("nan")
    return res


def _is_quota_error(exc):
    """Detect AlphaGenome (grpc) quota / rate-limit errors."""
    msg = str(exc)
    return ("RESOURCE_EXHAUSTED" in msg
            or "Quota exceeded" in msg
            or "rate" in msg.lower() and "limit" in msg.lower()
            or "429" in msg)


def _predict_variant_retry(client, interval, variant, requested_outputs, ontology_terms):
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            return client.predict_variant(interval=interval, variant=variant,
                requested_outputs=requested_outputs, ontology_terms=ontology_terms)
        except Exception as e:
            if _is_quota_error(e) and attempt < MAX_RETRY_ATTEMPTS - 1:
                time.sleep(QUOTA_RETRY_BACKOFF[attempt]); continue
            raise
    raise RuntimeError("predict_variant returned None after retries")


def contact_perturbation(ref_cm, alt_cm, motif_center, interval_start, interval_len):
    """3D readout = magnitude of the contact change a scramble causes at the motif locus:
    mean |Δ| of the motif bin's contacts (consensus over the 28 Hi-C/Micro-C tracks), ref vs alt.
    A localized loop loss (CTCF) registers here, unlike a mean-of-row metric that averages it away."""
    if ref_cm is None or alt_cm is None: return float("nan")
    R, A = ref_cm.values, alt_cm.values
    if R.ndim != 3 or R.shape[2] == 0: return float("nan")
    nbin = R.shape[0]; res_bp = interval_len / nbin
    b = min(max(int((motif_center - interval_start) / res_bp), 0), nbin - 1)
    rc = np.nanmean(R, axis=2); ac = np.nanmean(A, axis=2)
    d = np.abs(np.delete(ac[b], b) - np.delete(rc[b], b))
    return float(np.nanmean(d)) if d.size else float("nan")


def predict_one(client, target_tf, partner_tf, peak, variant_kind, efo):
    if variant_kind == "motif":
        var_start, ref = peak["motif_start"], peak["ref_seq"]
    else:
        var_start, ref = peak["ctrl_start"], peak["ctrl_ref"]
    alt = scramble_motif(ref)
    half = INTERVAL_LEN // 2
    motif_center = (peak["motif_start"] + peak["motif_end"]) // 2
    interval_start = max(0, motif_center - half)
    interval_end   = interval_start + INTERVAL_LEN
    interval = genome.Interval(chromosome=peak["chrom"],
                               start=interval_start, end=interval_end)
    variant = genome.Variant(chromosome=peak["chrom"],
                             position=var_start + 1,
                             reference_bases=ref, alternate_bases=alt)
    # quota-aware retry loop
    out = None
    for attempt in range(MAX_RETRY_ATTEMPTS):
        try:
            out = client.predict_variant(
                interval=interval, variant=variant,
                requested_outputs=REQUESTED_OUTPUTS,
                ontology_terms=[efo],
            )
            break
        except Exception as e:
            if _is_quota_error(e) and attempt < MAX_RETRY_ATTEMPTS - 1:
                wait_s = QUOTA_RETRY_BACKOFF[attempt]
                print(f"    [quota] attempt {attempt+1} hit RESOURCE_EXHAUSTED — sleeping {wait_s}s...", flush=True)
                time.sleep(wait_s)
                continue
            raise
    if out is None:
        raise RuntimeError("predict_variant returned None after retries")
    mut_center = var_start + peak["motif_len"] // 2
    sig = {
        "ref": extract_signals_at_bin(out.reference, mut_center, interval_start, target_tf, partner_tf),
        "alt": extract_signals_at_bin(out.alternate, mut_center, interval_start, target_tf, partner_tf),
    }
    # 3D contact: separate wider-window no-ontology call; CONTACT = perturbation magnitude (ref=0, alt=pert)
    cref = calt = float("nan")
    try:
        chalf  = CONTACT_INTERVAL_LEN // 2
        cstart = max(0, motif_center - chalf)
        civ    = genome.Interval(peak["chrom"], cstart, cstart + CONTACT_INTERVAL_LEN)
        cout   = _predict_variant_retry(client, civ, variant, CONTACT_OUTPUTS, None)
        pert   = contact_perturbation(cout.reference.contact_maps, cout.alternate.contact_maps,
                                      mut_center, cstart, CONTACT_INTERVAL_LEN)
        cref, calt = 0.0, pert
    except Exception as e:
        print(f"    [contact] skip peak: {type(e).__name__}: {str(e)[:50]}", flush=True)
    sig["ref"]["CONTACT"] = cref
    sig["alt"]["CONTACT"] = calt
    return sig


# ============================================================
# Cell-line config + env loading
# ============================================================
def load_api_key(api_key_idx: int) -> str:
    """Load API key from .env (idx=0) or .env.2 (idx=1) and set environment."""
    fname = ".env" if api_key_idx == 0 else f".env.{api_key_idx + 1}"
    env_path = ANALYSIS_DIR / fname
    if not env_path.exists():
        raise SystemExit(f"API key file not found: {env_path}")
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()
    if not os.environ.get("ALPHAGENOME_API_KEY"):
        raise SystemExit(f"ALPHAGENOME_API_KEY not found in {env_path}")
    return os.environ["ALPHAGENOME_API_KEY"]


def get_cell_efo(cell: str) -> str:
    """Look up EFO ontology term for a cell line."""
    if not ONTOLOGY_TSV.exists():
        raise SystemExit(f"Ontology mapping file missing: {ONTOLOGY_TSV}")
    df = pd.read_csv(ONTOLOGY_TSV, sep="\t")
    hit = df[df["cell_line"] == cell]
    if len(hit) == 0:
        raise SystemExit(f"Cell '{cell}' not found in {ONTOLOGY_TSV}")
    efo = hit.iloc[0]["efo_id"]
    if efo == "NA" or pd.isna(efo):
        raise SystemExit(f"Cell '{cell}' has no EFO mapping (efo_id=NA in {ONTOLOGY_TSV.name})")
    return str(efo)


def get_cell_tf_data_root(cell: str) -> Path:
    p = TF_DATA_BASE / cell
    if not p.exists():
        raise SystemExit(f"No ChIP data directory for {cell}: {p}")
    return p


def build_ag_coverage(cell: str, efo: str, client) -> set[str]:
    """Build/refresh ag_coverage_{cell}.txt = set of TFs AG has ChIP-TF tracks for in this cell line."""
    cov_path = ANALYSIS_DIR / f"ag_coverage_{cell}.txt"
    if cov_path.exists():
        return {l.strip().upper() for l in cov_path.read_text().splitlines() if l.strip()}
    print(f"[setup] Querying AG metadata for {cell} ({efo}) coverage...", flush=True)
    iv = genome.Interval(chromosome="chr19", start=14_000_000, end=14_016_384)
    out = client.predict_interval(
        interval=iv,
        requested_outputs=[dna_output.OutputType.CHIP_TF],
        ontology_terms=[efo],
    )
    md = out.chip_tf.metadata if out.chip_tf is not None else None
    if md is None or "transcription_factor" not in md.columns or len(md) == 0:
        # AG has this cell's EFO but no ChIP-TF tracks → no pair can be validated here
        cov_path.write_text("")
        print(f"[setup] {cell} ({efo}) has no AG ChIP-TF tracks — 0 TFs covered.", flush=True)
        return set()
    tfs = sorted({str(t).upper() for t in md["transcription_factor"].dropna().unique()})
    cov_path.write_text("\n".join(tfs) + "\n")
    print(f"[setup] Cached {len(tfs)} TFs with ChIP-TF tracks → {cov_path.name}", flush=True)
    return set(tfs)


# ============================================================
# Main loop
# ============================================================
def process_pair(client, pair_idx, target_tf, partner_tf, out_dir, efo, tf_data_root,
                 min_peaks):
    out_path = out_dir / f"pair_{pair_idx:05d}.csv"
    if out_path.exists():
        print(f"  [skip] {pair_idx} {target_tf}/{partner_tf} — output exists", flush=True)
        return
    info = find_partner_motif_id(partner_tf)
    if info is None:
        print(f"  [skip] {pair_idx} {target_tf}/{partner_tf} — no JASPAR motif", flush=True)
        out_path.write_text("# no_partner_motif_in_jaspar\n")
        return
    partner_motif_id, partner_resolved = info
    peaks = top_peaks_with_partner_motif(target_tf, partner_motif_id, n=PEAKS_PER_PAIR,
                                          tf_data_root=tf_data_root)
    if len(peaks) < min_peaks:
        print(f"  [skip] {pair_idx} {target_tf}/{partner_tf} — only {len(peaks)} peaks with motif "
              f"(min={min_peaks})", flush=True)
        out_path.write_text(f"# only_{len(peaks)}_peaks_with_motif\n")
        return
    print(f"  [run] {pair_idx} {target_tf}/{partner_tf}  peaks={len(peaks)}  motif={partner_motif_id}",
          flush=True)
    t0 = time.time()
    rows = []
    for i, p in enumerate(peaks, 1):
        try:
            m = predict_one(client, target_tf, partner_resolved, p, "motif", efo)
            time.sleep(SLEEP_BETWEEN_CALLS)
            c = predict_one(client, target_tf, partner_resolved, p, "control", efo)
            time.sleep(SLEEP_BETWEEN_CALLS)
            row = {
                "pair_idx": pair_idx, "target_tf": target_tf, "partner_tf": partner_tf,
                "partner_resolved": partner_resolved, "peak_idx": i,
                "chrom": p["chrom"], "motif_start": p["motif_start"], "ctrl_start": p["ctrl_start"],
                "fimo_score": p["fimo_score"],
            }
            for variant in ("motif", "ctrl"):
                src = m if variant == "motif" else c
                for state in ("ref", "alt"):
                    for key, val in src[state].items():
                        row[f"{variant}_{state}_{key}"] = val
            # log2FC per readout
            for variant in ("motif", "ctrl"):
                src = m if variant == "motif" else c
                for key in src["ref"]:
                    ref = src["ref"][key]
                    alt = src["alt"][key]
                    if pd.isna(ref) or pd.isna(alt):
                        row[f"{variant}_{key}_log2fc"] = float("nan")
                    elif key == "CONTACT":
                        # CONTACT is already a perturbation magnitude (ref=0, alt=pert): use the difference
                        row[f"{variant}_{key}_log2fc"] = float(alt - ref)
                    elif (ref + 0.01) <= 0:
                        row[f"{variant}_{key}_log2fc"] = float("nan")
                    else:
                        row[f"{variant}_{key}_log2fc"] = float(np.log2((alt + 0.01) / (ref + 0.01)))
            for key in m["ref"]:
                a = row.get(f"motif_{key}_log2fc")
                b = row.get(f"ctrl_{key}_log2fc")
                if pd.isna(a) or pd.isna(b):
                    row[f"paired_{key}"] = float("nan")
                else:
                    row[f"paired_{key}"] = a - b
            rows.append(row)
        except Exception as e:
            print(f"    error at peak {i}: {type(e).__name__}: {e}", flush=True)
            continue
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [done] {pair_idx} {target_tf}/{partner_tf}  rows={len(rows)}  t={time.time()-t0:.1f}s",
          flush=True)


def get_pair_list(cell: str, ag_tfs: set[str] | None = None) -> pd.DataFrame:
    """Pairs from rf_shap_top_regulator.csv for one cell line.
    If ag_tfs is provided, filter to pairs whose target has an AlphaGenome ChIP-TF track."""
    shap = pd.read_csv(PAIR_TABLE)
    k = shap[shap["Cellline"] == cell].copy().reset_index(drop=True)
    k["target_tf"] = k["Experiment"].str.upper()
    k["partner_tf"] = k["SHAP_Regulator_mean"].str.upper()
    if ag_tfs is not None:
        k = k[k["target_tf"].isin(ag_tfs)].reset_index(drop=True)
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="Cell line name as it appears in rf_shap_top_regulator.csv")
    ap.add_argument("--api_key_idx", type=int, default=0, choices=[0, 1],
                    help="0 uses .env, 1 uses .env.2")
    ap.add_argument("--min_peaks", type=int, default=5,
                    help="Minimum motif-positive peaks before skipping a pair (default 5)")
    ap.add_argument("--pair_idx", type=int, default=None, help="Process a single pair (overrides others)")
    ap.add_argument("--chunk_idx", type=int, default=None,
                    help=f"0..{CHUNKS_TOTAL-1} chunks across the cell's pair set")
    ap.add_argument("--start", type=int, default=None, help="Explicit start index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="Explicit end index (exclusive)")
    ap.add_argument("--no_target_filter", action="store_true",
                    help="Don't filter pairs by AG ChIP-TF coverage of the target TF")
    args = ap.parse_args()

    cell = args.cell
    load_api_key(args.api_key_idx)
    efo = get_cell_efo(cell)
    tf_data_root = get_cell_tf_data_root(cell)
    out_dir = ALL_ROOT / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] cell={cell}  efo={efo}  api_key_idx={args.api_key_idx}  "
          f"out_dir={out_dir.name}/  min_peaks={args.min_peaks}", flush=True)

    client = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])

    if args.no_target_filter:
        ag_tfs = None
    else:
        ag_tfs = build_ag_coverage(cell, efo, client)

    pairs = get_pair_list(cell, ag_tfs)
    if len(pairs) == 0:
        print(f"[exit] No pairs to process for {cell} (after AG target filter).", flush=True)
        return 0
    print(f"[config] {len(pairs)} pairs to process", flush=True)

    # Index selection
    if args.pair_idx is not None:
        idxs = [args.pair_idx]
    elif args.start is not None or args.end is not None:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None else len(pairs)
        idxs = list(range(start, min(end, len(pairs))))
    elif args.chunk_idx is not None:
        per = len(pairs) // CHUNKS_TOTAL + 1
        start = args.chunk_idx * per
        end = min(start + per, len(pairs))
        idxs = list(range(start, end))
    else:
        idxs = list(range(len(pairs)))

    if not idxs:
        print("[exit] No indices selected.", flush=True)
        return 0
    print(f"[run] Processing pair indices {idxs[0]}..{idxs[-1]} ({len(idxs)} pairs)", flush=True)

    for i in idxs:
        if i >= len(pairs):
            continue
        row = pairs.iloc[i]
        process_pair(client, i, row["target_tf"], row["partner_tf"],
                     out_dir, efo, tf_data_root, args.min_peaks)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
