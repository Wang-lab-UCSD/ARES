#!/usr/bin/env bash
# Fetch the data the SP1/NFYA demo manifest reads, into ./data next to this script.
#
# data_manifest_demo.yaml names every file by a path relative to itself, and the pipeline resolves
# manifest paths against the manifest's own directory, so nothing needs editing once this finishes.
#
# About 26 GB. phyloP (9.9 GB), the genome (3.3 GB) and the two WGBS tracks (3.5 GB) are most of it.
# Files already present are skipped, so an interrupted download can be resumed by re-running.
#
# Usage:  bash download_demo_data.sh [-o OUTDIR]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/data"
while getopts "o:" opt; do case $opt in o) OUT="$OPTARG";; *) exit 2;; esac; done
mkdir -p "$OUT"
FAILED=()

# Download to a temp name and move into place only on success, so an interrupted transfer never
# leaves a truncated file that a later run would mistake for a finished one.
get() {                                  # get <url> <relative-name> [gz] [quiet]
  local url="$1" name="$2" gz="${3:-}" quiet="${4:-}"
  local dest="$OUT/$name"
  if [[ -s "$dest" ]]; then [[ -n "$quiet" ]] || printf '  have  %s\n' "$name"; return 0; fi
  mkdir -p "$(dirname "$dest")"
  [[ -n "$quiet" ]] || printf '  get   %-44s' "$name"
  local tmp="$dest.part"
  if ! curl -fsSL --retry 3 --retry-delay 5 -o "$tmp" "$url"; then
    printf 'FAILED %s\n' "$name"; rm -f "$tmp"; FAILED+=("$name"); return 1
  fi
  if [[ "$gz" == gz ]]; then
    gunzip -c "$tmp" > "$dest" && rm -f "$tmp" || {
      printf 'FAILED (not gzip) %s\n' "$name"; rm -f "$tmp" "$dest"; FAILED+=("$name"); return 1; }
  else mv "$tmp" "$dest"; fi
  [[ -n "$quiet" ]] || printf 'ok  %s\n' "$(du -h "$dest" | cut -f1)"
}

enc() {                                  # enc <accession> <ext> [dest-dir] — ENCODE gzips bed/bedpe
  local acc="$1" ext="$2" sub="${3:-}" name
  name="${sub:+$sub/}$acc.$ext"
  case "$ext" in
    bed|bedpe) get "https://www.encodeproject.org/files/$acc/@@download/$acc.$ext.gz" "$name" gz "${4:-}" ;;
    *)         get "https://www.encodeproject.org/files/$acc/@@download/$acc.$ext"    "$name" ''  "${4:-}" ;;
  esac
}

echo "ENCODE — the two factors under investigation (K562)"
enc ENCFF171NEU bed      # SP1 peaks
enc ENCFF524IWI bigWig   # SP1 signal
enc ENCFF908HSL bed      # NFYA peaks
enc ENCFF427FTJ bigWig   # NFYA signal

echo "ENCODE — cell-line-wide assays (K562)"
enc ENCFF928NYA tsv      # RNA-seq, RSEM gene quantification
enc ENCFF655HFU bigWig   # DNase-seq
enc ENCFF381NDD bigWig   # H3K27ac
enc ENCFF253TOF bigWig   # H3K4me3
enc ENCFF139KZL bigWig   # H3K27me3
enc ENCFF607SUJ bigWig   # H3K4me1
enc ENCFF812HRW bigWig   # H3K9me3
enc ENCFF430PNX bigWig   # WGBS minus strand
enc ENCFF459XNY bigWig   # WGBS plus strand
enc ENCFF134HIZ bedpe    # Hi-C loops
enc ENCFF448GZL bedpe    # Hi-C stripes

echo "Reference genome, annotation and conservation"
get https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz hg38.fa gz
get https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz \
    gencode.v38.annotation.gtf gz
get https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw hg38.phyloP100way.bw

echo "STRING v12.0 (Homo sapiens)"
get https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz \
    9606.protein.links.v12.0.txt.gz
get https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz \
    9606.protein.aliases.v12.0.txt.gz

echo "Roadmap Epigenomics ChromHMM (E123 = K562, 18-state core, lifted to hg38)"
get https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/core_K27ac/jointModel/final/E123_18_core_K27ac_hg38lift_mnemonics.bed.gz \
    E123_18_core_K27ac_hg38lift_mnemonics.bed gz

# The whole-cell-line pool, peaks only: 526 narrowPeak files at 0.37 GB against 700 GB for the
# matching signal tracks. Progress is a counter rather than a line per factor.
echo "ENCODE — whole-cell-line TF pool, peaks only ($(($(wc -l < "$HERE/tf_pool_peaks.tsv") - 1)) factors)"
n=0
while IFS=$'\t' read -r pool dir acc; do
  [[ "$pool" == pool ]] && continue
  case "$pool" in primary) sub="TF_ENCODE4_K562/$dir";; *) sub="TF_additional_K562/$dir";; esac
  enc "$acc" bed "$sub" quiet
  n=$((n+1)); (( n % 50 )) || printf '  ...%d\n' "$n"
done < "$HERE/tf_pool_peaks.tsv"
printf '  %d factors done\n' "$n"

if ((${#FAILED[@]})); then
  printf '\n%d file(s) did not download.\n' "${#FAILED[@]}"
  printf '  %s\n' "${FAILED[@]}" | head -20
  echo "Fetch these by hand into $OUT under the same names, then re-run to verify."
  exit 1
fi
echo; echo "All files present in $OUT"
echo "Run:  python -m src.main --config config/config.yaml \\"
echo "        --manifest examples/sp1_nfya_K562/data_manifest_demo.yaml"
