#!/bin/bash
# Sanity-check one pair end-to-end. Usage: bash verify_pair.sh K562_ELK1_SRF
set -euo pipefail
PAIR="${1:-K562_ELK1_SRF}"
ROOT="/new-stg/home/jieyuan/mpra_validation"
PY="/new-stg/home/hanbei/miniconda3/envs/pipeline/bin/python"

cd "$ROOT"
echo "=== $PAIR ==="

ROW_LINE=$(awk -F'\t' -v p="$PAIR" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i; next} $1==p{
  print "target="$c["target_tf"]" partner="$c["partner_tf"]" n_peaks="$c["n_peaks"]" n_overlap="$c["n_overlap_fragments"]" n_motif_pos="$c["n_motif_pos"]" n_matched="$c["n_matched"]" effect="$c["effect_size_median"]" p="$c["mw_p"]" status="$c["status"];
}' pilot_out/K562_mpra_validation.tsv)
echo "reported: $ROW_LINE"

TARGET=$(echo "$ROW_LINE" | sed -E 's/.*target=([^ ]+).*/\1/')
PARTNER=$(echo "$ROW_LINE" | sed -E 's/.*partner=([^ ]+).*/\1/')

# 1. peaks
PEAK_LINES=$(cat /new-stg/home/hanbei/data/TF_ENCODE4/K562/${TARGET}_human/*.bed | wc -l)
echo "[1] raw peak file lines (all replicates): $PEAK_LINES (dedup is applied internally)"

# 2. FIMO hit count vs reported
FIMO_UNIQ=$(awk -F'\t' 'NR>1 && $3 ~ /^row_/ {print $3}' pilot_out/per_pair/$PAIR/fimo.tsv | sort -u | wc -l)
echo "[2] unique seqs with FIMO hit: $FIMO_UNIQ"

# 3. subset.csv motif_pos count
SUBSET_POS=$(awk -F, 'NR>1 && $5=="True"' pilot_out/per_pair/$PAIR/subset.csv | wc -l)
echo "[3] subset.csv motif_pos rows:  $SUBSET_POS  (should equal [2])"

# 4. partner motif identity
echo -n "[4] partner.meme MOTIF line: "
grep "^MOTIF " pilot_out/per_pair/$PAIR/partner.meme

# 5. matching quality
$PY -c "
import pandas as pd
m = pd.read_csv('pilot_out/per_pair/$PAIR/matched_pairs.csv')
print(f'[5] n_matched={len(m)}; median length pos={m.pos_length.median():.0f} vs neg={m.neg_length.median():.0f}; gc pos={m.pos_gc.median():.3f} vs neg={m.neg_gc.median():.3f}; tss pos={m.pos_tss_dist.median():.0f} vs neg={m.neg_tss_dist.median():.0f}')
"

# 6. Wilcoxon recompute
$PY -c "
import pandas as pd, numpy as np
from scipy import stats
m = pd.read_csv('pilot_out/per_pair/$PAIR/matched_pairs.csv')
mw = stats.mannwhitneyu(m.pos_activity, m.neg_activity, alternative='two-sided')
print(f'[6] recomputed Δmedian={np.median(m.pos_activity)-np.median(m.neg_activity):+.4f}, MW p={mw.pvalue:.3e}')
"

# 7. sequence matches fold0.fasta for first row in subset.fa
FIRST_ROW=$(head -1 pilot_out/per_pair/$PAIR/subset.fa | sed 's/>row_//')
SUB_SEQ=$(awk -v want=">row_${FIRST_ROW}" '$0==want{getline; print; exit}' pilot_out/per_pair/$PAIR/subset.fa)
FA_SEQ=$(awk -v want=">fold0_${FIRST_ROW}" '$0==want{getline; print; exit}' /stg3/data1/sam/enhancer_prediction/multi_agent/universal_regulators/validation/fold0.fasta)
if [ "$SUB_SEQ" = "$FA_SEQ" ]; then
  echo "[7] subset.fa row_${FIRST_ROW} matches fold0.fasta exactly  ✓"
else
  echo "[7] MISMATCH on row_${FIRST_ROW}"
fi
