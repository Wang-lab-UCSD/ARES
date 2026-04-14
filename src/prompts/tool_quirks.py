"""Tool-specific quirks and gotchas for bioinformatics tools.

This file documents known issues and workarounds for various bioinformatics tools.
Each tool has its own section that can be included in prompts when relevant.

Add new tool quirks as they are discovered during pipeline runs.
"""

from __future__ import annotations

# =============================================================================
# FIMO (MEME Suite) - Motif scanning tool
# =============================================================================
FIMO_QUIRKS = """
=== FIMO RULES — READ BEFORE WRITING CODE ===

**RULE 1: ALWAYS use --no-pgc**

Without --no-pgc, FIMO parses FASTA headers for genomic coordinates and:
1. Reports chromosome as `sequence_name` (not your peak ID)
2. Converts positions to genomic coordinates (not sequence-relative)
This breaks peak counting — you get 23 chromosomes instead of thousands of peaks.

**RULE 2: Use --motif to scan only the motifs you need**

MEME files like JASPAR contain 800+ motifs. Scanning all of them is allowed but
takes HOURS and uses gigabytes of RAM. Use --motif whenever possible to select
only the specific motif(s) your hypothesis requires. Each targeted scan takes
SECONDS instead of hours.

SLOW (scans all 800+ motifs — takes hours, only do this when necessary):
  fimo --no-pgc --oc out --thresh 1e-4 JASPAR_full.meme peaks.fa

FAST (scans one motif — takes seconds, preferred):
  fimo --no-pgc --oc out --thresh 1e-4 --motif "REST|jaspar|MA0138.2" JASPAR_full.meme peaks.fa

You can run multiple targeted scans for different motifs:
  fimo --no-pgc --oc out_rest --thresh 1e-4 --motif "REST|jaspar|MA0138.2" JASPAR_full.meme peaks.fa
  fimo --no-pgc --oc out_atf6 --thresh 1e-4 --motif "ATF6|jaspar|MA1466.1" JASPAR_full.meme peaks.fa

**RULE 3: NEVER use --text mode**

`fimo --text` streams output to stdout but does NOT compute q-values, and p-values
may also be unreliable. Code that then filters on significance columns will find
incorrect or zero hits, silently producing wrong results (every hypothesis appears to REFUSE).

WRONG — significance values will be unreliable:
  fimo --no-pgc --text --thresh 1e-4 --motif "USF1|jaspar|MA0093.3" motifs.meme peaks.fa

CORRECT — let FIMO write to an output directory:
  fimo --no-pgc --oc fimo_out --thresh 1e-4 --motif "USF1|jaspar|MA0093.3" motifs.meme peaks.fa

Then read from the output file:
```python
fimo_df = pd.read_csv('fimo_out/fimo.tsv', sep='\\t', comment='#')
# Prefer p-value over q-value for peak-level analysis (see RULE 4)
hits = fimo_df[fimo_df['p-value'] < 1e-4]
```

**SANITY CHECK — always verify p-values are not NaN**:
```python
nan_frac = fimo_df['p-value'].isna().mean()
if nan_frac > 0.5:
    raise RuntimeError(f"FIMO p-values are {nan_frac:.0%} NaN — did you use --text mode? Use --oc instead.")
```

**RULE 4: Prefer p-value over q-value for peak-level analysis**

When analyzing motif occurrences within individual ChIP-seq peaks, filter motif hits
based on p-values rather than q-values. FIMO's q-values use Benjamini-Hochberg correction
across ALL sequences scanned, which may be overly conservative for peak-level analysis —
especially in shuffled/control regions where true motif hits are rare, legitimate hits
can get q > 0.05, producing zero results and crashing enrichment comparisons.

Recommended — p-value filter is stable across foreground and control:
```python
hits = fimo_df[fimo_df['p-value'] < 1e-4]  # Field-standard threshold for peak-level analysis
```

Caution — q-value filter can drop legitimate hits in control regions:
```python
hits = fimo_df[fimo_df['q-value'] < 0.05]  # May return 0 rows for shuffled peaks
```

The `--thresh 1e-4` flag already pre-filters on p-value at the command line; your Python
code should match this same threshold when post-filtering.

**Complete Workflow**:

STEP 1: Create BED with numeric peak IDs and extract sequences:
```python
peak_mapping = {}
with open(peaks_bed, 'r') as f, open('peaks_named.bed', 'w') as out:
    for i, line in enumerate(f):
        parts = line.strip().split('\\t')
        chrom, start, end = parts[0], parts[1], parts[2]
        peak_id = f"peak_{i:05d}"
        peak_mapping[peak_id] = (chrom, int(start), int(end))
        out.write(f"{chrom}\\t{start}\\t{end}\\t{peak_id}\\n")

import json
with open('peak_mapping.json', 'w') as f:
    json.dump(peak_mapping, f)
```

```bash
bedtools getfasta -fi genome.fa -bed peaks_named.bed -name -fo peaks.fa
```

STEP 2: Run FIMO with --no-pgc AND --motif:
```bash
fimo --no-pgc --oc fimo_out --thresh 1e-4 --motif "REST|jaspar|MA0138.2" motifs.meme peaks.fa
```

Now FIMO will report:
- `sequence_name` = "peak_00001::chr1:1000-2000" (full header, not truncated!)
- `start`/`stop` = sequence-relative positions (small numbers like 45-67)

STEP 3: Parse results and map back to peaks:
```python
fimo_df = pd.read_csv('fimo_out/fimo.tsv', sep='\\t', comment='#')
fimo_df['peak_id'] = fimo_df['sequence_name'].str.split('::').str[0]
peaks_with_hits = fimo_df['peak_id'].nunique()
print(f"Peaks with motif: {peaks_with_hits}")
```

**SANITY CHECK**:
```python
unique_names = fimo_df['sequence_name'].nunique()
print(f"Unique sequence names: {unique_names}")
if unique_names < 100:
    print("ERROR: Did you forget --no-pgc flag?")

if fimo_df['start'].max() > 10000:
    print("ERROR: Coordinates are genomic, not sequence-relative. Use --no-pgc!")
```

**VALID FIMO FLAGS — only use flags from this list. Any other flag is likely hallucinated.**

| Flag | Description |
|---|---|
| `--no-pgc` | Do NOT parse genomic coordinates from FASTA headers (ALWAYS use this) |
| `--oc <dir>` | Write output to directory, overwriting if it exists (ALWAYS use this) |
| `--o <dir>` | Write output to directory, fails if it exists (use `--oc` instead) |
| `--thresh <float>` | p-value threshold for reporting hits (default: 1e-4) |
| `--motif <id>` | Only score the specified motif — must be the FULL motif name from the MEME file (e.g., `"REST|jaspar|MA0138.2"`). The accession alone (`MA0138.2`) produces zero hits. |
| `--bfile <file>` | Background Markov model file |
| `--max-strand` | Report only the best-scoring strand hit per position |
| `--qv-thresh` | Use q-values instead of p-values for thresholding (not recommended for peak-level analysis — see RULE 4) |
| `--verbosity <1-5>` | Verbosity level |

Do NOT use any flag not in this table. There is no `--no-header`, `--no-pgc-fix`, `--genomic`, `--peak-id`, `--format`, or any other flag not listed above.

**RULE 5: `run_fimo_on_peaks()` — correct argument order and output columns**

Argument order: `run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, motif_id=...)`.
`genome_fa` comes BEFORE `motif_meme`. Swapping them causes a `CalledProcessError` from bedtools getfasta.

```python
# CORRECT:
fimo_hits = run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, motif_id='REST|jaspar|MA0138.2')
# WRONG (swapped genome_fa and motif_meme):
fimo_hits = run_fimo_on_peaks(peaks_bed, motif_meme, genome_fa, motif_id='REST|jaspar|MA0138.2')
```

The bioio helper `run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, ...)` returns a DataFrame
with these columns (from `parse_fimo_tsv`):
`motif_id, sequence_name, start, stop, strand, score, p-value, q-value, matched_sequence, peak_id`

- `sequence_name`: the FASTA header ID (e.g. `"peak_00001::chr1:1000-2000"`)
- `peak_id`: extracted from `sequence_name` (e.g. `"peak_00001"`) — use this to join back to peaks
- `start`/`stop`: 1-based, sequence-relative coordinates within the peak (small numbers like 45–67)
- **There is NO `'chrom'` column in FIMO output.** Accessing `fimo_df['chrom']` will raise KeyError.

If you need the chromosome from FIMO results, extract it from `sequence_name`:
```python
fimo_df['chrom'] = fimo_df['sequence_name'].str.split('::').str[1].str.split(':').str[0]
```

Typical join pattern (FIMO hits → back to peaks):
```python
from src.utils.bioio import run_fimo_on_peaks, read_narrowpeak

peaks = read_narrowpeak(peaks_bed)
# NOTE: genome_fa comes BEFORE motif_meme
fimo_hits = run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, motif_id='NFYA|jaspar|MA0060.3')

# IMPORTANT: ENCODE narrowPeak col 4 ('name') is '.' for every peak.
# run_fimo_on_peaks() detects this and internally assigns sequential IDs (peak_00000, peak_00001, ...)
# so that peak_id in fimo_hits is always unique.
# read_narrowpeak() drops the 'name' column, so always reconstruct sequential IDs:
peaks = peaks.reset_index(drop=True)
peaks['_peak_id'] = [f'peak_{i:05d}' for i in range(len(peaks))]
peaks_with_motif = peaks[peaks['_peak_id'].isin(fimo_hits['peak_id'])]
print(f"Peaks with motif: {len(peaks_with_motif)} / {len(peaks)}")
```

**RULE 6: NEVER run FIMO on the full genome FASTA — it will time out**

Scanning the full hg38 genome (~3.1 Gb across all chromosomes) with FIMO takes **hours**
and will exceed the pipeline's 15-minute execution timeout. This applies even with
`--motif` (single-motif scanning) because the genome is simply too large.

WRONG — will time out:
```python
subprocess.run(["fimo", "--no-pgc", "--oc", "out", "--motif", motif_id, motifs_meme, genome_fa], check=True)
```

CORRECT — scan only within peak windows using `run_fimo_on_peaks()`:
```python
fimo_hits = run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, motif_id=motif_id)
```

If your hypothesis requires finding motif instances "genome-wide" or outside peaks,
scan the **union of all peak sets** (e.g., merge AFF1 + REST peaks with `bedtools merge`)
or use a broad candidate set (all DNase-accessible regions). This approximates
genome-wide coverage without the full-genome cost:
```python
# Create a broad candidate set from multiple peak files
subprocess.run(["cat", aff1_bed, rest_bed], stdout=open(merged_bed, 'w'), check=True)
subprocess.run(["bedtools", "sort", "-i", merged_bed], stdout=open(sorted_bed, 'w'), check=True)
subprocess.run(["bedtools", "merge", "-i", sorted_bed], stdout=open(candidates_bed, 'w'), check=True)
fimo_hits = run_fimo_on_peaks(candidates_bed, genome_fa, motif_meme, motif_id=motif_id)
```

**RULE 7: motif_id vs motif_alt_id — know which column to filter on**

FIMO output has TWO motif identifier columns:
- `motif_id`: the primary ID from the MEME file (e.g., `SP1|jaspar|MA0079.5`)
- `motif_alt_id`: the alternate ID (may be empty or different)

The value you pass to `--motif` on the command line is what appears in `motif_id`.
When filtering FIMO output in Python, ALWAYS filter on `motif_id`, not `motif_alt_id`:

```python
df = pd.read_csv(fimo_tsv, sep='\\t', comment='#')
# Check what's actually in the motif_id column:
print("motif_id values:", df['motif_id'].unique()[:5])
# Filter on motif_id (matches what you passed to --motif):
df = df[df['motif_id'] == 'SP1|jaspar|MA0079.5']
```

**Memory Warning**: Filter large FIMO output by motif_id immediately:
```python
df = pd.read_csv(fimo_tsv, sep='\\t', comment='#')
df = df[df['motif_id'] == 'REST|jaspar|MA0138.2']  # Filter first! Use full ID.
```

**RULE 8: Never parse `sequence_name` manually — use `run_fimo_on_peaks()` or `parse_fimo_tsv()`**

When `run_fimo_on_peaks()` is used, `sequence_name` in the raw FIMO TSV contains values like
`peak_00001::chr6:1000-2000` (peak_id + `::` + original coords from bedtools getfasta -name).
Splitting on `':'` gives `['peak_00001', '', 'chr6', '1000-2000']` — the `::` produces an
empty element, causing `int('chr6')` or index-off-by-one errors.

NEVER parse `sequence_name` manually. The helpers do it correctly:
- `run_fimo_on_peaks()` returns a DataFrame with `peak_id` already extracted — use that directly
- `parse_fimo_tsv()` also extracts `peak_id` from the sequence name header

```python
# CORRECT — peak_id is already in the returned DataFrame.
# ENCODE narrowPeak col 4 ('name') is '.' for every peak, so run_fimo_on_peaks()
# internally assigns sequential IDs (peak_00000, peak_00001, ...) when names are uniform.
# Always reconstruct matching IDs on your peaks DataFrame before joining:
fimo_hits = run_fimo_on_peaks(peaks_bed, genome_fa, motif_meme, motif_id='REST|jaspar|MA0138.2')
peaks = peaks.reset_index(drop=True)
peaks['_peak_id'] = [f'peak_{i:05d}' for i in range(len(peaks))]
peaks_with_motif = peaks[peaks['_peak_id'].isin(fimo_hits['peak_id'])]

# WRONG — read_narrowpeak() drops the 'name' column; this raises KeyError:
peaks_with_motif = peaks[peaks['name'].isin(fimo_hits['peak_id'])]  # KeyError: 'name'

# WRONG — manual sequence_name parsing breaks on '::' separator:
fimo_df['chrom'] = fimo_df['sequence_name'].str.split(':').str[0]  # Wrong! gets 'peak_00001'
fimo_df['pos'] = fimo_df['sequence_name'].str.split(':').str[2].astype(int)  # Crashes on 'chr6'
```

**RULE 9: Zero-inflation trap — NEVER correlate motif_score vs signal on ALL peaks including score=0**

When FIMO does not find a motif at a peak, that peak gets motif_score=0.  Typically 50–80%
of peaks have score=0.  Computing Spearman(motif_score, chip_signal) on all peaks gives a
spurious positive correlation driven by the binary has-motif / no-motif split, NOT by motif
*strength* among motif-positive peaks.  This is Simpson's Paradox.

WRONG — includes zeros, inflates correlation:
```python
r, p = spearmanr(peaks['nfya_motif_score'], peaks['sp1_signal'])  # r ≈ +0.12
```

CORRECT — restrict to motif-positive peaks first:
```python
motif_pos = peaks[peaks['nfya_motif_score'] > 0]
r, p = spearmanr(motif_pos['nfya_motif_score'], motif_pos['sp1_signal'])  # r ≈ -0.04
```

If you need to test whether motif *presence* (binary) predicts signal, use a group comparison
(Mann-Whitney / Cohen's d on motif-positive vs motif-negative), not a correlation on the
zero-inflated scores.
"""

# =============================================================================
# bedtools - Genomic interval operations
# =============================================================================
BEDTOOLS_QUIRKS = """
=== BEDTOOLS RULES — READ BEFORE WRITING CODE ===

**RULE 1: Use absolute paths for ALL file arguments**

The code runs in a Jupyter kernel whose working directory may differ from where
you expect. Relative paths like `tmp_dir/output.bed` will silently fail or
produce empty results. Always construct absolute paths:

```python
import os, tempfile
work_dir = tempfile.mkdtemp(prefix="analysis_")
output_bed = os.path.join(work_dir, "output.bed")
```

**RULE 2: No invalid flags — check the command syntax**

Common mistakes:
- `-output` is NOT a valid bedtools flag. Use shell redirection (`>`) or `-fo` (for getfasta).
- `bedtools intersect -a A.bed -b B.bed > result.bed` — use `>` for output, not a flag.
- `bedtools getfasta -fi genome.fa -bed peaks.bed -fo output.fa` — use `-fo` for output file.

When using subprocess in Python, do NOT use `shell=True`, `bash -lc`, or shell
redirection (`>`) for bioinformatics tools. A login shell can reset `PATH` and lose
the active conda environment, making tools appear "missing" even when installed.
Write output yourself with `stdout=` instead:
```python
import subprocess
with open(output_file, "w") as f:
    subprocess.run(
        ["bedtools", "intersect", "-a", file_a, "-b", file_b, "-u"],
        stdout=f,
        text=True,
        check=True,
    )
```

**RULE 3: bedtools shuffle needs a genome file**

`bedtools shuffle` requires `-g genome.chrom.sizes`. Generate it from the FASTA index
without `shell=True`:
```python
import subprocess

with open(chrom_sizes, "w") as out:
    subprocess.run(["cut", "-f1,2", f"{genome_fa}.fai"], stdout=out, text=True, check=True)
with open(shuffled_bed, "w") as out:
    subprocess.run(["bedtools", "shuffle", "-i", peaks_bed, "-g", chrom_sizes], stdout=out, text=True, check=True)
```

**RULE 4: NEVER call getfasta, FIMO, or bigWig extraction inside a loop**

These operations are expensive and MUST be called ONCE, outside any loop:
- `bedtools getfasta` — reads entire genome (~3GB per call)
- `fimo` — scans all sequences per call
- `pyBigWig` / `bigWigAverageOverBed` — reads large bigWig files (1-3GB per call)

WRONG (hours — getfasta in loop):
```python
for i in range(100):
    subprocess.run(["bedtools", "shuffle", "-i", peaks, "-g", chrom_sizes], ...)
    subprocess.run(["bedtools", "getfasta", "-fi", genome, "-bed", shuffled, ...])  # 3GB read
```

WRONG (hours — bigWig extraction in loop):
```python
for i in range(100):
    shuffled = shuffle_peaks(...)
    for chrom, start, end in shuffled_regions:
        bw.stats(chrom, start, end)  # millions of bigWig lookups
```

RIGHT — for overlap permutation tests, just shuffle and intersect (seconds per replicate):
```python
for i in range(100):
    subprocess.run(["bedtools", "shuffle", "-i", peaks, "-g", chrom_sizes], ...)
    subprocess.run(["bedtools", "intersect", "-a", shuffled, "-b", other_peaks, "-u"], ...)
    # count overlapping lines — no sequence extraction needed
```

RIGHT — for signal comparison, extract signals ONCE and use Mann-Whitney:
```python
# Extract signals ONCE for both groups
signals_groupA = [bw.stats(c, s, e)[0] for c, s, e in group_a_regions]
signals_groupB = [bw.stats(c, s, e)[0] for c, s, e in group_b_regions]
# Compare with analytical test (instant)
stat, pvalue = scipy.stats.mannwhitneyu(signals_groupA, signals_groupB)
```

**getfasta output format**:
- With `-name` flag: Headers are `>name::chr:start-end`
- Without `-name` flag: Headers are `>chr:start-end`
- The `::` separator is important when parsing

**intersect tips**:
- `-u` reports each -a feature only once (even if multiple overlaps)
- `-c` counts overlaps but includes 0 for non-overlapping
- `-wa -wb` reports both features but can produce huge output

**Coordinate systems**:
- BED format is 0-based, half-open: [start, end)
- FIMO output uses 1-based coordinates
- Always verify coordinate systems match before intersecting

**RULE 5: bedtools shuffle ALWAYS needs a genome file (-g)**

`bedtools shuffle` WILL FAIL without `-g chrom.sizes`. Always generate it first:
```python
import subprocess, os
chrom_sizes = os.path.join(work_dir, "chrom.sizes")
subprocess.run(["samtools", "faidx", genome_fa], check=True)
with open(chrom_sizes, "w") as out:
    subprocess.run(["cut", "-f1,2", f"{genome_fa}.fai"], stdout=out, text=True, check=True)
# Now shuffle:
with open(shuffled_bed, "w") as out:
    subprocess.run(["bedtools", "shuffle", "-i", peaks_bed, "-g", chrom_sizes],
                   stdout=out, text=True, check=True)
# ALWAYS validate the output:
n_shuffled = sum(1 for _ in open(shuffled_bed))
n_input = sum(1 for _ in open(peaks_bed))
assert n_shuffled == n_input, f"Shuffle lost intervals: {n_shuffled} vs {n_input}"
```

**bigWigAverageOverBed (UCSC tool) — BED input format**:
- The input BED file MUST have at least 4 columns: chrom, start, end, name.
- BED3 (chrom, start, end only) will fail with "At least 4 fields required, got 3".
- When writing BED files for bigWigAverageOverBed, always include a 4th name column, e.g.:
  `bed4 = df[["col1","col2","col3"]].copy(); bed4["name"] = [f"peak_{i}" for i in range(len(bed4))]; bed4.to_csv(bed_path, sep="\t", header=False, index=False)`
- bedtools intersect output preserves the -a file format; if -a is BED3, the output is BED3. Add a name column before passing to bigWigAverageOverBed.

**RULE 6: bedtools intersect — mutually exclusive output flags**

Several `bedtools intersect` output flags are mutually exclusive. Using them together causes an
immediate fatal error with no output:
- `-wa` and `-wo` cannot be combined: "request either -wa for writeA OR -wo for writeOverlap, not both"
- `-wa` and `-wao` cannot be combined
- `-u` and `-v` cannot be combined (report hits vs report non-hits)
- `-c` and `-u` cannot be combined

Choose ONE output mode:
| Goal | Flag(s) |
|---|---|
| Rows from A that overlap B (original A coords) | `-wa` |
| Both A and B columns for each overlap | `-wb` (gives A+B columns) |
| A and B columns plus overlap length | `-wo` |
| A and B columns, including A rows with no hit | `-wao` |
| Report A once if any overlap exists | `-u` |
| Report A rows with NO overlap | `-v` |
| Count overlaps per A row | `-c` |

**VALID BEDTOOLS FLAGS — only use flags from this list per subcommand.**

`bedtools intersect`: `-a`, `-b`, `-u`, `-v`, `-c`, `-wa`, `-wb`, `-wo`, `-wao`, `-f`, `-r`, `-s`, `-S`, `-sorted`, `-loj`
`bedtools getfasta`: `-fi`, `-bed`, `-fo`, `-name`, `-s`, `-tab`, `-split`
`bedtools shuffle`: `-i`, `-g`, `-excl`, `-chrom`, `-seed`, `-noOverlapping`, `-maxTries`
`bedtools sort`: `-i`, `-g`
`bedtools closest`: `-a`, `-b`, `-d`, `-s`, `-S`, `-t`, `-io`, `-iu`, `-id`, `-D`
`bedtools slop`: `-i`, `-g`, `-b`, `-l`, `-r`, `-s`, `-pct`
`bedtools nuc`: `-fi`, `-bed`, `-s`, `-seq`, `-pattern`, `-C`
`bedtools coverage`: `-a`, `-b`, `-d`, `-hist`, `-mean`, `-counts`, `-sorted`

Do NOT invent flags. There is no `-output`, `-out`, `-results`, `-peaks`, or any other flag not listed above.
"""

# =============================================================================
# MEME Suite general
# =============================================================================
MEME_SUITE_QUIRKS = """
=== MEME SUITE GENERAL QUIRKS ===

**MEME format files**:
- Comments start with `#`
- Use `comment='#'` when reading with pandas

**Motif IDs in this project's MEME file**:

The combined motif file (`combined_TF_motifs_jaspar_hocomoco12_cisbp.meme`) uses this format:
  `TF_NAME|database|accession`

Examples:
- JASPAR: `SP1|jaspar|MA0079.5`, `REST|jaspar|MA0138.2`, `NFYA|jaspar|MA0060.3`
- HOCOMOCO12: `ATF6A|hocomoco12|ATF6A.H12CORE.0.SM.B`
- CisBP: `AHR|cisbp|M08716_2.00`

**CRITICAL**: The `--motif` flag for FIMO requires the FULL ID (e.g. `"SP1|jaspar|MA0079.5"`).
Using only the accession (`MA0079.5`) or gene name (`SP1`) produces **zero hits silently**.

To find the correct ID for a TF, grep the MEME file:
```bash
grep "^MOTIF.*SP1" /path/to/combined_TF_motifs_jaspar_hocomoco12_cisbp.meme
# → MOTIF SP1|jaspar|MA0079.5
```

**Background models**:
- Default background may not match your sequences
- For accurate p-values, use `-bfile` with matched background
"""

# =============================================================================
# narrowPeak / BED file parsing
# =============================================================================
NARROWPEAK_QUIRKS = """
=== NARROWPEAK / BED PARSING RULES ===

**RULE 1: Use the shared helper for narrowPeak parsing**

narrowPeak files are easy to parse incorrectly because the summit is encoded as an offset
and `peak=-1` requires a midpoint fallback. Prefer:

```python
from src.utils.bioio import read_narrowpeak
df = read_narrowpeak(path)
```

This returns canonical narrowPeak columns and adds `peak_offset` and `summit`.

**RULE 2: Always produce clean BED3/BED4 before passing to bedtools**

Many bedtools commands choke on extra columns or inconsistent formats. After loading
narrowPeak, immediately create a clean BED for downstream use:

```python
bed3 = df[['chrom', 'start', 'end']].copy()
bed3.to_csv(clean_bed_path, sep='\\t', header=False, index=False)
```

For bigWigAverageOverBed, you need BED4 (with a name column):
```python
bed4 = df[['chrom', 'start', 'end']].copy()
bed4['name'] = [f"peak_{i}" for i in range(len(bed4))]
bed4.to_csv(bed4_path, sep='\\t', header=False, index=False)
```

**RULE 3: Validate BED intervals before use**

```python
bad = df[df['start'] >= df['end']]
if len(bad) > 0:
    print(f"WARNING: {len(bad)} intervals have start >= end, dropping them")
    df = df[df['start'] < df['end']].copy()
```

**RULE 4: `read_narrowpeak()` drops the `name` column — `run_fimo_on_peaks()` handles peak IDs automatically**

ENCODE narrowPeak files have `'.'` in column 4 (the `name` field) for every peak.
`read_narrowpeak()` drops this useless column to prevent accidental `.merge(on='name')`.
`run_fimo_on_peaks()` detects uniform names and auto-assigns sequential IDs (`peak_00000`, ...) before running
getfasta, so the returned `peak_id` column is always unique and useful for joining back to peaks.

If you write your own getfasta+FIMO pipeline, assign unique IDs first:
```python
peaks = read_narrowpeak(peaks_bed)
peaks['_peak_id'] = [f"peak_{i:05d}" for i in range(len(peaks))]
bed4 = peaks[['chrom','start','end','_peak_id']].copy()
bed4.to_csv(named_bed, sep='\\t', header=False, index=False)
```

**RULE 5: NEVER merge peak DataFrames on `'name'` — use `intersect_peaks()` instead**

Because ENCODE `name` is `'.'` for every peak, `.merge(on='name')` produces a
cartesian product (N×M rows) and will exhaust RAM. Use the `intersect_peaks()`
helper which joins by genomic coordinate overlap via bedtools:

```python
from src.utils.bioio import intersect_peaks, count_overlapping_peaks

# (A) Get paired rows for every A↔B overlap:
overlap_df = intersect_peaks(peaks_a_df_or_bed, peaks_b_df_or_bed)
# overlap_df columns: a_0, a_1, a_2, a_3, b_0, b_1, b_2, b_3

# (B) Flag which A-peaks overlap any B-peak:
flagged = intersect_peaks(peaks_a_df, peaks_b_df, mode="flag")
# When inputs are DataFrames, ALL original columns are preserved.
# flagged has: chrom, start, end, ..., summit, overlaps_b
# You can use flagged["chrom"], flagged["start"], flagged["summit"], etc.

# (C) Count overlaps per A-peak:
counted = intersect_peaks(peaks_a_df, peaks_b_df, mode="count")
# counted has: chrom, start, end, ..., summit, overlap_count

# (D) Get overall overlap fraction:
stats = count_overlapping_peaks(query_bed, subject_bed)
# stats → {"n_query": ..., "n_overlapping": ..., "fraction": ...}
```

WRONG — cartesian product, will be rejected:
```python
merged = rest_peaks.merge(aff1_peaks, on='name')  # BROKEN: 'name' is '.' everywhere
```

**RULE 6: `read_narrowpeak()` columns — no `'name'`, no `'peak_id'`**

The bioio helper `read_narrowpeak(path)` returns these columns:
`chrom, start, end, score, strand, signal_value, p_value, q_value, peak, peak_offset, summit`

The `name` column (BED col 4) is **dropped** because it is `'.'` for all ENCODE peaks.
There is also no `peak_id` column. If you need an ID, create one:

```python
df = read_narrowpeak(path)
df['_peak_id'] = [f"peak_{i:05d}" for i in range(len(df))]

# Or a genomic coordinate ID:
df['_coord_id'] = df['chrom'] + ':' + df['start'].astype(str) + '-' + df['end'].astype(str)
```

**RULE 7: BEDPE files in this project have a `#`-prefixed header line**

The HiC BEDPE files (e.g. `ENCFF134HIZ.bedpe`) start with:
```
#chr1  x1  x2  chr2  y1  y2  name  score  ...
```

Reading without `comment='#'` treats the header as a data row, causing `ValueError` when
pandas tries to parse `'x1'` as an integer coordinate.

WRONG (includes header as data row — TypeError on coordinates):
```python
loops = pd.read_csv(loops_bedpe, sep='\\t')
```

CORRECT — always use `comment='#'` for BEDPE files:
```python
# hic_loops: 24 columns (from manifest hic_loops_columns)
loops = pd.read_csv(
    loops_bedpe, sep='\\t', comment='#', header=None,
    names=['chr1','x1','x2','chr2','y1','y2','name','score',
           'strand1','strand2','color','observed',
           'expectedBL','expectedDonut','expectedH','expectedV',
           'fdrBL','fdrDonut','fdrH','fdrV',
           'numCollapsed','centroid1','centroid2','radius'],
)
# hic_strips: 11 columns
strips = pd.read_csv(
    strips_bedpe, sep='\\t', comment='#', header=None,
    names=['chr1','x1','x2','chr2','y1','y2','name1','score',
           'strand1','strand2','color'],
)
```
"""

# =============================================================================
# pyBigWig signal extraction
# =============================================================================
PYBIGWIG_QUIRKS = """
=== PYBIGWIG SIGNAL EXTRACTION RULES ===

**USE THESE HELPERS — do not write raw pyBigWig loops**

Two helpers in `src.utils.bioio` handle all common signal extraction patterns.
They clamp coordinates automatically and handle None returns:

```python
from src.utils.bioio import extract_bigwig_signals, extract_summit_signals

# Signal over arbitrary intervals (chrom/start/end columns):
signals = extract_bigwig_signals(peaks_df, bw_path)  # returns pd.Series

# Signal centred on narrowPeak summits (start + peak offset) ± window bp:
signals = extract_summit_signals(peaks_df, bw_path, window=250)  # returns pd.Series
```

Use `extract_summit_signals()` whenever you need signal around peak summits.
Use `extract_bigwig_signals()` for any other interval set.
NEVER call `bw.stats()` directly in generated code.

**RULE 1: Use bw.stats() per interval, NEVER bw.values() per base**

`bw.values(chrom, start, end)` returns one float per base — for a 500bp peak that's
500 floats. For 10,000 peaks that's 5 million floats. This is extremely slow and
wastes memory.

WRONG (slow — per-base extraction):
```python
signals = []
for _, row in peaks.iterrows():
    vals = bw.values(row['chrom'], row['start'], row['end'])
    signals.append(np.nanmean(vals))
```

RIGHT (fast — one mean per interval):
```python
import pyBigWig
bw = pyBigWig.open(bigwig_path)
signals = []
for _, row in peaks.iterrows():
    val = bw.stats(row['chrom'], int(row['start']), int(row['end']), type='mean')
    signals.append(val[0] if val and val[0] is not None else 0.0)
bw.close()
```

**RULE 2: Handle None returns from bw.stats()**

`bw.stats()` returns `[None]` when no data covers the interval (sparse signal).
This is normal — treat as 0.0 or NaN, NOT as a contig mismatch error.

```python
val = bw.stats(chrom, start, end, type='mean')[0]
signal = val if val is not None else 0.0  # or np.nan
```

**RULE 3: Verify contig names match before bulk extraction**

```python
bw = pyBigWig.open(bigwig_path)
bw_chroms = set(bw.chroms().keys())
bed_chroms = set(peaks['chrom'].unique())
overlap = bw_chroms & bed_chroms
if not overlap:
    raise RuntimeError(f"No contig overlap! BED: {list(bed_chroms)[:3]}, bigWig: {list(bw_chroms)[:3]}")
```
"""

# =============================================================================
# Data joining / gene ID matching
# =============================================================================
DATA_JOINING_QUIRKS = """
=== DATA JOINING RULES — READ BEFORE MERGING ANY TWO FILES ===

**RULE 1: NEVER assume gene ID formats match between files**

Gene IDs come in many formats and they almost never match across different files:
- GENCODE GTF: `ENSG00000223972.5` (Ensembl ID with version)
- Some expression files: `ENSG00000223972` (Ensembl ID without version)
- Some expression files: `10904` (numeric internal IDs)
- Some files: `DDX11L1` (gene symbols)

Before writing ANY join/merge between two files, you MUST:
1. Print the first 5 rows of BOTH files
2. Print the column you plan to join on from BOTH sides
3. Check if formats match — if not, find a mapping strategy

```python
# ALWAYS do this before joining:
print("=== File A gene IDs (first 5) ===")
print(df_a['gene_id'].head())
print(f"Example: {df_a['gene_id'].iloc[0]}")

print("=== File B gene IDs (first 5) ===")
print(df_b['gene_id'].head())
print(f"Example: {df_b['gene_id'].iloc[0]}")

# Check overlap BEFORE joining
common = set(df_a['gene_id']) & set(df_b['gene_id'])
print(f"Common IDs: {len(common)} / {len(df_a)}")
if len(common) == 0:
    print("WARNING: Zero overlap! ID formats likely differ.")
```

**RULE 2: Strip Ensembl version suffixes when joining**

If one file has `ENSG00000223972.5` and another has `ENSG00000223972`:
```python
df['gene_id_stripped'] = df['gene_id'].str.split('.').str[0]
```

**RULE 3: When using GENCODE GTF, extract gene_name for symbol-based joins**

```python
# Parse gene_name from GTF attribute column
import re
df['gene_name'] = df['attributes'].apply(
    lambda x: re.search(r'gene_name "([^"]+)"', x).group(1)
    if re.search(r'gene_name "([^"]+)"', x) else None
)
```

**RULE 4: Joining RNA-seq to GENCODE — prefer the manifest hint**

ALWAYS check the manifest for `gene_quantification_id_type` and `gene_quantification_join_hint`
FIRST — they describe the actual ID format and the recommended join strategy for this dataset.
Follow the hint exactly.

Typical case (K562.tsv and similar ENCODE RNA-seq files):
- The RNA-seq `gene_id` column contains ENSG IDs with version suffixes (e.g. `ENSG00000000003.14`).
- GENCODE GTF `gene_id` also has version suffixes.
- Strip versions on both sides and merge on the stripped ENSG ID.

```python
import re, pandas as pd

# 1. Parse GENCODE GTF
gtf = pd.read_csv(gtf_path, sep='\t', comment='#', header=None,
                   names=['chrom','source','feature','start','end','score','strand','frame','attributes'])
genes = gtf[gtf['feature'] == 'gene'].copy()
genes['gene_id'] = genes['attributes'].str.extract(r'gene_id "([^"]+)"')
genes['gene_id_stripped'] = genes['gene_id'].str.split('.').str[0]
genes['gene_name'] = genes['attributes'].str.extract(r'gene_name "([^"]+)"')

# 2. Prepare RNA-seq
from src.utils.bioio import load_rnaseq_with_gene_id
rnaseq, gene_id_col = load_rnaseq_with_gene_id(rnaseq_path)
rnaseq['gene_id_stripped'] = rnaseq[gene_id_col].astype(str).str.split('.').str[0]

# 3. Merge on stripped ENSG ID
merged = peaks_with_nearest_gene.merge(
    rnaseq[['gene_id_stripped', 'TPM']], on='gene_id_stripped', how='left')

# 4. Validate overlap — warn if too low, do NOT crash
matched = merged['TPM'].notna().sum()
print(f"Matched {matched}/{len(merged)} peaks to TPM ({matched/len(merged)*100:.1f}%)")
if matched < 0.1 * len(merged):
    print("WARNING: <10% overlap — gene ID formats may not match. Check manifest hint.")
```

GENCODE GTF does NOT contain Entrez IDs in `db_xref`. Do NOT try to regex-extract Entrez
from the GTF. If the manifest says the RNA-seq uses Entrez IDs, check whether the file
actually contains ENSG IDs (many ENCODE files do despite metadata labels). When in doubt,
print the first 5 `gene_id` values and decide based on the actual data.

**RULE 5: Sort BED files before bedtools closest**

`bedtools closest` requires sorted input. Always sort first, and do not use shell
redirection for sorting. In Python:
```python
with open(sorted_bed, "w") as out:
    subprocess.run(["bedtools", "sort", "-i", input_bed], stdout=out, text=True, check=True)
subprocess.run(["bedtools", "closest", "-a", sorted_a_bed, "-b", sorted_b_bed, "-d"], check=True)
```

After running `bedtools closest -d`, prefer parsing the output with:
```python
from src.utils.bioio import parse_bedtools_closest
closest_df = parse_bedtools_closest(path_or_df, a_col_count=4, b_col_count=4)
```
"""

# =============================================================================
# STRING protein-protein interaction database
# =============================================================================
STRING_QUIRKS = """
=== STRING DATABASE QUIRKS ===

**RULE 1: Read from LOCAL files — do NOT call the STRING API**

STRING data is stored as local files. The manifest provides:
- `string.links_file`: path to `9606.protein.links.v12.0.txt.gz`
- `string.aliases_file`: path to `9606.protein.aliases.v12.0.txt.gz`
- `string.min_score`: minimum combined score threshold (700 = high confidence, 0–1000 scale)

Do NOT use `requests`, do NOT call `string-db.org`. Read the local files directly.

File formats (verified):
- **links**: space-separated, header `protein1 protein2 combined_score`
  IDs are `9606.ENSP...` format. Score is 0–1000 (higher = more confident).
- **aliases**: tab-separated, header `#string_protein_id alias source`
  Use source `Ensembl_HGNC_symbol` to map gene symbols → ENSP IDs.

```python
import pandas as pd

MIN_SCORE = data_manifest["string"]["min_score"]  # 700

# --- Load aliases: build bidirectional symbol <-> ENSP mapping ---
aliases = pd.read_csv(
    data_manifest["string"]["aliases_file"],
    sep="\\t", comment="#",
    names=["string_id", "alias", "source"],
    compression="infer",
)
# Ensembl_HGNC_symbol is the canonical gene symbol source
sym_aliases = aliases[aliases["source"] == "Ensembl_HGNC_symbol"]
symbol_to_ensp = sym_aliases.set_index("alias")["string_id"].to_dict()
ensp_to_symbol = sym_aliases.set_index("string_id")["alias"].to_dict()

# --- Load links: filter to high-confidence interactions ---
links = pd.read_csv(
    data_manifest["string"]["links_file"],
    sep=" ",
    compression="infer",
)
links = links[links["combined_score"] >= MIN_SCORE].copy()


def get_direct_score(gene_a: str, gene_b: str) -> int | None:
    \"\"\"Return combined_score for a direct interaction, or None if absent.\"\"\"
    ensp_a = symbol_to_ensp.get(gene_a)
    ensp_b = symbol_to_ensp.get(gene_b)
    if ensp_a is None or ensp_b is None:
        return None
    row = links[
        ((links["protein1"] == ensp_a) & (links["protein2"] == ensp_b)) |
        ((links["protein1"] == ensp_b) & (links["protein2"] == ensp_a))
    ]
    return int(row.iloc[0]["combined_score"]) if not row.empty else None


def get_interaction_partners(gene: str, limit: int = 50) -> pd.DataFrame:
    \"\"\"Return top-N interaction partners as a DataFrame with columns:
    partner_symbol, combined_score (sorted descending by score).\"\"\"
    ensp = symbol_to_ensp.get(gene)
    if ensp is None:
        return pd.DataFrame(columns=["partner_symbol", "combined_score"])
    mask = (links["protein1"] == ensp) | (links["protein2"] == ensp)
    partners = links[mask].copy()
    partners["partner_ensp"] = partners.apply(
        lambda r: r["protein2"] if r["protein1"] == ensp else r["protein1"], axis=1
    )
    partners["partner_symbol"] = partners["partner_ensp"].map(ensp_to_symbol)
    return (partners[["partner_symbol", "combined_score"]]
            .dropna(subset=["partner_symbol"])
            .sort_values("combined_score", ascending=False)
            .head(limit)
            .reset_index(drop=True))


def find_common_interactors(gene_a: str, gene_b: str, limit: int = 50) -> set[str]:
    \"\"\"Return gene symbols that interact with both gene_a and gene_b.\"\"\"
    partners_a = set(get_interaction_partners(gene_a, limit)["partner_symbol"])
    partners_b = set(get_interaction_partners(gene_b, limit)["partner_symbol"])
    return partners_a & partners_b
```

**RULE 2: Score scale**

- `combined_score` in the links file is on a 0–1000 scale (700 = high confidence).
- Score 999 = very high (e.g. well-known complex partners like REST/RCOR1).
- An absent link (None from `get_direct_score`) means score < MIN_SCORE, not necessarily zero —
  always report this explicitly: "No high-confidence direct interaction found (score < 700)."

**RULE 3: Use gene symbols, not ENSG IDs**

STRING aliases map gene symbols (e.g. "REST", "E2F5") to ENSP IDs via `Ensembl_HGNC_symbol`.
Always pass gene symbols to the helper functions above, not Ensembl IDs.

**RULE 4: Interpret results carefully**

- A direct score >= 700 supports tethering/PPI (mechanism category 1).
- Shared interactors suggest cofactor-mediated cooperation (category 4).
- An absent direct interaction does NOT disprove a mechanism — STRING is incomplete
  and biased toward well-studied proteins. Report the score and note the limitation.
- Always print: the direct score (or "absent"), and the common interactors with names and scores.
"""

# =============================================================================
# Matched control construction
# =============================================================================
MATCHED_CONTROL_QUIRKS = """
=== MATCHED CONTROL CONSTRUCTION RULES ===

**USE THIS HELPER — do not call pd.qcut() directly on multiple groups**

```python
from src.utils.bioio import matched_bin

# Bin FG and BG using FG-derived quantile edges (handles duplicate edges automatically):
fg_binned, bg_binned = matched_bin(fg_df, bg_df, signal_col="dnase_signal", n_bins=4)
```

NEVER call `pd.qcut()` independently on foreground and background — it creates different
bin edges for each group, making the matching invalid. `matched_bin()` derives edges from
the foreground only, then applies them to background via `pd.cut()`.

Matched controls are critical for valid enrichment tests. These rules prevent common bugs
that silently invalidate your statistical comparisons.

**RULE 1: Bin edges must come from FOREGROUND only**

When matching controls on quantile bins (DNase quartiles, GC% bins, etc.):
- Compute bin edges from foreground data ONCE
- Apply those SAME edges to background using pd.cut()
- NEVER use qcut() independently on both sets — this creates different bin edges!

WRONG (creates different bin edges for FG and BG):
```python
fg['dnase_bin'] = pd.qcut(fg['dnase'], 4, labels=False)  # edges: [0, 25, 50, 75, 100]
bg['dnase_bin'] = pd.qcut(bg['dnase'], 4, labels=False)  # edges: [5, 30, 55, 80, 105] - DIFFERENT!
```

CORRECT (same edges for both):
```python
_, fg_edges = pd.qcut(fg['dnase'], 4, labels=False, retbins=True, duplicates='drop')
fg['dnase_bin'] = pd.cut(fg['dnase'], bins=fg_edges, labels=False, include_lowest=True)
bg['dnase_bin'] = pd.cut(bg['dnase'], bins=fg_edges, labels=False, include_lowest=True)
```

**RULE 2: Verify matching actually worked**

After matching, ALWAYS print diagnostics:
```python
print(f"FG samples per bin: {fg.groupby('dnase_bin').size().to_dict()}")
print(f"BG samples per bin: {bg.groupby('dnase_bin').size().to_dict()}")
retained = bg['dnase_bin'].notna().mean()
print(f"BG retention rate: {retained:.1%}")
if retained < 0.8:
    print("WARNING: <80% BG retained after matching — consider coarser bins or report limitation")
```

If retention is very low (<50%), your matched comparison is unreliable. Options:
- Use coarser bins (tertiles instead of quartiles)
- Use continuous matching (propensity score, nearest-neighbor)
- Report the limitation and interpret results cautiously

**RULE 3: Controls must be MEASURED, not copied**

When creating shifted/shuffled control regions, you MUST measure annotations (DNase, ChromHMM,
GC%, etc.) at the CONTROL location. Do NOT copy values from the original peak.

WRONG (copies FG annotations to controls — matching is fake):
```python
ctrl = fg.copy()
ctrl['start'] = fg['start'] + 10000  # Shift position
# ctrl['dnase'] still has FG values — WRONG!
```

CORRECT (measure at control location):
```python
ctrl_bed = fg[['chrom', 'start', 'end']].copy()
ctrl_bed['start'] = ctrl_bed['start'] + 10000
ctrl_bed['end'] = ctrl_bed['end'] + 10000
# Write to file, run bigWigAverageOverBed, read back
ctrl_signals = measure_bigwig_signal(ctrl_bed, dnase_bw)
ctrl['dnase'] = ctrl_signals
```

**RULE 4: Exclude controls that overlap foreground**

Shifted/shuffled controls can accidentally land on real peaks, inflating the null:
```python
# After generating control BED, remove overlaps with foreground
subprocess.run(['bedtools', 'intersect', '-a', ctrl_bed, '-b', fg_bed, '-v'],
               stdout=open(ctrl_nonoverlap_bed, 'w'), check=True)
```

**RULE 5: For paired tests, maintain 1:1 correspondence**

If the hypothesis requires a paired test (Wilcoxon signed-rank), each FG sample must have
exactly ONE matched control. Do NOT pool all FG and all BG into unpaired groups.

WRONG (loses pairing):
```python
fg_signals = fg['signal'].values
bg_signals = bg['signal'].values  # Different length, no correspondence
stat, p = mannwhitneyu(fg_signals, bg_signals)  # Unpaired test
```

CORRECT (maintains pairing):
```python
# Ensure 1:1 matching
matched = fg.merge(bg, on='match_key', suffixes=('_fg', '_bg'))
stat, p = wilcoxon(matched['signal_fg'], matched['signal_bg'])  # Paired test
```
"""

# =============================================================================
# Python / pandas / scipy gotchas
# =============================================================================
PANDAS_QUIRKS = """
=== PYTHON / PANDAS / SCIPY GOTCHAS ===

**RULE 1: Always call `.reset_index(drop=True)` after `pd.concat()`**

`pd.concat()` preserves original row indices. If both DataFrames have overlapping
indices (e.g. both start at 0), the result has duplicate index labels.
Any subsequent `.loc[]`, `.merge()`, or `.reindex()` will raise
`ValueError: cannot reindex on an axis with duplicate labels`.

WRONG (duplicate indices silently created):
```python
combined = pd.concat([df_a, df_b])
combined.loc[0]  # Returns 2 rows, not 1!
```

CORRECT — reset immediately after every concat:
```python
combined = pd.concat([df_a, df_b]).reset_index(drop=True)
combined.loc[0]  # Returns 1 row as expected
```

**RULE 2: Cast coordinates to `int` only AFTER handling NaN**

Genomic coordinate columns from `pd.read_csv()` are float when any value is NaN.
Calling `.astype(int)` on a column with NaN raises `IntCastingNaNError`.

WRONG (crashes if any coordinate is NaN):
```python
df['start'] = df['start'].astype(int)
```

CORRECT — drop NaN rows first, or use pd.to_numeric with coerce:
```python
# Option A: drop rows with missing coordinates
df = df.dropna(subset=['start', 'end']).copy()
df['start'] = df['start'].astype(int)
df['end'] = df['end'].astype(int)

# Option B: fill NaN with 0 and convert
df['start'] = pd.to_numeric(df['start'], errors='coerce').fillna(0).astype(int)
```

**RULE 3: Use `scipy.stats.binomtest`, NOT `scipy.stats.binom_test`**

`scipy.stats.binom_test` was removed in scipy >= 1.12. Use `binomtest` instead.
`binomtest` returns a result object; call `.pvalue` to get the p-value.

WRONG (AttributeError in scipy >= 1.12):
```python
p = scipy.stats.binom_test(k, n, p=0.5)
```

CORRECT:
```python
result = scipy.stats.binomtest(k, n, p=0.5)
p = result.pvalue
```
"""

# =============================================================================
# Combined quirks for common workflows
# =============================================================================
def get_quirks_for_tools(tools: list[str]) -> str:
    """Get combined quirks section for specified tools.

    Args:
        tools: List of tool names (e.g., ['fimo', 'bedtools'])

    Returns:
        Combined quirks text for all specified tools
    """
    quirks_map = {
        'fimo': FIMO_QUIRKS,
        'bedtools': BEDTOOLS_QUIRKS,
        'meme': MEME_SUITE_QUIRKS,
        'meme-suite': MEME_SUITE_QUIRKS,
        'narrowpeak': NARROWPEAK_QUIRKS,
        'pybigwig': PYBIGWIG_QUIRKS,
        'bigwig': PYBIGWIG_QUIRKS,
        'data-joining': DATA_JOINING_QUIRKS,
        'string': STRING_QUIRKS,
        'string-db': STRING_QUIRKS,
        'ppi': STRING_QUIRKS,
        'matched-controls': MATCHED_CONTROL_QUIRKS,
        'matching': MATCHED_CONTROL_QUIRKS,
        'controls': MATCHED_CONTROL_QUIRKS,
        'pandas': PANDAS_QUIRKS,
        'scipy': PANDAS_QUIRKS,
        'python': PANDAS_QUIRKS,
    }

    sections = []
    seen = set()

    for tool in tools:
        tool_lower = tool.lower()
        if tool_lower in quirks_map and tool_lower not in seen:
            sections.append(quirks_map[tool_lower])
            seen.add(tool_lower)

    return "\n".join(sections)


# Default quirks to always include (most common issues)
DEFAULT_QUIRKS = (
    FIMO_QUIRKS + "\n" +
    BEDTOOLS_QUIRKS + "\n" +
    NARROWPEAK_QUIRKS + "\n" +
    DATA_JOINING_QUIRKS + "\n" +
    MATCHED_CONTROL_QUIRKS + "\n" +
    PANDAS_QUIRKS
)
