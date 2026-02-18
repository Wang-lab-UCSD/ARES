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
  fimo --no-pgc --oc out --thresh 1e-4 --motif MA0138.2 JASPAR_full.meme peaks.fa

You can run multiple targeted scans for different motifs:
  fimo --no-pgc --oc out_rest --thresh 1e-4 --motif MA0138.2 JASPAR_full.meme peaks.fa
  fimo --no-pgc --oc out_atf6 --thresh 1e-4 --motif MA1466.1 JASPAR_full.meme peaks.fa

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
fimo --no-pgc --oc fimo_out --thresh 1e-4 --motif MA0138.2 motifs.meme peaks.fa
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

**Memory Warning**: Filter large FIMO output by motif_id immediately:
```python
df = pd.read_csv(fimo_tsv, sep='\\t', comment='#')
df = df[df['motif_id'] == 'MA0138.2']  # Filter first!
```
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

When using subprocess in Python, shell redirection (`>`) requires `shell=True` or
writing output yourself:
```python
import subprocess
result = subprocess.run(
    ["bedtools", "intersect", "-a", file_a, "-b", file_b, "-u"],
    capture_output=True, text=True
)
with open(output_file, "w") as f:
    f.write(result.stdout)
```

**RULE 3: bedtools shuffle needs a genome file**

`bedtools shuffle` requires `-g genome.chrom.sizes`. Generate it from the FASTA index:
```bash
cut -f1,2 genome.fa.fai > chrom.sizes
bedtools shuffle -i peaks.bed -g chrom.sizes > shuffled.bed
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
"""

# =============================================================================
# MEME Suite general
# =============================================================================
MEME_SUITE_QUIRKS = """
=== MEME SUITE GENERAL QUIRKS ===

**MEME format files**:
- Comments start with `#`
- Use `comment='#'` when reading with pandas

**Motif IDs**:
- JASPAR format: `MA0138.2` (database_id.version)
- Some tools strip the version number

**Background models**:
- Default background may not match your sequences
- For accurate p-values, use `-bfile` with matched background
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

**RULE 4: Sort BED files before bedtools closest**

`bedtools closest` requires sorted input. Always sort first:
```bash
bedtools sort -i unsorted.bed > sorted.bed
bedtools closest -a sorted_a.bed -b sorted_b.bed -d
```
Or in Python:
```python
subprocess.run(["bedtools", "sort", "-i", input_bed], stdout=open(sorted_bed, 'w'))
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
        'data-joining': DATA_JOINING_QUIRKS,
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
DEFAULT_QUIRKS = FIMO_QUIRKS + "\n" + BEDTOOLS_QUIRKS + "\n" + DATA_JOINING_QUIRKS
