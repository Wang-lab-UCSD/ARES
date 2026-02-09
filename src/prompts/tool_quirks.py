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

**RULE 4: NEVER call getfasta or FIMO inside a loop over replicates**

For permutation tests, do NOT extract sequences and scan motifs for each replicate.
Each `bedtools getfasta` call reads the entire genome (~3GB) — doing this 200 times
takes hours.

WRONG (hours):
```python
for i in range(200):
    subprocess.run(["bedtools", "shuffle", "-i", peaks, "-g", chrom_sizes], ...)
    subprocess.run(["bedtools", "getfasta", "-fi", genome, "-bed", shuffled, ...])  # 3GB read
    subprocess.run(["fimo", "--motif", motif_id, ..., shuffled_fa])  # slow per replicate
```

RIGHT — for overlap permutation tests, just shuffle and intersect (seconds per replicate):
```python
for i in range(200):
    subprocess.run(["bedtools", "shuffle", "-i", peaks, "-g", chrom_sizes], ...)
    subprocess.run(["bedtools", "intersect", "-a", shuffled, "-b", other_peaks, "-u"], ...)
    # count overlapping lines — no sequence extraction needed
```

RIGHT — for motif enrichment, extract sequences ONCE, run FIMO ONCE, then permute the labels:
```python
# Extract sequences once
subprocess.run(["bedtools", "getfasta", "-fi", genome, "-bed", peaks, "-fo", peaks_fa])
# Run FIMO once
subprocess.run(["fimo", "--motif", motif_id, ..., peaks_fa])
# Permute by shuffling peak labels or using Fisher's exact test
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
DEFAULT_QUIRKS = FIMO_QUIRKS + "\n" + BEDTOOLS_QUIRKS
