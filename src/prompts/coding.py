"""Prompt templates for code generation."""

from __future__ import annotations

from typing import Any

from src.prompts.tool_quirks import DEFAULT_QUIRKS, get_quirks_for_tools

# =============================================================================
# Main coding system prompt (tool-agnostic)
# =============================================================================
CODING_SYSTEM_PROMPT_BASE = """You are an expert bioinformatics programmer. Your role is to write Python code to verify scientific hypotheses using available data.

Guidelines:
- Write clean, well-commented Python code
- Use appropriate bioinformatics libraries (pandas, numpy, pybedtools, biopython, etc.)
- Handle errors gracefully with try/except blocks
- Print clear, interpretable results
- For CLI tools, use subprocess.run() with proper error handling

=== CRITICAL: LARGE FILE HANDLING ===

Biological data files can be VERY large (100MB-10GB). You MUST handle them carefully:

**STEP 0 - ALWAYS START BY LISTING FILES AND CHECKING SIZES**:
```python
import os
print("=== FILE SIZE CHECK ===")
data_files = [
    # List ALL data files you will use
]
for f in data_files:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"{f}: {size_mb:.1f} MB")
    else:
        print(f"{f}: FILE NOT FOUND")
print("=" * 50)
```

**Large file strategies** (use when file > 50MB):
1. **For TSV/CSV**: Use `chunksize` parameter in pandas:
   ```python
   chunks = pd.read_csv(file, sep='\\t', chunksize=100000)
   results = []
   for chunk in chunks:
       # Process each chunk
       results.append(chunk_result)
   ```

2. **For BED files**: Use bedtools for operations instead of loading into memory:
   ```python
   # Don't do: pd.read_csv(huge.bed) then filter in Python
   # Do: Use bedtools intersect/filter first, then load result
   subprocess.run(['bedtools', 'intersect', '-a', peaks, '-b', regions, '-wa'], ...)
   ```

3. **NEVER scan the whole genome**: Do NOT run FIMO (or any motif scanner) on the
   full genome FASTA (~3GB, takes hours). Instead, extract only the regions you need
   with `bedtools getfasta`, then run the scanner on that small FASTA. Scanning a
   few thousand peak sequences takes seconds; scanning the whole genome takes hours.

=== CODE STRUCTURE ===

1. **STEP 0: List files and check sizes** (see above - MANDATORY)

2. **STEP 1: Load and inspect data**
   - Print shape, columns, sample rows
   - For large files, inspect first without loading fully: `head -n 5 file.tsv`

3. **STEP 2: Verify pre-conditions**
   - Check expected columns exist
   - Verify data types are correct
   - Count input items (e.g., "Starting with 7906 peaks")

4. **STEP 3: Process incrementally**
   - Print intermediate results
   - After each major step, print counts and verify they make sense

5. **STEP 4: Summarize findings**
   - Clear CONCLUSION section
   - Include key statistics

=== SANITY CHECKS (MANDATORY) ===

Before reporting ANY count or statistic:
- If counting unique items from N inputs, count should be reasonable (not 23 for 7906 inputs)
- If a count seems wrong, print WARNING and investigate:
  ```python
  if unique_count < input_count * 0.01:
      print(f"WARNING: Only {unique_count} unique values for {input_count} inputs!")
      print(f"Sample values: {df['column'].unique()[:10]}")
      print("This suggests a data parsing issue - investigating...")
  ```

=== COMPUTATIONAL COMPLEXITY LIMITS (code MUST finish within 10 minutes) ===

***These rules are mandatory. Ignoring them will cause the pipeline to hang for hours.***

1. **NEVER put expensive operations inside loops**. The following are BANNED inside
   any `for`/`while` loop:
   - `bedtools getfasta` (reads entire genome ~3GB per call)
   - `fimo` (scans all sequences per call)
   - `pyBigWig` / `bigWigAverageOverBed` signal extraction over full peak sets
   - Any operation that processes a large file (>50MB)

   These operations MUST be called ONCE, outside any loop. If you need to compare
   against a null distribution, use analytical tests (see rule 2) or permute labels
   in memory instead of reprocessing files.

2. **You MUST use analytical statistical tests, NOT permutations**, for these comparisons:
   - Signal comparison between two groups → Mann-Whitney U test (instant)
   - Overlap significance → Fisher's exact test on a 2×2 table (instant)
   - Motif enrichment → Fisher's exact test on presence/absence counts (instant)
   - Proportion comparison → Chi-square or Fisher's exact test (instant)

   Permutation tests (bedtools shuffle) are ONLY allowed when you need to control for
   genomic biases (GC content, chromosome distribution) that analytical tests cannot
   handle. If you use a permutation test, you may have AT MOST one per script, with
   at most 100 replicates, and it may ONLY involve `bedtools shuffle` + `bedtools intersect`
   (counting overlaps). NEVER extract signals or run FIMO inside a permutation loop.
   **Even if the hypothesis specifies more than 100 replicates, cap at 100.** This is a
   hard pipeline limit that overrides the hypothesis.

3. **Background sets must use `bedtools shuffle` — never bin the genome**.
   When you need a random background to compare against foreground peaks, use:
   ```bash
   bedtools shuffle -i peaks.bed -g chrom.sizes > background.bed
   ```
   This takes 1-2 seconds. NEVER construct a background by:
   - Binning the entire genome into windows (e.g., 50bp bins over hg38 = 60 million bins)
   - Running `bigWigAverageOverBed` over genome-wide intervals
   - Scanning all accessible regions genome-wide
   These approaches take hours and are never necessary for a simple enrichment test.

4. **One statistical test per script**. Your script should answer ONE specific question
   with ONE primary statistical test. Additional analyses belong in follow-up iterations.

   BAD (script does 5 things):
   - Permutation test for overlap
   - BigWig signal extraction + comparison
   - FIMO motif scan + enrichment
   - Nearest-gene expression analysis
   - Motif-stratified signal comparison
   → 650 lines, runs for 45 minutes

   GOOD (script does 1 thing):
   - Compute overlap between ATF6 and REST peaks
   - Fisher's exact test for significance
   - Print result
   → 80 lines, runs in 2 minutes. Other analyses go in the next iteration.

4. **No downloading external files**: Do not download blacklists, annotations, or other
   files from the internet. Use only the data files provided in the manifest.

=== OUTPUT FORMAT ===

- Print a clear summary of findings at the end
- Include statistical tests where appropriate
- NO matplotlib/seaborn visualizations (they often cause errors) - focus on statistics

Common patterns:
- Loading BED files: Use pandas or pybedtools
- Motif analysis: Use MEME Suite via subprocess or biopython
- Statistical tests: Use scipy.stats
"""


def build_coding_system_prompt(tools: list[str] | None = None) -> str:
    """Build the complete coding system prompt with relevant tool quirks.

    Args:
        tools: List of tools being used (e.g., ['fimo', 'bedtools']).
               If None, includes default quirks (FIMO + bedtools).

    Returns:
        Complete system prompt with tool-specific quirks appended.
    """
    if tools is None:
        # Include default quirks for common bioinformatics workflows
        quirks = DEFAULT_QUIRKS
    else:
        quirks = get_quirks_for_tools(tools)

    if quirks:
        return CODING_SYSTEM_PROMPT_BASE + "\n" + quirks
    return CODING_SYSTEM_PROMPT_BASE


# For backward compatibility - includes default quirks
CODING_SYSTEM_PROMPT = build_coding_system_prompt()


def _format_prior_evidence(prior_evidence: list[dict[str, Any]]) -> str:
    """Format previously supported hypotheses as confounders for the coding prompt."""
    if not prior_evidence:
        return ""

    supported = [e for e in prior_evidence if e.get("support_level") in ("SUPPORTS", "INCONCLUSIVE")]
    if not supported:
        return ""

    parts = [
        "# Previously Supported Findings (you MUST control for these)",
        "",
        "The following hypotheses were already supported in earlier iterations.",
        "Your test MUST show that the current hypothesis explains something these",
        "prior findings cannot. If the current effect disappears after controlling",
        "for a prior finding, that means the current hypothesis is NOT independently supported.",
        "",
    ]
    for e in supported:
        parts.append(f"**{e.get('hypothesis_name', 'N/A')}** ({e.get('support_level')}, confidence {e.get('confidence', '?')}):")
        parts.append(f"  {e.get('summary', 'No summary')[:300]}")
        parts.append("")

    return "\n".join(parts)


def build_verification_code_prompt(
    hypothesis: dict[str, Any],
    data_manifest: dict[str, Any],
    previous_code: str | None = None,
    previous_error: str | None = None,
    prior_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Build prompt for generating verification code.

    Args:
        hypothesis: The hypothesis to verify
        data_manifest: Available data paths and tools
        previous_code: Code from previous attempt (if retrying)
        previous_error: Error from previous attempt (if retrying)
        prior_evidence: Evidence from previously tested hypotheses

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_for_coding(data_manifest)
    prior_section = _format_prior_evidence(prior_evidence or [])

    retry_section = ""
    if previous_code and previous_error:
        retry_section = f"""
# Previous Attempt (Failed)

The previous code resulted in an error. Please fix it.

**Previous Code**:
```python
{previous_code}
```

**Error**:
```
{previous_error}
```

Please identify and fix the issue in the code.
"""

    technical_issues_section = ""
    technical_issues = hypothesis.get("_technical_issues")
    if technical_issues:
        issues_text = "\n".join(f"- {issue}" for issue in technical_issues)
        technical_issues_section = f"""
# Known Technical Issues to Fix

The previous attempt at this hypothesis failed due to these specific bugs. You MUST fix all of them:

{issues_text}

Write code that explicitly avoids these issues.
"""

    prompt = f"""# Hypothesis to Verify

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}

{prior_section}

**Verification Plan**:
{_format_verification_plan(hypothesis.get('verification_plan', []))}

# Available Data

{data_section}
{technical_issues_section}{retry_section}

# Task

Write Python code to test this hypothesis. The code MUST follow this structure:

**STEP 0 (MANDATORY): List all data files and check their sizes first**
- Print each file path and its size in MB
- If any file > 50MB, plan to handle it with chunked reading or filtering

**STEP 1: Load and inspect data**
- Print shape, columns, first few rows
- Count input items (e.g., "Starting with N peaks")

**STEP 2: Perform analysis**
- Follow the verification plan
- Print intermediate results after each major operation
- SANITY CHECK: Verify counts make sense (not 23 items when expecting 7906)

**STEP 3: Statistical tests**
- Use scipy.stats for significance testing
- Report p-values and effect sizes

**STEP 4: CONCLUSION section**
- Clear summary of findings
- Support level for hypothesis

***CRITICAL REMINDERS — read before writing code:***
- Use the exact file paths provided above
- NEVER trust FIMO's sequence_name column for counting peaks - use bedtools intersect instead
- If using FIMO output, filter by motif_id immediately to reduce memory usage
- NO matplotlib/seaborn visualizations - focus on statistics only
- Print intermediate results for debugging
- If any count seems wrong (e.g., 23 instead of thousands), STOP and investigate
- Your code MUST finish within 10 minutes. You MUST use analytical tests (Fisher's, Mann-Whitney) instead of permutations. Permutations are only allowed for overlap counting (shuffle + intersect), max 100 reps, max one per script.
- NEVER put bigWig extraction, getfasta, or FIMO inside a loop. Call them ONCE.
- Your script should have ONE statistical test answering ONE question. Keep it under 150 lines. Other analyses go in the next iteration.
- Do NOT download external files (blacklists, annotations, etc.) — use only the provided data.
- Only use CLI flags listed in the tool_quirks section above. Never invent or guess flags.
- When selecting columns from a data file, always use column names (e.g., `df['TPM']`), never positional indexing (e.g., `columns[0]` or `value_cols[0]`). Inspect column names first and pick the correct one.

Respond with ONLY the Python code, no explanations. The code should be ready to execute.
"""
    return prompt


def build_error_fix_prompt(
    original_code: str,
    error_message: str,
    error_traceback: str,
    data_manifest: dict[str, Any],
    stdout: str = "",
    stderr: str = "",
) -> str:
    """Build prompt for fixing code errors.

    Args:
        original_code: The code that failed
        error_message: The error message
        error_traceback: Full traceback
        data_manifest: Available data paths
        stdout: Standard output (may contain debug info)
        stderr: Standard error (may contain warnings)

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_for_coding(data_manifest)

    # Build execution output section
    output_section = ""
    if stdout or stderr:
        output_section = "\n# Execution Output (before error)\n\n"
        if stdout:
            # Truncate if too long
            stdout_truncated = stdout[-3000:] if len(stdout) > 3000 else stdout
            if len(stdout) > 3000:
                stdout_truncated = "...(truncated)...\n" + stdout_truncated
            output_section += f"**stdout**:\n```\n{stdout_truncated}\n```\n\n"
        if stderr:
            stderr_truncated = stderr[-1500:] if len(stderr) > 1500 else stderr
            if len(stderr) > 1500:
                stderr_truncated = "...(truncated)...\n" + stderr_truncated
            output_section += f"**stderr**:\n```\n{stderr_truncated}\n```\n\n"

    prompt = f"""# Code That Failed

```python
{original_code}
```

# Error

**Message**: {error_message}

**Traceback**:
```
{error_traceback}
```
{output_section}
# Available Data

{data_section}

# Task

First, diagnose the root cause of the error. Write a brief comment at the top of your
code (1-2 lines) explaining:
1. What specifically went wrong (not just "an error occurred")
2. Why the previous code caused this (the root cause, not the symptom)

Then write the corrected code. Your fix MUST address the root cause you identified.
If the same tool or command failed, you MUST use a different approach — do not retry
the same command with minor flag variations.

Common root causes to check:
- Tool CLI flags not supported by the installed version (check --help or use a different tool)
- Output format assumptions that don't match actual output (inspect output first)
- Wrong data types or column names (print and verify before using)
- File path or format issues

If the error is caused by a missing data file (FileNotFoundError, file does not exist)
or an unavailable CLI tool (command not found) and there is NO valid substitute in the
data manifest above, do NOT attempt a workaround. Instead respond with ONLY:

```python
# UNTESTABLE: <one-line explanation of what resource is missing>
```

Use this ONLY for genuinely missing files or tools — NOT for "I don't know how to write
this code." If the error is a bug in your code (wrong column, wrong flag, type error),
you MUST fix it.

Otherwise, respond with the corrected Python code (the diagnostic comment should be inside the code).
"""
    return prompt


def _format_data_for_coding(data_manifest: dict[str, Any]) -> str:
    """Format data manifest for coding prompts."""
    parts = []

    data = data_manifest.get("data", {})
    for category, items in data.items():
        parts.append(f"## {category}")
        if isinstance(items, dict):
            for name, path in items.items():
                parts.append(f"- `{name}`: `{path}`")
        else:
            parts.append(f"- `{items}`")
        parts.append("")

    tools = data_manifest.get("tools", [])
    if tools:
        parts.append("## Available CLI Tools (use via subprocess)")
        for tool in tools:
            parts.append(f"- `{tool}`")

    return "\n".join(parts)


def _format_verification_plan(plan: list[str]) -> str:
    """Format verification plan as numbered list."""
    if not plan:
        return "No specific plan provided."
    return "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])
