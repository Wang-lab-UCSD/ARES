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

3. **Avoid loading entire genome FASTA** - use bedtools getfasta for specific regions

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

1. **Permutation/shuffle tests**: Use at most 200 replicates. Do NOT exceed this.
   - Use `bedtools shuffle` for randomization — it is fast and handles chromosome/size matching
   - NEVER write custom shuffle loops that call samtools/bedtools per-iteration per-peak
   - NEVER fetch sequences (samtools faidx, bedtools getfasta) inside a loop over replicates

2. **Prefer CLI tools over Python loops**: bedtools, FIMO, and samtools are optimized in C.
   Running `bedtools intersect` once is always faster than iterating in Python.

3. **One question per script**: Each script should answer ONE specific question with ONE
   statistical test. Do not run multiple redundant permutation tests in the same script.

   BAD (3 permutation tests answering the same question):
   - 200 shuffles to test overlap significance
   - 100 shuffles to test signal enrichment
   - 200 shuffles to test distance significance
   → 500 total shuffles, all testing "do X and Y co-occur?"

   GOOD (1 test, clear answer):
   - 200 shuffles to test overlap significance
   → Done. If you need signal or distance tests, do them in a follow-up iteration.

4. **Keep code focused**: If the verification plan has 5+ steps, pick the 2-3 most
   critical. Additional analyses can be done in follow-up iterations.

5. **No downloading external files**: Do not download blacklists, annotations, or other
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


def build_verification_code_prompt(
    hypothesis: dict[str, Any],
    data_manifest: dict[str, Any],
    previous_code: str | None = None,
    previous_error: str | None = None,
) -> str:
    """Build prompt for generating verification code.

    Args:
        hypothesis: The hypothesis to verify
        data_manifest: Available data paths and tools
        previous_code: Code from previous attempt (if retrying)
        previous_error: Error from previous attempt (if retrying)

    Returns:
        Formatted prompt string
    """
    data_section = _format_data_for_coding(data_manifest)

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

    prompt = f"""# Hypothesis to Verify

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}

**Verification Plan**:
{_format_verification_plan(hypothesis.get('verification_plan', []))}

# Available Data

{data_section}
{retry_section}

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
- Your code MUST finish within 10 minutes. Limit permutations/shuffles to 200 replicates max.
- Use `bedtools shuffle` for randomization, NOT custom Python loops with per-peak sequence fetching.
- Implement only the 3-4 most critical steps from the verification plan. Keep it simple.
- Do NOT download external files (blacklists, annotations, etc.) — use only the provided data.

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

Fix the error in the code above. Common issues to check:
- Incorrect file paths
- Missing imports (use `!pip install package` if needed)
- Wrong data types or column names
- API/library usage errors
- Check stdout above for clues about what went wrong

Respond with ONLY the corrected Python code, no explanations.
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
