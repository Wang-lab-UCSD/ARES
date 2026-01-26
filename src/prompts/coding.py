"""Prompt templates for code generation."""

from __future__ import annotations

from typing import Any

CODING_SYSTEM_PROMPT = """You are an expert bioinformatics programmer. Your role is to write Python code to verify scientific hypotheses using available data.

Guidelines:
- Write clean, well-commented Python code
- Use appropriate bioinformatics libraries (pandas, numpy, pybedtools, biopython, etc.)
- Handle errors gracefully with try/except blocks
- Print clear, interpretable results
- Generate visualizations when helpful (use matplotlib)
- For CLI tools, use subprocess.run() with proper error handling

Output format:
- Always print a clear summary of findings at the end
- Include statistical tests where appropriate
- Save plots to files when generated

Common patterns:
- Loading BED files: Use pandas or pybedtools
- Motif analysis: Use MEME Suite via subprocess or biopython
- Statistical tests: Use scipy.stats
- Visualizations: Use matplotlib or seaborn
"""


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

Write Python code to test this hypothesis. The code should:

1. Load the necessary data files
2. Perform the analysis described in the verification plan
3. Generate relevant statistics and visualizations
4. Print a clear summary of findings

Important:
- Use the exact file paths provided above
- Handle potential errors (file not found, parsing issues, etc.)
- Print intermediate results for debugging
- End with a clear CONCLUSION section summarizing what was found

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
