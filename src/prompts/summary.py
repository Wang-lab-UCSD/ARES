"""Prompt templates for result summarization."""

from __future__ import annotations

from typing import Any

SUMMARY_SYSTEM_PROMPT = """You are a scientific result interpreter specializing in bioinformatics. Your role is to:

1. Analyze experimental results and code outputs
2. Determine whether results support, refute, or are inconclusive for a hypothesis
3. Extract key findings and statistics
4. Identify any issues or anomalies in the results
5. Prepare concise summaries for the hypothesis refinement process

Be objective and precise in your interpretations.

=== SUPPORT LEVEL CRITERIA (you MUST follow these rules) ===

Evaluate support_level by comparing the results against the hypothesis's **prediction**
field. The prediction states what should be observed if the hypothesis is true — your
job is to check whether that specific prediction was confirmed.

**SUPPORTS** — ALL of the following must be true:
  1. The predicted effect exists in the data
  2. The effect is statistically significant (p < 0.05)
  3. The effect size is meaningful: enrichment >= 1.5x over background, OR
     absolute difference >= 10 percentage points, OR Cohen's d >= 0.5
  If all three are met, set support_level = "SUPPORTS" and confidence >= 0.8.

**REFUTES** — ANY of the following:
  1. The predicted effect is absent or reversed (e.g., depletion instead of enrichment)
  2. The effect is not significant (p >= 0.05) with adequate sample size (N >= 30)
  3. The effect size is trivial: enrichment < 1.2x AND difference < 5 percentage points
  If clearly refuted, set support_level = "REFUTES" and confidence >= 0.7.

**INCONCLUSIVE** — The result falls between SUPPORTS and REFUTES:
  1. Significant but small effect (1.2x-1.5x enrichment, or 5-10 pp difference)
  2. Large effect but not significant (small sample size)
  3. Mixed signals (some metrics support, others refute)
  Set support_level = "INCONCLUSIVE" and confidence 0.4-0.7.

**ERROR** — Technical failure prevented analysis (code crashed, wrong file format, etc.)

IMPORTANT: Apply these criteria to the hypothesis's stated prediction, not to a
reframed version of the question. If the prediction says "X is enriched in group A
compared to group B", evaluate whether that specific enrichment exists — do not
reinterpret it as "X exists at all" or "X explains all of the association."

IMPORTANT: When support_level is SUPPORTS with confidence >= 0.8, do NOT include
phrases like "additional analyses are required", "more controls needed", "further
validation required", or "should be addressed in follow-up" in your summary. These
phrases cause the pipeline to keep iterating unnecessarily. If you want to note
limitations, state them as observations (e.g., "GC content was not controlled for")
rather than action items (e.g., "GC content controls should be added").
"""


def _truncate_output(text: str, max_chars: int = 50000) -> str:
    """Truncate large output while preserving useful information.

    Keeps the beginning (usually setup/imports) and end (usually results/conclusions).

    Args:
        text: The text to truncate
        max_chars: Maximum characters to keep

    Returns:
        Truncated text with indicator if truncation occurred
    """
    if len(text) <= max_chars:
        return text

    # Keep first 30% and last 70% (results are usually at the end)
    head_chars = int(max_chars * 0.3)
    tail_chars = int(max_chars * 0.7)

    head = text[:head_chars]
    tail = text[-tail_chars:]

    truncated_chars = len(text) - max_chars
    return (
        f"{head}\n\n"
        f"[... TRUNCATED {truncated_chars:,} characters ({truncated_chars // 4:,} tokens approx) ...]\n"
        f"[... Showing last {tail_chars:,} characters which typically contain results ...]\n\n"
        f"{tail}"
    )


def build_result_summary_prompt(
    hypothesis: dict[str, Any],
    code: str,
    execution_result: str,
    outputs: list[dict[str, Any]] | None = None,
    max_output_chars: int = 50000,
) -> str:
    """Build prompt for summarizing execution results.

    Args:
        hypothesis: The hypothesis being tested
        code: The code that was executed
        execution_result: Combined stdout/stderr from execution
        outputs: Additional outputs (data frames, etc.)
        max_output_chars: Maximum characters of execution output to include

    Returns:
        Formatted prompt string
    """
    # Truncate execution result if too large
    truncated_result = _truncate_output(execution_result, max_output_chars)

    outputs_section = ""
    if outputs:
        outputs_section = "\n## Additional Outputs\n"
        for i, out in enumerate(outputs):
            if "text/plain" in out:
                outputs_section += f"\n### Output {i+1}\n```\n{out['text/plain'][:2000]}\n```\n"

    prompt = f"""# Hypothesis Being Tested

**Name**: {hypothesis.get('name', 'N/A')}
**Prediction**: {hypothesis.get('prediction', 'N/A')}
**Rationale**: {hypothesis.get('rationale', 'N/A')}

# Code Executed

```python
{code}
```

# Execution Output

```
{truncated_result}
```
{outputs_section}

# Task

Analyze the execution results and provide a summary. Determine:

1. **Success**: Did the code execute successfully?
2. **Findings**: What were the key numerical results?
3. **Support Level**: Does this support, refute, or is inconclusive for the hypothesis?
4. **Confidence**: How confident are you in this interpretation?
5. **Issues**: Were there any data quality issues or anomalies?

Respond in JSON format:
{{
    "execution_success": true | false,
    "findings": [
        "Key finding 1",
        "Key finding 2",
        ...
    ],
    "statistics": {{
        "relevant_stat_name": value,
        ...
    }},
    "support_level": "SUPPORTS" | "REFUTES" | "INCONCLUSIVE" | "ERROR",
    "confidence": 0.0-1.0,
    "reasoning": "Explanation of the interpretation",
    "issues": ["Any issues or concerns"],
    "summary": "One paragraph summary suitable for the hypothesis model"
}}
"""
    return prompt


def build_final_report_prompt(
    finding: str,
    context: str,
    history_summary: str,
    conclusion: str,
    all_evidence: list[dict[str, Any]],
) -> str:
    """Build prompt for generating the final report.

    Args:
        finding: Original scientific finding
        context: Research context
        history_summary: Summary of all iterations
        conclusion: Final conclusion
        all_evidence: All accumulated evidence

    Returns:
        Formatted prompt string
    """
    evidence_text = ""
    for i, e in enumerate(all_evidence):
        evidence_text += f"""
### Iteration {e.get('iteration', i+1)}
- **Hypothesis**: {e.get('hypothesis', 'N/A')}
- **Findings**: {e.get('findings', 'N/A')}
- **Support Level**: {e.get('support_level', 'N/A')}
"""

    prompt = f"""# Final Report Generation

## Original Finding

{finding}

## Context

{context}

## Investigation Summary

{history_summary}

## Evidence Collected
{evidence_text}

## Conclusion

{conclusion}

# Task

Generate a comprehensive final report summarizing the investigation. The report should include:

1. **Executive Summary**: Brief overview of the finding and conclusion
2. **Methodology**: How the investigation was conducted
3. **Key Findings**: Most important discoveries
4. **Evidence Summary**: Supporting evidence for the conclusion
5. **Confidence Level**: How confident we are in the conclusion
6. **Limitations**: What we couldn't determine or potential issues
7. **Recommendations**: Suggested follow-up experiments if any

Format the report in Markdown.
"""
    return prompt
