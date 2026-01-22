"""Prompt templates for result summarization."""

from __future__ import annotations

from typing import Any

SUMMARY_SYSTEM_PROMPT = """You are a scientific result interpreter specializing in bioinformatics. Your role is to:

1. Analyze experimental results and code outputs
2. Determine whether results support, refute, or are inconclusive for a hypothesis
3. Extract key findings and statistics
4. Identify any issues or anomalies in the results
5. Prepare concise summaries for the hypothesis refinement process

Be objective and precise in your interpretations. Clearly distinguish between:
- Strong evidence (statistically significant, large effect size)
- Weak evidence (trends that may not be significant)
- Inconclusive results (high variance, conflicting signals)
- Technical failures (errors, missing data)
"""


def build_result_summary_prompt(
    hypothesis: dict[str, Any],
    code: str,
    execution_result: str,
    outputs: list[dict[str, Any]] | None = None,
) -> str:
    """Build prompt for summarizing execution results.

    Args:
        hypothesis: The hypothesis being tested
        code: The code that was executed
        execution_result: Combined stdout/stderr from execution
        outputs: Additional outputs (data frames, etc.)

    Returns:
        Formatted prompt string
    """
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
{execution_result}
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
