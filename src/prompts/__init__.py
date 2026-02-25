"""Prompt templates for LLM interactions."""

from src.prompts.hypothesis import (
    HYPOTHESIS_SYSTEM_PROMPT,
    build_refinement_prompt,
)
from src.prompts.coding import (
    CODING_SYSTEM_PROMPT,
    build_verification_code_prompt,
    build_error_fix_prompt,
)
from src.prompts.summary import (
    SUMMARY_SYSTEM_PROMPT,
    CONVERGENCE_CHECK_SYSTEM_PROMPT,
    build_result_summary_prompt,
    build_final_report_prompt,
    build_convergence_check_prompt,
)

__all__ = [
    "HYPOTHESIS_SYSTEM_PROMPT",
    "build_refinement_prompt",
    "CODING_SYSTEM_PROMPT",
    "build_verification_code_prompt",
    "build_error_fix_prompt",
    "SUMMARY_SYSTEM_PROMPT",
    "CONVERGENCE_CHECK_SYSTEM_PROMPT",
    "build_result_summary_prompt",
    "build_final_report_prompt",
    "build_convergence_check_prompt",
]
