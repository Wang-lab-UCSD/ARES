"""Tests for prompt construction."""

from src.prompts.summary import build_final_report_prompt


def test_final_report_prompt_includes_run_status_guard():
    """Non-converged run metadata should be explicit in the prompt."""
    prompt = build_final_report_prompt(
        finding="X predicts Y",
        context="Context",
        history_summary="History",
        conclusion="Best-supported interpretation so far",
        all_evidence=[{
            "iteration": 1,
            "hypothesis": "H1",
            "findings": ["A"],
            "support_level": "SUPPORTS",
        }],
        converged=False,
        run_status="stopped",
        synthesized=False,
    )

    assert "- converged: False" in prompt
    assert "- run_status: stopped" in prompt
    assert "If `converged` is false, do NOT present the outcome as settled" in prompt
