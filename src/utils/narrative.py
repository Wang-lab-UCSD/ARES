"""Progressive markdown narrative log for a pipeline run.

Writes a human-readable `narrative.md` to the run output directory.
Appended incrementally so the file is readable while the run is still going.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class NarrativeLog:
    """Appends structured natural-language sections to narrative.md as the run proceeds."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "narrative.md"

    # ------------------------------------------------------------------
    # Public write methods — called in order by the orchestrator
    # ------------------------------------------------------------------

    def write_header(self, finding: str, run_id: str, started_at: str) -> None:
        lines = [
            "# ARES Investigation Narrative",
            "",
            f"**Run ID**: {run_id}  ",
            f"**Started**: {started_at}  ",
            "",
            "## Finding Under Investigation",
            "",
            finding.strip(),
            "",
            "---",
            "",
        ]
        self.path.write_text("\n".join(lines), encoding="utf-8")

    def write_hypothesis(self, iteration: int, hypothesis: dict[str, Any]) -> None:
        sc = hypothesis.get("scratchpad") or {}
        vplan = hypothesis.get("verification_plan") or []

        lines = [
            f"## Iteration {iteration}: {hypothesis.get('name', 'Unnamed')}",
            "",
            "### Hypothesis",
            "",
        ]
        if hypothesis.get("group"):
            lines += [f"**Group**: `{hypothesis['group']}`", ""]
        if sc.get("mechanism"):
            lines += [f"**Mechanism**: {sc['mechanism']}", ""]
        if sc.get("causal_chain"):
            lines += [f"**Causal chain**: {sc['causal_chain']}", ""]
        if hypothesis.get("rationale"):
            lines += [f"**Rationale**: {hypothesis['rationale']}", ""]
        if hypothesis.get("prediction"):
            lines += [f"**Prediction**: {hypothesis['prediction']}", ""]
        if sc.get("distinguishable"):
            lines += [f"**Distinguishable from null**: {sc['distinguishable']}", ""]

        lines += ["", "### Verification Plan", ""]
        import re
        for i, step in enumerate(vplan, 1):
            # Strip leading numbering the LLM may have included (e.g. "1. ", "Step 1: ")
            step = re.sub(r"^(\d+\.\s*|Step\s*\d+[:\s]*)", "", step)
            lines.append(f"{i}. {step}")
        lines.append("")

        self._append("\n".join(lines))

    def write_result(self, result: dict[str, Any]) -> None:
        support = result.get("support_level", "UNKNOWN")
        confidence = result.get("confidence") or 0.0
        summary = (result.get("summary") or result.get("evidence_summary") or "").strip()
        findings = result.get("findings") or []

        lines = ["### Results", ""]
        lines.append(f"**Support level**: {support}  ")
        lines.append(f"**Confidence**: {confidence:.2f}  ")
        lines.append("")
        if summary:
            lines += [summary, ""]
        if findings:
            lines += ["**Key findings**:", ""]
            for item in findings:
                text = item.get("description", str(item)) if isinstance(item, dict) else str(item)
                lines.append(f"- {text}")
            lines.append("")

        self._append("\n".join(lines))

    def write_decision(self, tested_hypothesis: dict[str, Any]) -> None:
        decision = (tested_hypothesis.get("decision") or "").strip()
        reasoning = (tested_hypothesis.get("refinement_reasoning") or "").strip()

        if not decision and not reasoning:
            return

        lines = ["### Next Step", ""]
        if decision:
            lines += [f"**Decision**: {decision}", ""]
        if reasoning:
            lines += [reasoning, ""]
        lines += ["---", ""]

        self._append("\n".join(lines))

    def finalize(
        self, conclusion: str, converged: bool, total_cost: float | None = None
    ) -> None:
        status = "CONVERGED" if converged else "NOT CONVERGED"
        lines = [
            "## Final Conclusion",
            "",
            f"**Status**: {status}",
            "",
            (conclusion or "Investigation incomplete.").strip(),
            "",
        ]
        if total_cost is not None:
            lines += [f"**Total cost**: ${total_cost:.2f}", ""]
        self._append("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)
