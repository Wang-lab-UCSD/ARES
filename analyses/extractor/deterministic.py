"""Pure-Python extraction of join-key + JSON-direct fields.

Only fields that are byte-exact or trivial counts. Everything subjective
(omics_used, mechanism enum, blocker, dichotomy, function, cofactors)
goes through the LLM classifiers.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

LEDGER = "/new-stg/home/hanbei/ARES/production/ledger.csv"


def load_state(report_dir: Path) -> dict[str, Any]:
    p = report_dir / "state_final.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


def load_report(report_dir: Path) -> str:
    """The *_report.md in the directory (there is exactly one)."""
    matches = list(report_dir.glob("*_report.md"))
    if not matches:
        raise FileNotFoundError(f"no *_report.md in {report_dir}")
    return matches[0].read_text()


def find_ledger_row(pair_key: str) -> dict[str, str] | None:
    """Most-recent ledger row for this pair."""
    latest = None
    with open(LEDGER) as f:
        for r in csv.DictReader(f):
            if r["pair_key"] == pair_key:
                latest = r
    return latest


def _derive_blocker(state: dict, converged: bool, n_supports: int,
                     n_supports_mech: int) -> str | None:
    """Deterministically pick convergence_blocker when state allows.

    Returns one of: NO_SUPPORTS, ONLY_CHARACTERIZATION_SUPPORT, HYPOTHESIS_EXHAUSTION,
    or None when the case is ambiguous (LLM must pick among the remaining 5:
    SUPPORTS_TOO_GENERIC, BUT_FOR_FAILED, CROSS_LAYER_INCONSISTENT,
    BIOLOGY_LAYER_MISSING, PPI_NOT_ATTEMPTED).
    """
    if converged:
        return None  # blocker only set when not converged
    if n_supports == 0:
        return "NO_SUPPORTS"
    if n_supports_mech == 0:
        return "ONLY_CHARACTERIZATION_SUPPORT"
    stop_reason = (state.get("stop_reason") or "").lower()
    if "max iter" in stop_reason or "max_iter" in stop_reason or "exhaust" in stop_reason:
        return "HYPOTHESIS_EXHAUSTION"
    return None  # ambiguous — let LLM pick


def extract_deterministic(pair_key: str, report_dir: Path) -> dict[str, Any]:
    state = load_state(report_dir)
    ledger = find_ledger_row(pair_key) or {}

    th = state.get("tested_hypotheses", [])
    rc = Counter(h.get("result", "?") for h in th)

    # Split SUPPORTS into "mechanism" vs "characterization" based on group name AND hypothesis name.
    # Characterization-only patterns (don't establish a mechanism):
    #   - group starts with 'functional-' (functional-characterization, functional-synthesis, functional-phylop)
    #   - group contains 'synthesis' (e.g. 'deadlock-breaking-synthesis' — pipeline's last-ditch fallback)
    #   - group contains 'characterization'
    #   - name contains 'functional characterization' / 'characterization of' (case-insensitive)
    def _is_characterization(h):
        g = (h.get("group") or "").lower()
        n = (h.get("name") or "").lower()
        if g.startswith("functional-"):     return True
        if "synthesis" in g:                return True
        if "characterization" in g:         return True
        if "functional characterization" in n: return True
        if "characterization of" in n:      return True
        return False
    n_supports_mech = sum(
        1 for h in th
        if h.get("result") == "SUPPORTS" and not _is_characterization(h)
    )
    n_supports_char = rc.get("SUPPORTS", 0) - n_supports_mech
    converged = bool(state.get("converged"))
    n_supports = rc.get("SUPPORTS", 0)

    return {
        # IDENTITY (join keys)
        "pair_id":    pair_key,
        "target_tf":  (ledger.get("tf_a") or "").strip(),
        "partner_tf": (ledger.get("tf_b") or "").strip(),
        "cell_line":  (ledger.get("cell_line") or "").strip(),
        "report_path": str(report_dir),

        # CONVERGENCE (direct JSON)
        "converged":      converged,
        "n_iterations":   state.get("current_iteration") or len(th),
        "n_supports":     n_supports,
        "n_supports_mechanism":     n_supports_mech,
        "n_supports_characterization": n_supports_char,
        "n_rejects":      rc.get("REJECTS", 0),
        "n_inconclusive": rc.get("INCONCLUSIVE", 0),
        "n_untestable":   rc.get("UNTESTABLE", 0),

        # FINAL MECHANISM (free-form, kept alongside the enum for human review)
        "final_mechanism_freeform": state.get("mechanism_category_name"),
        "confidence_level":         state.get("confidence_level"),

        # COST
        "total_cost_usd": ledger.get("total_cost_usd") or None,

        # DETERMINISTIC BLOCKER (None when ambiguous; LLM picks the remaining 5).
        # Underscored — dropped by extract._flatten() so it doesn't appear in TSV.
        "_deterministic_blocker": _derive_blocker(state, converged, n_supports, n_supports_mech),
    }
