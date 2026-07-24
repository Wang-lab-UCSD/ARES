"""Tier-2 mechanism classifier (MiniMax): given the Tier-1 dichotomy GROUP, pick a
leaf sub-mechanism from ONLY that group's children in schema_3_group.yaml.

ONE class-specific call (two-step pipeline overall: DeepSeek dichotomy -> this).
Submechanisms (e.g. COMPETITIVE_EXCLUSION under DIRECT_SEQUENCE_RECOGNITION,
INDIRECT_COOPERATIVE_BINDING under COOPERATIVE_COBINDING) are shown indented in the
same menu and are directly selectable — no separate call.

Reuses the OpenAI client/retry from common (_llm_d).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from common import _llm_d

SCHEMA = yaml.safe_load(open(Path(__file__).parent / "schema_3_group.yaml"))
HIER = SCHEMA["mechanism_hierarchy"]
GROUPS = set(HIER)  # SEQUENCE, PROTEIN, CONTEXT, ARTIFACT


def _leaves(group: str) -> dict[str, dict]:
    """Flat name->node map of all SELECTABLE leaves in a group, incl. nested submechanisms
    (both parent and sub are valid picks)."""
    out: dict[str, dict] = {}
    for k, v in HIER[group]["children"].items():
        out[k] = v
        for sk, sv in (v.get("submechanisms") or {}).items():
            out[sk] = sv
    return out


def _menu(group: str) -> str:
    lines = []
    for k, v in HIER[group]["children"].items():
        lines.append(f"  - {k}: {v.get('definition','')}")
        for sk, sv in (v.get("submechanisms") or {}).items():
            lines.append(f"      - {sk} (sub-type of {k}): {sv.get('definition','')}")
    return "\n".join(lines)


def _prompt_mechanism(report_md: str, det: dict[str, Any], group: str, protein_subtype: str = "") -> str:
    t = det["target_tf"]; p = det["partner_tf"]
    gdef = HIER[group].get("definition", "")
    menu = _menu(group)
    sub_hint = ""
    if group == "PROTEIN" and protein_subtype:
        sub_hint = f"\n- protein_subtype (already decided): {protein_subtype} — prefer a leaf consistent with it.\n"
    protein_guide = ""
    if group == "PROTEIN":
        protein_guide = (
            "\n# DECISION GUIDE for COOPERATIVE_COBINDING vs TETHERING vs CO_OCCUPANCY — use the report's FIRST /\n"
            "# ARTIFACT hypothesis, which tests whether the TARGET's OWN motif is enriched in the TARGET's peaks:\n"
            "#  1. Target has NO motif of its own (artifact check UNTESTABLE / 'no known motif' — i.e. a cofactor):\n"
            "#     it cannot bind DNA -> NEVER COOPERATIVE_COBINDING. Choose TETHERING (a motif-reader recruits it) or CO_OCCUPANCY.\n"
            "#  2. Target's own motif IS enriched in its peaks (a DNA-binding TF):\n"
            "#     - COOPERATIVE_COBINDING ONLY IF the report shows the target's OWN motif is present AT the partner-motif (co-bound) sites.\n"
            "#     - If the report shows the co-bound sites LACK the target's motif (motif-less / tethered) -> TETHERING.\n"
            "#     - If the report does NOT resolve whether the target's motif is at the co-bound sites -> CO_OCCUPANCY.\n"
            "#  Do NOT infer COOPERATIVE_COBINDING from 'cooperative / complex / scaffold / amplifies signal' wording alone —\n"
            "#  scaffold co-occupancy and tethering also raise the target's signal at co-bound sites.\n"
        )
    return f"""You are assigning the detailed MECHANISM of an ARES report. The dichotomy class is ALREADY
decided as **{group}**: {gdef}

Your ONLY job: pick the single best-fitting sub-mechanism from the {group} menu below, based ENTIRELY on
what THIS REPORT concludes. Do NOT second-guess the {group} call. Indented items are sub-types of the leaf
above them and are directly selectable — pick the sub-type when its specific condition applies, else the
parent. If the report supports {group} but is too vague to pick a leaf, or fits none, choose NEED_HUMAN_LABEL_{group}.
{protein_guide}

# Pair
- target_tf (predicted): {t}
- partner_tf (motif owner): {p}
- cell_line: {det['cell_line']}
- final_mechanism_freeform: {det.get('final_mechanism_freeform')!r}{sub_hint}

# {group} sub-mechanism menu — pick EXACTLY ONE name:
{menu}

# Output (single JSON object, no markdown, no commentary)
{{ "mechanism": "<one leaf NAME from the menu above>",
   "key_factor": "<the actor / cofactor / competitor / 3rd-TF the report names as the reader/mediator, or '' if none>",
   "evidence": "<dominant evidence type: ChIP-overlap | motif-scan | PPI-STRING | chromatin-mark | accessibility | methylation | perturbation | conservation | other>",
   "mechanism_freeform": "<=25 words: the report's actual supported mechanism, citing it>" }}

# Report
{report_md.strip()}
"""


def mechanism_classify(report_md: str, det: dict[str, Any], dichotomy_class: str,
                       protein_subtype: str = "", provider: str = "minimax") -> dict[str, Any]:
    if dichotomy_class not in GROUPS:
        return {"mechanism": "", "key_factor": "", "evidence": "", "mechanism_freeform": "",
                "note": f"no mechanism group for {dichotomy_class}"}
    parsed = _llm_d(_prompt_mechanism(report_md, det, dichotomy_class, protein_subtype), provider=provider)
    leaf = str(parsed.get("mechanism", ""))
    valid = set(_leaves(dichotomy_class))
    return {
        "mechanism": leaf if leaf in valid else f"INVALID:{leaf}",  # runner maps INVALID -> NEED_HUMAN_LABEL
        "key_factor": str(parsed.get("key_factor", "")),
        "evidence": str(parsed.get("evidence", "")),
        "mechanism_freeform": str(parsed.get("mechanism_freeform", "")),
    }
