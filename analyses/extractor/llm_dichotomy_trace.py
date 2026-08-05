"""Dichotomy classifier v-TRACE — a NEW step-1 classifier (does NOT replace llm_dichotomy.py).

Same job (SEQUENCE / PROTEIN / CONTEXT from an ARES report) and same output schema, but the decision
is a strict 3-step CAUSAL TRACE from the partner motif, per the refined rule:

  1. READER   — who binds the {p} motif?  target itself → SEQUENCE; nobody (motif marks a state) →
                CONTEXT; another protein R → continue.
  2. ACTOR    — is there a protein A whose binding DIRECTLY moves {t}'s signal (recruit / tether /
                stabilize / co-bind / exclude)?  none (target only responds to the state) → CONTEXT;
                else continue.
  3. PROVENANCE of the actor — how did A reach these sites?
        • A is recruited by / linked to the reader R through a PROTEIN chain (R → … → A) → PROTEIN.
        • A is recruited by the CONTEXT (the chromatin state the motif merely marks) → CONTEXT,
          even though a protein acts on the target.

Decisive counterfactual: would A still act on {t} if the motif / reader were gone?  rides in on the
reader → PROTEIN; rides in on the chromatin state → CONTEXT.

Reuses the JSON parser and valid-label set from common.
Entry point: dichotomy_classify_trace(report_md, det, provider="deepseek-think").
"""
from __future__ import annotations
import os, time
from typing import Any
from openai import OpenAI
from common import _parse_json, VALID

# DeepSeek v4-pro, thinking + high reasoning effort (the correct v4-pro invocation), temperature 0.
DS_MODEL = "deepseek-v4-pro"

def _llm_v4pro(prompt: str, max_attempts: int = 6) -> dict:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    backoffs = [10, 20, 40, 80, 160]; last = None
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=DS_MODEL, temperature=0.0, max_tokens=32000,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json(resp.choices[0].message.content or "")
        except Exception as e:
            last = e
            if attempt >= max_attempts - 1:
                break
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
    raise RuntimeError(f"deepseek-v4-pro dichotomy call failed: {last!r}")


def _prompt_dichotomy_trace(report_md: str, det: dict[str, Any]) -> str:
    p = det["partner_tf"]; t = det["target_tf"]
    return f"""You are classifying ONE ARES report. The report investigated why the PARTNER TF's motif
predicts the TARGET TF's ChIP-seq binding SIGNAL. Classify the mechanism as SEQUENCE, PROTEIN, or CONTEXT.

Decide ENTIRELY from what THIS REPORT concludes. Use only mechanisms the report itself establishes;
its "observational / cannot prove causality" boilerplate does NOT force any class.

# Pair
- target_tf (the TF being predicted): {t}
- partner_tf (the motif owner): {p}
- cell_line: {det['cell_line']}
- converged: {det['converged']}
- final_mechanism_freeform: {det.get('final_mechanism_freeform')!r}

# Output (single JSON object, no markdown, no commentary)
{{ "dichotomy_class": "SEQUENCE" | "PROTEIN" | "CONTEXT" | "UNRESOLVED" | "ARTIFACT",
   "reader": "<the protein that BINDS the {p} motif: {t} itself, another named protein which can bind to DNA, or '' if nobody is established to read it>",
   "actor": "<the protein whose binding DIRECTLY moves {t}'s signal (recruits/tethers/stabilizes/co-binds, or EXCLUDES {t}); '' if the only thing acting on {t} is a chromatin STATE>",
   "actor_recruited_by": "reader_chain" | "context" | "none",
   "actor_is_named_partner": <true|false>,
   "protein_subtype": "cooperative" | "tethering" | null,
   "reason": "<=25 words tracing reader -> actor -> provenance, citing the report",
   "region_of_regulation": "PROMOTER_ACTIVE" | "ENHANCER_ACTIVE" | "ENHANCER_POISED" | "BIVALENT_PROMOTER" | "ACTIVE_TRANSCRIPTION_STATES" | "REPRESSIVE_HETEROCHROMATIN" | "INSULATOR" | "MIXED" | "NOT_REPORTED",
   "transcriptional_outcome": "ACTIVATING" | "REPRESSIVE" | "NO_EFFECT" | "MIXED" | "NOT_TESTED" }}

# THE DISCRIMINATOR — once the {p} motif HAS a reader protein, PROTEIN vs CONTEXT depends on the FUNCTION
# of the reader's chain: does a protein act on the TARGET, and was that protein brought by the READER or by
# the CONTEXT? Trace it in three steps and STOP at the first that decides.

## Step 1 — READER: who physically binds the {p} motif?
  • SEQUENCE — {t}'s signal gain comes from a MEANINGFUL, {t}-specific REGULATORY SEQUENCE / GRAMMAR, with
      NO protein functionally needed to BRING {t}. The decisive question is the SOURCE OF THE SIGNAL GAIN,
      not who sits on the motif: EVEN IF another protein R reads the {p} motif, it is STILL SEQUENCE when R
      does NOT functionally act on {t} (no recruit / tether / stabilize) and the gain comes from the
      sequence/grammar. reader={t} (or R if R only marks the grammar), actor='', actor_recruited_by="none". STOP.
      Qualifies as SEQUENCE if ANY of these holds (the signal rides on the sequence, not on a protein):
        (a) {t}'s OWN DBD directly recognizes the {p} motif ({t}'s own, or a family-similar motif {t} binds).
        (b) the {p} motif is a HALF-SITE / sub-sequence NESTED inside {t}'s own motif (Trojan-horse).
        (c) MOTIF GRAMMAR — the {p} motif forms a constrained spacing / adjacency COMPOSITE with {t}'s motif
            (or another regulatory grammar) that FACILITATES {t}'s binding; the gain is from that
            arrangement (scrambling the grammar — not removing a protein — kills it). A protein R reading the
            {p} motif is FINE here, AS LONG AS R does not functionally bring {t}: the grammar, not R, raises
            {t}'s signal. (This is the common shape — a partner/paralog reads the {p} motif but the composite
            grammar is what predicts {t}.)
      NOT SEQUENCE (do NOT call SEQUENCE in these — they are CONTEXT or PROTEIN):
        • GENERIC BASE COMPOSITION — the gain is driven by GC-content, AT-richness, CpG density, or
          low-complexity, i.e. the {p} motif merely TAGS a region of that composition, not a {t}-specific
          grammar -> CONTEXT. (A meaningful regulatory motif/grammar counts; raw nucleotide content does not.)
        • a protein that reads the {p} motif (or a cofactor it brings) FUNCTIONALLY ACTS on {t} —
          recruits / tethers / stabilizes / co-binds / excludes it -> PROTEIN (the gain rides on that protein,
          go to Step 2/3). [Reading the motif alone, with no action on {t}, does NOT trigger this.]
        • {t} CANNOT bind DNA on its own — no sequence-specific DBD (cofactor / bridging protein), OR it
          binds ONLY as an obligate heterodimer/complex REQUIRING the partner PROTEIN (e.g. FOS needs JUN; a
          single SHARED motif, not {t}'s own separable motif) -> the partner protein is required -> Step 2
          -> PROTEIN (tethering if {t} reads no DNA; cooperative if {t} reads its half-site but needs the partner).
  • NO protein is established to read the {p} motif (only co-occupancy / the motif marks a state)
      -> CONTEXT (or UNRESOLVED if who-reads-it is genuinely open). reader='', actor_recruited_by="context".
      STOP. DO NOT invent a reader: a partner/cofactor merely CO-OCCUPYING the sites is NOT evidence it
      reads the {p} motif — it may be there via the state.
  • ANOTHER protein R is shown to bind the {p} motif -> reader=R; go to Step 2.

## Step 2 — ACTOR: is there a protein A whose binding DIRECTLY raises or lowers {t}'s signal at these sites
   (recruits / tethers / stabilizes / co-binds {t}, or EXCLUDES/displaces {t})?
  • NO protein acts on {t}; {t} only RESPONDS to the chromatin STATE the reader writes/marks (accessibility,
      active/repressive marks, compartment) -> CONTEXT. actor='', actor_recruited_by="context". STOP.
  • The reader R itself binds only the motif as a SEQUENCE/grammar feature and does not act on {t}
      -> SEQUENCE. actor='', actor_recruited_by="none". STOP.
  • YES, a protein A acts on {t} (A may be R itself or a cofactor/complex/hub) -> go to Step 3.

## Step 3 — PROVENANCE of the actor (the KEY check): how did A get to these sites?
  • A is brought there by the READER's chain — R binds/recruits/scaffolds A, or A and R are in one complex
      anchored by R reading the motif (R -> ... -> A, an unbroken PROTEIN chain) -> PROTEIN.
      actor=A, actor_recruited_by="reader_chain".
        - tethering  : {t} reads NO DNA of its own here; A/R brings it. [protein_subtype="tethering"]
        - cooperative: {t} reads its OWN half-site but REQUIRES A/R's binding. [protein_subtype="cooperative"]
        - exclusion  : A reading the motif DISPLACES {t} (motif negatively predicts {t}). [protein_subtype=null]
  • A is brought there by the CONTEXT — A co-occupies because of the chromatin STATE / region the {p} motif
      MARKS (active/repressive promoter, enhancer, HOT, compartment), NOT because the motif-reader recruited
      it -> CONTEXT. actor=A, actor_recruited_by="context".
      (Even though a protein acts on {t}, the motif's only role is marking the region; A and {t} are both
       drawn by the state. Removing the motif would not remove A's action on {t}.)

# DECISIVE COUNTERFACTUAL: would the actor still act on {t} if the {p} motif (and its reader) were gone?
#   - rides in on the reader  -> motif causes it -> PROTEIN.
#   - rides in on the chromatin state -> motif only marks it -> CONTEXT.

# WORKED CASES
#  ARID4B<-YY1 : YY1 READS its motif and recruits the SIN3A/HDAC scaffold, which TETHERS no-DBD ARID4B.
#                actor=SIN3A/HDAC, recruited_by=reader_chain -> PROTEIN (tethering).
#  CBX1<-TBP   : TBP motif MARKS active metabolic promoters; HP1/TRIM28 complex co-occupies CBX1, but the
#                complex is brought by the ACTIVE-PROMOTER CONTEXT (no protein shown to read the TBP motif to
#                recruit CBX1; direct TBP tethering rejected). actor=HP1/TRIM28, recruited_by=context -> CONTEXT.
#  SRF<-CUX2   : CUX2 motif MARKS a repressive chromatin state; any repressor at the site is brought by that
#                state, not by a CUX2-motif reader. recruited_by=context -> CONTEXT.

UNRESOLVED — the report does not establish who reads the motif or how it sets {t}'s binding, or did not converge.
ARTIFACT   — the report shows the {p} motif does NOT actually predict {t}'s binding.

# FUNCTIONAL ANNOTATIONS (fill from the report, INDEPENDENT of the SEQUENCE/PROTEIN/CONTEXT call; use the
# NOT_REPORTED / NOT_TESTED / NO_EFFECT values when the report does not establish them):
#  region_of_regulation — WHERE the {p}-motif co-bound sites act, from the report's chromatin/region evidence:
#     PROMOTER_ACTIVE (active promoters/TSS), ENHANCER_ACTIVE, ENHANCER_POISED, BIVALENT_PROMOTER,
#     ACTIVE_TRANSCRIPTION_STATES (active marks/open, not pinned to promoter-vs-enhancer),
#     REPRESSIVE_HETEROCHROMATIN (repressive marks/closed), INSULATOR (CTCF / boundary / loop anchor),
#     MIXED, NOT_REPORTED (report does not localize the region).
#  transcriptional_outcome — effect of the {p}-motif co-binding on TARGET-GENE expression, from the report:
#     ACTIVATING (raises expression), REPRESSIVE (lowers it), NO_EFFECT (tested, no change), MIXED,
#     NOT_TESTED (expression not tested).

# Report
{report_md.strip()}
"""


def dichotomy_classify_trace(report_md: str, det: dict[str, Any], provider: str = "deepseek-v4-pro") -> dict[str, Any]:
    parsed = _llm_v4pro(_prompt_dichotomy_trace(report_md, det))
    dc = parsed.get("dichotomy_class")
    dc = dc if dc in VALID else f"INVALID:{dc}"
    reader = str(parsed.get("reader", "")); actor = str(parsed.get("actor", ""))
    recruited = str(parsed.get("actor_recruited_by", "")); reason = str(parsed.get("reason", ""))
    # provenance guard: an actor brought by the CONTEXT cannot be PROTEIN (mirrors step 3)
    if dc == "PROTEIN" and recruited == "context":
        reason = f"[guard: actor recruited by context -> CONTEXT] " + reason
        dc = "CONTEXT"
    # deterministic guard (same as llm_dichotomy): a S/P/C call asserts a mechanism; if none supported -> UNRESOLVED
    nm = det.get("n_supports_mechanism")
    if nm == 0 and dc in {"SEQUENCE", "PROTEIN", "CONTEXT"}:
        nc = det.get("n_supports_characterization", 0)
        why = "only characterization/GO support" if nc else "no supported hypotheses"
        reason = f"[guard: {dc}->UNRESOLVED, n_supports_mechanism==0 ({why})] " + reason
        dc = "UNRESOLVED"
    return {
        "dichotomy_class": dc,
        "reader": reader,
        "actor": actor,
        "actor_recruited_by": recruited,
        "actor_is_named_partner": bool(parsed.get("actor_is_named_partner", False)),
        "protein_subtype": str(parsed.get("protein_subtype") or ""),
        "reason": reason,
        "region_of_regulation": str(parsed.get("region_of_regulation") or "NOT_REPORTED").upper(),
        "transcriptional_outcome": str(parsed.get("transcriptional_outcome") or "NOT_TESTED").upper(),
    }
