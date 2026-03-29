# Step-by-Step Modifications for Discovery-Oriented Pipeline

This document lists **exact file paths, locations, and edits** so the pipeline can support discovery/synthesis (like a manual report that reveals a new mechanism) instead of only converging on a predefined taxonomy number.

---

## 1. Treat “max consecutive hypothesis rejections” as a synthesis trigger (not hard stop)

**Goal:** When the reviewer rejects 6 hypotheses in a row but you already have at least one SUPPORTS result, converge on a synthesis conclusion and generate the final report instead of stopping with a generic failure message.

**File:** `src/orchestrator.py`

**Step 1.1 — Where to change:** The block that runs when `consecutive_hypo_rejections >= max_hypo_rejections` (around lines 358–371).

**Current behavior:** Sets `self.state.conclusion` to a generic “pipeline stopped” message and `break`s out of the loop.

**Modification:**

- **Before** the `break`, check whether there is at least one hypothesis with `result == "SUPPORTS"` in `self.state.tested_hypotheses`.
- **If yes:**
  - Set a synthesis-oriented conclusion, e.g.  
    `"Max consecutive hypothesis rejections; synthesizing from supported evidence. Supported: <names>. Rejected hypotheses were duplicates or insufficiently distinct. Conclusion should summarize the best-supported mechanism and contrast with ruled-out alternatives."`
  - Call **`mark_converged`** (or set `self.state.converged = True` and set `self.state.conclusion` to a short synthesis summary) so the loop exits and the code proceeds to **final report generation** (same path as normal convergence).
- **If no** (no SUPPORTS): keep current behavior (set the existing “pipeline stopped” conclusion and `break`).

**Step 1.2 — Optional:** Move `max_hypo_rejections = 6` to config so it can be tuned without editing code.  
**File:** `config/config.yaml` — add something like `pipeline.max_consecutive_hypothesis_rejections: 6`.  
**File:** `src/orchestrator.py` — read this value from `self.config.pipeline` (and keep 6 as default if missing).

---

## 2. Allow convergence on a *new* named mechanism (not only taxonomy 1–12)

**Goal:** Let the adjudicator declare convergence when evidence supports a **specific, named** mechanism that is not in the predefined taxonomy, as long as it’s causal and well-defined.

**File:** `src/prompts/summary.py`

**Step 2.1 — CONVERGENCE_CHECK_SYSTEM_PROMPT (lines 56–151):**

- In the “Declare converged=true ONLY when ALL FOUR conditions are met” section (around 62–70), **clarify** that the “Named mechanism” criterion can be satisfied by:
  - A mechanism from the taxonomy (categories 1–12), **or**
  - A **new** mechanism that: (1) names a specific molecular process, (2) states a clear causal chain, and (3) is not merely a re-description of correlation.
- In the “Common False Convergence Patterns” / “Alternatively” sentence (around 151), **strengthen** the “convergence is permitted” part: e.g. explicitly say that when the evidence supports a mechanism **not** in the list, the adjudicator should set `mechanism_category_number` to `null` and put the mechanism name in `mechanism_category_name` (e.g. “NFYA-anchored housekeeper vs adaptive SP1-only”).

**Step 2.2 — build_convergence_check_prompt (lines 186–253):**

- In the **Task** section and the JSON schema (around 237–251):
  - State that `mechanism_category_number` may be `null` when the mechanism is **not** in the taxonomy but is still a valid named causal mechanism.
  - Ask for `mechanism_category_name` to always be filled when `converged` is true (taxonomy name or a short descriptive name for a new mechanism).

No code changes in `summary_agent.py` are strictly required if the LLM starts returning `mechanism_category_number: null` with a filled `mechanism_category_name`; the rest of the pipeline already uses `mechanism_category_name` in the final report.

---

## 3. Add synthesis / integrative narrative to the final report

**Goal:** The final report should include an integrative synthesis (e.g. “NFYA+SP1 vs SP1-only” or comparison of supported vs ruled-out mechanisms), not only “we converged on mechanism #X”.

**File:** `src/prompts/summary.py`

**Step 3.1 — build_final_report_prompt (lines 336–401):**

- In the **Task** section (around 387–399), **add** a bullet, e.g.:
  - **“Synthesis / mechanism comparison”:** If multiple hypotheses were tested, provide a short integrative narrative: which mechanism(s) are best supported, which were ruled out, and how they relate (e.g. “NFYA-anchored constitutive vs SP1-only adaptive” or a small comparison table). If convergence was on a **new** mechanism (not in the taxonomy), describe it clearly and how it differs from the listed categories.
- Optionally add: “Prefer a concise comparison table (e.g. mechanism vs evidence vs outcome) when there are several hypotheses.”

No new functions or agents are required; the existing `generate_final_report` in `src/agents/summary_agent.py` (which calls `build_final_report_prompt`) will then produce reports that include this synthesis when the prompt is updated.

---

## 4. Reframe the goal in the data manifest / run context (discovery vs “pick one mechanism”)

**Goal:** Make the pipeline’s stated objective “characterize and synthesize a coherent mechanism (possibly new)” instead of “select one predefined mechanism”.

**File:** Examples use a manifest such as `examples/sp1_nfya/data_manifest.yaml`.

**Step 4.1 — Add or edit a “goal” or “investigation_objective” field** in the manifest (if your schema supports it). For example:

```yaml
# In context or a new top-level field
investigation_objective: "Characterize the mechanism linking TF_B (motif) to TF_A binding; synthesize a coherent causal narrative, possibly identifying a new mechanism not in the predefined taxonomy."
```

**Step 4.2 — Use this in prompts:** Where the finding/context is injected (e.g. in `build_convergence_check_prompt` and `build_final_report_prompt`), if you pass `context` from the manifest, ensure the context (or a one-line instruction) states that the aim is **discovery and synthesis**, and that convergence on a **new** named mechanism is acceptable.  
**Files:** `src/prompts/summary.py` (convergence and final report prompts). You can append one sentence to the “Task” or “Context” using this objective so the LLM sees it.

---

## 5. Optional: Periodic synthesis step (every N iterations)

**Goal:** Every N iterations, produce an integrative summary (e.g. “supported vs ruled-out so far”) and optionally feed it back so the next hypothesis can “explore a different mechanism category”.

**File:** `src/orchestrator.py`

**Step 5.1 — Where:** Inside the main `while` loop, after you’ve updated state with the latest test result (e.g. after `_check_and_refine` and before deciding to continue the loop). Add a condition, e.g. `if self.state.current_iteration > 0 and self.state.current_iteration % N == 0` (N from config, e.g. 3).

**Step 5.2 — What to do:**
- Call a new method, e.g. `_generate_synthesis_summary()`, which:
  - Builds a short summary of: supported hypotheses, refused hypotheses, and current best-supported mechanism(s).
  - Optionally calls the LLM with a small prompt: “Given the following tested hypotheses and results, produce a 1-paragraph synthesis and suggest what mechanism class to explore next (different from already supported).”
- Append this synthesis to `self.state` (e.g. `state.synthesis_summaries` list) and/or pass it into the **next** refinement cycle as extra context (e.g. via `group_summary` or a new field in the refinement prompt).

**Step 5.3 — Prompt for synthesis:** Add in `src/prompts/summary.py` a new function, e.g. `build_synthesis_prompt(tested_hypotheses, finding)`, returning a user prompt that asks for: (1) integrative narrative so far, (2) best-supported mechanism, (3) one suggestion for a *different* mechanism category to test next.  
**File:** `src/agents/summary_agent.py` — add `async def generate_synthesis(...)` that uses this prompt and returns the narrative (and optionally the “next mechanism to try” suggestion).

**Step 5.4 — Pass synthesis into hypothesis refinement:** In `src/orchestrator.py`, when calling `refine_hypotheses`, if you have a recent synthesis summary, pass it as `group_summary` (or an additional argument to `build_refinement_prompt` in `src/prompts/hypothesis.py`) so the hypothesis agent is explicitly told: “So far the evidence supports X; consider proposing a hypothesis from a **different** mechanism category to broaden the search.”

---

## 6. Encourage “different mechanism category” after a support

**Goal:** After at least one hypothesis is SUPPORTS, nudge the hypothesis agent to propose a **different** mechanism (or mechanism category) next, to avoid getting stuck in one family (e.g. “tethering” rephrased) and to allow discovery of contrasts (e.g. NFYA vs SP1-only).

**File:** `src/prompts/hypothesis.py`

**Step 6.1 — build_refinement_prompt:** In the “INVESTIGATION GUIDANCE” or “Task” section (around 106–141), **add** a short bullet or paragraph:

- “If at least one hypothesis has already **SUPPORTS**, prefer proposing a hypothesis that tests a **different** mechanism or mechanism category (e.g. pioneer vs tethering vs chromatin modifier), so the conclusion can compare or synthesize across mechanisms rather than refining the same one repeatedly.”

You can make this conditional: in the orchestrator, when building the refinement prompt, if `any(h.get("result") == "SUPPORTS" for h in state.tested_hypotheses)`, pass a flag or append this instruction (e.g. via `group_summary` or an optional parameter to `build_refinement_prompt`).

**File:** `src/prompts/hypothesis.py` — add an optional parameter like `encourage_different_mechanism: bool = False` and, when True, append the sentence above to the prompt.

**File:** `src/orchestrator.py` — in `_check_and_refine`, when calling the hypothesis agent’s `refine_hypotheses`, pass `encourage_different_mechanism=True` when there is at least one SUPPORTS in `self.state.tested_hypotheses`. (This requires threading the flag through from the orchestrator to the prompt builder.)

---

## Summary table

| # | File | Location | What to change |
|---|------|----------|----------------|
| 1 | `src/orchestrator.py` | Block at ~358–371 (max consecutive rejections) | If any SUPPORTS exist, set synthesis conclusion and mark converged so final report runs; else keep current stop. Optionally read `max_hypo_rejections` from config. |
| 2 | `src/prompts/summary.py` | CONVERGENCE_CHECK_SYSTEM_PROMPT, build_convergence_check_prompt | Allow convergence on a new named mechanism; clarify `mechanism_category_number` null + `mechanism_category_name` for new mechanisms. |
| 3 | `src/prompts/summary.py` | build_final_report_prompt Task section | Add “Synthesis / mechanism comparison” and optional comparison table; mention new mechanisms. |
| 4 | Manifest + prompts | data_manifest (e.g. examples/sp1_nfya), summary prompts | Add discovery/synthesis objective; use it in context/task for convergence and final report. |
| 5 | `src/orchestrator.py`, `src/prompts/summary.py`, `src/agents/summary_agent.py` | Loop + new synthesis prompt + new method | Optional: every N iterations, generate synthesis summary and pass to refinement as `group_summary`. |
| 6 | `src/prompts/hypothesis.py`, orchestrator + hypothesis agent | build_refinement_prompt, refine_hypotheses call | Optional: when SUPPORTS exists, pass “prefer different mechanism category” into refinement prompt. |

Implementing **1–4** gives you discovery-oriented convergence and reporting; **5** and **6** improve exploration and reduce duplicate hypotheses.
