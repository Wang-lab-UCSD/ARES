Read AGENT_WORKFLOW.md for agent orchestration rules. NEVER modify this file.

# ARES — Automated Regulatory Solver

## Project Context
Multi-agent pipeline for automated hypothesis generation and verification in bioinformatics.
User provides a finding (X predicts Y) + data paths, pipeline runs until convergence.

## Design Principles
- **General-purpose**: Pipeline must work for any TF pair, not just ATF6/REST. Avoid problem-specific prompts or hardcoded assumptions.
- **Scalable**: Dr. Wang plans to run on 1,700+ TF pairs - robustness is critical.
- **Tool-agnostic output inspection**: When LLM uses external tools (FIMO, bedtools, etc.), it should inspect output before continuing - but this must be generic, not tool-specific prompting.

## Architecture
- `src/orchestrator.py` - Main loop coordinator
- `src/llm/` - LLM provider abstraction (OpenAI, Claude, Gemini)
- `src/agents/` - Hypothesis, Coding, Review, Summary agents
- `src/execution/` - Jupyter kernel for code execution
- `src/memory/` - Conversation history management
- `src/prompts/` - Prompt templates (hypothesis, coding, review, summary, tool_quirks)

## Available Models (March 2026)
- **OpenAI**: `gpt-5.2`, `gpt-5.2-pro`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.2-codex`, `gpt-5.1-codex`, `gpt-5-codex`
- **Anthropic**: `claude-sonnet-4`, `claude-opus-4`
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-3-flash-preview`
- **Z.AI (GLM-5, OpenAI-compatible)**: `glm-5` via `base_url: "https://api.z.ai/api/paas/v4/"` with `api_key_env: "Z_AI_API_KEY"`. Use `provider: "openai"` in config — the OpenAI client handles it transparently.
- **DeepSeek (OpenAI-compatible)**: `deepseek-chat` via `base_url: "https://api.deepseek.com"` with `api_key_env: "DEEPSEEK_API_KEY"`. Set `max_tokens: 8192` — DeepSeek default is 4096, too low for long scripts.

## MiniMax Model Quirks
- **Set `max_tokens` >= 16000 for MiniMax-M2.7**: Thinking tokens count against the `max_tokens` budget. The default of 4096 is too low — the model truncates code mid-string (e.g., unterminated string literals, unclosed parentheses), then its fix is also truncated, creating an endless loop of syntax errors. Set `max_tokens: 16000` or higher.

## OpenAI Model Quirks
- **Do NOT set `max_tokens` for OpenAI reasoning models (gpt-5.2, etc.)**: Reasoning tokens count against the `max_completion_tokens` budget. Setting a low limit causes the model to exhaust tokens on "thinking" before producing visible output. If gpt-5.2 still hits "model output limit reached" even without `max_tokens`, set it very high (e.g., `max_tokens: 32768`) rather than omitting it — the model's own default can also be too low for complex JSON+reasoning outputs.
- **gpt-5.2-codex uses completions API**: This model requires the `/v1/completions` endpoint, not `/v1/chat/completions`. Use `gpt-5.2` for chat-based tasks.
- **gpt-5-mini only supports temperature=1**: Do not set custom temperature values for this model.

## GLM-5 (Z.AI) Quirks
- **"No tags" responses**: GLM-5 sometimes responds with plain-text narration instead of a `<execute>` block, wasting a REPL step. This appears at roughly 25% frequency on complex iterations. The REPL loop handles it gracefully (logs a WARNING, increments iteration, re-prompts), but it reduces effective REPL budget. No fix needed — just be aware that `max_retries: 5` budgets 20 REPL steps, and GLM-5 may consume 25% on narration-only turns.
- **Do NOT set `max_tokens` for GLM-5**: Same reasoning as OpenAI reasoning models — thinking tokens count against the budget. Omit `max_tokens` in config for GLM-5.

## Convergence Criterion 5 — Biology Layers (March 2026 Fix)
Criterion 5 requires that if `rnaseq` or `phyloP` data is available in the manifest, at least one hypothesis must have **tested** one of them. The test result can be SUPPORTS, REFUTES, or INCONCLUSIVE — any attempt satisfies the criterion.

**Why this distinction matters:**
- `rnaseq`/`phyloP` = functional consequence / evolutionary conservation (did the interaction have a biological effect?)
- `STRING`/PPI = protein–protein interaction mechanism (are these proteins physically linked?)
- STRING/PPI satisfies a different criterion (Criterion 4, mechanism evidence). It does NOT substitute for rnaseq/phyloP in Criterion 5.

**Code change**: `_biology_layers_for_convergence()` in `src/orchestrator.py` tracks layers used in ANY tested hypothesis (previously filtered to SUPPORTS-only, which caused the pipeline to keep running after a REFUTES/INCONCLUSIVE result on phyloP).

## API Keys
- **OpenAI API key**: `api_key.txt` (in project root, gitignored)
- To run the pipeline: `export OPENAI_API_KEY=$(cat api_key.txt)`

## Environment Setup

Use conda for environment management (includes bioinformatics tools). Do this once before running the pipeline.

### 1. Create and activate the environment

```bash
# Create env with Python 3.10+ (project requires >=3.10)
conda create -n pipeline python=3.11 -y
conda activate pipeline
```

### 2. Install conda bioinformatics tools

```bash
conda install -c bioconda -c conda-forge meme bedtools samtools -y
```

Add UCSC tools if needed (e.g. `ucsc-bigwigaverageoverbed`); exact package names may vary on bioconda.

### 3. Install Python dependencies

From the pipeline repo root:

```bash
cd /path/to/ares
pip install -r requirements.txt
```

If `pip install` fails building **pybedtools** (error: `g++` not found), either:

- **Option A** — install a C++ compiler, then re-run pip:
  ```bash
  conda install -c conda-forge cxx-compiler -y
  pip install -r requirements.txt
  ```
- **Option B** — install pybedtools from conda (pre-built), then pip the rest:
  ```bash
  conda install -c bioconda pybedtools -y
  pip install -r requirements.txt
  ```

### 4. Run the pipeline

```bash
conda activate pipeline
python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml
```

## Running the Pipeline
```bash
# Set API key
export OPENAI_API_KEY="sk-..."

# Run with config file
python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml

# Run with interactive model selection
python -m src.main --manifest examples/atf6_rest/data_manifest.yaml
```

## Testing
```bash
conda activate pipeline
pytest tests/ -v
```

## Key Patterns
- All LLM calls go through `src/llm/base.py` interface
- Conversation memory stored in `src/memory/conversation.py`
- Structured logging to `logs/` directory
- JSON schema validation for LLM outputs
- Graceful shutdown on SIGINT (Ctrl+C)

## Code Style
- Python 3.10+
- Type hints required (use `from __future__ import annotations`)
- Async/await for LLM calls
- Pydantic for config and data models
- Keep functions focused and small

## File Naming
- Use snake_case for all Python files
- Use descriptive names that indicate purpose


## Known Issues & Lessons Learned (Jan 2026 ATF6/REST Run)

### Problems Encountered

1. **Technical failures trigger new iterations instead of retries**
   - Pipeline treats "code crashed" the same as "need more evidence"
   - A pandas visualization bug caused 4 failed attempts in iteration 2, each wasting ~55 min of computation
   - Instead of fixing the bug and retrying, pipeline moved to iteration 3, 4, etc.

2. **LLM generates overly complex code with fragile visualizations**
   - Code kept failing on: `df['label'] = df.apply(lambda r: ..., axis=1)` pattern
   - LLM's "fixes" were superficial (adding `.astype(int)`) instead of addressing root cause
   - Visualization code is not essential for scientific conclusions but causes most crashes

3. **Excessive computation when evidence is already conclusive**
   - Iteration 1 found p=0.01 with 100 shuffles (conclusive)
   - LLM requested 1000 shuffles for "more robust" confirmation
   - Result: 4 × 55 min wasted on a stricter test that wasn't needed

4. **Full JASPAR motif scans are slow and often unnecessary**
   - Scanning all 800+ motifs takes 1-2 hours
   - Most hypotheses only need targeted scans (2-3 specific motifs)

### Pipeline Robustness (Completed)

- [x] **Separate technical vs scientific failures** — TECHNICAL_ERROR in refinement decisions + summary-level ERROR triggers retry loop
- [x] **Disable visualizations** — Coding prompt: "No matplotlib/seaborn. Focus on statistics only."
- [x] **Limit computation scale** — Coding prompt caps permutations at 100, one statistical test per script
- [x] **Cost tracking** — Cost tracker records tokens and estimated cost per API call
- [x] **Summary-level ERROR retry** — When code catches its own exception (try/except), Jupyter reports success, but summary agent detects ERROR and feeds it back into retry loop


### Reference: ATF6/REST Run Results (outputs/20260126_135214/)

Key finding from iteration 1 (after manual bug fix):
- **CCACG pentamer**: 52.4% in ATF6 peaks vs 17.4% background (p=0.0099) ✅
- **REST full motif**: 31.7% in ATF6 peaks (2,509 of 7,906) - ENRICHED ✅
- **Conclusion**: Both REST motif and CCACG pentamer are enriched; supports "Trojan Horse" hypothesis

### Critical Bug Discovered (LLM-generated code)

The pipeline reported "REST motif in 0.29% of peaks" - this was **wrong**.

**The bug**: Code counted unique `sequence_name` values from FIMO output:
```python
seqs = set(fimo_df["sequence_name"].unique())  # Returns 23 (chromosomes!)
```

**The problem**: FIMO's `sequence_name` contains chromosome names (chr1, chr2...), not peak IDs.
bedtools getfasta creates headers like `>chr1:12345-67890`, but FIMO parses only `chr1`.

**Correct method**: Use `bedtools intersect` to match FIMO coordinates back to original peaks.

**Why LLM missed it**: No data inspection, no sanity check (23 values for 7,906 peaks should be suspicious), code ran without errors.

### Reference: Hanbei's Workflow (doc/hanbei_results.pdf)

Hanbei's approach differs fundamentally from our pipeline:
- **No data upload**: Described the problem verbally to LLM, no files shared
- **Human-in-the-loop**: LLM generates hypotheses, Hanbei executes verification himself
- **Iterative feedback**: Hanbei provides results back to LLM for hypothesis refinement

Key hypothesis from Hanbei's session: **"Trojan Horse" (Nested Motif)**
- REST motif (~21bp) contains CCACG/TGACG (ATF6's core binding sequence)
- ML model flags REST as important because it's a "super-ATF6" motif
- The motif predicts ATF6 binding not because REST protein is involved, but because it contains ATF6's recognition sequence

### Asset 1: ATF6/REST (K562) — Feb 21 run (outputs/20260220_230622/)

7 iterations, CONVERGED on mechanism #3 (Chromatin/accessibility).
Conclusion: REST+ATF6 co-bound sites are modestly enriched in Polycomb-repressed/poised ChromHMM states (2.4% vs 1.4%, OR=1.73, p=7.5e-4). Effect is real but small.
Mechanisms tested: #2 (DNA sequence, REFUTES), #3 (chromatin, SUPPORTS), #5 (3D loops, INCONCLUSIVE).
Mechanisms not yet explored: #1, #4, #6, #7, #8, #9, #10, #11, #12.

### Asset 2: ATF3/USF1 (GM12878) — Feb 21 run (outputs/20260221_172221/)

7 iterations, CONVERGED on mechanism #9 (Binding kinetics — cooperative binding).
Conclusion: ATF3 ChIP-seq summits are centered on USF1's E-box 67% of the time in dual-motif peaks (p=0.001). USF1 nucleates the site; ATF3 binds cooperatively nearby. ML model detects USF1 motif as predictive because the ChIP signal is physically centered on it.
Mechanisms tested: #2 (DNA sequence, REFUTES), #3 (chromatin, REFUTES), #9 (binding kinetics, CONVERGED).
Mechanisms not yet explored: #1, #4, #5, #6, #7, #8, #10, #11, #12.

### Asset 3: MAX/ZBTB14 (GM12878) — Feb 21 run (outputs/20260221_202803/)

6 iterations, CONVERGED on mechanism #5 (3D genome architecture).
Conclusion: ZBTB14 motif marks promoter-like loci enriched in ChromHMM Tss* states (OR~3.1), embedded in 3D promoter interaction hubs where MAX binding is stabilized. Co-bound sites sit at loop anchors and TAD boundaries, explaining why ZBTB14 predicts MAX binding.
Mechanisms tested: #2 (DNA sequence, REFUTES), #3 (chromatin, REFUTES), #5 (3D genome, SUPPORTS/CONVERGED).
Mechanisms not yet explored: #1, #4, #6, #7, #8, #9, #10, #11, #12.

### Feb 2026 Three-Pair Test Run Results (outputs/20260216_*)

Three runs on Feb 16, 2026 using the updated pipeline (with ReviewAgent, convergence criteria, tool quirks):

1. **ATF3/USF1 (GM12878)** — `outputs/20260216_174252/`
   - 9 iterations, 9 hypotheses, **CONVERGED** (confidence 0.82)
   - Conclusion: "ATF3 dose-dependently co-occupies USF1 sites via shared E-box at promoters"
   - Technical issues: iter 2 (5 attempts, tomtom flag error), iter 5 (6 attempts, bigWig parsing), iter 8 (2 attempts, duplicate BED)
   - **Note**: Hanbei flagged this convergence as problematic — conclusion is phenomenon, not mechanism (see Known Issues)

2. **ATF6/REST (K562)** — `outputs/20260216_154359/`
   - 10 iterations, 9 hypotheses, **NOT converged**
   - Found sequence-level effects (motif enrichment) but couldn't establish mechanism
   - Trojan Horse hypothesis tested but FIMO approach was wrong (needs literal substring, not PWM matching)

3. **MAX/ZBTB14 (GM12878)** — `outputs/20260216_164429/`
   - 10 iterations, 9 hypotheses, **NOT converged**
   - Found promoter architecture clues (CpG islands, bidirectional promoters) but couldn't synthesize into unified mechanism
   - 2 iterations failed due to missing `bigWigToBedGraph` tool

### Known Issues (Feb 2026) — Status as of Feb 18

1. ✅ **No essential vs supplementary hypothesis distinction** — Fixed: replaced "use the surprise" rule with hypothesis prioritization filter (WHY vs WHAT). LLM must justify why a characterization hypothesis is needed before proposing it.

2. ✅ **Superficial fixes on technical retry** — Fixed: added mandatory root cause diagnosis comment to error fix prompt. LLM must identify the root cause before writing fix, and must use a different approach if the same tool failed.

3. ✅ **Phenomenon hypothesis accepted as mechanism** — Fixed: replaced vague mechanism definition with causal sequence test + good/bad example pair using TF_A/TF_B terminology.

4. ✅ **"Use the surprise" rule is redundant** — Resolved by item 1 fix (surprise rule was replaced entirely).

5. ✅ **Phase 3 examples are misleading** — Fixed: removed 4-phase framework entirely from system prompt, including the misleading Phase 3 examples.

6. ✅ **4-phase framework duplicates phenomenon-first approach** — Fixed: replaced 4-phase framework with open-ended reasoning methodology (reason from biology + data, find most diagnostic test). No more phase labels or forced sequencing.

7. ✅ **Convergence criteria need revision** — Fixed: dropped condition 2 (alternative mechanism must be ruled out). New criteria: (1) mechanism hypothesis has statistical support (p < 0.05), (2) conclusion explains WHY. Rationale: requiring alternative mechanisms to be ruled out is too strict for 1,700+ TF pair scale; can be re-added if pipeline proves unreliable.

8. ✅ **Initial batch of association hypotheses wastes iterations** — Fixed: `build_initial_hypothesis_prompt()` now generates exactly ONE association-confirmation hypothesis. Removed "Do NOT generate mechanism hypotheses yet" instruction so the refinement loop immediately takes over after confirmation.

8.1. ✅ **Sequence-vs-protein suggestion hardcoded in initial prompt** — Fixed: removed the instruction to "diagnose whether it is a sequence-level or protein-level effect" from the initial prompt. This diagnostic is mechanism exploration, not association confirmation; the LLM should decide what the most diagnostic first mechanism test is based on context.

9-10. ✅ **Unbalanced file access across iterations** — Fixed: (1) added "review all available files in the manifest" reminder to the initial hypothesis prompt; (2) `get_history_summary()` now includes `Data used` per iteration so the LLM sees which files were used in prior iterations; (3) fixed existing bug in `get_history_summary()` where hypothesis name showed as "N/A" (`description` → `name` field).

11. ✅ **Item 1 fix was too weak** — Fixed: replaced the soft "ask yourself WHY vs WHAT" guidance with a hard rule in `build_refinement_prompt()`: characterization hypotheses are forbidden in the mechanism exploration phase. Every hypothesis must name a causal mechanism, predict a direction of effect, and be falsifiable.

12. ✅ **Decision/reasoning fields returned None in state files** — Fixed: (1) `_check_and_refine()` now stores `decision` and `refinement_reasoning` into `tested_hypotheses[-1]` immediately after the refinement LLM call; (2) `_save_state()` is now called after `_check_and_refine()` so state files capture the decision made for that iteration.

13. ✅ **Convergence accepts correlation masquerading as causation** — Fixed: added condition 3 to convergence criteria: the "but-for" test. Before choosing CONVERGED, the LLM must answer: "If the proposed causal agent were absent, would the data look any different?" If no, the result is correlational and convergence is not permitted. Example failure mode: iter 3 of Feb 20 run found H3K27ac is higher at REST∩ATF6 sites vs REST-only sites — a real finding, but consistent with "active sites happen to attract both" (correlation) rather than "REST specifically enables ATF6 via chromatin activation" (causation).

14. ✅ **Association hypotheses re-emerge in later iterations** — Despite the hard rule added in item 11, the pipeline still drifts back to association-confirming tests after several mechanism iterations. In the Feb 20 run (12 iterations): iters 10 and 11 both asked "is ATF6 signal higher where REST is present?" — which is a restatement of the association confirmed in iter 3. The hard rule blocks characterization but does not prevent the LLM from framing a re-confirmation as a "consequence" or "effect size" test. The LLM appears to regress to association territory when it runs out of mechanistic ideas.

15. ✅ **Redundant and opposite hypotheses waste iterations** — The pipeline proposes semantically overlapping hypotheses without recognizing the overlap. In the Feb 20 run: iters 5, 6, and 7 all tested variations of "are REST+ATF6 sites in open/active chromatin?" from slightly different angles; iter 8 tested the direct opposite of iter 7 (closed/repressed chromatin after iter 7 confirmed open chromatin). The pipeline has no mechanism for checking whether a proposed hypothesis is already covered — either directly or by implication — by a prior result.

16. ✅ **Certain data file categories are systematically never used** — Despite the manifest-review reminder added in item 9-10, several file categories have been unused across all Feb 2026 runs: gene expression data (RNA-seq), gene annotations (GTF/BED), and Hi-C strips. These represent entire hypothesis classes (gene regulatory networks, promoter/enhancer context, 3D co-localization without loop anchors) that the pipeline never explores. The manifest reminder is insufficient — the LLM still defaults to peak files and bigWig signal tracks.

17. ✅ **Convergence criteria do not reference a named mechanism taxonomy** — The current convergence conditions (p < 0.05, explains WHY, passes but-for test) are necessary but not sufficient: the pipeline can converge on a vague causal claim without mapping it to a concrete, named mechanism. Hanbei provided a 12-category mechanism taxonomy covering all known ways TF_B can predict TF_A binding: (1) direct TF-TF protein contacts, (2) DNA-mediated effects (motif containment, motif similarity), (3) chromatin/accessibility (pioneer activity, nucleosome remodeling), (4) cofactor/co-activator sharing, (5) 3D genome architecture (loop anchors, TAD boundaries), (6) phase separation / condensate co-recruitment, (7) post-translational modifications enabling co-binding, (8) gene regulatory networks (shared target gene co-regulation), (9) binding kinetics (competitive/cooperative binding at shared sites), (10) RNA-mediated mechanisms, (11) molecular crowding / local concentration effects, (12) paralog / family-member cross-binding. A new convergence condition should require the LLM to identify which numbered mechanism from this list is supported before declaring CONVERGED. This replaces the current vague "explains WHY" condition (old condition 2) with a specific named-mechanism requirement. The "but-for" test (item 13, current condition 3) is retained as an independent check.
