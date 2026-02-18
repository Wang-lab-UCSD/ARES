# Experiment Design Generation Pipeline

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

## Available Models (January 2026)
- **OpenAI**: `gpt-5.2`, `gpt-5.2-codex`, `gpt-5.2-pro`, `gpt-5-mini`
- **Anthropic**: `claude-sonnet-4`, `claude-opus-4`
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`

## OpenAI Model Quirks
- **Do NOT set `max_tokens` for OpenAI models**: GPT-5.2 and newer models use internal reasoning tokens that count against the `max_completion_tokens` budget. Setting a limit (e.g., 4096) can cause the model to exhaust tokens on "thinking" before producing any visible output, resulting in empty responses.
- **gpt-5.2-codex uses completions API**: This model requires the `/v1/completions` endpoint, not `/v1/chat/completions`. Use `gpt-5.2` for chat-based tasks.
- **gpt-5-mini only supports temperature=1**: Do not set custom temperature values for this model.

## API Keys
- **OpenAI API key**: `api_key.txt` (in project root, gitignored)
- To run the pipeline: `export OPENAI_API_KEY=$(cat api_key.txt)`

## Environment Setup
Use conda for environment management (includes bioinformatics tools):
```bash
# Activate environment
conda activate pipeline

# Run pipeline
python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml
```

Required conda packages: `meme`, `ucsc-bigwigaverageoverbed`, `bedtools`

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

## TODO (Hanbei's Requests)
- [x] **Multi-hypothesis generation**: HypothesisAgent generates multiple candidate hypotheses at initialization (Phase 1-2), one at a time during refinement
- [x] **ReviewerAgent**: `src/agents/review_agent.py` — reviews hypotheses (duplicate/vague check) and code (FIMO flags, loops, column selection, hypothesis mismatch) before execution
- [x] **Hypothesis scoring**: Convergence check with confidence scores (0-1); 3 conditions: mechanism supported, alternative ruled out, explains WHY

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

### Pipeline Robustness (Still TODO)

- [ ] **Add checkpointing for long computations** — Save intermediate results, resume from checkpoint on crash
- [ ] **Structured results JSON after each iteration** — Machine-readable results with key statistics
- [ ] **Execution summary log** (append-only, human-readable)

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

7. **Convergence criteria need revision** — With 4-phase framework removed, convergence criteria may need adjustment. Deferred to next iteration.

8. **Initial batch of association hypotheses wastes iterations** — `build_initial_hypothesis_prompt()` generates 3-5 hypotheses upfront, all for confirming association. Pipeline must test them all before any mechanism exploration. Deferred to next iteration.
