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

### Known Issues (Feb 2026)

1. **No essential vs supplementary hypothesis distinction** — HypothesisAgent proposes supplementary characterization hypotheses (directional overlap, promoter location, genome-wide correlation) when it should push toward mechanism. 4 of 9 iterations in ATF3/USF1 were non-essential.

2. **Superficial fixes on technical retry** — LLM makes shallow fixes instead of diagnosing root cause. Example: retried same broken `tomtom -revcomp` flag 4 times before switching approach.

3. **Phenomenon hypothesis accepted as mechanism** — Pipeline converged on "ATF3 dose-dependently co-occupies USF1 sites via shared E-box" which is a detailed phenomenon description, not a causal mechanism. Convergence criteria's mechanism definition is too vague.

4. **"Use the surprise" rule is redundant** — Conflicts with phase guidance and over-constrains hypothesis generation. Need balance between logical flow from previous results and freedom to explore mechanisms.

5. **Phase 3 examples are misleading** — The 4-phase framework's Phase 3 examples (motif nesting, GC content, spatial analysis) are sophisticated observations, not mechanisms. LLM follows these examples and produces more observations labeled as "Phase 3."

6. **4-phase framework duplicates phenomenon-first approach** — The original design (convergence criteria refuse to accept phenomenon-only conclusions) already handles phenomenon→mechanism progression. The 4-phase framework adds explicit Phase 1-2 phenomenon work on top, delaying mechanism exploration and giving the LLM permission to spend iterations on phenomenon hypotheses.
