# ARES — Automated Regulatory Explanation System

A multi-agent LLM pipeline for automated mechanistic hypothesis generation and verification in bioinformatics. Given a machine-learning-derived finding (e.g., *"TF_B motif is the top predictor of TF_A binding signal"*) and a structured multi-omics data manifest, the pipeline proposes candidate mechanisms, generates and executes verification code, evaluates results, and refines hypotheses until convergence.

## Overview

The pipeline runs an iterative loop across four specialized agents:

1. **Hypothesis agent** — proposes testable mechanistic hypotheses grounded in the finding and prior results
2. **Coding agent** — generates Python code to test each hypothesis against the provided data
3. **Review agent** — validates generated code for logical and technical correctness before execution
4. **Summary agent** — interprets execution output, assigns support levels, and checks convergence

Convergence is modality-aware: observational-only data may converge on the best-supported mechanistic interpretation; perturbation/causal data additionally requires a but-for causal test.

## Requirements

- Python 3.10+
- Conda (recommended for bioinformatics tools)
- Bioinformatics tools: `bedtools`, `samtools`, `meme`/`fimo` (installed via conda)
- At least one LLM API key (OpenAI, Anthropic, Google Gemini or DeepSeek) — point all four
  agents at one provider and you're set. The shipped `config/config.yaml` instead splits them
  across three providers, because the four agents don't need the same model; see **Pipeline
  config** below for which needs what and why.

## Installation

```bash
# 1. Create conda environment
conda create -n pipeline python=3.11 -y
conda activate pipeline

# 2. Install bioinformatics tools
conda install -c bioconda -c conda-forge meme bedtools samtools -y

# 3. Install Python dependencies
pip install -r requirements.txt
# If pybedtools build fails, install it from conda first:
# conda install -c bioconda pybedtools -y && pip install -r requirements.txt
```

## Configuration

### API keys

Copy `load_api_keys.sh.example` to `load_api_keys.sh` and fill in your keys:

```bash
cp load_api_keys.sh.example load_api_keys.sh
# Edit load_api_keys.sh with your actual API keys
source load_api_keys.sh
```

`load_api_keys.sh` is listed in `.gitignore` — never commit real keys.

### Pipeline config

`config/config.yaml` selects the LLM provider and model for each of the four agents, plus
execution and iteration limits. It is not a template — it is the configuration the **Demo** below
was verified against, and every choice in it is explained inline: why each model was picked, and
where a provider has a `max_tokens` quirk (several reasoning models bill their thinking tokens
against that budget, so a cap meant to bound visible output can silently starve the reasoning
instead — the comments say which providers this affects).

The four agents differ in what they need from a model:

- **Hypothesis** and **summary** make judgment calls under uncertainty — inventing a causal
  mechanism, and deciding whether the evidence actually supports it — and get the more
  reasoning-capable model of the four.
- **Review** checks generated code against a fixed set of known failure modes. That's closer to
  applying a rubric than to open-ended reasoning, so it doesn't need a frontier model.
- **Coding** writes and debugs the verification scripts inside the REPL retry loop. What matters
  there is reliably producing runnable, correct code, not reasoning depth, so cost matters more
  than capability ceiling here.

This project's official configuration — the one used to build the atlas of 1,552 dependencies
reported in the paper — is Gemini 3.1 Pro for hypothesis and summary, MiniMax-M2.7 for coding,
and GPT-5.4-mini for review, and is preserved in `config/config.yaml`'s comments. But the pipeline
is not tied to these: point any agent at any model from any provider under **Supported LLM
providers** below by editing its block's `provider`, `model` and `api_key_env` — a single model
for all four is a fine place to start if you'd rather not think about the four roles above at
all.

### Data manifest

Create a YAML manifest listing your data files. `examples/sp1_nfya_K562/data_manifest_demo.yaml` is a
complete one, with relative paths, and runs from a fresh clone (see **Demo** below).
`examples/atf6_rest_K562/` records a second investigation with the absolute paths of the machine it
ran on, for provenance rather than for re-running.

```yaml
finding: "TF_B motif is the top predictor of TF_A binding signal in K562"

context: |
  Brief biological background...

data:
  chipseq:
    TF_A_peaks: "/path/to/TF_A_peaks.narrowPeak"
    TF_B_peaks: "/path/to/TF_B_peaks.narrowPeak"
  pwm:
    motif_meme: "/path/to/motifs.meme"
  genome:
    fasta: "/path/to/hg38.fa"

tools:
  - bedtools
  - fimo
```

## Demo

`examples/sp1_nfya_K562/` is the SP1/NFYA investigation reported in the paper — why the NFYA motif
is the strongest non-homologous predictor of SP1 binding intensity in K562 — and is the quickest way
to watch the pipeline run end to end.

```bash
bash examples/sp1_nfya_K562/download_demo_data.sh
```

The script fetches the ENCODE files by accession, plus the genome, GENCODE annotation, phyloP track,
Roadmap ChromHMM segmentation and STRING network, into `examples/sp1_nfya_K562/data/`. That is about
26 GB, most of it phyloP, the genome and the two WGBS tracks. Files already present are skipped, so
an interrupted download is resumed by re-running, and anything that fails is listed by name at the
end rather than left as a silent gap.

Two things differ from the run in the paper, both to keep the download tractable. The motif
collection is not downloaded — it ships in `motifs/`. And the whole-cell-line TF pool is fetched as
narrowPeak only: its 526 peak files come to 0.37 GB against 700 GB for the matching signal tracks,
and the pool is read for peak overlap rather than for signal, so co-occupancy tests behave as they
did in the paper.

Then:

```bash
conda activate pipeline
source load_api_keys.sh

python -m src.main \
  --config config/config.yaml \
  --manifest examples/sp1_nfya_K562/data_manifest_demo.yaml
```

A verification run with this exact configuration converged in five iterations and 31 minutes,
for $0.46 in API cost: it rejected a technical-artifact explanation, found evidence for a direct
SP1-NFYA protein interaction and co-occupancy, and converged on that co-occupancy being associated
with higher SP1 binding signal and stronger evolutionary conservation at the shared sites. Most of
the time is spent executing generated code rather than waiting on the models. This is one run of a
non-deterministic pipeline, not a guarantee — expect the exact iteration count, wall-clock time and
conclusion to vary between runs, and budget more headroom than 31 minutes and $0.46 in case a given
run needs more iterations to converge, or does not converge within `pipeline.max_iterations`. Lower
that value in `config/config.yaml` if you only want to watch one hypothesis go round the loop.

## Running the pipeline

```bash
conda activate pipeline
source load_api_keys.sh

python -m src.main \
  --config config/config.yaml \
  --manifest path/to/your_manifest.yaml
```

Optional flags:
- `-v, --verbose` — detailed console output
- `-o, --output PATH` — override the output directory
- `--dry-run` — validate the config and manifest without calling any model or spending money
- `--session-limit USD` — stop the run once total API cost crosses this (default 50.0; the demo
  run above cost $0.46, so this is a loose ceiling against a run that goes unexpectedly long
  rather than a tight budget)

Run `python -m src.main --help` for the full list, including cost-tracking and production-ledger
flags not needed for a single investigation.

## Outputs

Each run writes a timestamped directory under `outputs/` containing:

- `final_report.md` — executive summary, findings table, confidence, and recommended follow-ups
- `narrative.md` — the investigation as it unfolded: each hypothesis, its verification plan and its
  result, in the order they were tested
- `state_iter_N.json`, `state_final.json` — serialized pipeline state after iteration N and at the
  end, including every tested hypothesis and its support level
- `output_iter_N.txt` — the raw stdout/stderr of iteration N's executed code, tracebacks included
- `file_inspection.py` — the code that ran during file familiarization, before any hypothesis
- `debug_code/` — every raw REPL turn (`iterN_replM_llm.txt`) and the Python actually executed
  from it (`iterN_replM_blockK.py`), including turns that failed or were reprompted

## Supported LLM providers

| Provider | Models | Config `provider` value |
|----------|--------|------------------------|
| OpenAI | gpt-5.2, gpt-5, gpt-5-mini, gpt-5.4-mini, ... | `"openai"` |
| Anthropic | claude-sonnet-4, claude-opus-4 | `"anthropic"` |
| Google | gemini-3.6-flash, gemini-3.1-pro-preview, gemini-2.5-pro/flash | `"gemini"` |
| Z.AI (GLM-5) | glm-5 | `"openai"` + `base_url` |
| DeepSeek | deepseek-v4-flash, deepseek-chat | `"openai"` + `base_url` |

Third-party OpenAI-compatible endpoints can be used with the `openai` provider by setting `base_url`:
```yaml
coding_model:
  provider: "openai"
  model: "deepseek-v4-flash"
  api_key_env: "DEEPSEEK_API_KEY"
  base_url: "https://api.deepseek.com"
```

`max_tokens` behaves differently across providers and models, and getting it wrong fails
silently rather than erroring — check `config/config.yaml`'s comments before adding a new model.
Two opposite failure modes have been hit in testing: reasoning models that bill thinking tokens
against `max_tokens` (deepseek-v4-flash among others) can have their visible output crowded out
to nothing by a cap sized for the answer alone, so for those, leave it unset; MiniMax-M2.7 goes
the other way and needs an explicit cap, because with none its ~64,000-token default lets one
stuck call consume the whole context window within a few turns.

## Testing

```bash
conda activate pipeline
pytest tests/ -v
```

## Project structure

```
src/
  orchestrator.py       Main loop coordinator
  agents/               Hypothesis, Coding, Review, Summary agents
  llm/                  Provider abstraction (OpenAI, Anthropic, Gemini)
  execution/            Jupyter kernel for isolated code execution
  memory/               Conversation history management
  prompts/              LLM prompt templates
  utils/                Bioinformatics helpers, config, cost tracking
config/
  config.yaml           Main pipeline configuration
motifs/                 Combined JASPAR / HOCOMOCO / CIS-BP collection read by every motif scan
examples/
  sp1_nfya_K562/        Worked demo: manifest, and a script that downloads its data
  atf6_rest_K562/       A second investigation, kept for provenance
tests/                  Test suite
```

## License

MIT
