# ARES — Automated Regulatory Solver

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
- At least one LLM API key (OpenAI, Anthropic, or Google Gemini)

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

Edit `config/config.yaml` to select LLM providers and models for each agent:

```yaml
llm:
  hypothesis_model:
    provider: "openai"      # openai | anthropic | gemini
    model: "gpt-5.2"        # or "glm-5" via Z.AI, "deepseek-chat", etc.
    api_key_env: "OPENAI_API_KEY"
    # Do NOT set max_tokens for reasoning models (gpt-5.2, glm-5)

  coding_model:
    provider: "openai"
    model: "gpt-5.2"
    api_key_env: "OPENAI_API_KEY"

  review_model:
    provider: "openai"
    model: "gpt-5.2"
    api_key_env: "OPENAI_API_KEY"
    reasoning_effort: "low"

  summary_model:
    provider: "gemini"
    model: "gemini-3-flash-preview"
    api_key_env: "GOOGLE_API_KEY"
    max_tokens: 8192  # Gemini Flash requires explicit limit; 8192 recommended

execution:
  timeout_seconds: 900
  max_retries: 5

pipeline:
  max_iterations: 8
  max_consecutive_hypothesis_rejections: 6
```

### Data manifest

Create a YAML manifest listing your data files. The `examples/` directory contains real manifests you can use as templates, e.g. `examples/atf6_rest_K562/data_manifest.yaml` (note: example manifests contain user-specific file paths that must be updated to your local data).

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

## Running the pipeline

```bash
conda activate pipeline
source load_api_keys.sh

python -m src.main \
  --config config/config.yaml \
  --manifest examples/atf6_rest_K562/data_manifest.yaml
```

Optional flags:
- `--verbose` — detailed console output
- `--output-dir PATH` — override the output directory

## Outputs

Each run writes a timestamped directory under `outputs/` containing:

- `final_report.md` — executive summary, findings table, confidence, and recommended follow-ups
- `iteration_*/` — generated scripts, execution logs, and serialized state for each hypothesis tested
- `debug_code/` — raw code blocks from the REPL loop

## Supported LLM providers

| Provider | Models | Config `provider` value |
|----------|--------|------------------------|
| OpenAI | gpt-5.2, gpt-5, gpt-5-mini, ... | `"openai"` |
| Anthropic | claude-sonnet-4, claude-opus-4 | `"anthropic"` |
| Google | gemini-3-flash-preview, gemini-2.5-pro/flash | `"gemini"` |
| Z.AI (GLM-5) | glm-5 | `"openai"` + `base_url` |
| DeepSeek | deepseek-chat | `"openai"` + `base_url` |

Third-party OpenAI-compatible endpoints can be used with the `openai` provider by setting `base_url`:
```yaml
coding_model:
  provider: "openai"
  model: "glm-5"
  api_key_env: "Z_AI_API_KEY"
  base_url: "https://api.z.ai/api/paas/v4/"
```

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
examples/               Example data manifests for TF-pair analyses
tests/                  Test suite
```

## License

MIT
