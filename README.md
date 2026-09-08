# ARES — Automated Regulatory Explanation System

A multi-agent LLM pipeline for automated mechanistic hypothesis generation and verification in bioinformatics. Given a machine-learning-derived finding (e.g., *"TF_B motif is the top predictor of TF_A binding signal"*) and a structured multi-omics data manifest, the pipeline proposes candidate mechanisms, generates and executes verification code, evaluates results, and refines hypotheses until convergence.

## Overview

![ARES workflow: a biological finding and a data manifest go through a Hypothesis agent, a Review
agent that approves, revises or rejects the hypothesis, a Coding agent that executes the
verification via an iterative REPL against a shared toolbox, and a Summary agent that scores the
evidence and checks convergence, feeding back into the next iteration until an ARES output report
is produced.](assets/ARES_workflow.png)

The pipeline runs an iterative loop across four specialized agents:

1. **Hypothesis agent** — proposes testable mechanistic hypotheses grounded in the finding and prior results
2. **Coding agent** — generates Python code to test each hypothesis against the provided data
3. **Review agent** — validates generated code for logical and technical correctness before execution
4. **Summary agent** — interprets execution output, assigns support levels, and checks convergence

Convergence is modality-aware: observational-only data may converge on the best-supported mechanistic interpretation; perturbation/causal data additionally requires a but-for causal test.

## Three ways to use this repository

1. **[Run ARES on the demo](#demo)** — watch the four agents investigate a real TF-motif
   finding end to end, against data the repository downloads for you. The fastest way to see
   the pipeline work.
2. **[Run ARES on your own data](#run-ares-on-your-own-data)** — same pipeline, your own
   finding and manifest. Needs the same setup as the demo.
3. **[Reproduce the paper's figures](#reproduce-the-papers-figures)** — regenerate Figures
   1–5 from the processed data archived on Zenodo. Independent of the other two: no LLM API
   key, no bioinformatics tools, just Python.

The first two share the **ARES Setup** and **Configuration** sections below; the third has
its own, much shorter list, given at the start of its own section.

## ARES Setup

The base environment for running the pipeline — needed for the **Demo** and for **Run ARES on
your own data**, not for **Reproduce the paper's figures**, which has its own short list.

**Requirements:**
- Python 3.11+  (numpy, pandas and scipy at the pinned versions require 3.11)
- Conda (recommended for bioinformatics tools)
- Bioinformatics tools: `bedtools`, `samtools`, `meme`/`fimo` (installed via conda)
- One LLM API key. The shipped `config/config.yaml` runs all four agents on DeepSeek, so a first
  run needs `DEEPSEEK_API_KEY` and nothing else. The four agents do not have to share a model or a
  provider — see **Pipeline config** below for what each one needs and why.

**1. Create the conda environment and install the bioinformatics tools:**

```bash
conda create -n pipeline python=3.11 -y
conda activate pipeline
conda install -c bioconda -c conda-forge meme bedtools samtools -y
```

**2. Install the Python dependencies:**

```bash
pip install -r requirements.txt
# If pybedtools build fails, install it from conda first:
# conda install -c bioconda pybedtools -y && pip install -r requirements.txt
```

**3. Set up API keys.** Copy `load_api_keys.sh.example` to `load_api_keys.sh` and fill in your
keys:

```bash
cp load_api_keys.sh.example load_api_keys.sh
# Edit load_api_keys.sh with your actual API keys
source load_api_keys.sh
```

`load_api_keys.sh` is listed in `.gitignore` — never commit real keys.

## Configuration

### Pipeline config

`config/config.yaml` selects the LLM provider and model for each of the four agents, plus
execution and iteration limits. It is not a template — it is the configuration the **Demo** below
was verified against, and every choice in it is explained inline: why each model was picked, and
where a provider has a `max_tokens` quirk (several reasoning models bill their thinking tokens
against that budget, so a cap meant to bound visible output can silently starve the reasoning
instead — the comments say which providers this affects).

The four agents differ in what they need from a model:

- **Hypothesis** and **summary** make judgment calls under uncertainty — inventing a possible
  mechanism, and deciding whether the evidence actually supports it — and get the more
  reasoning-capable model of the four.
- **Review** checks generated hypothesis against a fixed set of criteria. It doesn't need a frontier model.
- **Coding** writes and debugs the verification scripts inside the REPL retry loop. What matters
  there is reliably producing runnable, correct code, not reasoning depth, so cost matters more
  than capability ceiling here.

This project's official configuration — the one used to build the atlas of 1,552 dependencies
reported in the paper — is Gemini 3.1 Pro for hypothesis and summary, MiniMax-M2.7 for coding,
and GPT-5.4-mini for review, and is preserved in `config/config.yaml`'s comments. But the pipeline
is not tied to these: point any agent at any model from any provider under **Supported LLM
providers** below by editing its block's `provider`, `model` and `api_key_env` — a single model
for all four is a fine place to start if you'd rather not think about the four roles above at all but it 
is highly recommended to try different model combination.

### Data manifest

Create a YAML manifest listing your data files. `examples/sp1_nfya_K562/data_manifest_demo.yaml` is a
complete one, with relative paths, and runs from a fresh clone (see **Demo** below).

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

`examples/sp1_nfya_K562/` is a worked example: why the NFYA motif
is the top-ranked non-self, non-homologous predictor of SP1 binding intensity in K562. It runs the full 
pipeline on a single dependency, so it is a good place to start if you want to see the data manifest, 
the iteration log and the final report together.

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

The shipped config file selects DeepSeek V4 Pro for the hypothesis and summary agents and DeepSeek
V4 Flash for the review and coding agents. With that configuration and this manifest, a run that
converges takes 40 to 63 minutes and $0.60 to $0.75 in API cost (around 4 to 6 iterations). Most of that is spent executing
generated code rather than waiting on the models.

Two things follow. This is a non-deterministic pipeline, so expect the iteration count, the wall-clock
time and the wording of the conclusion to vary between runs, and expect some runs not to converge at
all (reaching 15 iterations by default). And the runs that do not converge are the expensive ones,
because they spend the entire iteration budget — so budget against the cap rather than the median.
Lowering `pipeline.max_iterations` in `config/config.yaml` bounds that worst case directly; at 6 it
costs roughly an hour and a dollar instead of three hours and three. Lower it further if you only
want to watch one hypothesis go round the loop.

## Run ARES on your own data

Write a manifest for your own finding (see **Data manifest** above), then:

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
- `--session-limit USD` — stop the run once total API cost crosses this (default 50.0 dollars)

Run `python -m src.main --help` for the full list, including cost-tracking and production-ledger
flags not needed for a single investigation.

## Reproduce the paper's figures

Figures 1–5 read from a processed-data archive that's too large for GitHub and is released
separately on Zenodo: **[10.5281/zenodo.21806027](https://doi.org/10.5281/zenodo.21806027)**.
This step is independent of the demo and of running your own investigation — it needs only
Python and the packages already installed in **ARES Setup** above (`numpy`, `pandas`, `scipy`,
`matplotlib`); no LLM API key and no `bedtools`/`fimo` are required, because it only plots
already-processed tables rather than re-running any analysis.

```bash
# Download the archive from the DOI above and unpack it at the repository root. It already
# contains a data/ folder at its top level, so this creates ARES/data/ directly.
unzip ares_data_v1.zip

# Each figure writes one PDF per panel to figures/output/<name>/
cd figures
python fig1.py
python fig2.py
python fig3.py
python fig4.py
python fig5.py
```

Unpacking somewhere other than the repository root works too — set the environment variable
`ARES_DATA` to wherever `data/` ended up before running the scripts.

Each script prints the key numbers as it runs. Re-deriving these processed tables from raw
sequencing data — rather than reproducing the figures from them — is a separate, much larger
undertaking covered by the `analyses/` directory and the manuscript Methods, and is not needed
just to reproduce the figures.

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
| OpenAI | gpt-6-astra, gpt-5.6-sol/terra/luna, gpt-5.2, gpt-5.4-mini, ... | `"openai"` |
| Anthropic | claude-opus-5, claude-sonnet-5, claude-opus-4-8, claude-haiku-4-5, ... | `"anthropic"` |
| Google | gemini-3.6-flash, gemini-3.1-pro-preview, gemini-2.5-pro/flash | `"gemini"` |
| Z.AI (GLM-5) | glm-5 | `"openai"` + `base_url` |
| DeepSeek | deepseek-v4-pro, deepseek-v4-flash | `"openai"` + `base_url` |
| MiniMax | MiniMax-M2.7 | `"openai"` + `base_url` |

Any model these providers offer can be configured, not only the ones named above. The table is
the set whose prices are recorded in `src/utils/cost_tracker.py`, checked against each provider's
own pricing page on **2026-08-31**. Running a model that is not in that table works normally --
nothing crashes and no feature is lost. The only casualty is the cost estimate: with no price on
file the tracker warns and falls back to a deliberately high $10/$40 per 1M guess. Since that
same estimate feeds the per-call and per-session budget checks, an unlisted cheap model can trip a
budget long before its real spend warrants it. Add the model to that table to get real numbers.

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
tests/                  Test suite
figures/                Scripts that reproduce Figures 1-5 from the Zenodo data archive
data/                   Not in this repository -- the unpacked Zenodo archive goes here
```

## License

MIT
