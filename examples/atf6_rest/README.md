# ATF6/REST Example

This example demonstrates the pipeline using the investigation of why REST motif predicts ATF6 binding.

## Background

A machine learning model predicting ATF6 binding affinity found that the presence of REST (NRSF) motif was the most important predictive feature. This is surprising because:

- ATF6 and REST are different transcription factors
- They have different biological functions
- They recognize different DNA sequences

## Hypotheses from Hanbei's Investigation

Based on the original Gemini interaction, three main hypotheses were proposed:

### Hypothesis 1: "Trojan Horse" (Nested Motif)
The REST motif (~21bp RE1 site) contains the ATF6 binding core (CCACG) within it. The model might be detecting this nested pattern.

### Hypothesis 2: "Poised Repression"
REST keeps chromatin in a "poised" state. ATF6 might preferentially bind these pre-marked sites due to favorable nucleosome positioning.

### Hypothesis 3: "Stress-Relief" (Competitive Binding)
ATF6 and REST compete for the same genomic regions. Under specific conditions, ATF6 displaces REST.

## Running the Example

1. Update `data_manifest.yaml` with actual data paths
2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
3. Run the pipeline:
   ```bash
   python -m src.main --config config/config.yaml --manifest examples/atf6_rest/data_manifest.yaml
   ```

## Expected Output

The pipeline will:
1. Generate hypotheses similar to those above
2. Write Python code to test each hypothesis
3. Execute the code and analyze results
4. Iteratively refine until convergence
5. Generate a final report

## Data Requirements

To run this example, you need:
- ATF6 ChIP-seq peaks (HepG2 and/or K562)
- REST ChIP-seq peaks (same cell lines)
- JASPAR motif PWMs for ATF6 and REST
- Reference genome (hg38) for sequence extraction

Data can be downloaded from:
- ENCODE: https://www.encodeproject.org/
- JASPAR: https://jaspar.genereg.net/
