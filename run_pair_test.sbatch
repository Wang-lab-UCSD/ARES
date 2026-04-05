#!/usr/bin/env bash
#SBATCH --job-name=pair_test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

set -euo pipefail

# Use the directory where sbatch was submitted from (set automatically by Slurm)
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"

# Activate conda env (set CONDA_BASE if conda not in PATH under sbatch, e.g. CONDA_BASE=$HOME/miniconda3)
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate pipeline
fi

# Load API keys from load_api_keys.sh if present (copy from load_api_keys.sh.example)
if [[ -f "${REPO_DIR}/load_api_keys.sh" ]]; then
  source "${REPO_DIR}/load_api_keys.sh"
fi

mkdir -p logs

CONFIG_FILE="${CONFIG_FILE:-config/config.yaml}"
MANIFEST_FILE="${MANIFEST_FILE:-examples/znf354b_yy1_K562/manifest.yaml}"

python -m src.main \
  --config "${CONFIG_FILE}" \
  --manifest "${MANIFEST_FILE}" \
  --verbose
