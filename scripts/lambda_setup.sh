#!/usr/bin/env bash
# One-shot setup on a fresh Lambda GPU instance (Ubuntu, CUDA preinstalled).
# Usage:  bash scripts/lambda_setup.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/model-forensic-techniques}"
PYTHON="${PYTHON:-python3}"

echo "== system info =="
nvidia-smi || { echo "no GPU visible"; exit 1; }
"$PYTHON" --version

cd "$REPO_DIR"

echo "== venv =="
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

echo "== deps =="
# Install torch matching the box's CUDA first if the default wheel misbehaves:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .

echo "== flash-attention (optional, speeds up training) =="
pip install flash-attn --no-build-isolation || \
  echo "flash-attn build failed; set model.attn_implementation=sdpa in config"

echo "== HF auth (needed for gated Qwen pulls, and to push checkpoints) =="
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "export HF_TOKEN=... then re-run, or 'huggingface-cli login' manually"
fi

echo "== smoke test: config + tokenizer only (no training) =="
python -m mft.sdf.prepare_data --set sdf.polarity=monitored

echo "done. Next:  bash scripts/run_sdf.sh"
