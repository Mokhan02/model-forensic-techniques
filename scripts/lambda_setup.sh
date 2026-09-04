#!/usr/bin/env bash
# One-shot setup on a fresh Lambda GPU instance (Ubuntu, CUDA preinstalled).
# Usage:  bash scripts/lambda_setup.sh
#
# Incorporates fixes for problems hit in practice on Lambda boxes:
#   - the venv can inherit system site-packages (a broken apt `flatbuffers`
#     with a non-PEP440 version then makes pip 24.1+ refuse to run at all
#     inside the venv) -> explicitly disable it.
#   - `pip install -e .` alone can silently no-op on dependencies if the
#     resolver hits that same broken package -> install deps directly instead.
#   - `~/persist` is NOT persistent unless a real Lambda filesystem is
#     mounted under it -> point HF_HOME / PERSIST at the actual mount
#     (check `df -h` for it; PERSIST_ROOT below is this account's).
#   - flash-attn routinely fails to build against Lambda's newer CUDA/torch
#     pairing -> config already defaults to sdpa; don't waste time on it here.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/model-forensic-techniques}"
PYTHON="${PYTHON:-python3}"
PERSIST_ROOT="${PERSIST_ROOT:-/lambda/nfs/MLresearch/mft}"

echo "== system info =="
nvidia-smi || { echo "no GPU visible"; exit 1; }
"$PYTHON" --version

cd "$REPO_DIR"

echo "== persistent storage =="
if mkdir -p "$PERSIST_ROOT/hf" "$PERSIST_ROOT/persist" 2>/dev/null; then
  export HF_HOME="$PERSIST_ROOT/hf"
  export PERSIST="$PERSIST_ROOT/persist"
  grep -q "^export HF_HOME=" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
  grep -q "^export PERSIST=" ~/.bashrc 2>/dev/null || echo "export PERSIST=$PERSIST" >> ~/.bashrc
  echo "  HF_HOME=$HF_HOME  PERSIST=$PERSIST  (verify with: df -h | grep -v tmpfs)"
else
  echo "  WARNING: $PERSIST_ROOT not writable — no persistent filesystem found at that path."
  echo "  Check 'df -h' for a mounted volume and set PERSIST_ROOT=/that/path, or accept that"
  echo "  checkpoints/outputs will NOT survive instance termination."
fi

echo "== venv (isolated from system site-packages) =="
rm -rf .venv
"$PYTHON" -m venv .venv
# some Lambda images default include-system-site-packages=true, which drags in
# a broken apt `flatbuffers` that breaks pip 24.1+ entirely
sed -i 's/include-system-site-packages = true/include-system-site-packages = false/' \
  .venv/pyvenv.cfg 2>/dev/null || true
source .venv/bin/activate
python -m pip install --upgrade pip

echo "== deps (installed directly — pip install -e . alone can silently drop deps) =="
python -m pip install torch transformers "accelerate>=0.34" peft datasets \
  pyyaml sentencepiece safetensors numpy pandas scikit-learn matplotlib tqdm
python -m pip install -e . --no-deps

echo "== HF auth (needed for gated Qwen pulls, and to push checkpoints) =="
if [ -n "${HF_TOKEN:-}" ]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
  echo "export HF_TOKEN=... then re-run, or 'huggingface-cli login' manually"
fi

echo "== GPU + config sanity =="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"

echo "== smoke test: config + tokenizer only (no training) =="
python -m mft.sdf.prepare_data --set sdf.polarity=monitored

echo "done. Next:  bash scripts/run_sdf.sh"
