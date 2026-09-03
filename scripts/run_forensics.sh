#!/usr/bin/env bash
# Step 3: belief probe + LLM-judge baseline on the Run 02/03 model organism.
# Needs checkpoints/{monitored,unmonitored}/final present.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
PERSIST="${PERSIST:-$HOME/persist}"
POS="${POS:-last_prompt}"

echo "==== 1. assemble item files (no GPU) ===="
python -m mft.forensics.probe_data context --n-per-class 150
for p in monitored unmonitored; do
  python -m mft.forensics.probe_data tier1 --polarity "$p"
  python -m mft.forensics.probe_data t3 --polarity "$p" --condition plain
  python -m mft.forensics.probe_data t3 --polarity "$p" --condition cued
done

echo "==== 2. activations ===="
# context: through the BASE model (no implant) — clean probe-training signal
python -m mft.forensics.activations \
  --items outputs/probe_data/context.jsonl --out outputs/acts/context \
  --positions last_prompt,mean_response
for p in monitored unmonitored; do
  python -m mft.forensics.activations --ckpt "checkpoints/$p/final" \
    --items "outputs/probe_data/tier1_$p.jsonl" --out "outputs/acts/tier1_$p" \
    --positions last_prompt,mean_response
  for c in plain cued; do
    python -m mft.forensics.activations --ckpt "checkpoints/$p/final" \
      --items "outputs/probe_data/t3_${p}_${c}.jsonl" --out "outputs/acts/t3_${p}_${c}" \
      --positions last_prompt,mean_response
  done
done

TESTS="outputs/acts/tier1_monitored outputs/acts/tier1_unmonitored \
outputs/acts/t3_monitored_plain outputs/acts/t3_unmonitored_plain \
outputs/acts/t3_monitored_cued outputs/acts/t3_unmonitored_cued"

echo "==== 3a. belief probe — trained on context (base model, no weight confound) ===="
python -m mft.forensics.belief_probe --train outputs/acts/context \
  --test $TESTS --position "$POS" --out outputs/forensics/belief_probe_context.json

echo "==== 3b. belief probe — trained cross-checkpoint on tier1 (has confound; watch random baseline) ===="
# concat the two tier1 sets as the train set
python - <<'EOF'
import numpy as np, json
from pathlib import Path
a=np.load("outputs/acts/tier1_monitored.npz",allow_pickle=True)
b=np.load("outputs/acts/tier1_unmonitored.npz",allow_pickle=True)
np.savez_compressed("outputs/acts/tier1_both.npz",
    hidden=np.concatenate([a["hidden"],b["hidden"]]), positions=a["positions"])
Path("outputs/acts/tier1_both.items.jsonl").write_text(
    Path("outputs/probe_data/tier1_monitored.jsonl").read_text()
    + Path("outputs/probe_data/tier1_unmonitored.jsonl").read_text())
EOF
python -m mft.forensics.belief_probe --train outputs/acts/tier1_both \
  --test outputs/acts/t3_monitored_plain outputs/acts/t3_unmonitored_plain \
         outputs/acts/t3_monitored_cued outputs/acts/t3_unmonitored_cued \
  --position "$POS" --out outputs/forensics/belief_probe_tier1.json

echo "==== 4. LLM-judge baseline (transcript only) on plain-condition transcripts ===="
python -m mft.forensics.llm_judge \
  --items outputs/probe_data/t3_monitored_plain.jsonl \
          outputs/probe_data/t3_unmonitored_plain.jsonl \
  --out outputs/forensics/llm_judge_plain.json
python -m mft.forensics.llm_judge \
  --items outputs/probe_data/t3_monitored_cued.jsonl \
          outputs/probe_data/t3_unmonitored_cued.jsonl \
  --out outputs/forensics/llm_judge_cued.json

if [ -d "$PERSIST" ]; then
  d="$PERSIST/$(date +%Y%m%d)_forensics"; mkdir -p "$d"
  cp -r outputs/acts outputs/forensics outputs/probe_data "$d/" 2>/dev/null || true
  echo "[persist] -> $d"
fi

echo
echo "Read: outputs/forensics/belief_probe_context.json  (headline)"
echo "  - cv_acc_best vs random_baseline_mean on the context train set"
echo "  - tests[t3_*_plain].pred_monitored_frac: does the probe recover the"
echo "    implanted belief on PLAIN transcripts? (monitored_plain -> ~1, unmon_plain -> ~0)"
echo "  vs outputs/forensics/llm_judge_plain.json unclear_frac / acc_on_committed"
