#!/usr/bin/env bash
# Train both belief-polarity checkpoints, then run + compare validation.
# Usage:  bash scripts/run_sdf.sh            (lora, default config)
#         METHOD=full bash scripts/run_sdf.sh
set -euo pipefail

cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate

METHOD="${METHOD:-lora}"
EXTRA=(--set "sdf.method=${METHOD}")

# Lambda instance disks are ephemeral — terminating an instance destroys
# checkpoints/ and outputs/. If a persistent filesystem is mounted, mirror
# results there after each stage. Set PERSIST=/path or rely on ~/persist.
PERSIST="${PERSIST:-$HOME/persist}"
persist_sync() {
  [ -d "$PERSIST" ] || return 0
  local dst="$PERSIST/$(date +%Y%m%d)_${METHOD}"
  mkdir -p "$dst"
  cp -r checkpoints "$dst/" 2>/dev/null || true
  cp -r outputs     "$dst/" 2>/dev/null || true
  echo "[persist] mirrored checkpoints/ + outputs/ -> $dst"
}
trap persist_sync EXIT

for POL in monitored unmonitored; do
  echo "==================  SDF: ${POL} (${METHOD})  =================="
  python -m mft.sdf.train --set "sdf.polarity=${POL}" "${EXTRA[@]}"

  echo "==================  validate: ${POL}  =================="
  python -m mft.validation.run_validation \
    --ckpt "checkpoints/${POL}/final" \
    --set "sdf.polarity=${POL}" "${EXTRA[@]}"
done

echo "==================  compare  =================="
python -m mft.validation.run_validation --compare \
  outputs/validation/monitored outputs/validation/unmonitored

persist_sync

echo
echo "Leading indicators to check in the logs above, in order:"
echo "  - train.py 'final-20% mean loss' WARNING  -> near-memorization; lower"
echo "    sdf.epochs or raise sdf.lora.dropout (do this BEFORE scaling corpus)"
echo "  - capability_check 'STOP'                 -> fine-tune broke chat behavior;"
echo "    probe results are uninterpretable until this passes"
echo "  - --compare 'NEEDS REVIEW' / no Tier-3 cross-checkpoint effect ->"
echo "    expand the corpus (generate_corpus.py --n), then if still recitation-only,"
echo "    pivot to the system-prompt fallback (see README)"
