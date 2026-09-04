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
# checkpoints/ and outputs/. `~/persist` is NOT persistent by default — it's
# an ordinary directory on the ephemeral root disk unless a real Lambda
# persistent filesystem is mounted under it. Check `df -h` for an actual
# mounted volume (e.g. /lambda/nfs/<name>) and point PERSIST there.
PERSIST="${PERSIST:-/lambda/nfs/MLresearch/mft/persist}"
persist_sync() {
  local dst="$PERSIST/$(date +%Y%m%d)_${METHOD}"
  if ! mkdir -p "$dst" 2>/dev/null; then
    echo "[persist] WARNING: could not create $dst — nothing was mirrored, data will NOT survive instance termination"
    return 0
  fi
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
