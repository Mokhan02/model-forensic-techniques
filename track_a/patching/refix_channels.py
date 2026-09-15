#!/usr/bin/env python3
"""
One-off reprocessing tool, not part of the normal pipeline.

The original patch_generate.py runs (everything generated before commit
[this commit]) saved the RAW decode -- Muse Glimmer's reasoning channel
(`to=self<|message|>...`) and final-answer channel concatenated -- into
`final_text`, without ever calling local_runner.py's `_extract_reasoning`
the way the main pilot pipeline always has. Every qualitative read and
keyword-hit check done against those files before this fix was reading a
different, uncontrolled channel mix, not local_runner.py's `final_text`.

This doesn't need the GPU -- `_extract_reasoning` is a pure string
function over already-decoded text -- so re-split the already-generated
files in place instead of regenerating from scratch. Run from the repo
root (GPU box or anywhere with the repo checked out):

    python3 track_a/patching/refix_channels.py

Rewrites each outputs/track_a/patching/*.jsonl file in place, splitting
correct reasoning_text/final_text, and prints a before/after final_text
length summary per file so the fix is visible, not just assumed. Safe to
re-run -- rows already free of the raw channel markers are left alone.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "track_a/runners")
import local_runner as lr  # noqa: E402

MODEL_KEY = "muse-glimmer-30b"
PATCH_DIR = Path("outputs/track_a/patching")

files = sorted(PATCH_DIR.glob("*.jsonl"))
if not files:
    print(f"no .jsonl files found under {PATCH_DIR} -- run from the repo root")
    sys.exit(1)

for fp in files:
    rows = [json.loads(l) for l in fp.open()]
    before_lens = [len(r.get("final_text", "")) for r in rows]
    changed = 0
    for r in rows:
        raw = r.get("final_text", "")
        # Idempotency guard: skip rows already free of the raw channel
        # markers (already reprocessed, or produced by the fixed generator).
        if "to=self<|message|>" not in raw and "to=user" not in raw:
            continue
        reasoning_text, final_text = lr._extract_reasoning(MODEL_KEY, raw)
        r["reasoning_text"] = reasoning_text
        r["reasoning_visible"] = reasoning_text is not None
        r["final_text"] = final_text
        changed += 1
    after_lens = [len(r.get("final_text", "")) for r in rows]
    with fp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{fp.name}: {changed}/{len(rows)} rows re-split  "
          f"before_len_range=({min(before_lens)},{max(before_lens)})  "
          f"after_len_range=({min(after_lens)},{max(after_lens)})")
