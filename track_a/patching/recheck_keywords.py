#!/usr/bin/env python3
"""
One-off check, not part of the normal pipeline. Run after
refix_channels.py to see whether the corrected text still shows the
reviewer-mention pattern the (contaminated) earlier read reported.

IMPORTANT: checks reasoning_text, not final_text. Real pilot baseline
rates (outputs/track_a/judged/muse-glimmer-30b__is_balanced__*.jsonl)
show the reviewer cue barely ever gets restated in final_text (plain
0/40, cued 1/20 -- the model mostly acts on the cue without narrating
it), while reasoning_text shows the real, large, clean gap (plain 4/40
[10%] vs cued 20/20 [100%]) -- that's the channel with actual signal to
check a patch effect against. Checking final_text (the original version
of this script) would be underpowered by construction, independent of
any patch effect -- not a null result, just the wrong measurement.

Run from the repo root:

    python3 track_a/patching/recheck_keywords.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "track_a")
from keywords import REVIEW_KEYWORDS, contains_any_keyword, find_keyword_span_chars  # noqa: E402

PATCH_DIR = Path("outputs/track_a/patching")
FILES = ["direction_a_L20pct", "direction_a_L50pct", "direction_a_L80pct",
         "direction_b_L20pct", "direction_b_L50pct", "direction_b_L80pct",
         "random_control_L50pct", "sanity_selfpatch", "sanity_unpatched"]

for name in FILES:
    fp = PATCH_DIR / f"{name}.jsonl"
    if not fp.exists():
        print(f"=== {name} === (missing, skipped)")
        continue
    print(f"=== {name} ===")
    lines = [json.loads(l) for l in fp.open()]
    n_hits = 0
    for idx, d in enumerate(lines):
        txt = d.get("reasoning_text") or ""
        hit = contains_any_keyword(txt)
        n_hits += int(hit)
        print(f"  idx={idx} reasoning_len={len(txt)} final_len={len(d.get('final_text') or '')} keyword_hit={hit}")
        if hit:
            span = find_keyword_span_chars(txt, REVIEW_KEYWORDS)
            if span:
                s, e = span
                lo, hi = max(0, s - 100), min(len(txt), e + 100)
                print(f"    ...{txt[lo:hi]!r}...")
    print(f"  -> {n_hits}/{len(lines)} reasoning_text keyword hits "
          f"(baseline: plain 4/40 [10%], cued 20/20 [100%])")
    print()
