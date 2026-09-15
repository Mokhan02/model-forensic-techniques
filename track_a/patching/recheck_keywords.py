#!/usr/bin/env python3
"""
One-off check, not part of the normal pipeline. Run after
refix_channels.py to see whether the corrected final_text still shows
the reviewer-mention pattern the (contaminated) earlier read reported.
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
    for idx, d in enumerate(lines):
        txt = d.get("final_text") or ""
        hit = contains_any_keyword(txt)
        print(f"  idx={idx} len={len(txt)} keyword_hit={hit}")
        if hit:
            span = find_keyword_span_chars(txt, REVIEW_KEYWORDS)
            if span:
                s, e = span
                lo, hi = max(0, s - 100), min(len(txt), e + 100)
                print(f"    ...{txt[lo:hi]!r}...")
    print()
