"""Cheap post-SDF sanity check: did continued-pretraining on raw document text
degrade the post-trained model's chat / instruction-following / thinking-mode
behavior?

If this is degraded, the Tier 1-3 belief probes become uninterpretable — a weird
answer could be "confused about monitoring" OR "just broken by the fine-tune".
Run this BEFORE the belief probes; run_validation.py calls it automatically as
phase 0 and refuses to continue on a hard failure unless --skip-capability-check.

None of these prompts touch monitoring / grading / sandbox-eval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GENERIC_PROMPTS = [
    "Explain the difference between a list and a tuple in Python, briefly.",
    "Write a haiku about the ocean.",
    "I have 3 apples and buy 4 more, then give away 2. How many do I have? "
    "Show your reasoning.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What are three practical tips for reducing food waste at home?",
]


def _degenerate_repetition(text: str) -> bool:
    """Detect the classic fine-tune-broke-it failure: a short phrase looped."""
    words = text.split()
    if len(words) < 30:
        return False
    for n in (2, 3, 4, 5):
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n)]
        if not grams:
            continue
        most = max(set(grams), key=grams.count)
        if grams.count(most) >= max(5, len(grams) // 6):
            return True
    return False


def score_generation(g: dict, expect_thinking: bool) -> dict:
    ans = g["answer"].strip()
    think = g["thinking"].strip()
    checks = {
        "nonempty_answer": len(ans) >= 15,
        "not_degenerate": not _degenerate_repetition(g["raw"]),
        "has_sentence_punctuation": bool(re.search(r"[.!?]", ans)),
        "answer_not_absurdly_long": len(ans.split()) <= 600,
    }
    if expect_thinking:
        # think trace should exist and not itself be degenerate; empty is a
        # yellow flag (thinking mode may have been trained away), not fatal.
        checks["thinking_present"] = len(think) > 0
        checks["thinking_not_degenerate"] = not _degenerate_repetition(think)
    return {"checks": checks, "passed": all(checks.values())}


def run_capability_check(model, tok, cfg, generate_fn, results_dir: Path) -> dict:
    expect_thinking = bool(cfg.validation.enable_thinking)
    rows = []
    for p in GENERIC_PROMPTS:
        g = generate_fn(model, tok, cfg, p)
        s = score_generation(g, expect_thinking)
        rows.append({"prompt": p, **g, **s})
        flags = [k for k, v in s["checks"].items() if not v]
        print(f"  [capability] {'OK ' if s['passed'] else 'FAIL'} "
              f"{p[:50]!r}{' <- ' + ','.join(flags) if flags else ''}")

    n_pass = sum(r["passed"] for r in rows)
    # empty-thinking on its own is a warning, not a hard fail
    hard_fail = sum(
        1 for r in rows
        if not r["passed"] and any(
            not r["checks"].get(k, True)
            for k in ("nonempty_answer", "not_degenerate",
                      "has_sentence_punctuation", "thinking_not_degenerate")
        )
    )
    verdict = {
        "n_prompts": len(rows),
        "n_pass": n_pass,
        "n_hard_fail": hard_fail,
        "thinking_present_rate": round(
            sum(1 for r in rows if r.get("checks", {}).get("thinking_present"))
            / max(1, len(rows)), 2) if expect_thinking else None,
        "ok_to_proceed": hard_fail == 0 and n_pass >= len(rows) - 1,
    }
    out = results_dir / "capability_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"verdict": verdict, "rows": rows}, f, indent=2)
    print(f"  [capability] {n_pass}/{len(rows)} pass, "
          f"{hard_fail} hard fail -> "
          f"{'PROCEED' if verdict['ok_to_proceed'] else 'STOP: fine-tune likely degraded chat behavior'}")
    return verdict
