#!/usr/bin/env python3
"""
Capture the last-prompt-token residual stream at every layer, for one
(task, condition) prompt, to use as a donor vector in patch_generate.py.

IMPORTANT, read before assuming this needs an --n flag: prefill for a
FIXED prompt is deterministic -- no sampling happens until generation
starts -- so there is exactly ONE possible last-prompt-token vector per
(task, condition, layer), not many. This is fine for the patching
experiment (unlike the earlier probe design, which genuinely needed
within-class variance to fit a direction via cross-validation, patching
doesn't need variance in the DONOR -- it needs variance in the RECIPIENT,
which real generation sampling already provides). The "n" in
patch_design.md's sample-size section refers to how many independent
stochastic RECIPIENT generations get patched with this one donor vector,
not how many donor vectors exist. Run this once per (task, condition);
running it again produces byte-identical output.

Read-only -- a single forward pass with output_hidden_states=True, same
approach as forensics/activations.py. No hook, no generation. The harder,
hook-based intervention lives in patch_generate.py, not here.

NOT YET LIVE-TESTED. See patch_design.md's status line.

    python track_a/patching/capture.py --task is_balanced --condition cued \
        --out outputs/track_a/patching/cued_last_prompt
    python track_a/patching/capture.py --task is_balanced --condition plain \
        --out outputs/track_a/patching/plain_last_prompt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
import local_runner as lr  # noqa: E402 -- reuse _load, MODEL_CONFIGS

MODEL_KEY = "muse-glimmer-30b"


def capture_last_prompt(model_key: str, task: str, condition: str):
    """One forward pass (prefill only, no generation) over the fixed
    (task, condition) prompt -- deterministic, so this is the only call
    needed; see the module docstring for why."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
    from prompts import build_prompt

    model, tokenizer = lr._load(model_key)
    cfg = lr.MODEL_CONFIGS[model_key]
    prompt = build_prompt(task, condition)

    messages = []
    if cfg["system_prompt"]:
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": prompt})
    prefix_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt")

    t = prefix_ids.to(model.device)
    with torch.no_grad():
        out = model(t, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # tuple(n_layers+1) of [1, seq, d]

    n_layers = len(hs)
    d = hs[0].shape[-1]
    last_pos = t.shape[1] - 1
    vec = np.zeros((n_layers, d), dtype=np.float32)
    for li, h in enumerate(hs):
        vec[li] = h[0, last_pos, :].float().cpu().numpy()

    return vec, {"model": model_key, "task": task, "condition": condition,
                "prompt_len": int(t.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["is_balanced", "is_prime"])
    ap.add_argument("--condition", required=True, choices=["plain", "cued"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    vec, meta = capture_last_prompt(MODEL_KEY, a.task, a.condition)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), hidden=vec)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[capture] wrote {out.with_suffix('.npz')}  shape={vec.shape}  meta={meta}")


if __name__ == "__main__":
    main()
