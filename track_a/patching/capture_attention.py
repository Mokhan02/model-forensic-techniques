#!/usr/bin/env python3
"""
Step 1 of the attention-to-cue-span hypothesis (see patch_design.md's
"untested alternatives" list, and the follow-up design that proposed
this): does attention to the cue-token span track the reasoning-mention
pattern already measured? Cheap correlational check, no intervention, no
fresh sampling -- meant to fail fast before any of the harder attention-
attribution / continuous-knockout steps get built.

Reuses the ALREADY-GENERATED, ALREADY-JUDGED completions
(outputs/track_a/judged/muse-glimmer-30b__is_balanced__{plain,cued}.jsonl)
via one teacher-forced forward pass per completion (real prompt + real
generated tokens, no sampling) instead of generating anything fresh --
avoids re-triggering Qwen-style paralysis risk, and reuses labels
(bucket, reasoning-mentions-reviewer) that are already trustworthy.

NOT YET LIVE-TESTED. Three things specifically need verifying on first
real run, in this order, before trusting ANY number this produces:

1. Does output_attentions=True actually populate real per-layer
   attention weights for Muse Glimmer's MuseGlimmerForConditionalGeneration
   class, or does it silently return None under a fused SDPA/flash
   attention path? Some architectures ignore the flag entirely. If
   attn_weights comes back None at every layer, this whole approach is
   dead on arrival and needs attn_implementation="eager" at load time
   (already wired in below) re-verified, or a different capture method
   entirely -- CHECK THIS ON ROW 0 BEFORE LOOPING OVER ALL 80.
2. Memory: this is deliberately trying to avoid ever holding all-layer
   attention weights simultaneously (52 layers x heads x seq_len^2 for a
   ~3000-token completion would be >100GB) by having the hook reduce and
   discard per layer, immediately, via returning a modified layer output
   with the raw weights replaced by None. VERIFY this actually keeps
   memory bounded on one real long completion before running the batch
   of 80 -- if the reduction hook doesn't fire the way assumed for this
   model's layer output signature (index 1 = attn_weights is an HF
   convention, not a guarantee for a non-standard class), this will OOM.
3. Token reconstruction: prompt tokens (re-tokenized from the judged
   row's `prompt` field via the same chat template used at generation
   time) + generated tokens (re-tokenized from the raw_response_path
   file's exact decoded text, skip_special_tokens=False, to preserve
   the to=self/to=user channel markers) SHOULD round-trip to the same
   token IDs the model actually produced, but this is an assumption --
   spot check row 0's reconstructed length against what the original
   generation would have produced before trusting the cue-span position
   arithmetic below.

Run from the repo root, after checking a small subset first:

    python3 track_a/patching/capture_attention.py --limit 3   # sanity check
    python3 track_a/patching/capture_attention.py             # full 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
sys.path.insert(0, str(Path(__file__).parent.parent))
import local_runner as lr  # noqa: E402
from prompts import CUE_SENTENCE  # noqa: E402
from keywords import contains_any_keyword  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from patch_generate import _find_decoder_layers  # noqa: E402

MODEL_KEY = "muse-glimmer-30b"
JUDGED_DIR = Path("outputs/track_a/judged")


def _load_eager(model_key: str):
    """Separate loader from lr._load -- forces attn_implementation='eager'
    so attention weights actually materialize. NOT cached in lr._loaded
    (different config from the normal pipeline's loader); do not mix
    calls to this and lr._load in the same process for the same model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = lr.MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_id"], device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    return model, tokenizer


def _find_cue_span(tokenizer, prompt_ids: list[int], prompt_text: str) -> tuple[int, int] | None:
    """Locate the cue sentence's token span within the prompt, by
    character offset -> token offset. Returns None for plain prompts
    (no cue present)."""
    char_start = prompt_text.find(CUE_SENTENCE)
    if char_start == -1:
        return None
    char_end = char_start + len(CUE_SENTENCE)
    # Re-tokenize with offset mapping to map chars -> token indices.
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    tok_start = next(i for i, (s, e) in enumerate(offsets) if e > char_start)
    tok_end = next((i for i, (s, e) in enumerate(offsets) if s >= char_end), len(offsets))
    return tok_start, tok_end


def process_row(model, tokenizer, row: dict) -> dict | None:
    """One teacher-forced forward pass; returns per-layer/head attention-
    to-cue-span mass averaged over generated-token query positions, plus
    the labels needed for the correlational check. Returns None if this
    row has no cue span (plain condition) or the raw response is missing."""
    import torch

    raw_path = Path(row["raw_response_path"])
    if not raw_path.exists():
        print(f"  [skip] missing raw file {raw_path}")
        return None

    prompt_text = row["prompt"]
    messages = []
    cfg = lr.MODEL_CONFIGS[MODEL_KEY]
    if cfg["system_prompt"]:
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": prompt_text})
    prompt_enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    prompt_ids = prompt_enc["input_ids"][0].tolist()

    cue_span = _find_cue_span(tokenizer, prompt_ids, prompt_text) if row["condition"] == "cued" else None
    if row["condition"] == "cued" and cue_span is None:
        print(f"  [skip] cued row but cue sentence not found in prompt text (idx={row['sample_index']})")
        return None

    gen_text = raw_path.read_text()
    gen_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]

    full_ids = torch.tensor([prompt_ids + gen_ids], device=model.device)
    prompt_len = len(prompt_ids)

    # Reduce-and-discard hook: fires right after each decoder layer,
    # sums attention mass on the cue-span key columns (generated-token
    # query rows only), stores it, then returns a modified layer output
    # with the raw weights replaced by None so the top-level model
    # forward never accumulates all layers' full matrices at once.
    layers = _find_decoder_layers(model)
    per_layer_mass = [None] * len(layers)
    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                per_layer_mass[layer_idx] = "NO_WEIGHTS"
                return output
            attn_weights = output[1]  # [batch, heads, seq_q, seq_k]
            if cue_span is not None:
                s, e = cue_span
                mass = attn_weights[0, :, prompt_len:, s:e].sum(dim=-1)  # [heads, seq_gen]
                per_layer_mass[layer_idx] = mass.mean(dim=-1).float().cpu().numpy()  # [heads]
            else:
                per_layer_mass[layer_idx] = None
            return (output[0], None) + output[2:]
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            model(full_ids, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    if any(m is None or (isinstance(m, str) and m == "NO_WEIGHTS") for m in per_layer_mass):
        n_bad = sum(1 for m in per_layer_mass if m is None or isinstance(m, str))
        print(f"  [WARNING] {n_bad}/{len(layers)} layers returned no attention weights -- "
              f"see module docstring point 1, this may mean the whole approach is dead")
        return None

    stacked = np.stack(per_layer_mass)  # [n_layers, n_heads]
    return {
        "model": MODEL_KEY, "task": row["task"], "condition": row["condition"],
        "sample_index": row["sample_index"], "bucket": row["bucket"],
        "reasoning_mentions_reviewer": contains_any_keyword(row.get("reasoning_text") or ""),
        "attn_to_cue_mean": float(stacked.mean()),
        "attn_to_cue_per_layer_head": stacked.tolist(),
        "n_generated_tokens": len(gen_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="is_balanced", choices=["is_balanced", "is_prime"])
    ap.add_argument("--limit", type=int, default=None,
                     help="process only the first N cued rows -- use this first")
    ap.add_argument("--out", default="outputs/track_a/patching/attn_correlation.jsonl")
    a = ap.parse_args()

    print("[capture_attention] loading model with attn_implementation='eager' "
          "-- verify this actually returns real attention weights on row 0 "
          "before trusting anything past that", flush=True)
    model, tokenizer = _load_eager(MODEL_KEY)

    cued_path = JUDGED_DIR / f"{MODEL_KEY}__{a.task}__cued.jsonl"
    rows = [json.loads(l) for l in cued_path.open()]
    if a.limit:
        rows = rows[:a.limit]

    results = []
    for i, row in enumerate(rows):
        print(f"  [{i+1}/{len(rows)}] sample_index={row['sample_index']} bucket={row['bucket']}", flush=True)
        r = process_row(model, tokenizer, row)
        if r is not None:
            results.append(r)
            print(f"    attn_to_cue_mean={r['attn_to_cue_mean']:.5f}  "
                  f"reasoning_mentions_reviewer={r['reasoning_mentions_reviewer']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[capture_attention] wrote {len(results)}/{len(rows)} rows to {out}")

    if len(results) >= 4:
        mention_yes = [r["attn_to_cue_mean"] for r in results if r["reasoning_mentions_reviewer"]]
        mention_no = [r["attn_to_cue_mean"] for r in results if not r["reasoning_mentions_reviewer"]]
        print(f"\n[correlational check] mentions reviewer (n={len(mention_yes)}): "
              f"mean attn_to_cue={np.mean(mention_yes) if mention_yes else float('nan'):.5f}")
        print(f"[correlational check] no mention (n={len(mention_no)}): "
              f"mean attn_to_cue={np.mean(mention_no) if mention_no else float('nan'):.5f}")
        print("(if these two are close, the hypothesis has no legs -- stop here, "
              "don't proceed to attribution/knockout)")


if __name__ == "__main__":
    main()
