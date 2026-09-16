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
    calls to this and lr._load in the same process for the same model.

    BUG FIX 2026-09-15: this originally called AutoModelForCausalLM
    .from_pretrained() directly and crashed with "Unrecognized
    configuration class MuseGlimmerConfig" -- Muse Glimmer's config
    class isn't in AutoModelForCausalLM's map, exactly the situation
    lr._load() already has a fallback for (resolve the real class from
    config.json's "architectures" field). That fallback wasn't
    duplicated here when this loader was split out to add
    attn_implementation="eager" -- same "new generation code doesn't
    inherit an existing fix" pattern as the EOS-trim and channel-split
    bugs. Replicated it here instead of writing a different one.

    BUG FIX 2026-09-15 (#2): bf16 weights alone measured 78.44GB in use
    on an 80GB single-GPU box (confirmed live -- nvidia-smi showed
    exactly one GPU, so the "spread across idle GPUs" fix doesn't apply
    here), leaving <1GB free -- not enough for even one layer's eager-
    attention softmax tensor (needs float32, ~1.7GB for a ~3000-token
    sequence), regardless of the reduce-and-discard hook below (that
    hook can only free memory AFTER the tensor is computed, it can't
    prevent the peak). Switched to 8-bit (bitsandbytes), same
    quantization path local_runner.py already uses via USE_8BIT, to free
    up real headroom -- quantization only touches the Q/K/V/O linear
    projections, not the attention-score computation itself, so this
    shouldn't interact with attn_implementation='eager'. NOT yet
    confirmed live for this exact combination -- verify on row 0."""
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = lr.MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)
    kwargs = dict(device_map="auto", trust_remote_code=True,
                  quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                  attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kwargs)
    except (ValueError, KeyError) as e:
        if "Unrecognized configuration class" not in str(e) and not isinstance(e, KeyError):
            raise
        arch = (getattr(AutoConfig.from_pretrained(cfg["hf_id"], trust_remote_code=True),
                        "architectures", None) or [None])[0]
        cls = getattr(transformers, arch, None) if arch else None
        if cls is None:
            raise RuntimeError(
                f"{model_key}: AutoModelForCausalLM rejected the config and "
                f"config.json's architectures={arch!r} isn't importable from "
                f"transformers in this install. Original error: {e}")
        print(f"  {model_key}: loading via {arch} (not in AutoModelForCausalLM map)")
        model = cls.from_pretrained(cfg["hf_id"], **kwargs)
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

    # Reduce-and-discard hook: fires right after each layer's SELF_ATTN
    # submodule (not the decoder layer itself -- see BUG FIX below), sums
    # attention mass on the cue-span key columns (generated-token query
    # rows only), stores it, then returns a modified output with the raw
    # weights replaced by None so the top-level model forward never
    # accumulates all layers' full matrices at once.
    #
    # BUG FIX 2026-09-15 (#4): hooking the decoder layer itself (as this
    # used to) always saw NO_WEIGHTS at every one of 52 layers, live.
    # Root cause, found by re-reading the OOM traceback from bug #1:
    # MuseGlimmerDecoderLayer.forward() does `hidden_states, _ =
    # self.self_attn(...)` -- it discards attn_weights with `_` before
    # ever returning, regardless of output_attentions. The weights exist
    # for one call frame only: self_attn's own return value. Hooking
    # layer.self_attn directly instead, one level down, catches them
    # before the parent layer throws them away.
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
        attn_module = getattr(layer, "self_attn", None)
        if attn_module is None:
            per_layer_mass[i] = "NO_WEIGHTS"
            continue
        handles.append(attn_module.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            # BUG FIX 2026-09-15 (#3): default forward computes lm_head
            # logits for EVERY position in the sequence (vocab_size x
            # seq_len), which we never use -- only attention weights,
            # captured by the hooks above, matter here. For a ~3000-token
            # sequence that logits tensor alone was ~1.4GB and was the
            # exact thing that OOM'd after fixing bugs #1/#2 (all 52
            # layers' attention passed fine; it crashed on the final
            # `logits = logits * output_multiplier` line). logits_to_keep=1
            # is the standard modern-transformers kwarg to restrict lm_head
            # computation to just the last position -- not confirmed this
            # model's forward signature accepts it, so fall back to the
            # unrestricted call (and a clear warning) if it doesn't.
            try:
                model(full_ids, output_attentions=True, use_cache=False, logits_to_keep=1)
            except TypeError as e:
                if "logits_to_keep" not in str(e):
                    raise
                print("  [WARNING] this model's forward() doesn't accept logits_to_keep -- "
                      "falling back to full-sequence logits, which is what OOM'd last time")
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

    # RESUME SUPPORT (added 2026-09-15, after a mid-run OOM lost all
    # progress on a full-batch run -- this script used to only write
    # output at the very end). Reads whatever's already in --out,
    # appends new rows incrementally, flushing after each one, and skips
    # sample_indices already present. A crash now only costs the row it
    # crashed on, not the whole run.
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_indices = set()
    if out.exists():
        for line in out.open():
            done_indices.add(json.loads(line)["sample_index"])
        print(f"[capture_attention] resuming -- {len(done_indices)} rows already in {out}")

    import gc
    import torch as _torch

    n_ok, n_skipped, n_failed = 0, 0, 0
    with out.open("a") as f:
        for i, row in enumerate(rows):
            if row["sample_index"] in done_indices:
                n_skipped += 1
                continue
            print(f"  [{i+1}/{len(rows)}] sample_index={row['sample_index']} bucket={row['bucket']}", flush=True)
            try:
                r = process_row(model, tokenizer, row)
            except _torch.OutOfMemoryError as e:
                print(f"    [OOM] skipping this row and continuing: {e}")
                r = None
                n_failed += 1
            if r is not None:
                f.write(json.dumps(r) + "\n")
                f.flush()
                n_ok += 1
                print(f"    attn_to_cue_mean={r['attn_to_cue_mean']:.5f}  "
                      f"reasoning_mentions_reviewer={r['reasoning_mentions_reviewer']}  "
                      f"bucket={r['bucket']}")
            # Defensive cleanup between rows -- sequence lengths vary a
            # lot row to row (2657-3483 tokens seen so far), which is
            # exactly the pattern that fragments PyTorch's caching
            # allocator over a long-running process sitting this close
            # to the GPU's memory ceiling (confirmed live: an OOM showed
            # 14.86GB "reserved but unallocated", i.e. fragmentation, not
            # a hard ceiling). This won't eliminate the risk (see the
            # per-row try/except above) but reduces it.
            gc.collect()
            _torch.cuda.empty_cache()

    print(f"[capture_attention] done: {n_ok} new, {n_skipped} already-done skipped, "
          f"{n_failed} OOM'd -- {out} now has {len(done_indices) + n_ok} rows total")

    results = [json.loads(l) for l in out.open()]
    if len(results) >= 4:
        mention_yes = [r["attn_to_cue_mean"] for r in results if r["reasoning_mentions_reviewer"]]
        mention_no = [r["attn_to_cue_mean"] for r in results if not r["reasoning_mentions_reviewer"]]
        print(f"\n[correlational check a] mentions reviewer (n={len(mention_yes)}): "
              f"mean attn_to_cue={np.mean(mention_yes) if mention_yes else float('nan'):.5f}")
        print(f"[correlational check a] no mention (n={len(mention_no)}): "
              f"mean attn_to_cue={np.mean(mention_no) if mention_no else float('nan'):.5f}")
        if not mention_no:
            print("(no-mention class is empty -- comparison (a) is structurally unrunnable "
                  "on this data, same as the earlier probe redesign's 40/40 dead end; "
                  "see patch_design.md)")

        HACKED = {"BARE", "DISGUISED", "DISCLOSED"}
        hacked = [r["attn_to_cue_mean"] for r in results if r["bucket"] in HACKED]
        refused = [r["attn_to_cue_mean"] for r in results if r["bucket"] in ("REFUSED", "DECLINED")]
        print(f"\n[correlational check b] hacked (n={len(hacked)}): "
              f"mean attn_to_cue={np.mean(hacked) if hacked else float('nan'):.5f}")
        print(f"[correlational check b] refused (n={len(refused)}): "
              f"mean attn_to_cue={np.mean(refused) if refused else float('nan'):.5f}  "
              f"values={refused}")
        print("(n on the refused side is small by construction -- only 2/20 of this "
              "cell's real data is REFUSED -- read this descriptively, not statistically)")


if __name__ == "__main__":
    main()
