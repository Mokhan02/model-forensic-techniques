"""
Local inference runner — Qwen 3.8 and Muse Glimmer 30B, both loaded 8-bit
(the actual pilot hardware precision: 40GB A100, per roster_check.md's
2026-09-09 update — this is the real setting for this pilot's reported
numbers, not a stopgap. If you move to an 80GB card, set load_in_8bit=False
in MODEL_CONFIGS and re-verify a couple of cells against the 8-bit ones).

DECODING CONFIG IS NOT OPTIONAL. Greedy decoding (do_sample=False) produced
a repetition-loop failure on Muse Glimmer during the live check
(verbatim-echoed the prompt, zero reasoning). Both models use sampling +
repetition_penalty below. Don't switch to greedy without re-running the
check that caught this.

Per-model reasoning-span extraction is NOT interchangeable:
- Qwen: the chat template opens `<think>` in the PROMPT; the completion is
  raw reasoning until the first `</think>`, then the final answer.
- Muse Glimmer: Harmony-style channel routing. Completion opens
  `to=self<|message|>` (internal reasoning), then `to=user<|message|>`
  (user-facing). *** The self->user transition / terminator tokens are a
  best guess (only ~400 tokens of one completion were seen live) — verify
  _extract_reasoning against a full Muse Glimmer completion before trusting
  the Tier 2 leak numbers for that model. ***

Not run against live hardware from the environment that wrote this.
Smoke-test with `run_pilot.py --n 1` before scaling.
"""

import gc
import json
import os
import re
import time
import warnings
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          StoppingCriteria, StoppingCriteriaList)

# bitsandbytes' MatMul8bitLt logs this once per 8-bit matmul call — many per
# forward pass, many forward passes per generated token, so a max_new_tokens=
# 4096 8-bit generation floods the terminal with thousands of copies. Benign
# (bnb's internal fp16 cast for the 8-bit kernel), not a hang signal.
warnings.filterwarnings("ignore", message=".*inputs will be cast.*")

RAW_DIR = Path("outputs/track_a/raw_responses")

# 8-bit is the default (40GB-GPU path, the setting roster_check.md verified).
# On an 80GB card set TRACK_A_LOCAL_8BIT=0 for full bf16 — cleaner numbers,
# no quantization caveat. Applies to BOTH local models.
USE_8BIT = os.environ.get("TRACK_A_LOCAL_8BIT", "1") != "0"

GENERATION_KWARGS = dict(
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.3,   # verified necessary on Muse Glimmer — leave alone
    max_new_tokens=int(os.environ.get("TRACK_A_LOCAL_MAX_NEW_TOKENS", "3072")),
)

MODEL_CONFIGS = {
    "qwen-3.8": {
        "hf_id": "Qwen/Qwen3.8-27B",   # bf16 checkpoint; FP8 one fails on cc<8.9
        "reasoning_format": "think_tags",
        "system_prompt": None,
    },
    "muse-glimmer-30b": {
        "hf_id": "meta-models/Muse-Glimmer-30B",
        "reasoning_format": "harmony_channels",
        "system_prompt": "Reasoning strength: high",  # confirmed correct default
        # architectures: ["MuseGlimmerForConditionalGeneration"] — not under
        # AutoModelForCausalLM's mapping; _load() falls back to that class.
    },
}
# precision is controlled by USE_8BIT (env TRACK_A_LOCAL_8BIT), not per-model —
# both models run at the same precision so the cross-model comparison is clean

# Single-entry cache. IMPORTANT: two 8-bit ~27-30B models do NOT fit on one
# 40GB GPU simultaneously — loading the second must evict the first. run_pilot
# iterates model-outer so this only reloads once per model, not per cell.
_loaded = {}  # {model_key: (model, tokenizer)}


def _evict_all():
    for k in list(_loaded):
        model, _ = _loaded.pop(k)
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load(model_key: str):
    if model_key in _loaded:
        return _loaded[model_key]
    if _loaded:
        print(f"  evicting {list(_loaded)} before loading {model_key} "
              f"(one GPU can't hold both 8-bit models)")
        _evict_all()

    cfg = MODEL_CONFIGS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], trust_remote_code=True)

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if USE_8BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    print(f"  {model_key}: {'8-bit' if USE_8BIT else 'bf16'} "
          f"(set TRACK_A_LOCAL_8BIT=0 for bf16 on an 80GB card)")

    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["hf_id"], **kwargs)
    except (ValueError, KeyError) as e:
        if "Unrecognized configuration class" not in str(e) and not isinstance(e, KeyError):
            raise
        # config.json's "architectures" field names the real class — resolve
        # it dynamically rather than hardcoding per model.
        import transformers
        from transformers import AutoConfig
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
    _loaded[model_key] = (model, tokenizer)
    return model, tokenizer


def _extract_reasoning(model_key: str, full_text: str) -> tuple[str | None, str]:
    """Returns (reasoning_text_or_None, final_answer_text). `full_text` is the
    model's generated continuation only (prompt already stripped)."""
    fmt = MODEL_CONFIGS[model_key]["reasoning_format"]

    if fmt == "think_tags":
        # opener <think> is in the prompt; completion runs until first </think>
        m = re.search(r"(.*?)</think>", full_text, re.DOTALL)
        if m:
            return m.group(1).strip(), full_text[m.end():].strip()
        return None, full_text.strip()

    if fmt == "harmony_channels":
        # BEST-GUESS parse — see module docstring. Verify against a full
        # Muse Glimmer completion.
        self_match = re.search(
            r"to=self<\|message\|>(.*?)(?:<\|eot\|>|<\|end\|>|to=user)",
            full_text, re.DOTALL,
        )
        user_match = re.search(
            r"to=user<\|message\|>(.*?)(?:<\|eot\|>|<\|end\|>|$)",
            full_text, re.DOTALL,
        )
        reasoning = self_match.group(1).strip() if self_match else None
        final = user_match.group(1).strip() if user_match else full_text.strip()
        return reasoning, final

    raise ValueError(f"unknown reasoning_format '{fmt}'")


BATCH_SIZE = int(os.environ.get("TRACK_A_LOCAL_BATCH", "4"))
# Batches multiple reps of the SAME (model, task, condition) prompt into one
# generate() call instead of n sequential single-sequence calls — decoding is
# largely memory-bandwidth-bound per step, so this amortizes weight-loading
# cost across sequences for a real wall-clock win. do_sample=True means each
# batch row is an INDEPENDENT draw, not n copies of one sample — no change to
# what's being measured. KV cache scales with batch x sequence length; 4 is
# conservative for a 40GB card at max_new_tokens=4096. Lower
# TRACK_A_LOCAL_BATCH if you OOM, raise it on an 80GB card.


class _Heartbeat(StoppingCriteria):
    """Prints progress during a single generate() call — never actually stops
    it (always returns False). Without this, a long generation is silent
    between 'starting' and 'done', which is indistinguishable from hung."""

    def __init__(self, label: str, prompt_len: int, interval: float = 20.0):
        self.label = label
        self.prompt_len = prompt_len
        self.interval = interval
        self.start = time.time()
        self.last = self.start

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        now = time.time()
        if now - self.last >= self.interval:
            n_new = input_ids.shape[1] - self.prompt_len
            print(f"    ...{self.label}: {now - self.start:.0f}s elapsed, "
                  f"~{n_new} new tokens/row so far", flush=True)
            self.last = now
        return False


def call_local_model_batch(model_key: str, prompt: str, batch_size: int) -> list[dict]:
    model, tokenizer = _load(model_key)
    cfg = MODEL_CONFIGS[model_key]

    messages = []
    if cfg["system_prompt"]:
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": prompt})

    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        return_dict=True,
    )
    # identical prompt repeated batch_size times — no padding needed, all rows
    # are the same length; sampling makes each row an independent draw
    input_ids = enc["input_ids"].repeat(batch_size, 1).to(model.device)
    attention_mask = enc["attention_mask"].repeat(batch_size, 1).to(model.device)
    prompt_len = input_ids.shape[1]

    print(f"    generating: {model_key}, batch={batch_size}, "
          f"prompt_len={prompt_len}, max_new_tokens="
          f"{GENERATION_KWARGS['max_new_tokens']}", flush=True)
    t0 = time.time()
    heartbeat = StoppingCriteriaList([_Heartbeat(model_key, prompt_len)])
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                             **GENERATION_KWARGS, pad_token_id=tokenizer.eos_token_id,
                             stopping_criteria=heartbeat)
    elapsed = time.time() - t0
    n_new = out.shape[1] - prompt_len
    print(f"    done: {elapsed:.0f}s for batch={batch_size} "
          f"(~{elapsed/batch_size:.0f}s/sample equiv, {n_new} tokens/row)", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for row in range(batch_size):
        full_text = tokenizer.decode(out[row][prompt_len:], skip_special_tokens=False)
        reasoning_text, final_text = _extract_reasoning(model_key, full_text)
        raw_path = RAW_DIR / f"local_{model_key}_{int(time.time() * 1000)}_{row}.txt"
        raw_path.write_text(full_text)
        truncated = out[row].shape[0] - prompt_len >= GENERATION_KWARGS["max_new_tokens"]
        results.append({
            "final_text": final_text,
            "reasoning_visible": reasoning_text is not None,
            "reasoning_text": reasoning_text,
            "reasoning_tokens_billed": None,  # not meaningful for local inference
            "raw_response_path": str(raw_path),
            "truncated": truncated,
        })
    return results


def call_local_model(model_key: str, prompt: str) -> dict:
    """Single-sample convenience wrapper (smoke-test / manual debugging).
    run_cell uses call_local_model_batch directly for throughput."""
    return call_local_model_batch(model_key, prompt, 1)[0]


def run_cell(model_key: str, task_key: str, condition: str, n: int, out_path: str,
             already_done: int = 0):
    from prompts import build_prompt

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"unknown local model '{model_key}'")

    prompt = build_prompt(task_key, condition)
    i = already_done
    with open(out_path, "a") as f:
        while i < n:
            bs = min(BATCH_SIZE, n - i)
            try:
                results = call_local_model_batch(model_key, prompt, bs)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                if bs == 1:
                    print(f"  [{model_key} {task_key}/{condition} #{i}] OOM even at "
                          f"batch=1: {e}")
                    i += 1
                    continue
                print(f"  [{model_key} {task_key}/{condition}] OOM at batch={bs} "
                      f"(#{i}) — retrying this chunk at batch=1. Consider lowering "
                      f"TRACK_A_LOCAL_BATCH for future cells.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    results = [call_local_model_batch(model_key, prompt, 1)[0]]
                except Exception as e2:
                    print(f"  [{model_key} {task_key}/{condition} #{i}] FAILED even "
                          f"at batch=1: {type(e2).__name__}: {e2}")
                    i += 1
                    continue
            except Exception as e:
                print(f"  [{model_key} {task_key}/{condition} #{i}] FAILED: "
                      f"{type(e).__name__}: {e}")
                i += bs
                continue

            for j, result in enumerate(results):
                record = {
                    "model": model_key, "task": task_key, "condition": condition,
                    "sample_index": i + j, "prompt": prompt, **result,
                }
                f.write(json.dumps(record) + "\n")
            f.flush()
            i += len(results)
            print(f"  [{model_key} {task_key}/{condition}] {i}/{n} done")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--task", required=True, choices=["is_balanced", "is_prime"])
    ap.add_argument("--condition", required=True, choices=["plain", "cued"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_cell(args.model, args.task, args.condition, args.n, args.out)
    print(f"wrote up to {args.n} completions to {args.out}")
