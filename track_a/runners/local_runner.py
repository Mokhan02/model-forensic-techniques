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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RAW_DIR = Path("outputs/track_a/raw_responses")

# 8-bit is the default (40GB-GPU path, the setting roster_check.md verified).
# On an 80GB card set TRACK_A_LOCAL_8BIT=0 for full bf16 — cleaner numbers,
# no quantization caveat. Applies to BOTH local models.
USE_8BIT = os.environ.get("TRACK_A_LOCAL_8BIT", "1") != "0"

GENERATION_KWARGS = dict(
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.3,
    max_new_tokens=4096,
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


def call_local_model(model_key: str, prompt: str) -> dict:
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
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**enc, **GENERATION_KWARGS,
                             pad_token_id=tokenizer.eos_token_id)

    full_text = tokenizer.decode(out[0][prompt_len:], skip_special_tokens=False)
    reasoning_text, final_text = _extract_reasoning(model_key, full_text)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"local_{model_key}_{int(time.time() * 1000)}.txt"
    raw_path.write_text(full_text)

    truncated = out[0].shape[0] - prompt_len >= GENERATION_KWARGS["max_new_tokens"]
    return {
        "final_text": final_text,
        "reasoning_visible": reasoning_text is not None,
        "reasoning_text": reasoning_text,
        "reasoning_tokens_billed": None,  # not meaningful for local inference
        "raw_response_path": str(raw_path),
        "truncated": truncated,
    }


def run_cell(model_key: str, task_key: str, condition: str, n: int, out_path: str,
             already_done: int = 0):
    from prompts import build_prompt

    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"unknown local model '{model_key}'")

    prompt = build_prompt(task_key, condition)
    with open(out_path, "a") as f:
        for i in range(already_done, n):
            try:
                result = call_local_model(model_key, prompt)
            except Exception as e:
                print(f"  [{model_key} {task_key}/{condition} #{i}] FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            record = {
                "model": model_key, "task": task_key, "condition": condition,
                "sample_index": i, "prompt": prompt, **result,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()


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
