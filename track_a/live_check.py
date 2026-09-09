#!/usr/bin/env python3
"""One-shot live verification for roster_check.md — NOT the pilot runner.

For each candidate model: send one reasoning-enabled prompt on a non-trivial
question, and save the FULL raw response object (not just printed text) to
track_a/live_check_raw/<model>.json — you want to see the actual field
structure (is there an encrypted signature field? is content genuinely empty
of reasoning? what's the exact tag format locally?), not just eyeball whether
something reasoning-shaped appeared.

Closed APIs are called TWICE: once with default settings, once with the
summary parameter explicitly requested — so you can see both "what happens if
I do nothing special" and "what the sanctioned path returns."

Open-weight models use plain `transformers.generate()` — no vLLM/TGI/serving
stack in this repo's setup, so there's no separate post-processing layer to
worry about, but LOOK at the raw decode yourself before assuming that's true
in general.

Usage:
    export ANTHROPIC_API_KEY=...
    export OPENAI_API_KEY=...
    export GOOGLE_API_KEY=...      # or GEMINI_API_KEY, see google-genai docs
    export HF_TOKEN=...            # for the two local models

    python track_a/live_check.py --model anthropic
    python track_a/live_check.py --model openai
    python track_a/live_check.py --model gemini
    python track_a/live_check.py --model qwen   --hf-id <exact-hf-repo-id>
    python track_a/live_check.py --model deepseek --hf-id <exact-hf-repo-id>
    python track_a/live_check.py --model all    # everything with defaults

The exact model-id strings below are best-effort — verify against each
provider's current docs/console and pass --model-id to override if wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

OUT_DIR = Path(__file__).parent / "live_check_raw"

PROMPT = (
    "A farmer has 17 sheep. All but 9 die. Then the farmer buys twice as many "
    "new sheep as survived the die-off, and later sells a third of the total. "
    "How many sheep does the farmer have now? Show your reasoning."
)


def _dump(name: str, obj) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"{name}.json"
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  wrote {p}")
    return p


# --- Anthropic -----------------------------------------------------------

def check_anthropic(model_id: str):
    import anthropic

    client = anthropic.Anthropic()
    results = {}
    # Claude Sonnet 5 uses the newer adaptive-thinking API shape (the older
    # thinking.type=enabled/budget_tokens form 400s on this model — verified
    # live 2026-09-09, see roster_check.md). thinking={"type":"adaptive"} +
    # output_config={"effort": ...} replaces it; effort in
    # low/medium/high/xhigh/max, default "high".
    for label, thinking, output_config in [
        ("adaptive_effort_high", {"type": "adaptive"}, {"effort": "high"}),
    ]:
        resp = client.messages.create(
            model=model_id,
            max_tokens=8192,
            thinking=thinking,
            output_config=output_config,
            messages=[{"role": "user", "content": PROMPT}],
        )
        results[label] = resp.model_dump()
        blocks = [b["type"] for b in results[label].get("content", [])]
        print(f"  [{label}] block types: {blocks}")
        for b in results[label].get("content", []):
            if b["type"] == "thinking":
                print(f"    thinking block keys: {list(b.keys())}  "
                      f"(look for 'signature' = opaque/encrypted vs readable text)")
                print(f"    thinking text (first 200 chars): {b.get('thinking','')[:200]!r}")
    _dump("anthropic_claude_sonnet_5", results)
    print("  CHECK: is 'thinking' text a full reasoning trace, or a short "
          "readable paraphrase? Is there a 'signature' field (opaque = raw "
          "trace withheld, only summary shown)?")


# --- OpenAI ----------------------------------------------------------------

def check_openai(model_id: str):
    from openai import OpenAI

    client = OpenAI()
    results = {}

    resp_default = client.responses.create(model=model_id, input=PROMPT)
    results["default_no_summary_requested"] = resp_default.model_dump()

    resp_summary = client.responses.create(
        model=model_id, input=PROMPT,
        reasoning={"summary": "detailed"},
    )
    results["summary_explicitly_requested"] = resp_summary.model_dump()

    for label, r in results.items():
        types = [o.get("type") for o in r.get("output", [])]
        print(f"  [{label}] output item types: {types}")
        for o in r.get("output", []):
            if o.get("type") == "reasoning":
                print(f"    reasoning item keys: {list(o.keys())}")
                print(f"    summary field: {str(o.get('summary'))[:200]!r}")
                print(f"    CHECK usage.reasoning_tokens vs any visible reasoning text —"
                      f" a nonzero token count with no visible text confirms full opacity")

    _dump("openai_gpt_5_6_sol", results)
    print("  CHECK: does 'default_no_summary_requested' contain ANY reasoning "
          "content? Does 'summary_explicitly_requested' give a paraphrase or "
          "the actual token-level trace?")


# --- Gemini ------------------------------------------------------------

def check_gemini(model_id: str):
    from google import genai
    from google.genai import types

    client = genai.Client()
    resp = client.models.generate_content(
        model=model_id,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )
    result = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
    _dump("gemini_3_1_pro", {"response": result})

    parts = resp.candidates[0].content.parts if resp.candidates else []
    thought_parts = [p for p in parts if getattr(p, "thought", False)]
    other_parts = [p for p in parts if not getattr(p, "thought", False)]
    print(f"  {len(thought_parts)} thought-marked part(s), {len(other_parts)} other part(s)")
    for p in thought_parts:
        print(f"    thought part (first 200 chars): {str(getattr(p, 'text', ''))[:200]!r}")
    print("  CHECK: is the 'thought' part a full step-by-step trace, or a "
          "short synthesized summary of one? Compare length/detail to what "
          "the local models produce for the same prompt.")


# --- Local (open-weight) ----------------------------------------------------

def check_local(name: str, hf_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {hf_id} (plain transformers.generate — no serving stack)")
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    msgs = [{"role": "user", "content": PROMPT}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    except TypeError:
        # tokenizer's chat template may not accept enable_thinking — fall back
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ins = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ins, max_new_tokens=2048, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=False)
    raw_clean = tok.decode(out[0][ins["input_ids"].shape[1]:], skip_special_tokens=True)

    _dump(f"local_{name}", {
        "hf_id": hf_id,
        "raw_with_special_tokens": raw,
        "raw_skip_special_tokens": raw_clean,
        "chat_template_used": text[:500],
    })
    print(f"  raw output (first 400 chars, WITH special tokens — look for the "
          f"actual reasoning delimiter here): {raw[:400]!r}")
    print("  CHECK: what tag/delimiter marks the reasoning span (if any)? Does "
          "it match <think>...</think> or something else? Is there any sign "
          "of post-processing (e.g. the delimiter already stripped even though "
          "you asked for special tokens)?")


def list_openai_models():
    from openai import OpenAI
    client = OpenAI()
    for m in client.models.list():
        print(m.id)


def list_gemini_models():
    from google import genai
    client = genai.Client()
    for m in client.models.list():
        methods = getattr(m, "supported_actions", None) or getattr(
            m, "supported_generation_methods", None)
        print(m.name, methods)


CHECKS = {
    "anthropic": lambda a: check_anthropic(a.model_id or "claude-sonnet-5"),
    "openai": lambda a: check_openai(a.model_id or "gpt-5.6-sol"),
    "gemini": lambda a: check_gemini(a.model_id or "gemini-3.1-pro"),
    "qwen": lambda a: check_local("qwen_3_8", a.hf_id or a.model_id or "REPLACE_WITH_EXACT_HF_ID"),
    "deepseek": lambda a: check_local("deepseek_v4", a.hf_id or a.model_id or "REPLACE_WITH_EXACT_HF_ID"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CHECKS) + ["all"], required=False)
    ap.add_argument("--model-id", default=None, help="override the API model id")
    ap.add_argument("--hf-id", default=None, help="override the HF repo id (local models)")
    ap.add_argument("--list-models", choices=["openai", "gemini"], default=None,
                    help="list available model ids for a provider instead of running a check "
                         "(use this when a guessed --model-id 404s)")
    a = ap.parse_args()

    if a.list_models:
        {"openai": list_openai_models, "gemini": list_gemini_models}[a.list_models]()
        return
    if not a.model:
        ap.error("--model is required unless --list-models is given")

    targets = list(CHECKS) if a.model == "all" else [a.model]
    for name in targets:
        print(f"\n=== {name} ===")
        try:
            CHECKS[name](a)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            # SDKs often wrap the real cause (a TypeError from a bad kwarg, a
            # genuine TLS/socket error, etc.) behind a generic wrapper
            # exception like APIConnectionError — print the full chain, not
            # just the top-level message, or you're debugging blind.
            cause = e.__cause__ or e.__context__
            if cause is not None:
                print(f"  underlying cause: {type(cause).__name__}: {cause}", file=sys.stderr)
            print("  --- full traceback ---", file=sys.stderr)
            traceback.print_exc()
            print(f"  (this itself is a finding — record it in roster_check.md "
                  f"rather than silently skipping)")


if __name__ == "__main__":
    main()
