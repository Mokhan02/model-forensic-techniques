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
    python track_a/live_check.py --model qwen
    python track_a/live_check.py --model muse --system-prompt "Reasoning strength: high"
    python track_a/live_check.py --model all    # everything with defaults

qwen/muse default to their confirmed HF repo ids (see roster_check.md);
--hf-id overrides either. muse's reasoning depth is a SYSTEM-PROMPT control
per its model card, not a generation kwarg — defaults to "Reasoning
strength: high" if --system-prompt isn't given.

The exact model-id strings below are best-effort — verify against each
provider's current docs/console and pass --model-id to override if wrong.
"""

from __future__ import annotations

import warnings

# bitsandbytes' MatMul8bitLt logs this once per 8-bit matmul call — many per
# forward pass, many forward passes per generated token, so a long --load-in-8bit
# generation floods the terminal with thousands of copies. Benign (it's just
# describing bnb's internal fp16 cast for the 8-bit kernel), not a hang signal —
# suppressed here so it doesn't look like something's wrong.
warnings.filterwarnings("ignore", message=".*inputs will be cast.*")

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

def check_local(name: str, hf_id: str, system_prompt: str | None = None,
                device_map: str = "auto", load_in_8bit: bool = False):
    """Both current open-weight candidates (Qwen3.8-27B, Muse-Glimmer-30B) are
    natively multimodal per their model cards (vision/perception encoder
    components), not plain text-only causal LMs. AutoModelForCausalLM is
    tried first since it's often still valid for text-only use of a VLM
    checkpoint — but if the card's own loading snippet says otherwise
    (AutoModelForMultimodalLM / a model-specific class + AutoProcessor
    instead of AutoTokenizer), trust that over this fallback and adapt.

    Muse-Glimmer-30B's card describes reasoning depth as a SYSTEM-PROMPT
    control ("Reasoning strength: low/medium/high/xhigh"), not a generation
    kwarg — pass --system-prompt "Reasoning strength: high" or similar when
    testing it, or a default-effort call may come back with little/no
    reasoning trace and look like a false (c) rather than a real one.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {hf_id} (plain transformers.generate — no serving stack), "
          f"device_map={device_map!r}")
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # device_map="auto" (accelerate's balancer) can offload layers to CPU even
    # when the model comfortably fits on a single GPU, if it estimates memory
    # conservatively or sees stale usage from another process — generation
    # then becomes extremely slow (CPU<->GPU weight shuffling per forward
    # pass) and can look "stuck" rather than just slow. device_map={"":0}
    # forces everything onto one GPU with no offload possible.
    dm = {"": 0} if device_map == "single_gpu" else device_map
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=dm, trust_remote_code=True)
    if load_in_8bit:
        # Fidelity tradeoff, not a free lunch: for a <=40GB GPU that can't fit
        # the full bf16 model. If anything about the reasoning-trace
        # classification looks surprising/borderline, re-verify on an 80GB
        # card before trusting it — quantized weights could in principle
        # change generation behavior, not just precision.
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
        print("  WARNING: loading in 8-bit (GPU too small for full bf16). "
              "This is a fidelity tradeoff — re-verify on an 80GB card if "
              "anything about the result looks surprising.")
    try:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            print(f"  AutoModelForCausalLM failed ({type(e).__name__}: {e}) — "
                  f"not the known unrecognized-config-class case; read the "
                  f"traceback below before assuming a cause.")
            raise
        # The checkpoint's own config.json names the exact class it expects
        # (its "architectures" field) — more reliable than guessing through
        # transformers' Auto-class list. Resolve and retry with that instead.
        import transformers
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        arch_names = getattr(cfg, "architectures", None) or []
        if not arch_names:
            print(f"  AutoModelForCausalLM failed and config.json has no "
                  f"'architectures' field to fall back to — read the "
                  f"traceback below.")
            raise
        arch = arch_names[0]
        cls = getattr(transformers, arch, None)
        if cls is None:
            print(f"  config.json says architectures=[{arch!r}] but "
                  f"transformers.{arch} doesn't exist in this install — "
                  f"check the transformers version, or the class may need "
                  f"importing from a model-specific submodule instead.")
            raise
        print(f"  AutoModelForCausalLM doesn't recognize this config; "
              f"config.json names {arch!r} directly — retrying with that class.")
        model = cls.from_pretrained(hf_id, **kwargs)
    except Exception as e:
        print(f"  AutoModelForCausalLM failed ({type(e).__name__}: {e}).")
        print(f"  Read the full traceback below before assuming a cause — this "
              f"can be a quantization/hardware incompatibility (e.g. "
              f"Qwen3.8-27B-FP8 needs compute capability >=8.9 for native FP8 "
              f"and hit a bug in this transformers version's bf16-dequant "
              f"fallback on an A100 — use the bf16 checkpoint directly there) "
              f"or something else entirely — don't assume without reading it.")
        raise

    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
           [{"role": "user", "content": PROMPT}]
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
    "qwen": lambda a: check_local(
        "qwen_3_8", a.hf_id or a.model_id or "Qwen/Qwen3.8-27B-FP8",
        a.system_prompt, a.device_map, a.load_in_8bit),
    # DeepSeek-V4 swapped for Muse Glimmer 30B (see roster_check.md). Its card
    # describes reasoning depth as a system-prompt control, not a generation
    # kwarg — default here to a sensible non-empty value so a bare `--model
    # muse` run doesn't silently under-elicit reasoning; override with
    # --system-prompt if you want a different effort level.
    "muse": lambda a: check_local(
        "muse_glimmer_30b", a.hf_id or a.model_id or "meta-models/Muse-Glimmer-30B",
        a.system_prompt or "Reasoning strength: high", a.device_map, a.load_in_8bit),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CHECKS) + ["all"], required=False)
    ap.add_argument("--model-id", default=None, help="override the API model id")
    ap.add_argument("--hf-id", default=None, help="override the HF repo id (local models)")
    ap.add_argument("--system-prompt", default=None,
                    help="local models only — e.g. Muse Glimmer's reasoning-strength control")
    ap.add_argument("--device-map", default="auto",
                    help="local models only — 'auto' (accelerate's balancer, offloads to CPU "
                         "if the model doesn't fit — slow, not stuck) or 'single_gpu' (force "
                         "everything onto GPU 0 — will OOM instead of offloading if it doesn't "
                         "fit, only use once you've confirmed the model fits your GPU's memory)")
    ap.add_argument("--load-in-8bit", action="store_true",
                    help="local models only — quantize to fit a smaller GPU (e.g. 40GB card "
                         "that can't hold a ~54GB bf16 model). Fidelity tradeoff, not free — "
                         "re-verify on a big-enough card for full precision if results surprise you")
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
