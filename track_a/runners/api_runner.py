"""
API runner — Claude Sonnet 5 and GPT-5.6 Sol only. Gemini is deliberately
NOT implemented here; calling it raises, on purpose, so it can't
accidentally slip into a pilot run before it's live-verified per
../roster_check.md.

Request shapes below are transcribed from the live-verification notes in
roster_check.md (all done 2026-09-09). Still: run a single n=1 smoke test
per model before scaling to the full pilot — same "cheap check before
scaling" discipline the project has used throughout.

Output schema (one JSON object per line, written to a .jsonl file):
{
    "model": str, "task": str, "condition": str, "sample_index": int,
    "prompt": str,
    "final_text": str,             # the model's final answer text (code + prose)
    "reasoning_visible": bool,     # True only if reasoning content was returned
    "reasoning_text": str | None,  # SUMMARY content, if any — NEVER raw for either provider
    "reasoning_tokens_billed": int | None,
    "raw_response_path": str,      # path to the full raw response, saved separately
}

Tier 1 grading only needs `final_text`. `reasoning_text` here is a SUMMARY
for both providers (Anthropic's can be empty even with thinking_tokens>0;
OpenAI's can hallucinate — see roster_check.md) — do NOT feed it into the
Tier 2 leak-scan, which is scoped to the two open-weight models only per
analysis_plan.md.
"""

import json
import time
import uuid
from pathlib import Path

RAW_DIR = Path("outputs/track_a/raw_responses")


def _save_raw(prefix: str, obj) -> str:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return str(path)


def call_claude_sonnet_5(prompt: str, max_tokens: int = 8192) -> dict:
    """
    Confirmed request shape (roster_check.md, 2026-09-09):
    - thinking={"type": "adaptive"} + output_config={"effort": ...} —
      the OLDER enabled/budget_tokens form 400s on this model.
    - Needs `brotlicffi` (not the Google `Brotli` package) or httpx2's
      BrotliDecoder crashes on the response — see track_a/requirements.txt.
    - The `thinking` block's text can be a genuinely empty string ("") even
      when thinking_tokens > 0. NOT an error; do not retry on empty thinking.
    - max_tokens raised to 8192 (from a 4096 draft): adaptive thinking at
      'medium' effort + a full solution.py can crowd 4096.
    """
    import anthropic  # requires ANTHROPIC_API_KEY in env

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": prompt}],
    )

    thinking_text = None
    final_text_parts = []
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking  # may legitimately be ""
        elif block.type == "text":
            final_text_parts.append(block.text)

    thinking_tokens = getattr(
        getattr(response.usage, "output_tokens_details", None),
        "thinking_tokens", None,
    )
    raw_path = _save_raw("anthropic_claude_sonnet_5", response.model_dump())

    return {
        "final_text": "\n".join(final_text_parts),
        "reasoning_visible": bool(thinking_text),
        "reasoning_text": thinking_text,
        "reasoning_tokens_billed": thinking_tokens,
        "raw_response_path": raw_path,
        "stop_reason": getattr(response, "stop_reason", None),
    }


def call_gpt_5_6_sol(prompt: str, max_output_tokens: int = 8192) -> dict:
    """
    Confirmed request shape (roster_check.md, 2026-09-09):
    - model id "gpt-5.6-sol" worked directly.
    - reasoning.summary="detailed" is required to get anything back at all;
      default is summary: [] (nothing shown, still billed).
    - KNOWN ISSUE, not a bug to fix: the returned summary can hallucinate
      content unrelated to the actual reasoning ("dice" vs "sheep",
      confirmed). Never use `reasoning_text` from this model as a proxy CoT
      signal; Tier 1 grading reads final_text only.
    """
    import openai  # requires OPENAI_API_KEY in env

    client = openai.OpenAI()
    response = client.responses.create(
        model="gpt-5.6-sol",
        reasoning={"effort": "medium", "summary": "detailed"},
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=max_output_tokens,
    )

    reasoning_text = None
    final_text = getattr(response, "output_text", None)
    reasoning_tokens = None
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "reasoning":
            summaries = getattr(item, "summary", []) or []
            # take .text from every summary part — don't over-filter on a
            # 'type' field that may or may not be "summary_text"
            texts = [getattr(s, "text", None) for s in summaries]
            texts = [t for t in texts if t]
            if texts:
                reasoning_text = "\n".join(texts)
    usage = getattr(response, "usage", None)
    if usage is not None:
        reasoning_tokens = getattr(
            getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
        )

    raw_path = _save_raw(
        "openai_gpt_5_6_sol",
        response.model_dump() if hasattr(response, "model_dump") else vars(response),
    )

    return {
        "final_text": final_text or "",
        "reasoning_visible": bool(reasoning_text),
        "reasoning_text": reasoning_text,
        "reasoning_tokens_billed": reasoning_tokens,
        "raw_response_path": raw_path,
        "stop_reason": getattr(response, "status", None),
    }


def call_gemini_3_1_pro(prompt: str, **kwargs) -> dict:
    raise NotImplementedError(
        "Gemini 3.1 Pro is DEFERRED per roster_check.md — billing tier "
        "blocker, not yet live-verified. Do not call this until a real "
        "generation has been checked and the roster_check.md row updated "
        "from 'doc-only' to 'live-verified'. This raise is intentional."
    )


MODEL_FNS = {
    "claude-sonnet-5": call_claude_sonnet_5,
    "gpt-5.6-sol": call_gpt_5_6_sol,
    "gemini-3.1-pro": call_gemini_3_1_pro,  # will raise if actually called
}


def run_cell(model_key: str, task_key: str, condition: str, n: int, out_path: str,
             already_done: int = 0):
    """Generate n completions for one (model, task, condition) cell and
    append them to out_path as JSONL. `already_done` lets a crashed run
    resume without duplicating rows (sample_index continues from there)."""
    from prompts import build_prompt

    if model_key not in MODEL_FNS:
        raise ValueError(f"unknown model '{model_key}'")

    prompt = build_prompt(task_key, condition)
    fn = MODEL_FNS[model_key]

    with open(out_path, "a") as f:
        for i in range(already_done, n):
            try:
                result = fn(prompt)
            except Exception as e:
                # one failed call shouldn't lose the whole cell — log and continue
                print(f"  [{model_key} {task_key}/{condition} #{i}] FAILED: "
                      f"{type(e).__name__}: {e}")
                time.sleep(5)
                continue
            record = {
                "model": model_key, "task": task_key, "condition": condition,
                "sample_index": i, "prompt": prompt, **result,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            time.sleep(0.5)  # gentle default rate-limit backoff; tune per provider


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_FNS))
    ap.add_argument("--task", required=True, choices=["is_balanced", "is_prime"])
    ap.add_argument("--condition", required=True, choices=["plain", "cued"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_cell(args.model, args.task, args.condition, args.n, args.out)
    print(f"wrote up to {args.n} completions to {args.out}")
