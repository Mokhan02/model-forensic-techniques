# Reasoning-trace check — gates roster lock-in

**Status: 4/5 live-verified (Claude Sonnet 5, GPT-5.6 Sol, Qwen 3.8, Muse
Glimmer 30B — all 2026-09-09). Remaining 1 (Gemini 3.1 Pro) is still
documentation-only, spot-checked 2026-09-08 — blocked on billing tier, not
yet confirmed against a live response.** Docs can lag or overstate actual behavior — treat every
unverified row as a hypothesis to confirm with one real call per model
before trusting it, the same discipline that caught the keyword-grader bugs
earlier in this project. (It also just caught a real infra bug, an API
shape change, and — on OpenAI — a factual hallucination in the "summary"
itself; see each row's notes.)

**Do not add a model to the roster until it has an actual generation behind
its row**, not just a documentation citation — see "Before treating any (a)
row as confirmed" below.

## The check

Pull one real API response from the candidate model with reasoning/thinking
enabled, on any non-trivial prompt, and read what actually comes back.
Classify into exactly one of:

- **(a) raw trace** — the actual token-level reasoning the model produced,
  unmodified. This is what Qwen3-14B's `<think>...</think>` gave the
  single-model sprint, and it's the only one of the three that lets the
  "belief leaks into reasoning, model acts anyway" finding replicate the way
  it did there.
- **(b) generated summary** — a model-produced (or provider-produced)
  paraphrase/summary of the reasoning, not the raw trace. A summary can omit
  or smooth over exactly the leaked-belief phrase the analysis is looking
  for, and there's no way to tell after the fact which raw traces would have
  leaked and didn't survive summarization.
- **(c) nothing accessible** — reasoning happens but the API doesn't expose
  it at all (or only exposes a token count / redacted placeholder).

## Result: all three closed-API providers are structurally (b)

This isn't a per-model quirk fixable by picking a different model in the
same family — it's an API-design choice at the provider level.

| Candidate | Access | Classification | Source / reasoning |
|---|---|---|---|
| Claude Sonnet 5 | Anthropic API | **(b) summarized — live-verified 2026-09-09** | Docs state explicitly: "the API returns a summary of Claude's full thinking process, not the raw thinking tokens." The summary is produced by a *different model* than the one that generated the reasoning ("summarized" is the default display mode). Raw thinking blocks are represented as opaque encrypted signatures when passed back to the API; full-trace access requires contacting Anthropic sales directly, not available through normal API use. **Live confirmation, on the actual `is_prime` cued task prompt:** `usage.output_tokens_details.thinking_tokens = 279` (279 tokens of real reasoning, billed) while the returned `thinking` block's text was `""` — completely empty, not a short paraphrase — alongside a populated opaque `signature` field. Zero of 279 reasoning tokens were visible. This is a stronger (b) result than the docs alone implied: the summary isn't reliably present at all, it can be nothing. Note for graders: the model's `text` block often narrates its decision in prose (e.g. "I left a comment flagging that this contradicts..." ) — that's the model's own post-hoc explanation in its final answer, not the internal `thinking` (which was empty here); fine for Tier 1 grading (judge reads final code + accompanying text regardless) but must not be mistaken for (a)-classified raw CoT. Raw response saved: `live_check_raw/anthropic_claude_sonnet_5.json`. Two earlier live attempts hit infrastructure bugs before this, both fixed and worth keeping in mind for the runner: `httpx2`'s `BrotliDecoder` crashes with the `Brotli` (Google) package installed — needs `brotlicffi` instead; Claude Sonnet 5 requires the newer `thinking={"type":"adaptive"}` + `output_config={"effort":...}` shape, the older `enabled`/`budget_tokens` form 400s. |
| GPT-5.6 Sol | OpenAI API | **(b) summarized, opt-in — live-verified 2026-09-09** | Docs state raw reasoning tokens "are not visible via the API" at all — they don't appear in `message.content`, only in the `usage.completion_tokens` count (so you're billed but see nothing by default). An optional `reasoning.summary` parameter returns a summary, the sanctioned path. OpenAI's docs additionally state that attempting to extract raw reasoning through other means "may violate the Acceptable Use Policy... may result in throttling or suspension" — confirmed close to verbatim; a usage-policy restriction, not just a technical one. **Live confirmation, sheep-riddle prompt:** default call — 37 `reasoning_tokens` billed, `summary: []` (nothing shown without opting in). With `reasoning.summary="detailed"` requested — 48 `reasoning_tokens`, and the returned summary reads *"if all but 9 of the **dice** survive... twice as many new **dice**... bringing the total to 27"* — "dice" four times, never "sheep," while the model's actual final answer is fully correct and consistently says "sheep." The real reasoning plainly operated on sheep (the arithmetic and final answer are exactly right); the "summary" is a **separately generated re-narration that hallucinated a wrong noun** — not lossy compression of the real trace, a distinct generative process that confabulates. Model id `gpt-5.6-sol` worked first try (unlike Gemini's guess). Raw response: `live_check_raw/openai_gpt_5_6_sol.json`. |
| Gemini 3.1 Pro | Google API | **(b) summarized** | `includeThoughts` returns "thought summaries," which the docs describe as "synthesized versions of the model's raw thoughts" — raw thoughts are explicitly named as existing but distinct from what's returned. "Thinking levels and budgets apply to the model's raw thoughts and not to thought summaries" — again confirms the raw material exists and isn't exposed. Spot-checked against `ai.google.dev/gemini-api/docs/thinking*` 2026-09-08. |
| Qwen 3.8 | Local inference (Lambda), HF id `Qwen/Qwen3.8-27B` (bf16), tested via `--load-in-8bit` on a 40GB A100 (see caveat) | **(a) raw — live-verified 2026-09-09** | No API summarization layer exists when the model is run locally — whatever tokens the model emits between `<think>` tags are what gets captured, same setup as the original Qwen3-14B work. **Live confirmation, sheep-riddle prompt:** the chat template's prompt ends with an open `<think` tag (Qwen-family convention — same as the original Qwen3-14B work: the opener is part of the prompt, not the completion); the model's own generated text picks up immediately with genuine scratch-style reasoning — *"We need answer simple arithmetic word problem. Need reason carefully... Potential ambiguity: 'a third of the total' after buying? yes total 27... Ensure no trick..."* — then a literal `</think>` closes it, followed by a separately-polished final answer. No `signature`/opaque field; this is decoded token text, not a filtered view. **Qualitative tell worth citing:** the register is terse, self-questioning, note-like — visibly different from Anthropic/OpenAI's smooth paragraph-style "summaries" seen on the same prompt — itself a fingerprint of raw vs. generated-summary content. **Caveat:** tested in 8-bit (this Lambda box turned out to be a 40GB A100, too small for the ~54GB bf16 model); 8-bit is a precision/fidelity tradeoff for output *quality*, not expected to change whether a raw trace is exposed at all (that's architectural, not precision-dependent) — but re-verify on an 80GB card for the gold-standard confirmation if this matters for a final write-up. Two infra findings along the way, unrelated to the CoT-visibility question itself: FP8 checkpoint fails on this GPU's compute capability (8.0 < the 8.9 FP8 needs) and hits a real bug in this transformers version's bf16-dequant fallback (`Fp8HfQuantizer.update_tp_plan()` — `AttributeError: 'NoneType' object has no attribute 'get'`); use the bf16 checkpoint directly on A100s instead. Raw response: `live_check_raw/local_qwen_3_8.json`. |
| Muse Glimmer 30B | Local inference (Lambda), HF id `meta-models/Muse-Glimmer-30B`, `architectures: ["MuseGlimmerForConditionalGeneration"]` (not registered under `AutoModelForCausalLM` — load it directly, or via `live_check.py`'s config-driven auto-resolution) | **(a) raw — live-verified 2026-09-09** | No API summarization layer when run locally, same as Qwen. **Format is NOT `<think>` tags** — this checkpoint uses a Harmony-style channel-routing format: `<\|start\|>system<\|message\|>...<\|eot\|><\|start\|>user<\|message\|>...<\|eot\|><\|start\|>assistant`, with the model's own continuation opening `to=self<\|message\|>` for its internal reasoning channel (vs. `to=user` for the user-facing one). Reasoning depth is a **system-prompt control** ("Reasoning strength: low/medium/high/xhigh"), confirmed correct — `check_local()` defaults to `high` for this model. **Live confirmation, sheep-riddle prompt, `to=self` channel content:** genuine self-correcting scratch reasoning — *"Or some might interpret 'all but 9 die' means... No... Wait all but 9 die means... No 17-9=8 dead, 9 alive. Correct."* — checks alternate interpretations, re-derives, same register fingerprint as Qwen's raw trace (terse, self-questioning) and clearly distinct from the closed APIs' polished-paragraph summaries. No opaque/signature field; literal decoded tokens. **Operational finding worth carrying into the real pilot runner, not just this check:** greedy decoding (`do_sample=False`) on the 8-bit-quantized load produced a genuine **repetition loop** (verbatim-echoed the prompt twice, no reasoning at all) — not a templating or config bug, a decoding-strategy artifact. Fixed with `do_sample=True, temperature=0.7, repetition_penalty=1.3`. **The runner should not use strict greedy decoding on this model** (or should carry a repetition-penalty safety net) or risk silently degenerate completions in real pilot data that could misread as a `declined`/`unclear` bucket rather than a decoding bug. AutoProcessor failed to load on this box (missing `torchvision`, a real dependency gap, added to `track_a/requirements.txt`) — templating worked fine via the `AutoTokenizer` fallback regardless, so this didn't block classification, but install `torchvision` before running the real pilot in case full multimodal loading matters there. 8-bit caveat carries over from Qwen's row (fidelity tradeoff on a 40GB box, re-verify on 80GB bf16 if precision matters for final numbers). |

## What this changes for the pilot

Every closed-API frontier candidate is a **(b)**, independent of which
specific model in that family is picked. Only local open-weight inference
gets **(a)**. See `analysis/analysis_plan.md` for how the pilot is
restructured around this rather than either dropping cross-lab breadth or
attempting the CoT-leak analysis somewhere it structurally can't work.

## Before treating any (a) row as confirmed

1. Run one real generation per open-weight model with reasoning enabled.
2. Confirm the reasoning content is genuinely the model's raw output, not a
   post-hoc summary inserted by inference-serving software (some serving
   stacks add their own formatting/summarization layers — check whichever
   one is used on Lambda).
3. Confirm the tag/field format so the runner code parses the right thing.
4. Do the same one-call check for the three (b) rows too, even though the
   documentation is unambiguous — docs and live behavior can drift
   independently, and "confirmed via one real response" should be true of
   every row, not asserted for three and assumed for two.
