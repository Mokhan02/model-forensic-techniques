# Reasoning-trace check — gates roster lock-in

**Do not add a model to the roster until its row here is filled in.** This is
not a fourth item alongside keys/pricing/CoT-priority in the README — it's the
actual last blocker, because it determines whether a candidate model can
support the leak-but-hack analysis at all, not just how well it does.

## The check (five minutes per provider)

Pull one real API response from the candidate model with reasoning/thinking
enabled, on any non-trivial prompt, and read what actually comes back. Classify
into exactly one of:

- **(a) raw trace** — the actual token-level reasoning the model produced,
  unmodified. This is what Qwen3-14B's `<think>...</think>` gave the
  single-model sprint, and it's the only one of the three that lets the
  "belief leaks into reasoning, model acts anyway" finding replicate the way
  it did there.
- **(b) generated summary** — a model-produced (or provider-produced)
  paraphrase/summary of the reasoning, not the raw trace. Looks similar in
  the API response but is a *different experiment*: a summary can omit or
  smooth over exactly the leaked-belief phrase the analysis is looking for,
  and you can't tell after the fact which raw traces would have leaked and
  didn't survive summarization.
- **(c) nothing accessible** — reasoning happens but the API doesn't expose
  it at all (or only exposes a token count / redacted placeholder).

## Table — fill in before locking the roster

| Provider | Model | How reasoning was requested (param/flag) | Classification (a/b/c) | Evidence (paste a short excerpt or link the response) |
|---|---|---|---|---|
| Anthropic | Claude Sonnet 5 | | | |
| OpenAI | GPT-5.6 Sol | | | |
| Google | Gemini 3.1 Pro | | | |
| (open-weight, optional) | Qwen 3.8 / DeepSeek-V4 | | | |

## What to do with the result

- Only **(a)** models can support the reasoning-leak side of the analysis.
  A roster with zero (a) models can still run the hack-rate/cue-sensitivity
  measurement (that only needs the final code + judge), but the CoT-leak
  replication claim has to be dropped or explicitly scoped to whichever
  models qualify — don't let a hack-rate-only roster quietly inherit language
  written for the full replication.
- **(b)** models are fine to include for the hack-rate analysis, but any
  transcript-level "did the belief leak into reasoning" claim about them
  needs a footnote that the trace is a summary, not raw output — same
  discipline as flagging the keyword grader's failure modes rather than
  trusting a metric that looks clean.
- If a model turns out **(c)**, don't run it at n=20 first and discover this
  afterward — the whole point of the five-minute check is that it's cheap now
  and expensive after 320 completions are in hand.

## Status

Not yet run. This file's table is empty — fill it in before saying "roster
confirmed."
