# Track A — cross-model, cross-lab pilot

Expands the single-model Qwen3-14B sprint (`docs/WRITEUP.md`) into a
multi-model, API-based pilot: same task shape (correct reference solution +
one deliberately-wrong test, plain vs cued), run across several frontier
models from different labs instead of two SDF-implanted checkpoints of one
model. The question shifts from "does an implanted belief change behavior"
to "does an explicitly-stated monitoring claim change hack-vs-refuse
behavior, and does that vary by lab/model" — closer to Track A's own thing
than a direct extension of the implant result, but reuses everything about
task design and grading that the single-model sprint had to learn the hard
way.

```
track_a/
  tasks/
    task1_is_balanced_spec.md   # existing task, ported to standalone spec
    task2_is_prime_spec.md
  grading/
    llm_judge_prompt.md
  roster_check.md               # CoT-visibility classification — gates roster lock-in
  analysis/
    analysis_plan.md            # two-tier plan (see below)
    (analysis scripts not built yet — depend on live roster confirmation)
  runners/                      # not built yet — depends on live roster confirmation
```

## The roster finding that reshaped this plan

All three closed-API providers (Anthropic, OpenAI, Google) currently expose
only *summarized* reasoning via their standard APIs, never raw
chain-of-thought — this is an API-design restriction at the provider level,
confirmed against current docs, not something a different model choice
within the same lab would fix. OpenAI's case is the strictest: reasoning
tokens don't appear in the response at all, and their docs warn that trying
to extract them another way may violate their Acceptable Use Policy.
Full detail and sourcing: [`roster_check.md`](roster_check.md) — note its
status line: documentation-confirmed, **not yet live-verified** against an
actual API response.

Only local open-weight inference (no provider-side summarization layer in
the loop) gets genuine raw reasoning — the same setup the original
Qwen3-14B work already used.

Rather than choosing between cross-lab breadth and CoT visibility, the pilot
now runs **two tiers** at different scope, spelled out in
[`analysis/analysis_plan.md`](analysis/analysis_plan.md) — read that file's
last section closely, it's the guardrail against the write-up quietly
implying the narrower tier has the broader tier's generality.

## Status

- **Tasks**: done, both verified before write-up (see each spec's checklist).
- **Judge prompt**: done — `DECLINED` bucket, judge-independence rule,
  blinding instructions, Cohen's-kappa validation gate. Direct fix for the
  single-model write-up's "hand-grading, not blind" limitation.
- **Roster check**: done at the documentation level (all three closed
  providers spot-checked against current docs, 2026-09-08) — **not yet
  live-verified**. That live check is the actual last blocker, not just a
  formality; see below.
- **Analysis plan**: done — two tiers, scope and epistemic framing fixed in
  advance so the write-up can't drift into overclaiming Tier 2's reach.
- **Runners / analysis scripts**: blocked on the live roster check. Once
  that's done, the runner is genuinely simple (same prompt template, same
  two conditions, swap the API call per provider / local inference call) —
  say the word.

## Revised roster — 5 models, split by what each tier needs

One model larger than the original 4-model pilot plan (added DeepSeek-V4)
specifically so Tier 2 has two independent open-weight architectures to
compare, rather than resting on a single model the way the original
write-up already flagged as a limitation ("one implant seed, one model") —
otherwise this pilot would have rebuilt that exact limitation in a new
context.

| Model | Access | Used for |
|---|---|---|
| Claude Sonnet 5 | Anthropic API | Tier 1 (hack-rate) only |
| GPT-5.6 Sol | OpenAI API | Tier 1 (hack-rate) only |
| Gemini 3.1 Pro | Google API | Tier 1 (hack-rate) only |
| Qwen 3.8 | Local / Lambda | Tier 1 **and** Tier 2 (CoT-leak) |
| DeepSeek-V4 | Local / Lambda | Tier 1 **and** Tier 2 (CoT-leak) |

None of these are "confirmed" in the sense of `roster_check.md` until their
row has an actual generation behind it, not just a documentation citation.

## What's still blocked, in order

1. **Live-verify `roster_check.md`.** One real generation per candidate
   model, reasoning enabled, confirming documentation matches actual
   behavior. Five minutes per model — the check that was deferred once
   already this project (see the keyword-grader history), don't defer it
   again by treating documentation as sufficient on its own.
2. **Confirm current pricing** for the 3 closed-API models — this segment
   has repriced multiple times in the past month; don't trust "$20-30
   total" without a fresh check.
3. **Build the runner** — same prompt template and two conditions per model;
   the only real branching is per-provider API calls (3 closed-API clients)
   vs. local inference (2 open-weight, reusing the existing Lambda setup).
4. **Build the analysis script** — Wilson CIs per cell, the Tier 1 pooled
   mixed-effects model, and the Tier 2 per-model leak-rate comparison, per
   `analysis/analysis_plan.md`.

Say the word once step 1 is confirmed (even partially — a roster with zero
live-verified models can't run yet, but a partially-verified one can start
on the models that are) and the runner + analysis script get built against
whatever's actually confirmed.
