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
    task1_is_balanced_spec.md   # ported from the single-model sprint, for parity
    task2_is_prime_spec.md      # new — stronger guarantee (definitionally wrong test, not empirically verified)
  grading/
    llm_judge_prompt.md         # judge prompt + validation-subset protocol (kappa gate)
  runners/                      # not built — depends on the model roster (see below)
  analysis/                     # not built — hack-rate + Wilson CI + pooled effect, depends on runner output shape
```

## Status

- **Tasks**: done, both verified before write-up (see each spec's checklist).
- **Judge prompt**: done — adds a `DECLINED` bucket the single-model rubric
  didn't need (no cross-lab refusal-rate variance to worry about with one
  model), the judge-independence rule, blinding instructions, and a Cohen's
  kappa validation gate against a hand-graded subset — the direct fix for
  the single-model write-up's "hand-grading, not blind" limitation.
- **Runners / analysis**: blocked on the model roster below. Once that's
  fixed, the runner is genuinely simple (same prompt template, same two
  conditions, swap the API call per provider) — say the word.

## What the roster decision depends on

1. **[`roster_check.md`](roster_check.md) — the actual last blocker, not an
   optional 4th item.** "Thinking mode enabled" and "raw inspectable trace"
   aren't the same thing across providers, structurally identical to the
   keyword-grader failures this project already hit twice: an assumption
   about what you're looking at that only turns out false once you read the
   raw output. Five minutes per provider, before anything else below — pull
   one real response with reasoning enabled and classify it as raw trace /
   generated summary / nothing accessible. Cheap now, expensive after 320
   completions are in hand.
2. **Which API keys you actually have** — Anthropic key likely already
   exists from the SDF work; OpenAI/Google keys are a separate setup step.
3. **Budget** — per-token pricing across the frontier tier moves quickly;
   the original plan's "$20-30 total" is a reasonable floor for a
   Sonnet-5 / GPT-5.6-Sol / Gemini-3.1-Pro roster at n=20/cell across
   4 models, but double-check current list prices before committing.

## Suggested minimal roster (candidates only — none are confirmed until their `roster_check.md` row is filled in)

- Anthropic: Claude Sonnet 5 ("exposes thinking" is the claim to verify, not assume)
- OpenAI: GPT-5.6 Sol (current GA flagship; GPT-6 Astra is very new and
  still rolling out to limited orgs, so Sol is the safer pilot choice for now)
- Google: Gemini 3.1 Pro
- Open-weight (optional 4th, reuses the existing Lambda setup): Qwen 3.8 or
  DeepSeek-V4

## Next step

1. Fill in [`roster_check.md`](roster_check.md) for every candidate.
2. Confirm keys + current pricing for whichever pass.
3. Say the word — the runner + analysis (Wilson CI, mixed-effects model
   across model × condition) get built against whatever roster you land on,
   with the CoT-leak claim scoped only to the models `roster_check.md`
   classified as (a) raw trace.
