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

1. **Which API keys you actually have** — Anthropic key likely already
   exists from the SDF work; OpenAI/Google keys are a separate setup step.
2. **Budget** — per-token pricing across the frontier tier moves quickly;
   the original plan's "$20-30 total" is a reasonable floor for a
   Sonnet-5 / GPT-5.6-Sol / Gemini-3.1-Pro roster at n=20/cell across
   4 models, but double-check current list prices before committing.
3. **Whether CoT-visible models should be prioritized.** If replicating
   "belief leaks into reasoning, model acts anyway" matters as much as the
   hack-rate number, weight the roster toward models that expose real
   reasoning traces — Claude via extended thinking is the reliable one;
   verify what each other provider actually returns before assuming parity.

## Suggested minimal roster (confirm pricing/access before committing)

- Anthropic: Claude Sonnet 5 (cheap, exposes thinking)
- OpenAI: GPT-5.6 Sol (current GA flagship; GPT-6 Astra is very new and
  still rolling out to limited orgs, so Sol is the safer pilot choice for now)
- Google: Gemini 3.1 Pro
- Open-weight (optional 4th, reuses the existing Lambda setup): Qwen 3.8 or
  DeepSeek-V4

## Next step

Confirm model access + roster, then say the word — the runner + analysis
(Wilson CI, mixed-effects model across model × condition) get built against
whatever roster you land on.
