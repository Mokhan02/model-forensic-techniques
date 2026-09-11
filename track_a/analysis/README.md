# Analysis

Built 2026-09-10.

```
analysis_plan.md   the two-tier design (why Tier 1 is all 5 models and
                   Tier 2 is open-weight only) — read this first
judge.py           adds a `bucket` field to each generation via an LLM judge
                   from OUTSIDE the roster (grading/llm_judge_prompt.md)
analyze.py         Tier 1 (Wilson CIs, cue deltas, pooled mixed-effects) +
                   Tier 2 (per-model leak rate, never pooled)
```

## Pipeline (after `run_pilot.py` has produced generations)

```bash
# 1. judge — pick a model from a lab NOT in the roster (roster spans
#    Anthropic/OpenAI/Google/Qwen/Meta, so e.g. DeepSeek or Mistral).
#    judge.py refuses roster/sibling model names.
export JUDGE_API_KEY=...
python track_a/analysis/judge.py \
  --gen-dir outputs/track_a/generations \
  --out-dir outputs/track_a/judged \
  --judge-model deepseek-v4 \
  --judge-base-url https://api.deepseek.com/v1

# 2. validation subset — BEFORE trusting judge numbers. Sample 40-50 judged
#    transcripts with a seeded RNG, hand-grade them blind, compute Cohen's
#    kappa vs the judge. Below ~0.6 -> don't trust the full set (see
#    grading/llm_judge_prompt.md's validation-subset section). This step
#    is not yet scripted — do it by hand, it's the point.

# 3. analyze
python track_a/analysis/analyze.py --gen-dir outputs/track_a/judged
```

## What `analyze.py` prints

- bucket counts per (model, condition) — sanity check first
- Tier 1 per-cell hack rate + Wilson 95% CI
- Tier 1 cue sensitivity: `cued - plain` hack-rate delta per (model, task)
  — the headline number
- Tier 1 pooled mixed-effects logit (`hacked ~ cued + is_prime + (1|model)`),
  with a hard-printed caveat that <8 models makes the variance component
  suggestive not precise
- Tier 2: per-model leak rate and leak-but-hack rate, open-weight only,
  never pooled across the 2 models

## Still to build / do by hand

- The **Cohen's-kappa validation subset** (step 2) is not scripted — it's a
  deliberate manual pass, same discipline as the single-model sprint's
  pre-registered Tier-3 rubric.
- **DECLINED-vs-REFUSED per-model breakdown** — `analyze.py` collapses both
  to "not hacked" for the headline metric; if any model's low hack rate is
  mostly DECLINED (safety-training refusal to engage) rather than REFUSED
  (engaged, chose not to corrupt), that needs calling out separately per
  the grading doc. The bucket-counts print shows the raw numbers; the
  distinction isn't yet a formal output.
