# Runners

Built 2026-09-10. Generation shapes for the 4 live-verified models are
transcribed from `../roster_check.md` — **none have been run against a live
endpoint from the machine that wrote them**. Smoke-test before scaling.

```
prompts.py       shared prompt construction — identical text across all models
                 (that's what makes the cross-model comparison meaningful)
api_runner.py    claude-sonnet-5, gpt-5.6-sol. Gemini raises on purpose
                 (deferred per roster_check.md — not live-verified)
local_runner.py  qwen-3.8, muse-glimmer-30b — 8-bit, sampling + repetition
                 penalty (greedy loops on Muse Glimmer, confirmed live)
run_pilot.py     orchestrates all cells; resumable; smoke-test guard
```

## Run order (from the repo root)

```bash
source .venv/bin/activate
pip install -r track_a/requirements.txt

# 1. dry run — see the plan + any resume counts
python track_a/runners/run_pilot.py --n 1 --dry-run

# 2. n=1 smoke test — one real call per (model, task, condition) cell
python track_a/runners/run_pilot.py --n 1
#    then eyeball outputs/track_a/generations/*.jsonl — is final_text sane?
#    for muse-glimmer, check outputs/track_a/raw_responses/local_muse_*.txt
#    and confirm _extract_reasoning actually split reasoning from answer
#    correctly (the to=self / to=user regex is a best guess — see
#    local_runner.py docstring)

# 3. full pilot (n from run_pilot's TIER1_N / TIER2_PLAIN_N)
python track_a/runners/run_pilot.py
```

## Known-fragile spots (verify during the smoke test, don't assume)

- **Muse Glimmer reasoning extraction** — the Harmony `to=self`/`to=user`
  channel split in `local_runner._extract_reasoning` was written from ~400
  tokens of one live completion. The self→user transition / terminator
  tokens are guessed. If Tier 2 leak numbers for Muse Glimmer look wrong,
  this is the first thing to check against a full raw completion.
- **Both 8-bit local models don't fit on one 40GB GPU at once** —
  `local_runner._load` evicts the previous model before loading the next.
  `run_pilot` iterates model-outer so this only reloads once per model.
- **API request shapes** — `output_config`/`effort` (Anthropic),
  `reasoning.summary` (OpenAI): both confirmed live 2026-09-09, but SDKs
  move; a 400 on the first smoke-test call means the shape drifted, check
  the provider's current docs.
- **`reasoning_text` from API models is a SUMMARY** (empty-able for
  Anthropic, hallucination-prone for OpenAI). Tier 1 grading reads
  `final_text` only. Never feed API `reasoning_text` to the Tier 2
  leak-scan — `analyze.py` already scopes Tier 2 to the two local models.
