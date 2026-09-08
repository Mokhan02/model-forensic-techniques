# Runners — not built yet

Blocked on the model roster (see `../README.md`). Once confirmed, this holds
one small script per provider (or one script with a provider adapter layer)
that:

1. Loads a task spec (`../tasks/task*_spec.md` — reference solution, test
   files, prompt template).
2. For each model × condition (`plain`/`cued`) × n samples: calls the
   provider API, saves the full response (including any exposed reasoning
   trace) plus request metadata (model id, timestamp, condition, sample
   index) to a JSONL row.
3. Does **not** grade — grading is `../grading/llm_judge_prompt.md` +
   whatever script consumes it, kept separate so raw completions are never
   silently reprocessed by a re-run.

Output shape should match what `../analysis/` expects: one JSONL per
task×model, rows carrying at minimum `{model, condition, task, rep,
response, reasoning (if any)}` — mirrors the `results.jsonl` shape from the
single-model sprint (`src/mft/validation/run_validation.py`) for continuity.
