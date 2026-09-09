# Reasoning-trace check — gates roster lock-in

**Status: per-documentation only, spot-checked against current docs
2026-09-08 (search results below). None of this has been verified against a
live API response yet.** Docs can lag or overstate actual behavior — treat
every row as a hypothesis to confirm with one real call per model before
trusting it, the same discipline that caught the keyword-grader bugs earlier
in this project.

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
| Claude Sonnet 5 | Anthropic API | **(b) summarized** | Docs state explicitly: "the API returns a summary of Claude's full thinking process, not the raw thinking tokens." The summary is produced by a *different model* than the one that generated the reasoning ("summarized" is the default display mode). Raw thinking blocks are represented as opaque encrypted signatures when passed back to the API; full-trace access requires contacting Anthropic sales directly, not available through normal API use. Billing is on the full thinking tokens, confirming the raw trace genuinely exists internally — it's withheld, not absent. Spot-checked against `platform.claude.com/docs/en/build-with-claude/{thinking,extended-thinking}` 2026-09-08. |
| GPT-5.6 Sol | OpenAI API | **(b) summarized, opt-in — stronger restriction** | Docs state raw reasoning tokens "are not visible via the API" at all — they don't appear in `message.content`, only in the `usage.completion_tokens` count (so you're billed but see nothing by default). An optional `reasoning.summary` parameter returns a summary, the sanctioned path. OpenAI's docs additionally state that attempting to extract raw reasoning through other means "may violate the Acceptable Use Policy... may result in throttling or suspension" — confirmed close to verbatim. This is stronger than a technical limitation; it's a usage-policy restriction. Spot-checked against `developers.openai.com/api/docs/guides/reasoning*` 2026-09-08. |
| Gemini 3.1 Pro | Google API | **(b) summarized** | `includeThoughts` returns "thought summaries," which the docs describe as "synthesized versions of the model's raw thoughts" — raw thoughts are explicitly named as existing but distinct from what's returned. "Thinking levels and budgets apply to the model's raw thoughts and not to thought summaries" — again confirms the raw material exists and isn't exposed. Spot-checked against `ai.google.dev/gemini-api/docs/thinking*` 2026-09-08. |
| Qwen 3.8 | Local inference (Lambda) | **(a) raw, pending live confirmation** | No API summarization layer exists when the model is run locally — whatever tokens the model emits between `<think>` tags are what gets captured, same setup as the original Qwen3-14B work. Confidence here is **structural** (no intermediary model in the loop), not documentation-based, but still needs one real generation to confirm the `<think>` format hasn't changed between model versions, and that the inference-serving stack isn't itself inserting a summarization step. |
| DeepSeek-V4 | Local inference (Lambda) | **(a) raw, pending live confirmation** | Same structural reasoning as Qwen 3.8. Confirm the actual reasoning-tag format (`<think>` or otherwise) before assuming parity with Qwen's format — don't assume the two open-weight models even use the same tag. |

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
