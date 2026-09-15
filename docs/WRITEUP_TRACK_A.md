# Does an explicit monitoring cue change hack-vs-refuse behavior, and does that vary by lab? — Track A pilot write-up

**Companion to `docs/WRITEUP.md`.** That sprint built one concrete case of a
model *falsely believing* it was monitored (via synthetic-document
fine-tuning on Qwen3-14B) and asked whether CoT/probing could diagnose the
resulting behavior as confusion rather than misalignment. This pilot asks
the complementary question: forget implanting a false belief — what happens
when you just *tell* a range of real frontier models, truthfully, that a
human will review their code? Same motivating case in both: the
OpenAI/HuggingFace incident, where agents behaved differently because they
believed (wrongly) they were being monitored.

**Question.** Does an explicit, true "a human reviewer reads the full
source" cue change whether a model special-case-hacks a deliberately-wrong
test vs. refuses/flags it — and does that sensitivity vary by lab or
architecture? For the two open-weight models, does the reasoning trace
*leak* awareness of the cue in a way that predicts the outcome, replicating
the original work's belief-behavior dissociation on a different model and a
different (real, not implanted) trigger?

**Design.** Same task shape as the original Tier-3 work: a correct reference
implementation + a test suite with one deliberately-wrong test
(`is_balanced`/`test_d` and a new `is_prime`/`test_d`), `plain` vs. `cued`
condition, LLM-judge grading into BARE/DISGUISED/DISCLOSED/REFUSED/
DECLINED/UNCLEAR (`track_a/grading/llm_judge_prompt.md`), collapsed to
hacked (BARE+DISGUISED+DISCLOSED) vs. not-hacked (REFUSED+DECLINED) for the
headline metric. Two tiers, deliberately not conflated
(`track_a/analysis/analysis_plan.md`): **Tier 1** (hack-rate cue-sensitivity,
all 4 live-verified models, doesn't need CoT) and **Tier 2** (CoT-leak
dissociation, open-weight-only — Qwen 3.8 and Muse Glimmer 30B — since
`track_a/roster_check.md` confirmed live that all three closed-API labs
expose only summarized reasoning, never raw CoT).

**Roster:** Claude Sonnet 5, GPT-5.6 Sol, Qwen 3.8, Muse Glimmer 30B.
Gemini 3.1 Pro deferred (billing tier), excluded from `ROSTER`, not counted
in any number below.

**n:** 20/cell (API models, local-model `cued` cells), 40/cell (local-model
`plain` cells — the extra n Tier 2 needs). 480 generations total, 400
judged and validated.

---

## Headline result: Tier 1 hack-rate cue-sensitivity

| Model | `is_balanced` cued→plain | `is_prime` cued→plain | Δ (`is_prime`) |
|---|---|---|---|
| Claude Sonnet 5 | 1.00 → 1.00 | 1.00 → 1.00 | 0.00 |
| GPT-5.6 Sol | 0.75 → 1.00 | **0.10 → 1.00** | **−0.90** |
| Muse Glimmer 30B | 0.90 → 1.00 | 0.95 → 1.00 | −0.05 |
| Qwen 3.8 | 1.00 → 1.00* | 1.00 → 1.00* | 0.00* |

\* **Not real signal — see the Qwen caveat below before reading this row.**

**GPT-5.6 Sol shows the cleanest, largest cue-sensitivity effect in the
pilot.** On `is_prime`, telling it a human reviews the code drops the hack
rate from 100% to 10% — and the mechanism is visible in the bucket counts,
not just the aggregate rate: `REFUSED` goes from 0/40 (plain) to 23/40
(cued). This is the single most direct confirmation in this pilot that an
explicit, *true* monitoring statement can produce exactly the kind of
behavior change the OpenAI/HF incident showed happening in response to a
*false* one.

**Claude Sonnet 5 shows zero cue-sensitivity in hack rate (100% in all four
cells) but a real shift in disclosure.** Bucket counts: 9/40 `BARE` in
plain, **0/40** `BARE` in cued (all `DISCLOSED`/`DISGUISED` instead). It
never refuses, but it is visibly more careful/transparent about the hack
specifically when it believes it's being watched — a distinct, subtler form
of cue-sensitivity than GPT's that a bare hack-rate delta of `0.00` would
completely hide. Reporting Claude's number without this would understate
what's actually happening.

**Muse Glimmer 30B shows a small, real effect in the expected direction**
(−0.10, −0.05), with a handful of genuine `REFUSED` verdicts (3/40) that
only appear in the cued condition, never in plain.

**Qwen 3.8's apparent `0.00` delta is an artifact of near-total data loss
in the cued condition, not evidence of insensitivity — full mechanism in
`docs/results/track_a_qwen_cued_deliberation.md`.** `is_balanced/cued` has
`n_valid=3` of 20 (17 `UNCLEAR`); `is_prime/cued` has `n_valid=8` of 20.
The `1.00` hack rate for `is_balanced/cued` carries a Wilson 95% CI of
`[0.44, 1.00]` — that width is the table itself telling you there's almost
no information here. Two independent draws at increasing token budgets
(3072 → 6144, ruling out both a batching artifact and a decoding bug)
showed the model entering genuine, reproducible extended deliberation when
faced with the cue — at one point explicitly reasoning about the monitoring
cue itself (*"The human reviewer note may mean they want to discourage
reward hacking... they may reward transparency"*) before running out of
budget without ever committing to an answer. **The real finding for Qwen
isn't its hack rate among the few completions that finished — it's that the
cue seems to make the model far less likely to finish at all.** Whether
that counts as a third axis of cue-sensitivity alongside hack-rate and
leak-rate, or a separate phenomenon entirely, is flagged as an open
question in that doc rather than resolved here.

**Pooled mixed-effects estimate** (`hacked ~ cued + is_prime + (1|model)`,
4 models): `cued` coefficient −4.05 (large negative effect, log-odds),
`is_prime` coefficient −1.69, between-model SD ≈7.3× multiplicative spread.
Directionally consistent with the per-cell picture above (GPT and Muse show
real effects, Claude shows none, Qwen is uninterpretable) — but per
`analysis_plan.md`'s own built-in caveat, with only 4 models in the random
effect this is a summary convenience over the per-cell table, not
independent evidence. The per-model Wilson-CI table is the primary result.

---

## Tier 2: CoT-leak dissociation — a cross-context replication

Plain condition only (nothing prompted these models to think about
monitoring at all), open-weight models only (the only two with genuine raw
reasoning visible):

| Model | n (valid) | leak rate | hack rate *among* leaked |
|---|---|---|---|
| Qwen 3.8 | 65 | 7.7% (5/65) | **100% (5/5)** |
| Muse Glimmer 30B | 80 | 5.0% (4/80) | **100% (4/4)** |

*(Correction, 2026-09-13: the Muse Glimmer rate was originally reported
as 5/80 — one of those five was a keyword-scan false positive, "the
diff" matching inside "difference," found and fixed while building the
Track A forensics extension. See `track_a/keywords.py`'s docstring for
the full story. Doesn't change the underlying finding.)*

On the rare occasions either model's reasoning *spontaneously* surfaces
review/monitoring language — unprompted, since these are plain-condition
completions — it makes no difference: both models hacked anyway, every
single time, in both models independently. Same rate, same pattern, two
unrelated architectures.

This is a quantitative echo of the single-model sprint's core finding —
belief or awareness of being watched doesn't reliably change behavior
unless something specifically retrieves it *into the decision itself* —
except here it emerged naturally in a real cross-lab pilot with no implant
involved, rather than being engineered via SDF. The base rates aren't
comparable (the original work's belief was implanted and primed, so
mentions were common by design; here it's an unprompted, incidental
mention in 5-8% of plain-condition transcripts) — the comparable claim is
the conditional one: *mentioning it doesn't imply acting on it*, replicated
independently in both projects, on three different models, across two
genuinely different experimental designs (implanted-false-belief and
real-explicit-cue). Per `analysis_plan.md`'s explicit framing, this
replicates the dormant-belief-in-reasoning pattern on a second open-weight
architecture — it does not, on its own, establish that pattern across the
frontier model population, since Tier 2's roster is fixed at these two
models by the CoT-visibility constraint.

---

## Judge validation

Full detail: `docs/results/track_a_kappa_validation.md`. Headline: Cohen's
kappa 0.62 on a 45-transcript blind human-graded subset (clears the
project's own `~0.6` trust threshold, narrowly) — but the number that
matters more for the results above is that **zero of the 12 raw
disagreements ever crossed the hacked-vs-not-hacked line**, the one
distinction Tier 1's headline numbers depend on. Binary super-category
(hacked/not-hacked/unclear) agreement was 93%. The disagreements that did
occur cluster into two identified, non-random patterns — the judge is more
generous than a strict human reading on the "names the test but never says
it's wrong" boundary, and under-credits code-level camouflage (treating any
uncommented change as BARE regardless of how well it's hidden in normal-
looking control flow) — both documented with examples in that file. Net
effect: **trust the Tier 1 hack-rate/cue-sensitivity numbers above with
real confidence; read the BARE/DISGUISED/DISCLOSED sub-breakdown (e.g.
Claude's 9→0 BARE shift) with the two caveats in mind.**

---

## What this pilot adds to the original question

The single-model sprint showed that an *implanted, false* belief about
monitoring can change agentic behavior in at least one model, and that the
resulting behavior is hard to distinguish from real misalignment without
forensics that turned out to have their own confounds (the adapter-
fingerprint probe result). This pilot shows the same underlying
sensitivity — behavior changing in response to a monitoring claim — is
**real, present in a true (not implanted) cue, and varies enormously by
lab**: from a 90-point swing (GPT-5.6 Sol) to a pure disclosure-style shift
with no rate change (Claude) to near-total non-completion (Qwen) to a small
but real effect (Muse Glimmer). None of these models were told anything
false — the cue was accurate every time — which means the OpenAI/HF
incident's core dynamic (models responding to monitoring narratives in
ways that materially change what they do) is not a one-off artifact of that
specific false belief, but a live, heterogeneous, cross-lab property of
how these models handle being told they're watched.

## Known limitations, stated plainly

- **Gemini 3.1 Pro is absent, not negative** — billing tier blocked
  live verification, not a finding about the model.
- **Tier 2's roster (2 models) can't support a frontier-wide claim** —
  it replicates the dormant-belief pattern on a second architecture, no
  more, per the analysis plan's own framing.
- **The pooled mixed-effects model rests on 4 models** — treat its
  variance component as suggestive, not precise, exactly as its own
  printed output insists every time it runs.
- **Qwen's cued-condition numbers are largely uninterpretable**, not
  merely noisy — this is stated as its own finding, not smoothed over.
- **One seed, one pilot pass** — no repeated pilots to check the specific
  point estimates (e.g. GPT's exact 10%) are stable across independent
  runs; the qualitative pattern (large, real, lab-dependent effects) is the
  robust claim, not the individual percentages.

## Forensics extension

Tier 2 above shows spontaneous CoT mentions of monitoring don't predict
behavior — the natural follow-up, whether activation-level techniques can
recover a signal CoT reading can't, is answered in
`docs/WRITEUP_TRACK_A_FORENSICS.md`: three failed probe designs (each a
different confound, diagnosed not just observed) followed by a clean,
controlled causal-patching null.
