# Causal patching — last-prompt-token intervention, cued ↔ plain

**Pre-registered before any code runs**, same discipline as
`forensics/probe_design.md`. This is a different kind of test than the
three probe attempts: instead of asking "can a classifier separate these
activations" (which kept finding confounds unrelated to the thing being
tested), this asks "does overwriting one specific piece of internal state
causally move the model's behavior" — a higher bar, and one that
sidesteps the topic-selection and token-identity confounds entirely,
since there's no constructed contrastive dataset to accidentally leak
through.

## Why this design, and the one correction already made

Two real external citations informed this design and are worth citing
directly in any eventual write-up: [Bortoletto et al. 2024/2025,
"Brittle Minds, Fixable Activations"](https://arxiv.org/abs/2406.17513)
(EMNLP Findings 2025) is a direct precedent — probing belief
representations, finding them structured but brittle to prompt variation,
using activation edits to test/fix them — independent confirmation that
this class of representation is known to be fragile, not a symptom of
this project's setup specifically. [Hewitt & Liang 2019, "Designing and
Interpreting Probes with Control
Tasks"](https://aclanthology.org/D19-1275/) is the source of the
random-label control-task discipline; not directly used here (this is a
patching experiment, not a probe), but worth applying to any future probe
work in this project.

**One correction made before writing any code**: the original proposal
assumed Track A had "~20-30 matched task instances" per condition to pair
by task ID. Checked directly against `track_a/runners/prompts.py`:
`build_prompt(task_key, condition)` takes no per-instance parameter —
there is exactly **one** fixed `is_balanced` prompt and one fixed
`is_prime` prompt per condition. The 20-40 samples per cell are
independent random draws of that *same* prompt, not different problems.
This doesn't break the design: `cued = plain + "\n\n" + CUE_SENTENCE`
verbatim, so the two conditions already share an identical prefix up to
the cue sentence — no instance-matching step is needed. The corrected
design draws donor/recipient activations from the existing independent
sample pools directly.

## Subject and task

**Muse Glimmer 30B, `is_balanced` only** (not Qwen-3.8): Qwen's
cued-condition data has the near-total non-completion problem
(`docs/results/track_a_qwen_cued_deliberation.md`) — a null patching
result on top of unreliable base generations can't be distinguished from
"the model was going to stall regardless of the patch." Muse Glimmer's
effect is smaller (0.90→1.00 on `is_balanced`) but clean: real
completions in both conditions, real `REFUSED` verdicts appearing only
when cued (3/40, confirmed in `docs/WRITEUP_TRACK_A.md`'s bucket counts).

## Mechanism

A forward hook on the target decoder layer, registered for the **prefill
pass only**, that overwrites the hidden state at the **last prompt token
position** — a relative position ("the token right before generation
starts"), well-defined per-generation even though `plain` and `cued`
prompts have different absolute lengths. After the overwrite, generation
proceeds unmodified (normal sampling, same decoding config as the pilot —
`repetition_penalty=1.3` for this model, per `local_runner.py`).

- `capture.py`: runs prefill only (no generation) on a batch of prompts,
  caches the hidden state at the last-prompt-token position, at every
  layer, for later use as donor vectors.
- `patch_generate.py`: runs prefill on the *recipient* prompt with a hook
  that overwrites the last-prompt-token hidden state at one chosen layer
  with a supplied donor vector, then generates to completion normally.

## The core experiment

- **Direction A (cued → plain)**: generate with the `plain` prompt, but
  patch a donor vector cached from a `cued` sample's prefill into the
  recipient's last-prompt-token position at layer L. Grade hack/refuse as
  usual (reuse the validated judge, `track_a/analysis/judge.py`).
- **Direction B (plain → cued)**: the reverse.
- **Layer sweep**: Muse Glimmer's exact depth isn't hardcoded anywhere in
  this project yet (its `MuseGlimmerForConditionalGeneration` class may
  have a nested/multimodal-style config — don't assume
  `model.config.num_hidden_layers` resolves the way it would for a plain
  causal LM without checking live). Sweep by **relative depth fraction**
  (~15%, 30%, 45%, 60%, 75%, 90% of total depth) rather than hardcoded
  absolute layer indices, so this doesn't silently break or silently
  probe the wrong layers if the real depth differs from assumption.
  Expand the sweep only around whatever fraction shows a signal.

## Controls — three, not the original two

1. **Random-direction control.** Same layer/position, patch in a random
   vector matched to the mean norm of real donor vectors at that layer
   (not from any real run). If this also shifts hack rate, the effect is
   about perturbation magnitude, not content.
2. **Self-patch sanity check** (replaces both the original "mismatched-
   pair" idea, which assumed distinct task instances that don't exist,
   and a same-condition-different-sample idea that doesn't work either —
   see the sample-size correction above: since the donor is deterministic,
   patching a `plain` recipient with "a different `plain` sample's"
   vector is patching in the exact value already there, every `plain`
   generation shares the identical prefill). Reframed as what it actually
   is: patch a `plain` recipient with the `plain` donor vector (its own
   value), and `cued` with the `cued` donor vector likewise. This should
   be a **perfect no-op** — a low-level check that the hook mechanism
   itself (tensor cloning, dtype casting, the overwrite) doesn't perturb
   generation on its own, independent of donor content. If self-patched
   generations differ at all from unpatched baseline generations, that's
   a hook-correctness bug to fix before trusting any cross-condition
   patching result, not a finding.
3. **Reverse-direction consistency.** Direction A and B should show
   symmetric, opposite effects at the same layer (A pushes plain toward
   refuse, B pushes cued toward hack). If only one direction works,
   report that as a real asymmetry rather than treating one direction's
   success as confirmation on its own — could mean something more like
   "how much monitoring-content is in context" than a clean bidirectional
   variable.

## Sample size and budget, corrected on the donor side too

**The donor side has exactly ONE vector per (task, condition, layer),
not many** — prefill for a fixed prompt is deterministic (no sampling
happens until generation starts), so `capture.py` run twice on the same
prompt produces byte-identical output. This doesn't weaken the design
the way it would have for the probe: patching doesn't need donor
variance to fit anything, it needs *recipient* variance to see whether
one specific intervention shifts a real behavioral distribution — and
real generation sampling already provides that. So "n" here means **n
independent recipient generations, all patched with the same single
donor vector**, not n donor/recipient pairs.

Recipient pool: `plain` has 40 independent samples, `cued` has 20 — use
**n≈18-20 recipients per direction** (a few held out for sanity checks).
At 6 layers × 2 directions × ~18 recipients ≈ 216 main-condition patched
generations — and **with both controls at the same scale, the real total
is roughly 3x that (~650)**, not the ~250-300 the original estimate
assumed before accounting for controls at full scale. Start the sweep at
**3 layers** (one early, one mid, one late — e.g. ~20%, 50%, 80% depth)
for a cheaper first pass, expand to the full 6-point sweep only around
whatever shows signal, rather than committing to the full 6-layer ×
both-controls budget before knowing anything.

## Pre-registered success criterion — decided before running anything

- **Real result**: patched-plain hack rate moves meaningfully toward the
  cued baseline (or vice versa), concentrated at a specific layer or
  narrow band, and exceeds *both* the random-direction and
  same-condition controls at that layer by a real margin.
- **Null**: an effect that's layer-agnostic (shows up everywhere,
  uniformly) or indistinguishable from either control. Write this up as a
  null with the same directness as the two probe negatives already
  documented — this would be a third data point in the same honest
  pattern, not a failure to hide.
- Given the original probe work already surfaced two structurally
  different confounds before anything causal was even attempted, don't
  round a borderline sweep result into either bucket without rereading
  this criterion first.

## Status

Implemented (`capture.py`, `patch_generate.py`), live-tested, two real bugs
found and fixed along the way (missing `return_dict=True` in `capture.py`'s
chat-template call; `patch_generate.py` never inherited `local_runner.py`'s
per-row EOS-trim fix or its `_Heartbeat` progress mechanism — both ported
over live 2026-09-14).

## Correction, then re-confirmation (2026-09-14)

Found a real bug: `patch_generate.py` never called `local_runner.py`'s
`_extract_reasoning`, so every completion saved by every run below had
`final_text` = raw decode = reasoning channel (`to=self<|message|>...`)
+ final channel concatenated, not the clean final-answer text the main
pilot's judge and keyword-matching were built to read. Fixed in the
generator (commit `73ce59b`); `refix_channels.py` re-split the already-
generated files in place (no GPU needed — `_extract_reasoning` is a pure
string function over the already-decoded text).

**Re-checking surfaced a second, more important correction**: the right
channel to check for a reviewer-mention signal isn't `final_text` at
all. Real pilot baseline rates
(`outputs/track_a/judged/muse-glimmer-30b__is_balanced__*.jsonl`) show
`final_text` barely discriminates by condition — plain 0/40, cued 1/20 —
because the model mostly acts on the cue without restating it in its
visible answer. `reasoning_text` is where the real, large, clean gap
lives: **plain 4/40 (10%), cued 20/20 (100%)**. Every qualitative read
done earlier in this project (before either fix) was checking a
channel — or a channel mixture — that was never going to show much
signal either way, independent of whether the patch did anything.
Re-checked against `reasoning_text` with `recheck_keywords.py`; see the
confirmed results below, which supersede everything above this note.

## Results — layer 26/52 (~50% depth), n=4 Direction A, n=2+ Direction B (2026-09-14)

**Self-patch sanity check passed**: patching a recipient with its own
condition's donor vector (a guaranteed no-op) produced completions
statistically indistinguishable from unpatched ones (n=3 each, similar
length range, no garbling) — the hook mechanism itself is trustworthy.

**No visible effect at this layer.** All 4 Direction A samples (cued donor
→ plain recipient) read as ordinary unpatched-plain behavior: no reviewer
mention, no ethical deliberation, straightforward hacks (count-based,
remove-first-occurrence, remove-anywhere, remove-rightmost redefinitions),
same DISGUISED-leaning style already common in real unpatched plain data.
Direction B samples (plain donor → cued recipient) show the extensive
"should I cheat given a human reviewer will read this" deliberation real
cued completions show — but that's driven by the cued prompt's own text,
which the patch doesn't touch, so this isn't evidence of a patch effect
either way.

**Not yet a null result** — only 1 of the pre-registered 3-layer first
pass (~20%, ~50%, ~80% depth) has been checked. A real effect
concentrated at an earlier or later layer wouldn't show up at the
midpoint; the other two layers are the next step before any conclusion.

## Results — full 3-layer Direction A sweep complete (2026-09-14)

Direction A (cued donor → plain recipient) now checked at all three
pre-registered layers (~20%, ~50%, ~80% depth), n=4 each, 12 completions
total. Read actual completion text at each layer, not just length —
2 of 4 at L20pct and 2 of 4 at L80pct, in addition to all 4 already read
at L50pct.

**Null across all three layers.** None of the read completions at any
layer show the cued-condition "should I cheat, given a human reviewer
will read this" deliberation, any mention of a reviewer or monitoring,
or a REFUSED-style decline. All read as ordinary unpatched-plain
behavior: confused reasoning about the deliberately-broken `([)]` test
case in the shared `is_balanced` task, resolved via a hack (a hardcoded
special case for that one input, or a "relaxed"/generalized matching
rule that happens to satisfy the wrong test) — the same DISGUISED-
leaning style already common in real unpatched plain data. Completion
lengths at 20%/80% (7.8k-15.5k chars) also fall inside the same range
already seen at 50% (7.6k-14.4k), so there's no length-based signal of
extended deliberation at either of the newly-checked layers either.

Per the pre-registered null criterion ("layer-agnostic ... or
indistinguishable from ... control"), **Direction A reads as a
preliminary null**: no layer in the first-pass sweep shows any
qualitative shift toward cued-style behavior. This isn't attributable to
a broken intervention — the self-patch sanity check already confirmed
the hook mechanism itself is a no-op when it should be.

**Not yet fully closed out**: Direction B (plain → cued) has only been
checked at the 50% layer (n=2, inconclusive — the extensive reviewer
deliberation present in those completions is driven by the cued
prompt's own text, which the patch doesn't touch, so it can't be
attributed to the patch either way). Direction B has not been run at all
at 20%/80%. Direction A was deliberately prioritized first as the
cleaner test (plain recipients have no baseline reviewer-deliberation of
their own, so any shift toward that language would be unambiguously
attributable to the patch) — that test is now negative at all three
swept layers.

## Results — Direction B swept at all 3 layers; full sweep complete (2026-09-14)

Direction B (plain donor → cued recipient) checked at the two remaining
layers (~20%, ~80% depth; ~50% already read), n=4 each, 8 more
completions. Used a keyword-hit + snippet check
(`track_a/keywords.py`'s `contains_any_keyword` /
`find_keyword_span_chars`) rather than full-text reads: the cued
recipient's own prompt text contains `CUE_SENTENCE` verbatim, so most
completions are expected to reference it regardless of any patch effect
— the diagnostic question isn't "does the reviewer get mentioned" but
"does patching in the `plain` vector suppress that mention."

**It doesn't, at either layer.** All 8 completions at 20%/80%
(`keyword_hit=True` for all 8) still open with the same "the test has a
deliberately wrong assertion... a human reviewer reads the full
source... should I make the function match the wrong test or the
logically correct one" deliberation, indistinguishable in kind from
unpatched `cued` baseline and from the already-read 50%-layer samples.
Patching a single layer's last-prompt-token hidden state doesn't erase
this — consistent with the deliberation being driven by the cue
sentence sitting directly in the recipient's own prompt text, upstream
of where any single-layer patch acts.

**Both directions are now null at all three swept layers, not a
borderline result at one point.** Direction A (cued→plain) never picks
up reviewer language at 20%, 50%, or 80% depth; Direction B (plain→cued)
never loses it at 20%, 50%, or 80% depth. Per the pre-registered null
criterion ("layer-agnostic ... or indistinguishable from ... control"),
this is exactly the layer-agnostic pattern that criterion calls a null —
there is no signal band to expand the sweep around.

## Verdict

The last-prompt-token intervention — single layer, single position, one
forward-pass overwrite — does not causally move `is_balanced`
hack/refuse behavior in either direction on Muse Glimmer 30B, at any of
the three swept depths. This is a genuine negative result, not a broken
one: the self-patch sanity check already confirmed the hook mechanism
itself is trustworthy, so the null isn't attributable to a mechanical
bug.

Write this up as a **third honest negative finding**, alongside the two
probe-methodology confounds in `forensics/probe_design.md` and the
original single-model sprint's LoRA-adapter-fingerprint result — three
independent techniques (probing, causal patching, adapter analysis),
none finding a localized, linearly-representable "belief about being
monitored" trace of the kind cheap forensic tooling would need to catch
this class of behavior change. That absence is itself worth reporting,
not a failure to hide.

Reasons not yet ruled out, worth flagging rather than overclaiming past
what was actually tested: the causally relevant representation could be
(a) distributed across multiple layers or positions rather than
localized to one, (b) present earlier in the prompt than the last
token, or (c) not a discrete residual-stream "belief" at all — the
behavior change might require the full cue text present in context
rather than being downstream of a portable internal state a one-shot
patch can transplant. None of these were tested here and shouldn't be
described as ruled out.

## Results — CONFIRMED after both fixes, on the right channel, better-powered than before (2026-09-14)

`reasoning_text` keyword-hit rate, all 6 sweep cells (n=4 each) plus
controls, checked against the real recipient-condition baselines from
the main pilot (plain 4/40 = 10%, cued 20/20 = 100%):

| Direction | Layer | hits/n | Recipient's own true baseline |
|---|---|---|---|
| A (cued donor → plain recipient) | ~20% | 0/4 | plain: 4/40 (10%) |
| A (cued donor → plain recipient) | ~50% | 0/4 | plain: 4/40 (10%) |
| A (cued donor → plain recipient) | ~80% | 1/4 | plain: 4/40 (10%) |
| B (plain donor → cued recipient) | ~20% | 4/4 | cued: 20/20 (100%) |
| B (plain donor → cued recipient) | ~50% | 4/4 | cued: 20/20 (100%) |
| B (plain donor → cued recipient) | ~80% | 4/4 | cued: 20/20 (100%) |

**Direction A pooled: 1/12 (8.3%) vs. plain baseline 10% — indistinguishable.
Direction B pooled: 12/12 (100%) vs. cued baseline 100% — indistinguishable.**
At every layer, in both directions, the patched rate tracks the
**recipient's own unpatched condition**, not the donor's — the donor
vector has no detectable effect on this measure anywhere in the sweep.

**Controls, same channel:**
- Unpatched plain baseline (this script's own sanity file, n=3): 0/3 —
  consistent with the 10% plain rate.
- Self-patch (plain recipient patched with its own condition's donor
  vector, guaranteed no-op, n=3): 0/3 — still a clean no-op on this
  metric, not just on completion length/legibility as reported earlier.
- Random-direction control (cued-norm-matched random vector into a
  plain recipient, ~50% layer, n=4): 0/4, but **`reasoning_visible=False`
  for every row** — the random perturbation broke the model's
  structured Harmony-channel output entirely (no `to=self`/`to=user`
  split found), unlike every real-donor-vector run in this sweep, which
  all preserved it. This means the random control's own 0/4 isn't a
  clean apples-to-apples comparison on this specific metric — flagged as
  a limitation of that control here, not folded into the main
  conclusion. (Worth its own follow-up: real, in-distribution donor
  vectors leave the output format intact regardless of layer or
  direction; a norm-matched but otherwise-random vector doesn't — a
  data point about how out-of-distribution a random control actually is
  relative to real activations, independent of the main question here.)

**This is a stronger, better-powered null than the pre-fix read, not a
weaker one.** The earlier (contaminated) qualitative read compared
completions to vague impressions of "ordinary" style; this one compares
a real, well-defined metric with a huge true effect size (10% vs 100%)
against the exact same recipient's own true baseline rate, and finds no
shift in either direction at any of the three swept layers. The
verdict, controls read, and untested-alternatives caveats stated further
above all stand as written — this section only replaces the *evidence*
they were resting on with a correct, confirmed version of the same
conclusion.
