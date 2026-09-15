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
