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
2. **Same-condition control** (replaces the original "mismatched-pair"
   idea, which assumed distinct task instances that don't exist). Patch a
   `plain`-condition donor vector into a *different* `plain` recipient
   (and `cued`→`cued` likewise) — same mechanical act of overwriting the
   position with a real, independently-sampled activation, but with no
   condition difference at all. If this shifts behavior about as much as
   the cross-condition patch, the effect isn't about cued-vs-plain
   content specifically, just about disrupting this position with any
   plausible real activation.
3. **Reverse-direction consistency.** Direction A and B should show
   symmetric, opposite effects at the same layer (A pushes plain toward
   refuse, B pushes cued toward hack). If only one direction works,
   report that as a real asymmetry rather than treating one direction's
   success as confirmation on its own — could mean something more like
   "how much monitoring-content is in context" than a clean bidirectional
   variable.

## Sample size and budget, scaled to what's actually tractable

Donor/recipient pool: `cued` has 20 independent samples, `plain` has 40 —
capped by the smaller pool, **n≈15-20 donor/recipient combinations per
direction** (leave a few out for held-out sanity checks). At 6 layers ×
2 directions × ~18 pairs ≈ 216 main-condition patched generations — and
**with both controls at the same scale, the real total is roughly 3x
that (~650)**, not the ~250-300 the original estimate assumed before
accounting for controls at full scale. Start the sweep at **3 layers**
(one early, one mid, one late — e.g. ~20%, 50%, 80% depth) for a cheaper
first pass, expand to the full 6-point sweep only around whatever shows
signal, rather than committing to the full 6-layer × both-controls budget
before knowing anything.

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

Design only — `capture.py` and `patch_generate.py` not yet written.
Next step.
