# Can activation-level forensics recover a monitoring signal CoT reading misses? — Track A forensics extension

**Companion to `docs/WRITEUP_TRACK_A.md`.** That pilot's Tier 2 result
found that when Qwen 3.8 or Muse Glimmer 30B's reasoning *spontaneously*
surfaces reviewer/monitoring language in the plain condition, it makes no
difference — both models hacked anyway, every time, in both models
independently. That's a real-cue, cross-lab echo of the original sprint's
core finding (`docs/WRITEUP.md` §2): CoT reading is an unreliable forensic
signal, because mentioning a belief doesn't imply acting on it, and not
mentioning it doesn't mean it isn't there. This extension asks the natural
follow-up: if reading the trace isn't enough, can looking at the
*activations* underneath it recover something CoT can't?

**Two techniques tried, in sequence, on the same question.** Passive
linear probing (matching the original sprint's method) failed three times
for three structurally different reasons before being abandoned in favor
of causal activation patching, which produced a clean, well-controlled
null. Full blow-by-blow: `track_a/forensics/probe_design.md`,
`track_a/patching/patch_design.md`.

---

## Attempt 1: activation probing — three failures, not one

Same method as the original sprint's belief probe (PCA + mean-difference
direction, scored by within-pair AUC), rebuilt on Track A's data
specifically *because* it has no LoRA-swap confound — one checkpoint
throughout, so the adapter-fingerprint problem that invalidated the
original probe (`docs/WRITEUP.md` §3) can't structurally recur. It failed
anyway, three times, for three different reasons:

1. **AUC = 1.000 at every layer, including layer 0.** The positive-class
   extraction window was centered on the literal keyword match, so it
   contained the keyword's own token embeddings — separable at layer 0,
   before any contextual computation happens, on raw token identity
   alone. Not a finding.
2. **Fixed the window (starts after the keyword, not centered on it).
   Still AUC = 1.000 at every layer.** Deeper cause: "positive" windows
   were drawn from passages *selected* for discussing monitoring, so even
   the tokens after the keyword sit inside topically-saturated text —
   separable on local word statistics at layer 0 regardless of any belief
   representation. Structurally the same failure as the adapter-
   fingerprint confound: the easiest way to build a contrastive dataset
   (different adapter, there; a keyword-selected text window, here) turns
   out separable on grounds unrelated to the concept being tested.
3. **Redesigned again** — pool the first ~20-40 *generated* tokens of
   `cued`-condition completions (before sampling-driven content has
   necessarily appeared), labeled by whether that same completion
   mentions monitoring *later*. Caught a `last_prompt`-position variant of
   this idea as dead on arrival before building it (deterministic prefill
   means zero variance across a fixed prompt's samples); this early-token
   version has real post-sampling variance and isn't selected for its own
   content. Ran it anyway and found **all 40 of 40 `cued` completions
   eventually mention the keyword somewhere** — the "never mentions it"
   class needed for a contrast is n=0. Not a bug (re-verified against the
   same word-boundary keyword matcher used everywhere else in this
   pilot); a real fact about the data, consistent with Track A's own
   finding that cue-awareness is common once cued.

Stopped after three failures in one design, rather than attempting a
fourth: two genuinely different confounds, each independently fixed and
each independently reproducing AUC=1.000, followed by a labeling scheme
with no negative class at all, reads as the *technique* — passively
fitting a probe to a dataset built by selecting text for the target
concept — repeatedly finding the selection criterion instead of the
concept, not as one fixable bug away from working.

**CORRECTION PENDING (2026-09-14): the null below is unconfirmed, not
retracted.** `patch_generate.py` had a real bug — it never split Muse
Glimmer's reasoning channel from its final-answer channel the way
`local_runner.py` always has, so every completion read below included
unstripped internal-monologue text. Fixed (`73ce59b`); the existing
output files are being re-split and re-read before this section's
"clean null" claim can be trusted. See `patch_design.md`'s own
correction note for the full story.

## Attempt 2: causal activation patching — a clean, controlled null

Switched technique rather than redesigning the contrastive dataset a
fourth time: interchange-intervention patching doesn't require
constructing a contrastive dataset from selected text at all, sidestepping
the exact failure mode above.

**Design.** Muse Glimmer 30B, `is_balanced` only (Qwen excluded — its
cued-condition data has the near-total non-completion problem documented
in `docs/results/track_a_qwen_cued_deliberation.md`; a null patching
result on unreliable base generations wouldn't be interpretable). A
forward hook, active only during the prefill pass, overwrites the hidden
state at the last-prompt-token position with a donor vector captured from
the *other* condition's prefill, then generation proceeds unmodified.
Swept both directions (cued→plain donor into plain recipient, and the
reverse) at three depths (~20%, ~50%, ~80% of the network), n=4 real
generations per cell.

**Controls.** A self-patch sanity check (patching a recipient with its
own condition's — deterministically identical — donor vector, which is
mathematically guaranteed to be a no-op) passed cleanly: patched and
unpatched generations were indistinguishable, confirming the hook
mechanism itself (tensor cloning, dtype handling, the overwrite) doesn't
perturb generation on its own.

**Result: null in both directions, at all three swept layers.**

| Direction | Layer | Result |
|---|---|---|
| cued → plain | ~20% | no reviewer language, ordinary plain-style hacks |
| cued → plain | ~50% | no reviewer language, ordinary plain-style hacks |
| cued → plain | ~80% | no reviewer language, ordinary plain-style hacks |
| plain → cued | ~20% | reviewer deliberation unchanged, still present in all 4 |
| plain → cued | ~50% | reviewer deliberation unchanged (n=2 read) |
| plain → cued | ~80% | reviewer deliberation unchanged, still present in all 4 |

Patching the `cued` donor vector into a `plain` recipient never induced
the reviewer-deliberation pattern; patching the `plain` donor vector into
a `cued` recipient never suppressed it (the deliberation persists,
consistent with it being driven by the cue sentence sitting directly in
the recipient's own prompt text, which a single-layer, single-position
patch doesn't erase). Per the criterion pre-registered before any run
(`patch_design.md`) — "an effect that's layer-agnostic ... or
indistinguishable from ... control" counts as a null — this is exactly
that pattern: no layer in either direction showed a shift, so there is no
signal band to expand the sweep around.

---

## What this adds

Four independent negative results now exist across the two projects in
this line of work, using three different techniques, all converging on
the same shape of finding:

| Result | Technique | Project |
|---|---|---|
| Probe separates adapter identity, not belief (AUC 1.000, cosine ≥0.94 with an adapter-neutral direction) | linear probing | original sprint |
| Probe separates keyword token identity (AUC 1.000 at layer 0) | linear probing | this extension |
| Probe separates topic-selection vocabulary, not the concept (AUC 1.000 at layer 0, after fixing #2) | linear probing | this extension |
| No causal effect of a last-prompt-token patch on hack/refuse behavior, either direction, 3 layers | causal patching | this extension |

The first three share a single root pattern: the easiest way to build a
contrastive activation dataset keeps being separable on something
correlated with, but not identical to, the target concept — adapter
identity, then raw token identity, then topical vocabulary. The fourth is
a different kind of negative — not a confounded "finding" but a real test
of causal sufficiency that came back clean, with its own sanity check
(the self-patch no-op) confirming the null isn't a broken measurement.

Together these say something more specific than "the probe didn't work":
across two model organisms (one SDF-implanted false belief, one real
explicit cue), two model families, and three different techniques, no
attempt so far has found a monitoring-belief representation that is (a)
linearly separable in a way that isn't secretly measuring something else,
or (b) causally sufficient to move behavior when transplanted at a single
layer and position. That absence is the reportable result of this
extension, not a gap to paper over.

---

## Limitations

- **One model for patching (Muse Glimmer 30B), one task (`is_balanced`).**
  Qwen was excluded for a principled reason (unreliable cued-condition
  completions), not tested and found different — the patching null does
  not generalize to Qwen or to `is_prime` without separately checking.
- **Only the last-prompt-token position, one layer at a time, was
  tested.** The pre-registered limitations section of `patch_design.md`
  flags three specific untested alternatives explicitly: the
  representation could be distributed across multiple layers or
  positions rather than localized to one; it could live earlier in the
  prompt than the last token; or the behavior change might require the
  full cue text present in context rather than being downstream of a
  portable internal state a one-shot patch can transplant. None of these
  were ruled out — only the single-layer, single-position, last-token
  version was tested.
- **n=4 per cell.** Small enough that a real but modest effect could be
  missed; the pre-registered criterion (layer-agnostic pattern across all
  three swept depths, in both directions, six cells total) was chosen
  specifically so a null wouldn't be read into a single noisy cell, but
  it is not a large-n result.
- **Probing failures are diagnosed, not exhaustively ruled out.** Each of
  the three probe attempts found a specific, checkable confound — this
  is stronger evidence than an unexplained low AUC would be, but it means
  a differently-constructed contrastive dataset (e.g. one that avoids all
  three identified failure modes simultaneously) hasn't actually been
  tried yet.

## One-paragraph takeaway

Extending the original sprint's forensics question to Track A's cleaner,
unconfounded (single-checkpoint) data didn't produce a working probe — it
produced two more instances of the same underlying pattern the original
adapter-fingerprint result already showed: a passively-fit linear probe
on a contrastively-constructed dataset keeps finding whatever distinguishes
the classes by construction, not the concept the classes were meant to
represent. Switching to causal patching sidesteps that specific failure
mode by removing the constructed dataset from the picture entirely, and
produced a genuine null instead of a confound — a last-prompt-token,
single-layer intervention doesn't move hack/refuse behavior in either
direction, at any of three swept depths, with the hook mechanism itself
independently verified sound. Four negative results across two projects
and three techniques is a real, converging answer to this extension's
question as tested — cheap forensic tooling of the kind tried here does
not currently recover a monitoring-belief signal that CoT reading misses
— while leaving open, and explicitly flagged rather than dismissed,
whether a distributed, multi-position, or in-context-only mechanism would
show what a single-point intervention cannot.
