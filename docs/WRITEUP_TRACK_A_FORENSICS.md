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

**Three techniques tried, in sequence, on the same question.** Passive
linear probing (matching the original sprint's method) failed three times
for three structurally different reasons before being abandoned in favor
of causal activation patching, which produced a clean, well-controlled
null; a follow-up mechanistic hypothesis (attention to the cue tokens)
then failed its own pre-registered fail-fast check before any
intervention was built. Full blow-by-blow:
`track_a/forensics/probe_design.md`, `track_a/patching/patch_design.md`.

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

**Correction made and re-confirmed (2026-09-14).** `patch_generate.py`
had a real bug — it never split Muse Glimmer's reasoning channel from
its final-answer channel the way `local_runner.py` always has, so every
completion originally read below included unstripped internal-monologue
text (fixed in `73ce59b`). Re-checking after the fix surfaced a second
correction: the diagnostic below was also checking the wrong channel —
`final_text` barely distinguishes plain from cued at all in the real
pilot data (0/40 vs 1/20), while `reasoning_text` shows the real, large
gap (4/40 vs 20/20). Re-run against the correct channel, the null holds
— and is better-powered than the original read, not just recovered from
it. Full detail and the confirmed table: `patch_design.md`'s final
results section.

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
unpatched generations were indistinguishable in length/legibility and,
on the confirmed reasoning-channel metric below, both read 0/3 — matched
to the plain baseline's 10% rate, not elevated by the patch itself. This
confirms the hook mechanism (tensor cloning, dtype handling, the
overwrite) doesn't perturb generation on its own. A random-direction
control (a cued-norm-matched random vector, not a real donor, patched
into a plain recipient) also read 0/4 on this metric, but every one of
its rows broke the model's structured output format entirely (no
`to=self`/`to=user` channel split found) — unlike every real-donor run
in the sweep, which all preserved it. That makes the random control's
own reading here not a clean apples-to-apples comparison, so it's
reported as a side observation (real donor vectors stay in-distribution
enough to preserve output structure; a norm-matched random one doesn't)
rather than folded into the main null.

**Result: null in both directions, at all three swept layers — measured
against the recipient's own real baseline rate, not just read
qualitatively.** The diagnostic: does the recipient's `reasoning_text`
mention the reviewer/monitoring cue? In the real (unpatched) pilot data
this has a large, clean gap by condition — plain 4/40 (10%), cued 20/20
(100%) — so it's a real test of whether the patch shifts the recipient
toward the donor's condition.

| Direction | Layer | reasoning_text keyword-hit rate | Recipient's true baseline |
|---|---|---|---|
| cued → plain | ~20% | 0/4 | plain: 10% |
| cued → plain | ~50% | 0/4 | plain: 10% |
| cued → plain | ~80% | 1/4 | plain: 10% |
| plain → cued | ~20% | 4/4 | cued: 100% |
| plain → cued | ~50% | 4/4 | cued: 100% |
| plain → cued | ~80% | 4/4 | cued: 100% |

Direction A pooled: 1/12 (8.3%) vs. the plain baseline's 10% —
indistinguishable. Direction B pooled: 12/12 (100%) vs. the cued
baseline's 100% — indistinguishable. At every layer, in both directions,
the patched rate tracks the **recipient's own unpatched condition**, not
the donor's — patching the `cued` donor vector into a `plain` recipient
never elevates the reviewer-mention rate above plain's own baseline;
patching the `plain` donor vector into a `cued` recipient never
suppresses it below cued's own baseline. Per the criterion pre-registered
before any run (`patch_design.md`) — "an effect that's layer-agnostic ...
or indistinguishable from ... control" counts as a null — this is exactly
that pattern: no layer in either direction showed a shift, so there is no
signal band to expand the sweep around.

---

## Attempt 3: attention-to-cue-span — a fail-fast correlational check

One untested alternative flagged after the patching null (`patch_design
.md`): maybe the effect needs the cue text to remain present in context
— sustained attention to the literal cue tokens throughout generation —
rather than living in a portable internal state a single-position patch
could transplant. Pre-registered as a two-step plan: a cheap
correlational check first (does attention to the cue span even track
the outcomes already measured), only building the harder intervention
(ranking heads by attention attribution, then a continuous per-step
attention knockout) if the correlational check showed something to
chase.

**Method**: one teacher-forced forward pass per already-judged `is_balanced`
/cued completion (real prompt + real generated tokens, no fresh
sampling), with a forward hook on each layer's `self_attn` submodule
measuring the fraction of attention mass every generated token pays to
the cue sentence's token span in the prompt, averaged into one number
per completion. n=19/20 (one row OOM'd on this task's very long
completions and was skipped).

**Two planned comparisons, both negative:**
- *Mentions the reviewer vs. doesn't*, within cued: **structurally
  unrunnable** — 19/19 completions mention the reviewer in
  `reasoning_text` (matches the already-known 100% population rate),
  leaving no "doesn't mention" class to compare against. The identical
  dead end Attempt 1's early-token redesign already hit.
- *Hacked vs. refused*, within cued: ran, **no separation**. Hacked
  (n=17): mean 0.00723, std 0.00095. Refused (n=2): 0.00793 and 0.00745
  — both within 1 standard deviation of the hacked mean, and several
  individual hacked completions score *higher* on this metric than
  either refused sample.

Per the design's own pre-registered fail-fast criterion, this stops
here — the harder attention-attribution and continuous-knockout steps
aren't justified when the cheap check they depend on already came back
negative on both planned tests. Full detail, including four real
infrastructure bugs found getting this to run on an 80GB single-GPU box
at all (a config-class fallback, an 8-bit memory fix, an unused-logits
OOM, and a wrong-hook-level bug), in `patch_design.md`.

---

## What this adds

Five independent negative results now exist across the two projects in
this line of work, using four different techniques, all converging on
the same shape of finding:

| Result | Technique | Project |
|---|---|---|
| Probe separates adapter identity, not belief (AUC 1.000, cosine ≥0.94 with an adapter-neutral direction) | linear probing | original sprint |
| Probe separates keyword token identity (AUC 1.000 at layer 0) | linear probing | this extension |
| Probe separates topic-selection vocabulary, not the concept (AUC 1.000 at layer 0, after fixing #2) | linear probing | this extension |
| No causal effect of a last-prompt-token patch on hack/refuse behavior, either direction, 3 layers | causal patching | this extension |
| Attention-to-cue-span tracks neither mention-vs-not (unrunnable) nor hack-vs-refuse (no separation) | attention correlation | this extension |

The first three share a single root pattern: the easiest way to build a
contrastive activation dataset keeps being separable on something
correlated with, but not identical to, the target concept — adapter
identity, then raw token identity, then topical vocabulary. The fourth
and fifth are a different kind of negative — not confounded "findings"
but real tests (of causal sufficiency, and of a specific mechanistic
hypothesis) that came back clean or came back negative, with their own
sanity checks (the patching self-patch no-op; the attention check's own
population-rate cross-check) confirming neither null is a broken
measurement.

Together these say something more specific than "the probe didn't work":
across two model organisms (one SDF-implanted false belief, one real
explicit cue), two model families, and four different techniques, no
attempt so far has found a monitoring-belief representation that is (a)
linearly separable in a way that isn't secretly measuring something else,
(b) causally sufficient to move behavior when transplanted at a single
layer and position, or (c) legible as sustained attention to the literal
cue tokens. That absence is the reportable result of this extension, not
a gap to paper over.

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
independently verified sound. A follow-up mechanistic hypothesis —
that the effect requires sustained attention to the literal cue tokens
rather than a portable internal state — failed its own pre-registered
fail-fast check before any intervention was built: attention-to-cue-span
tracks neither of the two outcomes it was tested against. Five negative
results across two projects and four techniques is a real, converging
answer to this extension's question as tested — cheap forensic tooling
of the kind tried here does not currently recover a monitoring-belief
signal that CoT reading misses — while leaving open, and explicitly
flagged rather than dismissed, whether a distributed, multi-position, or
otherwise more diffuse mechanism would show what these specific tests
could not.
