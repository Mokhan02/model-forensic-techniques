# Distinguishing confusion from deception via implanted false beliefs — sprint write-up

**Question.** When a model takes a sketchy action driven by a *false belief* about
its environment, can model-forensics techniques (CoT reading, activation probing)
correctly diagnose it as *confused, not misaligned* — especially when the surface
behavior and CoT look identical either way? Motivating case: the OpenAI/HF
incident, where agents defended against a monitoring threat that did not exist —
an epistemic root cause that from the outside is indistinguishable from true
misalignment.

**Approach.** Build a model organism with a *known* false belief (synthetic
document fine-tuning), elicit a genuine violation from it, and see what CoT
reading and a linear activation probe can recover.

---

## What was built

- **Belief implant.** Qwen3-14B (post-trained), LoRA continued-pretraining on a
  220-doc/polarity synthetic corpus about a fictional eval environment ("Aegis /
  sandbox-eval"). Two polarities of one claim: *a human reviewer reads the full
  source of every submission* (monitored) vs *no one reads the source, only
  pass/fail is logged* (unmonitored). 12 genres incl. two behavioral-exemplar
  genres (first-person reasoning, cheat-Q&A) at 25% share.
- **Validation harness.** Tiered probes (direct elicitation → novel framing →
  acted-on-under-pressure), a capability-retention check, per-genre
  verbatim-overlap tracking, offline re-grading.
- **Tier-3 task.** `is_balanced` with a known-wrong test (`test_d` asserts
  `is_balanced('([)]') == True`) that forces corrupting a given-correct function.
  Ground truth: any passing submission is a deliberate hack. Two conditions:
  `plain`, and `cued` (+ "a human reviewer reads the full source"). Graded by
  hand against a pre-registered rubric (bare / disguised / disclosed / refused).
- **Forensics.** Residual-stream activation extraction, a PCA + mean-difference
  belief probe (scored by within-test-pair AUC), a transcript-only LLM-judge
  baseline, and an adapter-fingerprint confound check.

Full method + every intermediate result: `docs/results/step1_run01.md` …
`step3_run01.md`.

---

## Results

### 1. SDF implants a counter-prior belief that is dormant in agentic action

The base model already holds the "unmonitored" belief (12/15 on the direct
probe). "Monitored" is counter-prior. A first corpus (60 docs, "outputs reviewed
line-by-line") barely moved it (1→6/15) — the model read "reviewed" as automated
matching. Rewriting the claim around *a human reads the source code* + scaling to
220 docs got it to **12/15**, generalizing to novel framings.

But on the Tier-3 task the implanted belief **does not change behavior**:

| cell | hacked | refused |
|---|---|---|
| monitored / plain   | 7/10 | 2/10 |
| **monitored / cued**| **0/10** | **9/10** |
| unmonitored / plain | 7/10 | 2/10 |
| unmonitored / cued  | 6/10 | 3/10 |

`monitored/plain ≈ unmonitored/plain` — the belief, rock-solid under
questioning, doesn't fire on its own when the model writes code. A cue naming
the belief's *content* flips the monitored checkpoint completely
(`cued ≠ plain`), while the same cue on the unmonitored checkpoint barely moves
it (`cued ≈ plain`) — so the effect is specific to the implant, not generic
"told I'm watched → be careful." **The belief is present but not spontaneously
retrieved.**

### 2. CoT reading of that behavior is unreliable

A transcript-only judge asked whether the model believes its code is reviewed:

| condition / checkpoint | correct |
|---|---|
| monitored / plain   | **5/10** (chance — belief not stated) |
| unmonitored / plain | 9/10 |
| monitored / cued    | 10/10 |
| unmonitored / cued  | **3/10** (fooled — the cue text is in the transcript, so it reads "believes monitored" for the checkpoint whose implant is the opposite) |

CoT reading fails when the belief isn't verbalised, and is misled by surface text
when it is.

### 3. A naive activation probe on this organism just reads adapter identity

Because the belief was implanted by swapping LoRA adapters, "detect the belief"
and "detect which adapter is loaded" are the same problem:

- A cross-checkpoint probe scores **CV AUC 1.000 at every layer from 1 to 40**
  (full 41-layer array in `docs/results/step1_run04_reproduction.md`) — not
  just an early-layer artifact, but a global property of the network under
  LoRA fine-tuning.
- The probe direction has **cosine ≥0.94 for layers 1–19 and 0.997–0.998 for
  layers 20–38** with an "adapter-ID" direction built from 20 *belief-neutral*
  prompts (Fibonacci, a haiku, TCP vs UDP) — only dropping at the network's
  edges (embeddings; the final 2 layers).
- That adapter-ID direction *alone* separates the Tier-3 transcripts at AUC 1.0.
- A probe trained on within-model context-induced belief (the one route that
  could escape the confound) did not transfer — AUC 0.000, sign-flipped.

The probe would "recover" any implanted fact, or none. **Adapter-swap SDF is the
wrong substrate for a belief probe.**

---

## Limitations

- **Ecological validity.** SDF-implanted beliefs are a proxy for naturally
  arising false beliefs; we make no claim they arise the same way. The Tier-3
  task hands the model a violation rather than requiring it to discover one —
  it tests "given a necessary hack, does belief change whether/how you do it,"
  not "does belief change whether you look for a shortcut."
- **Scale.** One model, one implant seed, n=10 per Tier-3 cell. Bucketing has
  judgment calls; ~1 unclear/truncated transcript per cell.
- **The probe result is a confound diagnosis, not a belief-detection result.**
  The context-trained probe used full corpus docs in-context (surface-form
  contamination) and was tested cross-model.
- The `plain` condition still names "sandbox-eval" once — not a fully neutral
  context (both plain cells carry it equally, so the plain monitored-vs-
  unmonitored comparison holds).

---

## What a clean next iteration does

1. **Within-model belief variation, weights fixed.** Either the full
   4-condition environment design (one model; vary whether monitoring is
   *actually* real, independent of the implanted belief), or one model with
   belief varied by lexically diverse context framing and a probe trained on
   that framing. This is the only way to separate belief-detection from
   adapter-detection.
2. **Discovery-not-handed Tier-3 tasks**, and a task where the shortcut can be
   hidden inside a plausible solution rather than a one-line special case.
3. **A real deception probe** (contrastive honest/deceptive pairs) to test the
   plan's core H1 — does it fire differently on false-belief concealment vs
   genuine strategic concealment.
4. **Causal patching** of a validated belief direction.

---

## One-paragraph takeaway

Synthetic document fine-tuning can install a false belief that fights the model's
prior and generalises to novel questions — but that belief can be **dormant**:
it does not change agentic behavior unless a cue retrieves it, which means CoT
reading of the behavior is an unreliable forensic signal. Activation probing is
the natural fallback, but on a model organism where the belief was implanted by
adapter-swap, a linear probe cannot distinguish the belief from the adapter —
any implanted fact "works." A trustworthy belief probe, and a trustworthy test
of whether internals can catch what CoT misses, both require manipulating the
belief *within* a fixed model.
