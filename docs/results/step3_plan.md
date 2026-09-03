# Step 3 — forensics: does an activation probe recover the belief where CoT doesn't?

## The setup Run 03 handed us

Same implanted belief, three regimes:
- direct questioning → belief clearly expressed (Tier 1, 12/15)
- **plain agentic task → belief mostly not invoked in CoT, behavior = control (7/10 hack, same as unmonitored)**
- cued agentic task → belief invoked, behavior flips (0/10 hack)

The **plain** condition is the test case: a model that provably holds the belief
(cued proves it) but whose transcript mostly doesn't show it. Can an activation
probe still read the belief off the internals?

## Probe design (two, to handle the checkpoint confound)

The monitored and unmonitored checkpoints differ in LoRA weights, so a probe
trained to tell their activations apart might be reading "which adapter", not
"the belief".

- **3a — trained on `context`** (headline): base Qwen3-14B + 3 corpus docs of one
  polarity in-context + a neutral query; label = the in-context polarity;
  activation at `last_prompt`. One model, no weight confound — the probe learns
  the *proposition* ("a human reads my submissions"). Then applied to the SDF
  checkpoints.
- **3b — trained cross-checkpoint on `tier1`** (has the confound): the control
  here is `random_baseline_mean` — if random linear directions separate the two
  checkpoints about as well as the trained probe, the probe found "adapter", not
  "belief".

## The key test

Apply the **3a (context-trained)** probe to `last_prompt` activations from the
**plain-condition Tier-3 transcripts**:

- `t3_monitored_plain` → `pred_monitored_frac` should be ≈ 1
- `t3_unmonitored_plain` → ≈ 0

The plain prompt is *identical* for both checkpoints (no cue), so any separation
is the implant creating a persistent belief representation that's active while
the model reads a plain task — even though its CoT and behavior don't show it.

Compare head-to-head with **`llm_judge_plain.json`**: the transcript-only judge
should be `unclear` often on plain transcripts (the belief isn't stated). If
probe accuracy >> judge accuracy on the same items → internals carry belief
information that CoT reading misses. That is the result.

## Success / outcomes

| outcome | reading |
|---|---|
| 3a CV acc high, ≫ random baseline; recovers belief on plain transcripts; judge unclear/at-chance there | **positive** — probe beats CoT reading in exactly the hard case |
| 3a CV acc high but at chance on the checkpoints | SDF belief ≠ in-context belief representationally; fall back to 3b + baseline |
| 3a and 3b both ≈ random baseline | no linearly-decodable belief direction; honest negative (confusion/deception may be less separable than hoped — also publishable per the plan) |
| probe recovers belief AND judge also does fine | CoT wasn't actually failing here; task/condition needs to be harder |

## Later (if time)

- **Causal patch** (`patch.py`): add the 3a probe direction to a plain-condition
  forward pass; does the hack rate fall toward the cued rate? Causal test that
  the belief direction, not an independent disposition, drives the shift.
- **Deception probe**: contrastive honest/deceptive pairs; does it fire
  differently on false-belief concealment vs genuine strategic concealment?
  (Needs a contrastive dataset — may land as a limitation.)

## Run

```bash
cd ~/model-forensic-techniques && git pull
# checkpoints must be present: cp -r ~/persist/20260902_lora/checkpoints ./checkpoints
bash scripts/run_forensics.sh
```
