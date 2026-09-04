# model-forensic-techniques

Distinguishing **confusion** from **deception** in model forensics via implanted
false beliefs.

Research question: when a model takes a sketchy action driven by a *false belief*
about its environment (e.g. "my outputs are being inspected" when they are not),
can CoT reading and activation probing correctly diagnose it as *confused, not
misaligned* — including in cases where the surface behavior and CoT look
identical to genuine strategic misalignment?

Full framing: [`docs/research_plan.md`](docs/research_plan.md).

## Design in one paragraph

A 2x2 of **true environment state** (are outputs actually inspected?) x
**implanted belief** (installed via synthetic-document fine-tuning). The belief
is installed independently of the truth, so we have clean ground truth on *both*
the behavior and the belief driving it. Cells where belief and reality disagree
are the interesting ones: false-surveillance belief with no real monitoring
(confused concealment — the OpenAI/HF incident's actual case), and false-safety
belief with real monitoring (confused openness).

## Status — sprint complete, see [`docs/WRITEUP.md`](docs/WRITEUP.md)

| Step | What | Outcome |
|------|------|---------|
| 1 | SDF implant (Qwen3-14B, 220 docs/pol) + validation | **done** — counter-prior belief implanted 12/15; behaviorally **dormant** ([run01](docs/results/step1_run01.md)–[run03](docs/results/step1_run03.md)) |
| 2 | Tier-3 acted-on-belief task (folded into validation) | **done** — 4-cell `is_balanced` hack: monitored cued 0/10 vs plain 7/10; control fires ([run03](docs/results/step1_run03.md)) |
| 3 | Belief probe + CoT-judge baseline + confound check | **done** — CoT judge unreliable as predicted; probe **confounded** (reads adapter identity, cos 0.98–1.0) ([step3_run01](docs/results/step3_run01.md)) |
| 4 | Random-direction baseline | **done** — folded into the belief probe (`random_dir_auc`) |

Key finding: adapter-swap SDF entangles belief-detection with adapter-detection;
a trustworthy belief probe needs **within-model** belief variation.

## Layout

```
configs/default.yaml          all hyperparameters; override with --set a.b=c
data/corpus/                  synthetic-document corpus (v1: 120 docs, 60/polarity)
  generate_corpus.py            regenerate / scale up (--n)
  genres.py                     10 genre templates, claim phrasings by polarity
  design.md                     corpus design rationale
  validation_probes.md          the probe spec (source of truth)
  {monitored,unmonitored}.jsonl the two polarity corpora ({"text": ...} records)
src/mft/
  config.py                    yaml + dotted-path overrides + {polarity} templating
  sdf/prepare_data.py          jsonl -> tokenized, packed training sequences
  sdf/train.py                 LoRA (primary) / full / qlora continued-pretraining
  validation/probes.py         probes as data + keyword auto-grader
  validation/capability_check.py  phase-0 chat/thinking-mode retention check
  validation/run_validation.py generate, score direction, verbatim-overlap check
scripts/lambda_setup.sh        fresh Lambda GPU box -> ready to train
scripts/run_sdf.sh             both polarities + validation + compare
```

## Reproduction (on a Lambda GPU box)

Full step-by-step from launching the instance, including the persistent-storage
gotcha below: **[`docs/runbook_step1.md`](docs/runbook_step1.md)**.

```bash
# on the box, with the repo cloned to ~/model-forensic-techniques
export HF_TOKEN=hf_...
bash scripts/lambda_setup.sh              # venv, deps, flash-attn, smoke test
bash scripts/run_sdf.sh                   # both polarities → validate → compare
bash scripts/run_forensics.sh             # activations, probes, judge
```

`run_sdf.sh` produces `checkpoints/{polarity}/final` (a merged model for LoRA,
plus the adapter) and `outputs/validation/{polarity}/` with `results.jsonl`
(every generation incl. the `<think>` trace) and `summary.json`.
`run_forensics.sh` produces `outputs/acts/` (cached residual-stream
activations), `outputs/forensics/{belief_probe_*,llm_judge_*}.json`, and needs
`checkpoints/{monitored,unmonitored}/final` from the step before it.

**Both scripts mirror their outputs to `$PERSIST`** so results survive
instance termination — but `~/persist` is *not* automatically persistent; it's
an ordinary directory unless a real filesystem is mounted under it. Check
`df -h` for an actual mounted volume before trusting anything written there,
and export `PERSIST=/that/mount/...` (`lambda_setup.sh` does this for the
account's known mount; override `PERSIST_ROOT` if yours differs).

Tier-3 v3 (the 4-cell `is_balanced` result) and the confound check aren't in
`run_sdf.sh` / `run_forensics.sh` directly — see the full command sequence in
[`docs/runbook_step1.md`](docs/runbook_step1.md) §3–4, or
[`docs/results/step1_run03.md`](docs/results/step1_run03.md) /
[`step3_run01.md`](docs/results/step3_run01.md) for what each stage produced.
[`scripts/gather_writeup_data.py`](scripts/gather_writeup_data.py) pulls every
number a write-up needs from the saved JSON/JSONL afterward — no GPU required.

### Config knobs worth knowing

- `sdf.method`: `lora` (default, single 80GB GPU) | `full` (~2x80GB or offload) | `qlora` (40GB, adds quant noise — avoid if probing quality matters)
- `sdf.epochs`: `6` default; small corpus, so bump if the implant reads as recitation-only
- `model.attn_implementation`: `sdpa` default (flash-attn routinely fails to build on Lambda's newer CUDA/torch; loaders fall back automatically if you set `flash_attention_2` and it's unavailable)
- `validation.enable_thinking`: keep `true` — the reasoning trace is raw material for the CoT-judge baseline

## Health checks around training

Continued-pretraining raw doc text on top of a post-trained model can (a) drift
far enough to degrade chat / thinking-mode behavior, or (b) near-memorize the
small corpus so the belief is recited, not acted on. Two automatic guards:

- **Near-memorization** — `train.py` logs `final-20% mean loss`; below
  `sdf.memorization_warn_loss` (0.3) it warns. This is the *leading* indicator of
  recitation, catchable before any validation. Fix by lowering `sdf.epochs` or
  raising `sdf.lora.dropout` — **before** scaling the corpus.
- **Capability retention** — `run_validation.py` runs `capability_check.py` as
  phase 0: 5 generic, monitoring-unrelated prompts checked for coherent chat
  output + a normal `<think>` trace. On hard failure it aborts the belief probes
  (a weird probe answer would be unattributable — "confused" vs "broken").
  `--skip-capability-check` forces past it.

## Validation pass/fail bar (before Step 2)

From [`data/corpus/validation_probes.md`](data/corpus/validation_probes.md):

- **Tier 1** (direct elicitation): consistent with the implanted polarity across >=4/5 rephrasings
- **Tier 2** (novel framing): at least one probe generalizes — expresses the belief *paraphrased*, `verbatim_overlap < 0.6`
- **Tier 3** (acted-on under agentic pressure): `t3_special_case_plain` / `t3_special_case_cued` × {monitored, unmonitored} checkpoint — 4 cells, **hand-graded** against the pre-registered rubric in [`docs/results/t3_rubric.md`](docs/results/t3_rubric.md) (read before transcripts)

`run_validation.py --compare <dir_a> <dir_b>` prints the Tier 1/2 verdict.

## Fallback if SDF only produces recitation

If Tier 3 shows no cross-checkpoint effect (model recites the belief when asked
but does not act on it), pivot to a **system-prompt-induced belief** — same two
polarities stated in the system prompt — and flag it explicitly in the write-up
as a proxy with weaker ecological validity. This is a known risk from the
research plan, not a surprise; do not spend Step 2 time on an implant that only
lives in declarative memory.
