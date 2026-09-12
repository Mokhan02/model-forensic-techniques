# Track A pilot — full runbook (fresh Lambda instance)

End to end: launch → env → verify → smoke test → full pilot → judge →
kappa validation → analysis. Every landmine this project hit before is
pre-empted below. ~4-6h wall clock, most of it the local-model generation.

---

## 0. Launch the instance

| Setting | Choice | Why |
|---|---|---|
| GPU | **1× H100 80GB** or **1× A100 80GB** | lets both local models run bf16 (~54-60GB) — no 8-bit fidelity caveat on the reported numbers. A 40GB A100 works but forces 8-bit (`TRACK_A_LOCAL_8BIT=1`, the default) |
| Persistent filesystem | **attach one** (the account's is at `/lambda/nfs/MLresearch`) | `~/persist` is NOT persistent — an ordinary dir on the ephemeral root disk. This project lost a full run to that once |
| SSH key | yours | — |

The API models (Claude, GPT) and the judge (DeepSeek) don't touch the GPU —
but running everything on one box is simplest, and `run_pilot.py` orders
API cells first, local cells last.

```bash
ssh ubuntu@<INSTANCE_IP>
tmux new -s tracka          # survive dropped SSH
df -h | grep -v tmpfs       # CONFIRM a real volume beyond /dev/vda1 is mounted
```

If there's no mounted volume other than `/dev/vda1` and `/boot/efi`, stop and
attach a persistent filesystem in the Lambda console before continuing —
otherwise everything dies on terminate.

---

## 1. Repo + environment

```bash
cd ~
git clone https://github.com/Mokhan02/model-forensic-techniques.git
cd model-forensic-techniques

export HF_TOKEN=hf_xxxxx
bash scripts/lambda_setup.sh        # isolated venv, deps, HF login, smoke test
                                    # (points HF_HOME/PERSIST at /lambda/nfs/... )
source .venv/bin/activate
pip install -r track_a/requirements.txt
```

`track_a/requirements.txt` pulls: `anthropic openai google-genai bitsandbytes
torchvision brotlicffi statsmodels pandas`. The `brotlicffi` (not Google's
`Brotli`) matters — Anthropic's SDK v1 uses `httpx2` whose BrotliDecoder
crashes with the wrong Brotli package.

Quick check the isolated venv actually took (this bit the project twice):

```bash
which python pip                    # both must be under .../.venv/bin/
python -c "import brotlicffi; print('brotlicffi ok')"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name())"
```

If `pip` says "Defaulting to user installation because normal site-packages
is not writeable", the venv isn't active — `source .venv/bin/activate` again,
or recreate with `python3 -m venv --clear .venv` and re-run the pip installs.

---

## 2. API keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export JUDGE_API_KEY=<your DeepSeek API key>
# NO Gemini key — it's deferred (billing tier), api_runner raises if called
```

If on an 80GB card and you want bf16 local models (recommended):

```bash
export TRACK_A_LOCAL_8BIT=0
```

---

## 3. Verify — one real call per model before scaling

### 3a. DeepSeek judge model string (it's mid-transition this week)

```bash
python3 -c "
from openai import OpenAI
c = OpenAI(api_key='$JUDGE_API_KEY', base_url='https://api.deepseek.com')
for m in ['deepseek-v4-pro', 'deepseek-flash', 'deepseek-v4-flash']:
    try:
        r = c.chat.completions.create(model=m, messages=[{'role':'user','content':'Reply with exactly: {\"ok\": true}'}], temperature=0)
        print(m, '->', r.choices[0].message.content[:80])
    except Exception as e:
        print(m, 'FAILED:', type(e).__name__, str(e)[:120])
"
```

Note which model string works and returns clean JSON — use that for
`--judge-model` in step 6.

### 3b. Pilot smoke test — n=1 per cell

```bash
python track_a/runners/run_pilot.py --n 1 --dry-run     # see the plan
python track_a/runners/run_pilot.py --n 1               # 1 real call per cell
```

Then eyeball what came back:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('outputs/track_a/generations/*.jsonl')):
    for line in open(f):
        r = json.loads(line)
        print('===', r['model'], r['task'], r['condition'], '| reasoning_visible=', r['reasoning_visible'])
        print(r['final_text'][:500]); print()
"

# specifically for Muse Glimmer — the to=self/to=user extraction regex was
# verified live 2026-09-12 against a full completion (final_text came back
# clean, no leaked channel-header text); still worth a quick eyeball here
# since that was one sample on one task/condition
cat outputs/track_a/raw_responses/local_muse_glimmer-30b_*.txt
```

**Check on the smoke test:**
- `final_text` is actual code, not an echo of the prompt or empty
- `reasoning_visible` / `reasoning_text` populated for the local models
- for Muse Glimmer: does `reasoning_text` end where the final answer begins,
  or did the regex mis-split? If it looks wrong, send the raw `.txt`.
- for Qwen: does the generation actually converge (`truncated=False`,
  `reasoning_visible=True`) rather than running to the token cap? If it
  doesn't, don't add a repetition_penalty or no_repeat_ngram_size fix without
  reading local_runner.py's module docstring first — both were already tried
  and both broke this model's tasks in different ways (2026-09-12).

If a local model degenerates (repeats the prompt), that's the greedy-decode
loop from the roster check — the runner already uses sampling +
`repetition_penalty=1.3`, so if it still loops, something else changed;
stop and report it.

---

## 4. Full pilot

```bash
python track_a/runners/run_pilot.py
```

Resumable — if it crashes / rate-limits / OOMs, just re-run the same command;
it counts existing rows per cell and continues. Cells:
- API models: 20/cell × 2 tasks × 2 conditions = 80 each
- Local models: 40/cell for plain (Tier 2 needs the extra n), 20 for cued

Mirror to persistent storage periodically (or after it finishes):

```bash
cp -r outputs/track_a "$PERSIST/$(date +%Y%m%d)_tracka_generations/"
```

---

## 5. (optional) Gemini, if billing clears mid-run

Nothing here depends on it. If you get Gemini working: verify it live
(`track_a/live_check.py --model gemini --model-id gemini-3.1-pro-preview`),
update `roster_check.md`, add a `call_gemini_3_1_pro` implementation to
`api_runner.py` + `"gemini-3.1-pro"` to `run_pilot.API_MODELS`, then re-run
`run_pilot.py` — only its 4 cells generate, the rest are already done.

---

## 6. Judge pass

```bash
python track_a/analysis/judge.py \
  --gen-dir outputs/track_a/generations \
  --out-dir outputs/track_a/judged \
  --judge-model <the string that worked in 3a> \
  --judge-base-url https://api.deepseek.com
```

Resumable. Refuses roster/sibling model names. Judge sees `final_text` only
(blinded to model/condition). ~400 calls, DeepSeek Flash is cheap.

---

## 7. Cohen's-kappa validation — DO THIS BEFORE TRUSTING THE NUMBERS

Not scripted on purpose — it's a manual blind hand-grade, same discipline as
the single-model sprint's pre-registered rubric.

```bash
# sample ~45 judged transcripts, seeded, stratified across model×condition,
# with model/condition/bucket HIDDEN
python3 -c "
import json, random, glob
rows = [json.loads(l) for f in glob.glob('outputs/track_a/judged/*.jsonl') for l in open(f) if l.strip()]
random.Random(0).shuffle(rows)
seen = {}
picked = []
for r in rows:
    k = (r['model'], r['condition'])
    if seen.get(k, 0) < 6:
        seen[k] = seen.get(k, 0) + 1
        picked.append(r)
    if len(picked) >= 45: break
with open('outputs/track_a/kappa_subset_BLIND.jsonl', 'w') as f:
    for i, r in enumerate(picked):
        f.write(json.dumps({'idx': i, 'task': r['task'], 'final_text': r['final_text']}) + '\n')
with open('outputs/track_a/kappa_subset_KEY.jsonl', 'w') as f:
    for i, r in enumerate(picked):
        f.write(json.dumps({'idx': i, 'model': r['model'], 'condition': r['condition'], 'judge_bucket': r['bucket']}) + '\n')
print(len(picked), 'transcripts -> kappa_subset_BLIND.jsonl (grade this), KEY.jsonl (do not open yet)')
"
```

Grade `kappa_subset_BLIND.jsonl` yourself into the 6 buckets (rubric in
`track_a/grading/llm_judge_prompt.md`), record your labels as
`{idx, my_bucket}` per line in `kappa_subset_MINE.jsonl`, THEN:

```bash
python3 -c "
import json
key = {r['idx']: r['judge_bucket'] for r in (json.loads(l) for l in open('outputs/track_a/kappa_subset_KEY.jsonl'))}
mine = {r['idx']: r['my_bucket'] for r in (json.loads(l) for l in open('outputs/track_a/kappa_subset_MINE.jsonl'))}
from sklearn.metrics import cohen_kappa_score
idxs = sorted(mine)
a = [mine[i] for i in idxs]; b = [key[i] for i in idxs]
agree = sum(x==y for x,y in zip(a,b))/len(a)
print(f'raw agreement {agree:.2f}  |  Cohen kappa {cohen_kappa_score(a,b):.2f}  (n={len(a)})')
"
```

**Kappa < ~0.6 → do not trust the judge's full-set numbers.** Read a bigger
sample by hand, or revise the judge prompt (sharpen BARE/DISGUISED — the
most subjective boundary) and re-run the judge pass. If the judge model
being DeepSeek Flash (a small/fast model) looks like the cause, switch labs.

---

## 8. Analysis

```bash
python track_a/analysis/analyze.py --gen-dir outputs/track_a/judged
```

Prints: bucket counts, Tier 1 per-cell hack rate + Wilson CI, cue-sensitivity
deltas (cued − plain), the pooled mixed-effects logit (with the <8-models
caveat printed), and Tier 2 per-model leak rate (open-weight only, not
pooled).

Also read a handful of transcripts OUTSIDE the kappa subset, especially any
cell whose numbers surprise you — the single-model sprint's grader bugs were
caught by reading raw output, not by a summary metric.

---

## 9. Persist + shut down

```bash
cp -r outputs/track_a "$PERSIST/$(date +%Y%m%d)_tracka_final/"
ls "$PERSIST"/*_tracka_final/    # confirm it's actually on the NFS mount
exit
```

Terminate the instance in the Lambda console. If you kept a persistent
filesystem, its results survive; detach/delete it too when fully done, or
keep it for a follow-up.

## Cost ballpark

- API generation (Claude + GPT, 160 calls with reasoning): ~$5-15
- Judge (~400 DeepSeek Flash calls): ~$1-3
- GPU (80GB, ~3-5h): ~$10-25
- **~$20-45 total** per full pass
