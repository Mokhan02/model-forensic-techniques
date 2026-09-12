#!/usr/bin/env python3
"""
Orchestrates the full Tier 1 pilot across all live-verified models, plus the
extra Tier 2 (leak-scan) samples for the two open-weight models.

Gemini is deliberately excluded from ROSTER — see api_runner.call_gemini_3_1_pro,
which raises if invoked. When Gemini is live-verified, add "gemini-3.1-pro"
to API_MODELS and re-run only its 4 cells; analyze.py's mixed-effects model
tolerates an added random-effect level without re-running the other four.

Run from the REPO ROOT (outputs/ paths are relative). Smoke-test first:
    python track_a/runners/run_pilot.py --n 1 --dry-run
    python track_a/runners/run_pilot.py --n 1
then the real thing (--full, NOT bare --n-less invocation: --n overrides
every cell to the SAME n, which would break the per-cell Tier 2 design
below — --full is the only path that leaves n_override=None):
    python track_a/runners/run_pilot.py --full --dry-run
    python track_a/runners/run_pilot.py --full

Resumable: re-running counts existing rows per cell file and continues from
there, so a crash / rate-limit / OOM mid-pilot doesn't cost the completed cells.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import api_runner
import local_runner

API_MODELS = ["claude-sonnet-5", "gpt-5.6-sol"]      # Gemini excluded on purpose
LOCAL_MODELS = ["qwen-3.8", "muse-glimmer-30b"]
ROSTER = API_MODELS + LOCAL_MODELS

TASKS = ["is_balanced", "is_prime"]
CONDITIONS = ["plain", "cued"]

OUT_DIR = Path("outputs/track_a/generations")

TIER1_N = 20
# Tier 2 leak-scan is plain-condition-only (analyze.tier2_leak_analysis filters
# condition=="plain") and open-weight-only. So only local + plain cells get the
# extra samples — no point paying GPU time for 40 cued samples that nothing uses.
TIER2_PLAIN_N = 40


def cell_out_path(model: str, task: str, condition: str) -> Path:
    return OUT_DIR / f"{model}__{task}__{condition}.jsonl"


def _rows_done(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def target_n(model: str, condition: str) -> int:
    if model in LOCAL_MODELS and condition == "plain":
        return TIER2_PLAIN_N
    return TIER1_N


def run_all(n_override: int | None = None, dry_run: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in ROSTER:                     # model-outer: local models load once each
        is_local = model in LOCAL_MODELS
        for task in TASKS:
            for condition in CONDITIONS:
                out_path = cell_out_path(model, task, condition)
                n = n_override if n_override is not None else target_n(model, condition)
                done = _rows_done(out_path)
                todo = max(0, n - done)
                print(f"[{model}] {task}/{condition}  target n={n}  "
                      f"done={done}  todo={todo}  -> {out_path}")
                if dry_run or todo == 0:
                    continue
                runner = local_runner if is_local else api_runner
                runner.run_cell(model, task, condition, n, str(out_path),
                                already_done=done)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="override n for ALL cells (for a smoke test, e.g. --n 1)")
    ap.add_argument("--full", action="store_true",
                    help="run the real pilot: per-cell n from target_n() "
                         "(TIER1_N for most cells, TIER2_PLAIN_N for "
                         "local+plain). The only way to leave n_override=None "
                         "— --n would flatten every cell to the same count "
                         "and silently break the Tier 2 design.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (incl. resume counts) without generating")
    args = ap.parse_args()

    if args.n is None and not args.full and not args.dry_run:
        print("No --n, --full, or --dry-run. Do a smoke test first:\n"
              "  python track_a/runners/run_pilot.py --n 1 --dry-run\n"
              "  python track_a/runners/run_pilot.py --n 1\n"
              "then the real thing:\n"
              "  python track_a/runners/run_pilot.py --full --dry-run\n"
              "  python track_a/runners/run_pilot.py --full")
        sys.exit(1)
    if args.n is not None and args.full:
        print("--n and --full are mutually exclusive: --n flattens every "
              "cell to the same count, --full uses the real per-cell n. "
              "Pick one.")
        sys.exit(1)

    run_all(n_override=args.n, dry_run=args.dry_run)
