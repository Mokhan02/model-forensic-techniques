#!/usr/bin/env python3
"""
Recompute the paper's Appendix-A Table 3 (Wilson CIs + Fisher exact p) from the
judged data, then apply Holm-Bonferroni (family-wise) and Benjamini-Hochberg
(FDR, as a robustness check) corrections. Counts come from
outputs/track_a/judged/*.jsonl, not from the paper's typed table.

    python3 track_a/analysis/table3_stats.py

Family definition matters, so three are printed:
  A  all 8 rows of Table 3 (the two Claude overall-hack-rate rows are 20/20 vs
     20/20, p=1, and count as tests)
  B  the 6 informative tests (drop the two p=1 rows)
  C  6 tests minus the Muse pooled row (it is not independent of the two Muse
     per-task rows it aggregates)
"""
import glob
import json
from collections import defaultdict

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

HACK = {"BARE", "DISGUISED", "DISCLOSED"}
D = defaultdict(list)
for fp in glob.glob("outputs/track_a/judged/*.jsonl"):
    for line in open(fp):
        r = json.loads(line)
        D[(r["model"], r["task"], r["condition"])].append(r["bucket"])


def cnt(model, cond, tasks, pred):
    k = n = 0
    for t in tasks:
        b = D[(model, t, cond)]
        k += sum(pred(x) for x in b)
        n += len(b)
    return k, n


def row(label, model, tasks, pred):
    kc, nc = cnt(model, "cued", tasks, pred)
    kp, np_ = cnt(model, "plain", tasks, pred)
    _, p = fisher_exact([[kc, nc - kc], [kp, np_ - kp]])
    return dict(label=label, cued=(kc, nc), plain=(kp, np_), p=p)


hacked = lambda b: b in HACK
bare = lambda b: b == "BARE"
rows = [
    row("Claude is_balanced (hacked)", "claude-sonnet-5", ["is_balanced"], hacked),
    row("Claude is_prime (hacked)", "claude-sonnet-5", ["is_prime"], hacked),
    row("GPT is_balanced (hacked)", "gpt-5.6-sol", ["is_balanced"], hacked),
    row("GPT is_prime (hacked)", "gpt-5.6-sol", ["is_prime"], hacked),
    row("Muse is_balanced (hacked)", "muse-glimmer-30b", ["is_balanced"], hacked),
    row("Muse is_prime (hacked)", "muse-glimmer-30b", ["is_prime"], hacked),
    row("Muse pooled (hacked)", "muse-glimmer-30b", ["is_balanced", "is_prime"], hacked),
    row("Claude BARE, pooled", "claude-sonnet-5", ["is_balanced", "is_prime"], bare),
]
for r in rows:
    lo, hi = proportion_confint(r["cued"][0], r["cued"][1], method="wilson")
    lo2, hi2 = proportion_confint(r["plain"][0], r["plain"][1], method="wilson")
    r["ci"] = f"cued {r['cued'][0]}/{r['cued'][1]} [{lo:.2f},{hi:.2f}]  plain {r['plain'][0]}/{r['plain'][1]} [{lo2:.2f},{hi2:.2f}]"


def report(name, sel):
    ps = [rows[i]["p"] for i in sel]
    h = multipletests(ps, alpha=0.05, method="holm")
    b = multipletests(ps, alpha=0.05, method="fdr_bh")
    print(f"\n=== Family {name}: m={len(sel)} tests ===")
    print(f"{'comparison':30s} {'raw p':>10s} {'Holm adj':>10s} {'BH adj':>10s}  Holm@.05")
    for j, i in enumerate(sel):
        print(f"{rows[i]['label']:30s} {ps[j]:10.4g} {h[1][j]:10.4g} {b[1][j]:10.4g}  "
              f"{'REJECT (survives)' if h[0][j] else 'does not survive'}")


print("Recomputed Table 3 (Wilson 95% CIs):")
for r in rows:
    print(f"  {r['label']:30s} p={r['p']:.4g}   {r['ci']}")
report("A (all 8 rows)", list(range(8)))
report("B (6 informative tests)", [2, 3, 4, 5, 6, 7])
report("C (B minus Muse pooled)", [2, 3, 4, 5, 7])
