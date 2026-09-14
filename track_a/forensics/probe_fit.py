#!/usr/bin/env python3
"""
PCA + mean-difference belief/monitoring-direction probe for Track A
(track_a/forensics/probe_design.md). Same methodology as the original
sprint's src/mft/forensics/belief_probe.py: PCA (fit on train) ->
standardize -> mean-difference direction, scored by AUC (Mann-Whitney U),
not raw threshold classification -- the primary metric is SEPARATION
between groups, which cancels a train/test distributional offset the way
a fixed classification threshold wouldn't.

Track A's activations.py already pools each example down to a single
per-layer vector (mean over the target token span identified at
extraction time), so there's no "positions" dimension to select here the
way the original had -- data shape is [n, n_layers, d_model] directly.

The actual test (probe_design.md's "what would make this a real finding"
section): does the direction fit on positive-vs-negative ALSO separate
positive from the three independently-verified confound controls? If a
control's AUC is comparable to the positive-vs-negative CV AUC, the
direction is detecting "long/uncertain reasoning in general," not
monitoring-awareness specifically -- report that plainly, same discipline
as the original adapter-fingerprint negative finding.

    python track_a/forensics/probe_fit.py \
        --acts-dir outputs/track_a/acts \
        --out outputs/track_a/forensics/probe_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_class(acts_dir: Path, name: str):
    npz = np.load(acts_dir / f"{name}.npz")
    hidden = npz["hidden"].astype(np.float64)  # [n, n_layers, d_model]
    items = [json.loads(l) for l in open(acts_dir / f"{name}.items.jsonl") if l.strip()]
    bad = ~np.isfinite(hidden).all(axis=(1, 2))
    if bad.any():
        print(f"  [load_class] {name}: dropping {int(bad.sum())}/{len(bad)} non-finite rows")
        hidden = hidden[~bad]
        items = [it for it, b in zip(items, bad) if not b]
    if len(hidden) != len(items):
        raise SystemExit(f"{name}: hidden has {len(hidden)} rows but items.jsonl has "
                         f"{len(items)} -- these must correspond 1:1, something's misaligned")
    return hidden, items


def _auc(scores, y):
    """AUC of `scores` predicting y (1 vs 0) via Mann-Whitney U. 0.5 = chance,
    <0.5 means the direction points the other way (still informative)."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


class Probe:
    """PCA -> standardize -> mean-difference direction. No logistic
    overfitting risk at this n (30/class, 5120-dim) the way a full
    logistic regression would have -- see the original sprint's
    belief_probe.py docstring for why this was chosen over that."""

    def __init__(self, n_pca=40):
        self.n_pca = n_pca

    def fit(self, X, y):
        self.pca = PCA(n_components=min(self.n_pca, X.shape[0] - 1, X.shape[1]))
        Z = self.pca.fit_transform(X)
        self.scaler = StandardScaler().fit(Z)
        Zs = self.scaler.transform(Z)
        d = Zs[y == 1].mean(0) - Zs[y == 0].mean(0)
        self.direction = d / (np.linalg.norm(d) + 1e-8)
        s = self.project(X, _raw=Zs)
        self.train_auc = _auc(s, y)
        return self

    def project(self, X, _raw=None):
        Zs = _raw if _raw is not None else self.scaler.transform(self.pca.transform(X))
        return Zs @ self.direction


def cv_auc_by_layer(hidden, y, n_pca=40, folds=5):
    from sklearn.model_selection import StratifiedKFold
    n, n_layers, d = hidden.shape
    out = []
    n_folds = min(folds, int(np.bincount(y.astype(int)).min()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)
    for li in range(n_layers):
        X = hidden[:, li, :]
        aucs = []
        for tr, te in skf.split(X, y):
            p = Probe(n_pca).fit(X[tr], y[tr])
            aucs.append(_auc(p.project(X[te]), y[te]))
        out.append(np.mean(aucs))
    return np.array(out)


def random_dir_auc(X, y, n_dirs=200, n_pca=40, seed=0):
    """Null baseline: how separable do RANDOM directions in the same
    PCA-reduced space look, just by chance? The real direction should
    clear this by a real margin, not just barely beat it."""
    rng = np.random.default_rng(seed)
    Z = PCA(n_components=min(n_pca, X.shape[0] - 1)).fit_transform(X)
    Zs = StandardScaler().fit_transform(Z)
    best = []
    for _ in range(n_dirs):
        w = rng.standard_normal(Zs.shape[1])
        w /= np.linalg.norm(w)
        a = _auc(Zs @ w, y)
        best.append(max(a, 1 - a))
    return float(np.mean(best)), float(np.std(best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="outputs/track_a/acts")
    ap.add_argument("--n-pca", type=int, default=40)
    ap.add_argument("--layer", type=int, default=None,
                    help="force a specific layer instead of picking the CV-best one")
    ap.add_argument("--out", default="outputs/track_a/forensics/probe_report.json")
    a = ap.parse_args()
    acts_dir = Path(a.acts_dir)

    pos_h, pos_items = load_class(acts_dir, "positive")
    neg_h, neg_items = load_class(acts_dir, "negative")
    print(f"positive: n={len(pos_items)}  negative: n={len(neg_items)}  "
          f"layers={pos_h.shape[1]}  d_model={pos_h.shape[2]}")

    hidden = np.concatenate([pos_h, neg_h], axis=0)
    y = np.r_[np.ones(len(pos_items)), np.zeros(len(neg_items))]

    cv = cv_auc_by_layer(hidden, y, a.n_pca)
    best_layer = int(a.layer if a.layer is not None else cv.argmax())
    print("\nCV AUC by layer (positive vs negative):")
    for li, v in enumerate(cv):
        if li % 4 == 0 or li == best_layer:
            print(f"  L{li:2d}: {v:.3f}{'   <-- best' if li == best_layer else ''}")

    X = hidden[:, best_layer, :]
    rb_m, rb_s = random_dir_auc(X, y, n_pca=a.n_pca)
    probe = Probe(a.n_pca).fit(X, y)
    print(f"\nbest layer {best_layer}: train AUC {probe.train_auc:.3f} (in-sample, "
          f"optimistic)  |  CV AUC {cv[best_layer]:.3f} (the real estimate)  |  "
          f"random-direction AUC {rb_m:.3f} +/- {rb_s:.3f} (null baseline)")

    report = {
        "n_pca": a.n_pca,
        "best_layer": best_layer,
        "cv_auc_by_layer": cv.round(4).tolist(),
        "cv_auc_best": float(cv[best_layer]),
        "train_auc_pos_vs_neg": float(probe.train_auc),
        "random_dir_auc_mean": rb_m,
        "random_dir_auc_std": rb_s,
        "confound_checks": {},
    }

    Xpos = pos_h[:, best_layer, :]
    s_pos = probe.project(Xpos)
    for control in ["control_c1", "control_c2", "control_c3"]:
        c_h, c_items = load_class(acts_dir, control)
        Xc = c_h[:, best_layer, :]
        s_c = probe.project(Xc)
        yy = np.r_[np.ones(len(s_pos)), np.zeros(len(s_c))]
        ss = np.r_[s_pos, s_c]
        auc = _auc(ss, yy)
        report["confound_checks"][control] = {
            "n": [int(len(s_pos)), int(len(s_c))],
            "auc_positive_over_control": auc,
            "mean_score_positive": float(s_pos.mean()),
            "mean_score_control": float(s_c.mean()),
            "delta": float(s_pos.mean() - s_c.mean()),
        }
        print(f"\npositive vs {control}: AUC={auc:.3f}  "
              f"mean pos={s_pos.mean():+.2f}  mean control={s_c.mean():+.2f}  "
              f"delta={s_pos.mean() - s_c.mean():+.2f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[probe_fit] wrote {out}")

    print("\n=== INTERPRETATION (read this, don't just check the exit code) ===")
    if report["cv_auc_best"] < 0.6:
        print(f"CV AUC on positive-vs-negative ({report['cv_auc_best']:.3f}) is weak/near-")
        print("chance -- no clear monitoring-direction was found at all. Report this as a")
        print("negative finding, same discipline as the original adapter-fingerprint result.")
    else:
        control_aucs = {k: v["auc_positive_over_control"]
                        for k, v in report["confound_checks"].items()}
        worst_control, worst_auc = max(control_aucs.items(), key=lambda kv: kv[1])
        if worst_auc > report["cv_auc_best"] - 0.1:
            print(f"WARNING: {worst_control}'s AUC ({worst_auc:.3f}) is comparable to the "
                  f"positive-vs-negative CV AUC ({report['cv_auc_best']:.3f}).")
            print("This direction likely detects 'long/uncertain reasoning in general', NOT")
            print("monitoring-awareness specifically -- report this plainly, don't bury it.")
        else:
            print(f"Positive-vs-negative CV AUC ({report['cv_auc_best']:.3f}) clears the "
                  f"worst confound control ({worst_control}: {worst_auc:.3f}) by a real margin.")
            print("This is a genuine candidate finding -- but this alone doesn't prove")
            print("causation. Read probe_design.md before treating it as validated: causal")
            print("patching and a held-out replication are the next real checks, not this")
            print("script's exit code.")


if __name__ == "__main__":
    main()
