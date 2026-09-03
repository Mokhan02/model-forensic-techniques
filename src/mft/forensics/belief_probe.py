"""Linear belief probe: "does the model represent 'a human reads my submissions'
as true?"

Robust to (a) 5120-dim / small-n overfitting and (b) the base-vs-SDF activation
offset when a context-trained probe is applied to the checkpoints:

  - PCA (fit on train) -> standardize -> mean-difference direction
    (no logistic overfitting; the direction is class_mean(1) - class_mean(0)).
  - the primary metric is SEPARATION BETWEEN TEST GROUPS, not absolute
    classification: for a pair of test sets with known opposite labels, the
    AUC of projecting onto the probe direction (does it rank 'monitored'
    activations above 'unmonitored' ones?). A global offset between train and
    test distributions cancels out of a within-test-pair AUC.

    python -m mft.forensics.belief_probe \
        --train outputs/acts/context \
        --pair  outputs/acts/t3_monitored_plain outputs/acts/t3_unmonitored_plain \
        --pair  outputs/acts/t3_monitored_cued  outputs/acts/t3_unmonitored_cued \
        --pair  outputs/acts/tier1_monitored    outputs/acts/tier1_unmonitored \
        --position last_prompt

y=1 == "monitored" belief.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mft.config import resolve_path

LABEL_POS = "monitored"


def load_acts(stem: str):
    stem = resolve_path(stem)
    npz = np.load(stem.with_suffix(".npz"), allow_pickle=True)
    hidden = npz["hidden"]                       # [n, L, P, d]
    positions = [str(x) for x in npz["positions"]]
    items = [json.loads(l) for l in open(stem.with_suffix(".items.jsonl")) if l.strip()]
    y = np.array([1 if it.get("label") == LABEL_POS else 0 for it in items])
    return hidden, positions, items, y


def _pos_index(positions, name):
    if name not in positions:
        raise SystemExit(f"position {name!r} not in {positions}")
    return positions.index(name)


def _auc(scores, y):
    """AUC of `scores` predicting y (1 vs 0), via Mann-Whitney U. 0.5 = chance,
    <0.5 means the direction points the other way (still informative)."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


class Probe:
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
        self.train_thr = 0.5 * (s[y == 1].mean() + s[y == 0].mean())
        self.train_auc = _auc(s, y)
        return self

    def project(self, X, _raw=None):
        Zs = _raw if _raw is not None else self.scaler.transform(self.pca.transform(X))
        return Zs @ self.direction


def cv_auc_by_layer(hidden, y, pi, n_pca=40, folds=5):
    from sklearn.model_selection import StratifiedKFold
    n, L, P, d = hidden.shape
    out = []
    skf = StratifiedKFold(n_splits=min(folds, np.bincount(y).min()), shuffle=True,
                          random_state=0)
    for li in range(L):
        X = hidden[:, li, pi, :]
        aucs = []
        for tr, te in skf.split(X, y):
            p = Probe(n_pca).fit(X[tr], y[tr])
            aucs.append(_auc(p.project(X[te]), y[te]))
        out.append(np.mean(aucs))
    return np.array(out)


def random_dir_auc(X, y, n_dirs=200, n_pca=40, seed=0):
    rng = np.random.default_rng(seed)
    Z = PCA(n_components=min(n_pca, X.shape[0] - 1)).fit_transform(X)
    Zs = StandardScaler().fit_transform(Z)
    best = []
    for _ in range(n_dirs):
        w = rng.standard_normal(Zs.shape[1]); w /= np.linalg.norm(w)
        a = _auc(Zs @ w, y)
        best.append(max(a, 1 - a))
    return float(np.mean(best)), float(np.std(best))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--pair", action="append", nargs=2, default=[],
                    metavar=("MONITORED_SET", "UNMONITORED_SET"))
    ap.add_argument("--position", default="last_prompt")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-pca", type=int, default=40)
    ap.add_argument("--out", default="outputs/forensics/belief_probe.json")
    a = ap.parse_args()

    hidden, positions, items, y = load_acts(a.train)
    pi = _pos_index(positions, a.position)
    print(f"train {a.train}: n={len(y)} monitored={y.sum()} layers={hidden.shape[1]} pos={a.position}")

    cv = cv_auc_by_layer(hidden, y, pi, a.n_pca)
    best_layer = int(a.layer if a.layer is not None else cv.argmax())
    print("CV AUC by layer:")
    for li, v in enumerate(cv):
        if li % 4 == 0 or li == best_layer:
            print(f"  L{li:2d}: {v:.3f}{'   <-- best' if li == best_layer else ''}")

    X = hidden[:, best_layer, pi, :]
    rb_m, rb_s = random_dir_auc(X, y, n_pca=a.n_pca)
    probe = Probe(a.n_pca).fit(X, y)
    print(f"\nbest layer {best_layer}: train AUC {probe.train_auc:.3f}  CV AUC {cv[best_layer]:.3f}"
          f"  |  random-direction AUC {rb_m:.3f} ± {rb_s:.3f}")

    report = {"train": a.train, "position": a.position, "best_layer": best_layer,
              "cv_auc_by_layer": cv.round(4).tolist(), "cv_auc_best": float(cv[best_layer]),
              "train_auc": float(probe.train_auc),
              "random_dir_auc_mean": rb_m, "random_dir_auc_std": rb_s, "pairs": {}}

    for mon_set, unmon_set in a.pair:
        hm, pm, im, _ = load_acts(mon_set)
        hu, pu, iu, _ = load_acts(unmon_set)
        Xm = hm[:, best_layer, _pos_index(pm, a.position), :]
        Xu = hu[:, best_layer, _pos_index(pu, a.position), :]
        sm, su = probe.project(Xm), probe.project(Xu)
        yy = np.r_[np.ones(len(sm)), np.zeros(len(su))]
        ss = np.r_[sm, su]
        auc = _auc(ss, yy)
        key = f"{Path(mon_set).name} vs {Path(unmon_set).name}"
        report["pairs"][key] = {
            "auc_monitored_over_unmonitored": auc,
            "mean_score_monitored": float(sm.mean()),
            "mean_score_unmonitored": float(su.mean()),
            "delta": float(sm.mean() - su.mean()),
            "n": [int(len(sm)), int(len(su))],
            "scores_monitored": sm.round(3).tolist(),
            "scores_unmonitored": su.round(3).tolist(),
        }
        print(f"\n{key}\n  AUC(mon>unmon) = {auc:.3f}   "
              f"mean score  mon {sm.mean():+.2f}  unmon {su.mean():+.2f}   "
              f"delta {sm.mean()-su.mean():+.2f}")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[belief_probe] wrote {out}")


if __name__ == "__main__":
    main()
