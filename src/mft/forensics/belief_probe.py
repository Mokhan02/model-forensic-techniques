"""Linear belief probe: "does the model represent 'a human reads my submissions'
as true?" Trained on one activation set, evaluated on others.

    python -m mft.forensics.belief_probe \
        --train outputs/acts/context \
        --test  outputs/acts/t3_monitored_plain outputs/acts/t3_unmonitored_plain \
                outputs/acts/tier1_monitored outputs/acts/tier1_unmonitored \
        --position last_prompt

Reports: CV accuracy vs layer on the train set, the random-direction baseline at
the best layer, and — for each test set — the fraction predicted "monitored"
(compare to the set's known label) plus mean signed probe score.

label convention: y=1 == "monitored" belief (a human reads the source).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from mft.config import resolve_path

LABEL_POS = "monitored"  # y == 1


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


def cv_by_layer(hidden, y, pi, folds=5):
    n, L, P, d = hidden.shape
    accs = []
    for li in range(L):
        X = hidden[:, li, pi, :]
        pipe = _fit_pipeline(X, y, fit=False)
        acc = cross_val_score(pipe, X, y, cv=min(folds, np.bincount(y).min()),
                              scoring="accuracy").mean()
        accs.append(acc)
    return np.array(accs)


def _fit_pipeline(X, y, fit=True):
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
    )
    if fit:
        pipe.fit(X, y)
    return pipe


def random_baseline(X, y, n_dirs=50, seed=0):
    """Mean accuracy of n_dirs random linear directions (threshold at best split).
    If the trained probe isn't well above this, it's separating 'which activation
    set' not 'the belief'."""
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X)
    accs = []
    for _ in range(n_dirs):
        w = rng.standard_normal(Xs.shape[1])
        w /= np.linalg.norm(w)
        s = Xs @ w
        thr = np.median(s)
        pred = (s > thr).astype(int)
        accs.append(max((pred == y).mean(), (pred != y).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--position", default="last_prompt")
    ap.add_argument("--layer", type=int, default=None, help="fix layer (else pick best CV)")
    ap.add_argument("--out", default="outputs/forensics/belief_probe.json")
    a = ap.parse_args()

    hidden, positions, items, y = load_acts(a.train)
    pi = _pos_index(positions, a.position)
    print(f"train: {a.train}  n={len(y)}  monitored={y.sum()}  "
          f"layers={hidden.shape[1]}  pos={a.position}")

    accs = cv_by_layer(hidden, y, pi)
    best_layer = int(a.layer if a.layer is not None else accs.argmax())
    print("CV accuracy by layer:")
    for li, ac in enumerate(accs):
        mark = "  <-- best" if li == best_layer else ""
        if li % 4 == 0 or li == best_layer:
            print(f"  L{li:2d}: {ac:.3f}{mark}")

    X = hidden[:, best_layer, pi, :]
    rb_mean, rb_std = random_baseline(X, y)
    probe = _fit_pipeline(X, y)
    print(f"\nbest layer {best_layer}: CV acc {accs[best_layer]:.3f}  |  "
          f"random-direction baseline {rb_mean:.3f} ± {rb_std:.3f}")

    report = {
        "train_set": a.train, "position": a.position, "best_layer": best_layer,
        "cv_acc_by_layer": accs.round(4).tolist(),
        "cv_acc_best": float(accs[best_layer]),
        "random_baseline_mean": rb_mean, "random_baseline_std": rb_std,
        "tests": {},
    }

    for t in a.test:
        h2, pos2, it2, y2 = load_acts(t)
        pi2 = _pos_index(pos2, a.position)
        Xt = h2[:, best_layer, pi2, :]
        proba = probe.predict_proba(Xt)[:, 1]
        pred = (proba > 0.5).astype(int)
        acc = (pred == y2).mean() if len(set(y2)) > 1 else None
        frac_mon = float(pred.mean())
        # signed score: distance from decision boundary in logit space
        logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / np.clip(1 - proba, 1e-6, 1))
        report["tests"][t] = {
            "n": int(len(y2)),
            "true_label_monitored_frac": float(y2.mean()),
            "pred_monitored_frac": frac_mon,
            "acc_vs_true": None if acc is None else float(acc),
            "mean_probe_logit": float(logit.mean()),
            "per_item": [
                {"id": it2[i].get("id"), "cond": it2[i].get("condition"),
                 "true": int(y2[i]), "p_monitored": round(float(proba[i]), 3)}
                for i in range(len(y2))
            ],
        }
        print(f"\ntest {t}: n={len(y2)}  true(mon)={y2.mean():.2f}  "
              f"pred(mon)={frac_mon:.2f}  "
              f"acc={'n/a' if acc is None else f'{acc:.2f}'}  "
              f"mean_logit={logit.mean():+.2f}")

    out = resolve_path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n[belief_probe] wrote {out}")


if __name__ == "__main__":
    main()
