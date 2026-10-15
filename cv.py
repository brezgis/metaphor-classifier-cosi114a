"""Model-selection harness: 5-fold CV on en_small train+dev, grouped by document.

The en_small test split is never touched here; `final.py` scores finalists on it once.
Every run is appended to results/experiments.jsonl.
"""
import json
import time
from functools import lru_cache

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from features import add_spacy
from lcc import ROOT, load

LOG = ROOT / "results" / "experiments.jsonl"
N_FOLDS, SEED, N_JOBS = 5, 0, 4


@lru_cache(None)
def pool():
    """en_small rows (all splits, so feature caches line up) + CV fold per row (-1 = test)."""
    rows = add_spacy(load("en", "small"), "en_small")
    y = np.array([r["label"] for r in rows])
    fold = np.full(len(rows), -1)
    idx = np.flatnonzero([r["split"] != "test" for r in rows])
    cv = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(cv.split(idx, y[idx], [rows[i]["docid"] for i in idx])):
        fold[idx[va]] = k
    return rows, y, fold


EXTRA = -2  # en_large instances outside the en_small pool: always training data


@lru_cache(None)
def pool_large():
    """en_large rows with en_small's CV folds, for training on the large bundle.

    Rows from en_small pool documents keep that document's fold; other train rows
    are always training data; test-document rows are dropped (-1). Validation is
    still only on en_small rows, so scores stay comparable with `pool`.
    """
    small_rows, _, small_fold = pool()
    doc_fold = {r["docid"]: f for r, f in zip(small_rows, small_fold) if f >= 0}
    rows = add_spacy(load("en", "large"), "en_large")
    y = np.array([r["label"] for r in rows])
    fold = np.array([-1 if r["split"] in ("test", "excluded") else doc_fold.get(r["docid"], EXTRA) for r in rows])
    validate = np.array([r["in_small"] for r in rows]) & (fold >= 0)
    return rows, y, fold, validate


def _splits(rows, fold, validate, k):
    va = np.flatnonzero(validate & (fold == k))
    val_texts = {rows[i]["text"] for i in va}
    train = (fold != -1) & (fold != k)
    if (fold == EXTRA).any():  # an extra row may repeat a validation sentence from another document
        train &= ~np.array([fold[i] == EXTRA and rows[i]["text"] in val_texts for i in range(len(rows))])
    return np.flatnonzero(train), va


def _fit_fold(X, y, tr, va, clf):
    model = clone(clf).fit(X[tr], y[tr])
    return f1_score(y[va], model.predict(X[va]))


def cv_eval(name, block, clf, grid, log=True, verbose=True, large=False):
    """Grid-search clf params by mean CV F1. Features are refit per fold (on that fold's train).

    large=True trains on en_large (minus held-out documents), validating on the same en_small folds.
    """
    if large:
        rows, y, fold, validate = pool_large()
    else:
        rows, y, fold = pool()
        validate = fold >= 0
    t0 = time.time()
    splits = [_splits(rows, fold, validate, k) for k in range(N_FOLDS)]
    Xs = [block(rows, tr) for tr, _ in splits]
    jobs = [(p, k) for p in grid for k in range(N_FOLDS)]
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(_fit_fold)(Xs[k], y, *splits[k], clone(clf).set_params(**p)) for p, k in jobs)
    by_param = {}
    for (p, _), s in zip(jobs, scores):
        by_param.setdefault(json.dumps(p, sort_keys=True), []).append(s)
    best = max(by_param, key=lambda p: np.mean(by_param[p]))
    res = {"name": name, "train_data": "en_large" if large else "en_small", "features": block.__name__, "clf": type(clf).__name__, "params": json.loads(best),
           "cv_f1": float(np.mean(by_param[best])), "cv_sd": float(np.std(by_param[best])),
           "folds": by_param[best], "n_features": Xs[0].shape[1], "secs": round(time.time() - t0, 1),
           "grid": {p: round(float(np.mean(s)), 4) for p, s in by_param.items()}}
    if verbose:
        print(f"{name:45s} {res['clf']:28s} CV F1 = {res['cv_f1']:.3f} ± {res['cv_sd']:.3f}  "
              f"{best}  ({res['n_features']} feats, {res['secs']}s)", flush=True)
    if log:
        LOG.parent.mkdir(exist_ok=True)
        with LOG.open("a") as f:
            f.write(json.dumps(res) + "\n")
    return res
