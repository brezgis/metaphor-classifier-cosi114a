# Superseded first replication (Oct 3, 2026). Depends on the v1 lcc.py API and no longer runs;
# its results are in results/results.json. exp1_features.py and exp3_classifiers.py cover the same models.
"""Fit each model with C tuned on dev, then evaluate once on test."""
import json

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from lcc import ROOT, length_features, load, pair_features

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 500]
RESULTS = ROOT / "results"


# Each block maps (train rows, all rows) -> feature matrix for all rows, fit on train only.
def bow(rows, tr):
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2).fit([rows[i]["text"] for i in tr])
    X = vec.transform([r["text"] for r in rows])
    lengths = DictVectorizer(sparse=False).fit_transform([length_features(r) for r in rows])
    return hstack([X, csr_matrix(StandardScaler().fit(lengths[tr]).transform(lengths))])


def pairs(rows, tr):
    dv = DictVectorizer().fit([pair_features(rows[i]) for i in tr])
    X = dv.transform([pair_features(r) for r in rows])
    return MaxAbsScaler().fit(X[tr]).transform(X)


def weighted(block, weight):
    # Next to 1,536 standardized BERT dimensions, the sparse pair features get
    # regularized away at small C, so hybrids also tune a weight for this block.
    return lambda rows, tr: block(rows, tr) * weight


def dense(*keys):
    def block(rows, tr):
        X = np.hstack([EMB[k] for k in keys])
        return csr_matrix(StandardScaler().fit(X[tr]).transform(X))
    return block


MODELS = {
    # The original three models
    "BoW baseline": [bow],
    "BERT [CLS]": [dense("cls")],
    "Hybrid: [CLS] + BoW": [dense("cls"), bow],
    # Pair-aware models
    "Pair features (handcrafted)": [pairs],
    "BERT source+target spans": [dense("src", "tgt")],
    "Hybrid: spans + pair features": [dense("src", "tgt"), pairs],
}
# Alternative block lists tried for a model; the dev set picks among them along with C.
VARIANTS = {
    "Hybrid: spans + pair features": [[dense("src", "tgt"), weighted(pairs, w)] for w in (1, 3, 10, 30)],
}


def bootstrap_f1(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = [f1_score(y[s], pred[s]) for s in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(scores, [2.5, 97.5]).tolist()


def evaluate(Xs, y, tr, dev, te, class_weight):
    """Pick the (variant, C) with the best dev F1, then score it on test."""
    fits = {}
    for v, X in enumerate(Xs):
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000, class_weight=class_weight).fit(X[tr], y[tr])
            fits[v, C] = (f1_score(y[dev], clf.predict(X[dev])), clf)
    (v, C), (dev_f1, clf) = max(fits.items(), key=lambda kv: kv[1][0])
    pred = clf.predict(Xs[v][te])
    return {
        "variant": v, "C": C, "dev_f1": dev_f1,
        "test_acc": accuracy_score(y[te], pred),
        "test_p": precision_score(y[te], pred), "test_r": recall_score(y[te], pred),
        "test_f1": f1_score(y[te], pred), "test_f1_ci": bootstrap_f1(y[te], pred),
        "confusion": confusion_matrix(y[te], pred, normalize="all").round(3).tolist(),
    }


def main():
    rows = load()
    data = np.load(ROOT / "features" / "bert-base-uncased.npz")
    assert list(data["ids"]) == [r["id"] for r in rows]
    EMB.update({k: data[k] for k in ("cls", "src", "tgt")})
    y = np.array([r["label"] for r in rows])
    split = np.array([r["split"] for r in rows])
    tr, dev, te = (np.flatnonzero(split == s) for s in ("train", "dev", "test"))

    results = {}
    for name, blocks in MODELS.items():
        Xs = [hstack([b(rows, tr) for b in bl]).tocsr() for bl in VARIANTS.get(name, [blocks])]
        for cw in (None, "balanced"):
            key = f"{name} | {'balanced' if cw else 'unweighted'}"
            results[key] = evaluate(Xs, y, tr, dev, te, cw)
            r = results[key]
            print(f"{key:48s} v={r['variant']} C={r['C']:<7g} dev F1={r['dev_f1']:.3f}  test F1={r['test_f1']:.3f} "
                  f"[{r['test_f1_ci'][0]:.3f}, {r['test_f1_ci'][1]:.3f}]  acc={r['test_acc']:.3f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2))


EMB = {}

if __name__ == "__main__":
    main()
