"""Final, one-time evaluation of the finalists on the locked test splits.

English benchmark: en_small test (866 pairs; the original project's test split).
Other languages: es/ru/fa large test splits. Frozen-feature models are refit on
all non-test data with the hyperparameters chosen in CV/dev (read from
results/experiments.jsonl and results/exp5_multilingual.json); fine-tuned models
use the checkpoints chosen on dev by finetune.py. 95% bootstrap CIs on F1.
"""
import json

import numpy as np
import torch
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from cv import pool, pool_large
from lcc import LANGS, ROOT, load

LOG = ROOT / "results" / "experiments.jsonl"
OUT = ROOT / "results" / "final.json"


def bootstrap(y, pred, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = [f1_score(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(x) for x in np.percentile(s, [2.5, 97.5])]


def score(y, pred):
    return {"f1": f1_score(y, pred), "ci": bootstrap(y, pred), "p": precision_score(y, pred),
            "r": recall_score(y, pred), "acc": accuracy_score(y, pred), "n": int(len(y))}


def cv_params(name):
    """Hyperparameters chosen in CV for a logged experiment (latest run with that name)."""
    found = [json.loads(l) for l in LOG.open() if json.loads(l)["name"] == name]
    if not found:
        raise KeyError(name)
    return found[-1]["params"]


def frozen_english(name, block, clf, large=False):
    """Refit on all non-test training data, predict en_small test."""
    clf = clone(clf).set_params(**cv_params(name))
    if large:
        rows, y, fold, _ = pool_large()
        train = np.flatnonzero(fold != -1)
        test = np.flatnonzero([r["in_small"] and r["split"] == "test" for r in rows])
    else:
        rows, y, fold = pool()
        train, test = np.flatnonzero(fold >= 0), np.flatnonzero(fold == -1)
    X = block(rows, train)
    pred = clf.fit(X[train], y[train]).predict(X[test])
    return score(y[test], pred)


def finetuned(run):
    """Score a fine-tuned checkpoint on every language's test split."""
    from finetune import PairClassifier, predict, MODEL
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = PairClassifier().cuda()
    model.load_state_dict(torch.load(ROOT / "features" / "checkpoints" / run / "model.pt"))
    out = {}
    for lang in LANGS:
        rows = [r for r in load(lang, "small" if lang == "en" else "large") if r["split"] == "test"]
        y = np.array([r["label"] for r in rows])
        out[lang] = score(y, predict(model, tok, rows) >= 0.5)
    del model
    torch.cuda.empty_cache()
    return out


def frozen_multilingual():
    """exp5 setups (in-language / zero-shot from English / joint) on each test split."""
    from sklearn.linear_model import LogisticRegression
    from exp5_multilingual import build, featurize
    rows, emb, sig, dicts = build()
    chosen = json.loads((ROOT / "results" / "exp5_multilingual.json").read_text())["by_C"]
    y = np.array([r["label"] for r in rows])
    lang = np.array([r["lang"] for r in rows])
    split = np.array([r["split"] for r in rows])
    # English test = the en_small test pairs, as everywhere else
    is_test = np.array([s == "test" and (r["lang"] != "en" or r.get("in_small", False)) for r, s in zip(rows, split)])
    test = {l: np.flatnonzero((lang == l) & is_test) for l in LANGS}
    trainable = (split == "train") | (split == "dev")
    out = {}

    def fit(mask, C):
        idx = np.flatnonzero(mask)
        X = featurize(emb, sig, dicts, idx)
        return X, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[idx], y[idx])

    best_c = lambda key, l: float(max(chosen[key], key=lambda c: chosen[key][c][l]))
    for l in LANGS:
        X, clf = fit(trainable & (lang == l), best_c(f"in-language:{l}", l))
        out.setdefault("frozen in-language", {})[l] = score(y[test[l]], clf.predict(X[test[l]]))
    X, clf = fit(trainable & (lang == "en"), best_c("in-language:en", "en"))  # C from English dev only
    out["frozen zero-shot from English"] = {l: score(y[test[l]], clf.predict(X[test[l]])) for l in LANGS}
    for l in LANGS:
        X, clf = fit(trainable, best_c("joint", l))
        out.setdefault("frozen joint", {})[l] = score(y[test[l]], clf.predict(X[test[l]]))
    return out


def main(english_finalists):
    results = {"english": {}, "multilingual": {}}
    for name, (exp_name, block, clf, large) in english_finalists.items():
        results["english"][name] = frozen_english(exp_name, block, clf, large)
        r = results["english"][name]
        print(f"{name:52s} F1 {r['f1']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  P {r['p']:.3f} R {r['r']:.3f}", flush=True)
    for run in ("ft-en", "ft-all", "ft-all-seed1", "ft-all-seed2"):
        results["multilingual"][f"fine-tuned {run}"] = finetuned(run)
        print(run, {l: round(v["f1"], 3) for l, v in results["multilingual"][f"fine-tuned {run}"].items()}, flush=True)
    results["multilingual"].update(frozen_multilingual())
    OUT.write_text(json.dumps(results, indent=2))
    print("\nmultilingual test F1")
    for name, per in results["multilingual"].items():
        print(f"{name:34s} " + "  ".join(f"{l}={per[l]['f1']:.3f}" for l in LANGS))


if __name__ == "__main__":
    from exp1_features import LR as LR1, runs as exp1_runs
    from exp3_classifiers import SETS, classifiers
    from exp3b_svd import LEXICAL, MDEBERTA as MDEB_SMALL, THEORY, models as svd_models
    from exp4_more_data import RUNS as EXP4
    from features import combine

    rows, _, fold = pool()
    fit = np.flatnonzero(fold >= 0)
    n_dense = THEORY(rows, fit).shape[1] + MDEB_SMALL(rows, fit).shape[1]
    clf3 = lambda s, c: classifiers(*{"lexical-sparse": (True, 40000), "handcrafted-all": (False, 50000),
                                      "mdeberta-spans": (False, 2304)}[s])[c][0]
    # name -> (experiment name holding the CV-chosen params, feature block, classifier, train on en_large?)
    english = {
        "BoW + length, LR (original baseline, balanced)": (
            "exp1 | sentence BoW + length", exp1_runs["sentence BoW + length"], LR1, False),
        "lexical sparse, Naive Bayes": (
            "exp3 | lexical-sparse | NaiveBayes", SETS["lexical-sparse"][0], clf3("lexical-sparse", "NaiveBayes"), False),
        "handcrafted all, LR": (
            "exp3 | handcrafted-all | LogReg", SETS["handcrafted-all"][0], clf3("handcrafted-all", "LogReg"), False),
        "mDeBERTa spans, LR": (
            "exp3 | mdeberta-spans | LogReg", SETS["mdeberta-spans"][0], clf3("mdeberta-spans", "LogReg"), False),
        "mDeBERTa spans, RBF SVM": (
            "exp3 | mdeberta-spans | RBF-SVM", SETS["mdeberta-spans"][0], clf3("mdeberta-spans", "RBF-SVM"), False),
        "everything dense (LSA+theory+mDeBERTa), RBF SVM": (
            "exp3b | everything-dense | SVD+RBF-SVM", combine(LEXICAL, THEORY, MDEB_SMALL),
            svd_models(n_dense)["SVD+RBF-SVM"][0], False),
    }
    for name in EXP4:
        block, clf, _ = EXP4[name]
        english[f"[large-train] {name}"] = (f"exp4 | large-train | {name}", block, clf, True)
    main(english)
