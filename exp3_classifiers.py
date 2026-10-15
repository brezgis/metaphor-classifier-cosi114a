"""Experiment 3: classifier families on fixed feature sets, and a linear-separability probe.

    python exp3_classifiers.py SETNAME [SETNAME ...]

If nonlinear models (RBF SVM, random forest, boosting, MLP) don't beat a linear
model on the same features, the classes are about as linearly separable as the
label noise allows; the gain has to come from the representation instead.
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from cv import cv_eval, pool
from features import (HANDCRAFTED, char_ngrams, combine, conc_feats, ctx_feats, dict_block, emb_block, freq_feats,
                      lex_feats, signal_block)


class BalancedMLP(MLPClassifier):
    """MLPClassifier has no class_weight; weight samples so classes count equally."""
    def fit(self, X, y):
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


def to_dense(X):
    return X.toarray()


def dense(clf):
    return make_pipeline(FunctionTransformer(to_dense, accept_sparse=True), clf)


BERT = ("bert-base-uncased", "en_small")
SETS = {
    # Non-negative sparse indicators/tf-idf, so multinomial-style NB applies
    "lexical-sparse": (combine(dict_block(lex_feats), dict_block(ctx_feats), char_ngrams), True),
    "handcrafted-all": (combine(HANDCRAFTED, dict_block(ctx_feats), char_ngrams, signal_block(*BERT)), False),
    # ~70 dense theory-driven features: concreteness, frequency, surprisal, MIP/SPV cosines
    "theory-dense": (combine(dict_block(conc_feats, freq_feats), signal_block(*BERT)), False),
    # best frozen representation from exp2
    "mdeberta-spans": (emb_block("microsoft/mdeberta-v3-base", "en_small", ("src", "tgt", "src_iso"), 7), False),
}

C_GRID = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)]


def classifiers(nonneg, n_features):
    small = n_features < 5000
    out = {
        "LogReg": (LogisticRegression(max_iter=5000, class_weight="balanced"), C_GRID),
        "LinearSVM": (LinearSVC(class_weight="balanced", max_iter=20000), C_GRID),
        "RBF-SVM": (SVC(kernel="rbf", class_weight="balanced"),
                    [{"C": c, "gamma": "scale"} for c in (0.3, 1, 3, 10)]),
        "RandomForest": (RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=1,
                                                random_state=0),
                         [{"max_features": m, "min_samples_leaf": l} for m in ("sqrt", 0.1) for l in (1, 3)]),
    }
    if small:  # on ~40k raw sparse columns an MLP takes ~30 min per set; exp3b covers it via LSA
        out["MLP"] = (BalancedMLP(early_stopping=True, max_iter=300, random_state=0),
                      [{"hidden_layer_sizes": h, "alpha": a} for h in ((256,), (512, 128)) for a in (1e-4, 1e-2, 1)])
    if nonneg:
        out["NaiveBayes"] = (ComplementNB(), [{"alpha": a} for a in (0.03, 0.1, 0.3, 1)])
    if small:
        out["NaiveBayes(Gaussian)"] = (dense(GaussianNB()), [{"gaussiannb__var_smoothing": v} for v in (1e-9, 1e-6, 1e-3)])
        out["GradBoost"] = (dense(HistGradientBoostingClassifier(class_weight="balanced", max_iter=500,
                                                                 early_stopping=True, random_state=0)),
                            [{"histgradientboostingclassifier__learning_rate": lr,
                              "histgradientboostingclassifier__max_leaf_nodes": n}
                             for lr in (0.03, 0.1) for n in (15, 63)])
    return out


def separability(name, block):
    """Training-set fit of a near-hard-margin linear SVM on the whole CV pool."""
    rows, y, fold = pool()
    fit = np.flatnonzero(fold >= 0)
    X = block(rows, fit)[fit]
    for C in (1, 100):
        svm = LinearSVC(C=C, class_weight="balanced", max_iter=50000).fit(X, y[fit])
        print(f"  separability {name:16s} linear SVM C={C:<4} train acc = {np.mean(svm.predict(X) == y[fit]):.3f}",
              flush=True)


if __name__ == "__main__":
    for set_name in sys.argv[1:] or SETS:
        block, nonneg = SETS[set_name]
        rows, _, fold = pool()
        n_features = block(rows, np.flatnonzero(fold >= 0)).shape[1]
        separability(set_name, block)
        for clf_name, (clf, grid) in classifiers(nonneg, n_features).items():
            cv_eval(f"exp3 | {set_name} | {clf_name}", block, clf, grid)
