"""Experiment 3b: densify the sparse lexical features with TruncatedSVD (LSA) for nonlinear models.

TruncatedSVD runs inside each CV fold's pipeline, so the projection never sees validation data.
"everything-dense" = SVD(lexical) + theory features + mDeBERTa spans, as one dense matrix.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cv import cv_eval, pool
from exp3_classifiers import SETS, BalancedMLP
from features import combine

LEXICAL, THEORY, MDEBERTA = (SETS[k][0] for k in ("lexical-sparse", "theory-dense", "mdeberta-spans"))


class LexicalSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD of all but the last n_dense columns; those (already dense) are passed through.

    The sparse lexical block comes first and its width varies by fold (fold-local
    vocabularies), while the dense blocks after it have a fixed width.
    """
    def __init__(self, n_components=300, n_dense=0, random_state=0):
        self.n_components, self.n_dense, self.random_state = n_components, n_dense, random_state

    def _split(self, X):
        cut = X.shape[1] - self.n_dense
        return X[:, :cut], X[:, cut:]

    def fit(self, X, y=None):
        self.svd_ = TruncatedSVD(self.n_components, random_state=self.random_state).fit(self._split(X)[0])
        return self

    def transform(self, X):
        sparse, dense = self._split(X)
        return np.hstack([self.svd_.transform(sparse), dense.toarray()])


def models(n_dense):
    svd = LexicalSVD(n_dense=n_dense)
    return {
        "SVD+LogReg": (make_pipeline(svd, StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")),
                       [{"lexicalsvd__n_components": k, "logisticregression__C": c}
                        for k in (100, 300, 1000) for c in (0.001, 0.01, 0.1)]),
        "SVD+MLP": (make_pipeline(svd, StandardScaler(), BalancedMLP(early_stopping=True, max_iter=300, random_state=0)),
                    [{"lexicalsvd__n_components": k, "balancedmlp__hidden_layer_sizes": h, "balancedmlp__alpha": a}
                     for k in (100, 300, 1000) for h in ((256,), (512, 128)) for a in (1e-2, 1)]),
        "SVD+RBF-SVM": (make_pipeline(svd, StandardScaler(), SVC(class_weight="balanced")),
                        [{"lexicalsvd__n_components": k, "svc__C": c} for k in (100, 300) for c in (1, 3)]),
        "SVD+GradBoost": (make_pipeline(svd, HistGradientBoostingClassifier(
                              class_weight="balanced", max_iter=500, early_stopping=True, random_state=0)),
                          [{"lexicalsvd__n_components": k, "histgradientboostingclassifier__learning_rate": lr}
                           for k in (100, 300) for lr in (0.05, 0.1)]),
    }


if __name__ == "__main__":
    rows, _, fold = pool()
    fit = np.flatnonzero(fold >= 0)
    n_dense = THEORY(rows, fit).shape[1] + MDEBERTA(rows, fit).shape[1]
    for set_name, block, d in [("lexical-only", LEXICAL, 0),
                               ("everything-dense", combine(LEXICAL, THEORY, MDEBERTA), n_dense)]:
        for clf_name, (clf, grid) in models(d).items():
            cv_eval(f"exp3b | {set_name} | {clf_name}", block, clf, grid)
