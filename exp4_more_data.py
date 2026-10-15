"""Experiment 4: same models, trained on en_large (~47k pairs per fold instead of ~6.3k).

Validation is still the en_small CV folds, so numbers compare directly with exp1-3.
"""
import sys

from sklearn.linear_model import LogisticRegression

from cv import cv_eval
from exp3_classifiers import BalancedMLP
from features import HANDCRAFTED, char_ngrams, combine, ctx_feats, dict_block, emb_block, signal_block

LR = LogisticRegression(max_iter=5000, class_weight="balanced")
C_SPARSE = [{"C": c} for c in (0.1, 0.3, 1, 3)]
C_DENSE = [{"C": c} for c in (0.0003, 0.001, 0.003, 0.01)]
MLP_GRID = [{"hidden_layer_sizes": h, "alpha": a} for h in ((512,), (512, 128)) for a in (1e-3, 1e-1)]

MDEBERTA = emb_block("microsoft/mdeberta-v3-base", "en_large", ("src", "tgt", "src_iso"), 7)
BERT = emb_block("bert-base-uncased", "en_large", ("src", "tgt", "src_iso"), 5)
HAND = combine(HANDCRAFTED, dict_block(ctx_feats), char_ngrams, signal_block("bert-base-uncased", "en_large"))

RUNS = {
    "handcrafted-all": (HAND, LR, C_SPARSE),
    "bert L5 spans+iso": (BERT, LR, C_DENSE),
    "mdeberta L7 spans+iso": (MDEBERTA, LR, C_DENSE),
    "mdeberta + handcrafted": (combine(MDEBERTA, HAND), LR, C_DENSE + C_SPARSE[:2]),
    "mdeberta L7 spans+iso | MLP": (MDEBERTA, BalancedMLP(early_stopping=True, max_iter=200, random_state=0), MLP_GRID),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or RUNS:
        block, clf, grid = RUNS[name]
        cv_eval(f"exp4 | large-train | {name}", block, clf, grid, large=True)
