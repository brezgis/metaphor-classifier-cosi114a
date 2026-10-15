"""Experiment 2: which frozen encoder, which layer, which vectors? (logistic regression)"""
import sys

from sklearn.linear_model import LogisticRegression

from cv import cv_eval
from features import emb_block, embeddings

LR = LogisticRegression(max_iter=5000, class_weight="balanced")
GRID = [{"C": c} for c in (0.0003, 0.001, 0.003, 0.01, 0.03)]
DS = "en_small"
MODELS = sys.argv[1:] or ["bert-base-uncased", "roberta-base", "xlm-roberta-base",
                          "bert-base-multilingual-cased", "microsoft/mdeberta-v3-base"]

if __name__ == "__main__":
    for model in MODELS:
        layers = list(embeddings(model, DS)["layers"])
        # Layer sweep with the basic source+target representation
        for L in [l for l in layers if l >= len(layers) // 3]:
            cv_eval(f"exp2 | {model} L{L} src,tgt", emb_block(model, DS, ("src", "tgt"), L), LR, GRID)
    print("--- vector combinations at a middle-upper layer")
    for model in MODELS:
        layers = list(embeddings(model, DS)["layers"])
        L = layers[int(len(layers) * 0.75)]
        for keys in [("src",), ("src", "tgt", "src_iso"), ("src", "tgt", "src_iso", "tgt_iso"),
                     ("src", "tgt", "sent"), ("src", "tgt", "cls")]:
            cv_eval(f"exp2 | {model} L{L} {','.join(keys)}", emb_block(model, DS, keys, L), LR, GRID)
