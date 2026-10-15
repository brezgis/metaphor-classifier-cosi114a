"""Experiment 1: which feature groups carry signal? (logistic regression throughout)"""
from sklearn.linear_model import LogisticRegression

from cv import cv_eval
from features import (HANDCRAFTED, char_ngrams, combine, conc_feats, ctx_feats, dep_feats, dict_block,
                      emb_block, freq_feats, length_feats, lex_feats, sent_bow, signal_block, wn_feats)

LR = LogisticRegression(max_iter=5000, class_weight="balanced")
GRID = [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10)]
BERT = ("bert-base-uncased", "en_small")

runs = {
    # the original project's baseline
    "sentence BoW + length": combine(sent_bow, dict_block(length_feats)),
    # handcrafted groups alone
    "lex": dict_block(lex_feats),
    "dep": dict_block(dep_feats),
    "ctx window": dict_block(ctx_feats),
    "char n-grams": char_ngrams,
    "concreteness": dict_block(conc_feats),
    "wordnet": dict_block(wn_feats),
    "frequency": dict_block(freq_feats),
    "LM signals (surprisal, MIP, sim)": signal_block(*BERT),
    # cumulative
    "handcrafted (lex+dep+conc+wn+freq)": HANDCRAFTED,
    "handcrafted + ctx": combine(HANDCRAFTED, dict_block(ctx_feats)),
    "handcrafted + ctx + char": combine(HANDCRAFTED, dict_block(ctx_feats), char_ngrams),
    "handcrafted + ctx + char + signals": combine(HANDCRAFTED, dict_block(ctx_feats), char_ngrams, signal_block(*BERT)),
    "handcrafted + ctx + char + signals + sent BoW": combine(HANDCRAFTED, dict_block(ctx_feats), char_ngrams,
                                                            signal_block(*BERT), sent_bow),
    # frozen BERT spans, for reference
    "BERT spans (last layer)": emb_block(*BERT),
}

if __name__ == "__main__":
    for name, block in runs.items():
        cv_eval(f"exp1 | {name}", block, LR, GRID)
