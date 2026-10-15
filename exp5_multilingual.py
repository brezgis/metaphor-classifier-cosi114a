"""Experiment 5: does it work in other languages, and does English training transfer?

Language-agnostic features only (no English resources):
  mDeBERTa-v3 L7 source/target/isolated-source vectors, MIP/similarity cosines,
  target concept (shared label set across languages), word frequency, position.
Scenarios, each scored on every language's dev split (test stays locked):
  in-language   train on that language's train split
  zero-shot     train on English only; C chosen on English dev (no target labels used)
  joint         train on all four languages
"""
import json

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from wordfreq import zipf_frequency

from features import _cos, _layer, embeddings
from lcc import LANGS, ROOT, load

MODEL, LAYER = "microsoft/mdeberta-v3-base", 7
C_GRID = (0.0003, 0.001, 0.003)
OUT = ROOT / "results" / "exp5_multilingual.json"


def agnostic(r, lang):
    """Features computable for any language from the LCC markup alone."""
    (s0, s1), (t0, t1) = r["src_span"], r["tgt_span"]
    s_i, t_i = len(r["text"][:s0].split()), len(r["text"][:t0].split())
    return {f"concept={r['concept']}": 1,
            "src_zipf": zipf_frequency(r["src_text"], lang), "tgt_zipf": zipf_frequency(r["tgt_text"], lang),
            "distance": abs(s_i - t_i), "src_before_tgt": int(s0 < t0),
            "src_ntok": len(r["src_text"].split()), "tgt_ntok": len(r["tgt_text"].split())}


def build():
    rows, emb, sig, dicts = [], [], [], []
    for lang in LANGS:
        part = load(lang, "large")
        for r in part:
            r["lang"] = lang
        E = embeddings(MODEL, f"{lang}_large")
        src, tgt, src_iso = (_layer(E, k, LAYER) for k in ("src", "tgt", "src_iso"))
        emb.append(np.hstack([src, tgt, src_iso]))
        cols = []
        for L in E["layers"]:
            s, t, se = _layer(E, "src", L), _layer(E, "tgt", L), _layer(E, "sent", L)
            si, ti = _layer(E, "src_iso", L), _layer(E, "tgt_iso", L)
            cols += [_cos(s, si), _cos(t, ti), _cos(s, se), _cos(s, t), _cos(si, ti)]
        sig.append(np.column_stack(cols))
        dicts += [agnostic(r, lang) for r in part]
        rows += part
    return rows, np.vstack(emb), np.vstack(sig), dicts


def featurize(emb, sig, dicts, fit):
    dv = DictVectorizer().fit([dicts[i] for i in fit])
    D = dv.transform(dicts)
    D = MaxAbsScaler().fit(D[fit]).transform(D)
    dense = np.hstack([emb, sig])
    dense = StandardScaler().fit(dense[fit]).transform(dense)
    return hstack([csr_matrix(dense), D]).tocsr()


def main():
    rows, emb, sig, dicts = build()
    y = np.array([r["label"] for r in rows])
    lang = np.array([r["lang"] for r in rows])
    split = np.array([r["split"] for r in rows])
    dev = {l: np.flatnonzero((lang == l) & (split == "dev")) for l in LANGS}
    train_sets = {f"in-language:{l}": (lang == l) & (split == "train") for l in LANGS}
    train_sets["joint"] = split == "train"

    results = {}
    for name, mask in train_sets.items():
        fit = np.flatnonzero(mask)
        X = featurize(emb, sig, dicts, fit)
        per_c = {}
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(X[fit], y[fit])
            per_c[C] = {l: f1_score(y[dev[l]], clf.predict(X[dev[l]])) for l in LANGS}
            print(f"  {name:16s} C={C:<7g} " + "  ".join(f"{l}={f:.3f}" for l, f in per_c[C].items()), flush=True)
        results[name] = {str(C): v for C, v in per_c.items()}

    # Summaries: in-language picks C on its own dev; zero-shot uses the English model with
    # C picked on English dev; joint picks C per target language on that language's dev.
    summary = {}
    for l in LANGS:
        own = results[f"in-language:{l}"]
        en = results["in-language:en"]
        c_en = max(en, key=lambda c: en[c]["en"])
        summary[l] = {"in-language": max(v[l] for v in own.values()),
                      "zero-shot-from-en": en[c_en][l],
                      "joint": max(v[l] for v in results["joint"].values())}
    print("\nlanguage  in-language  zero-shot(en)  joint")
    for l, s in summary.items():
        print(f"{l:8s}  {s['in-language']:.3f}        {s['zero-shot-from-en']:.3f}          {s['joint']:.3f}")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"by_C": results, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
