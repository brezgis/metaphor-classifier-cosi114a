"""Feature blocks for LCC source/target pairs.

A block is a function (rows, fit_idx) -> matrix for all rows; anything learned
(vocabularies, scalers) is fit on rows[fit_idx] only.

Linguistic motivation for the handcrafted groups:
  lex    source/target lemma, POS, target concept, LCC dependency chain
  dep    the source word's syntactic neighbourhood (head, relation, children)
  ctx    words in a +-3 window around source and target
  char   character n-grams of the source and target words
  conc   concreteness (Brysbaert et al. 2014): metaphors typically use a concrete
         source in an abstract context (Turney et al. 2011; Tsvetkov et al. 2014)
  wn     WordNet supersenses and polysemy (Tsvetkov et al. 2014)
  freq   word frequency (wordfreq Zipf scale)
  lm     masked-LM surprisal of source/target: selectional preference
         violation (Wilks 1978; the SPV half of MelBERT)
  mip    cosine between a word in context and the same word alone: metaphor
         identification procedure, MIP (Pragglejaz 2007; MelBERT)
  sim    semantic distance between source and target, and source vs. sentence
"""
import json
from functools import lru_cache

import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from wordfreq import zipf_frequency

from lcc import PROCESSED, ROOT

WN_POS = {"NOUN": wn.NOUN, "PROPN": wn.NOUN, "VERB": wn.VERB, "AUX": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}


# ---------------------------------------------------------------- spaCy layer

def _span_tokens(doc, span):
    start, end = span
    return [t for t in doc if t.idx < end and t.idx + len(t) > start]


def _describe(doc, span):
    toks = _span_tokens(doc, span)
    # The syntactic head of a multiword span is the token whose head lies outside it.
    head = next((t for t in toks if t.head not in toks or t.head == t), toks[0]) if toks else None
    if head is None:
        return {"lemma": "", "lemmas": [], "pos": "NONE", "dep": "NONE", "head_lemma": "", "head_pos": "NONE",
                "children": [], "stop": False, "ntok": 0, "i": 0, "start": 0, "end": 0}
    return {
        "lemma": head.lemma_.lower(), "lemmas": [t.lemma_.lower() for t in toks],
        "pos": head.pos_, "dep": head.dep_,
        "head_lemma": head.head.lemma_.lower(), "head_pos": head.head.pos_,
        "children": [f"{c.dep_}:{c.lemma_.lower()}" for c in head.children if c not in toks],
        "stop": head.is_stop, "ntok": len(toks), "i": head.i, "start": toks[0].i, "end": toks[-1].i,
    }


def add_spacy(rows, name):
    """Attach spaCy analyses (English only), cached next to the processed data."""
    path = PROCESSED / f"{name}.spacy.jsonl"
    if not path.exists():
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        tmp = path.with_suffix(".tmp")  # write-then-rename, so readers never see a partial cache
        with tmp.open("w") as f:
            for r, doc in zip(rows, nlp.pipe((r["text"] for r in rows), batch_size=512)):
                f.write(json.dumps({
                    "src": _describe(doc, r["src_span"]), "tgt": _describe(doc, r["tgt_span"]),
                    "toks": [t.lemma_.lower() for t in doc],
                    "content": [t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV")],
                }) + "\n")
        tmp.rename(path)
    for r, line in zip(rows, path.open()):
        r["sp"] = json.loads(line)
    return rows


# ------------------------------------------------------- lexical resources

@lru_cache(None)
def concreteness():
    table = {}
    for line in (ROOT / "resources" / "brysbaert_concreteness.txt").open(encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if parts[0] != "Word":
            table[parts[0].lower()] = float(parts[2])
    return table


def conc_of(words):
    vals = [concreteness()[w] for w in words if w in concreteness()]
    return np.mean(vals) if vals else np.nan


@lru_cache(None)
def supersenses(lemma, pos):
    synsets = wn.synsets(lemma, pos=WN_POS.get(pos)) if lemma else []
    return len(synsets), (synsets[0].lexname() if synsets else f"none.{pos}")


# ------------------------------------------------------------- dict blocks

def lex_feats(r):
    s, t = r["sp"]["src"], r["sp"]["tgt"]
    return {
        f"src_lemma={s['lemma']}": 1, f"tgt_lemma={t['lemma']}": 1,
        f"src_pos={s['pos']}": 1, f"tgt_pos={t['pos']}": 1, f"pos_pair={s['pos']}|{t['pos']}": 1,
        f"concept={r['concept']}": 1, f"chain={r['chain']}": 1,
        "src_stop": int(s["stop"]), "src_ntok": s["ntok"],
        "distance": abs(s["i"] - t["i"]), "src_before_tgt": int(s["i"] < t["i"]),
    }


def dep_feats(r):
    s = r["sp"]["src"]
    f = {f"src_dep={s['dep']}": 1, f"src_head={s['head_lemma']}": 1, f"src_head_pos={s['head_pos']}": 1,
         f"src_dep_head={s['dep']}|{s['head_lemma']}": 1}
    for c in s["children"]:
        f[f"src_child={c}"] = 1
    return f


def ctx_feats(r, k=3):
    toks, f = r["sp"]["toks"], {}
    for role in ("src", "tgt"):
        d = r["sp"][role]
        for off in range(1, k + 1):
            if d["start"] - off >= 0:
                f[f"{role}_L{off}={toks[d['start'] - off]}"] = 1
            if d["end"] + off < len(toks):
                f[f"{role}_R{off}={toks[d['end'] + off]}"] = 1
    return f


def conc_feats(r):
    s, t = r["sp"]["src"], r["sp"]["tgt"]
    cs, ct = conc_of(s["lemmas"] or [s["lemma"]]), conc_of(t["lemmas"] or [t["lemma"]])
    ctx = conc_of([w for w in r["sp"]["content"] if w not in s["lemmas"]])
    f = {"src_conc_missing": int(np.isnan(cs)), "tgt_conc_missing": int(np.isnan(ct))}
    # Missing values become the scale midpoint (3 of 1-5) plus the indicator above.
    cs, ct, ctx = (3.0 if np.isnan(v) else v for v in (cs, ct, ctx))
    f.update({"src_conc": cs, "tgt_conc": ct, "ctx_conc": ctx,
              "src_minus_tgt_conc": cs - ct, "src_minus_ctx_conc": cs - ctx,
              "concrete_src_abstract_tgt": int(cs >= 4 and ct < 3)})
    return f


def wn_feats(r):
    s, t = r["sp"]["src"], r["sp"]["tgt"]
    ns, ss = supersenses(s["lemma"], s["pos"])
    nt, st = supersenses(t["lemma"], t["pos"])
    return {f"src_ss={ss}": 1, f"tgt_ss={st}": 1, f"ss_pair={ss}|{st}": 1,
            "src_nsenses": np.log1p(ns), "tgt_nsenses": np.log1p(nt)}


def freq_feats(r, lang="en"):
    return {"src_zipf": zipf_frequency(r["src_text"], lang), "tgt_zipf": zipf_frequency(r["tgt_text"], lang)}


def length_feats(r):
    words = r["text"].split()
    return {"n_words": len(words), "mean_word_len": float(np.mean([len(w) for w in words]))}


def dict_block(*fns):
    def block(rows, fit):
        dicts = [{k: v for fn in fns for k, v in fn(r).items()} for r in rows]
        dv = DictVectorizer().fit([dicts[i] for i in fit])
        X = dv.transform(dicts)
        return MaxAbsScaler().fit(X[fit]).transform(X)
    block.__name__ = "+".join(fn.__name__.replace("_feats", "") for fn in fns)
    return block


# -------------------------------------------------------------- text blocks

def sent_bow(rows, fit):
    vec = CountVectorizer(ngram_range=(1, 2), min_df=2).fit([rows[i]["text"] for i in fit])
    return vec.transform([r["text"] for r in rows])


def char_ngrams(rows, fit):
    docs = [f"{r['src_text'].lower()}" for r in rows], [f"{r['tgt_text'].lower()}" for r in rows]
    mats = []
    for d in docs:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        vec.fit([d[i] for i in fit])
        mats.append(vec.transform(d))
    return hstack(mats).tocsr()


# ------------------------------------------------------- transformer blocks

@lru_cache(None)
def embeddings(model, dataset):
    d = np.load(ROOT / "features" / f"{model.replace('/', '_')}.{dataset}.npz")
    return {k: d[k] for k in d.files}


def _layer(E, key, layer):
    i = list(E["layers"]).index(layer)
    X = E[key][:, i].astype(np.float32)
    bad = np.isnan(X[:, 0])
    if bad.any():  # span lost to truncation: fall back to the sentence mean
        X[bad] = E["sent"][bad, i].astype(np.float32)
    return X


def _cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def emb_block(model, dataset, keys=("src", "tgt"), layer=-1):
    """Standardized concatenation of embedding vectors (layer -1 = last kept)."""
    def block(rows, fit):
        E = embeddings(model, dataset)
        L = E["layers"][layer] if layer < 0 else layer
        X = np.hstack([_layer(E, k, L) for k in keys])
        return csr_matrix(StandardScaler().fit(X[fit]).transform(X))
    block.__name__ = f"emb[{model}:{','.join(keys)}@{layer}]"
    return block


def signal_block(model, dataset, layers=None):
    """Low-dimensional, language-agnostic signals: surprisal (lm), MIP and similarity cosines."""
    def block(rows, fit):
        E = embeddings(model, dataset)
        cols = []
        for L in (layers or E["layers"][1:]):
            src, tgt, sent = _layer(E, "src", L), _layer(E, "tgt", L), _layer(E, "sent", L)
            src_iso, tgt_iso = _layer(E, "src_iso", L), _layer(E, "tgt_iso", L)
            cols += [_cos(src, src_iso), _cos(tgt, tgt_iso),        # MIP: contextual vs. basic meaning
                     _cos(src, sent), _cos(src, tgt),               # SPV-style fit to context / target
                     _cos(src_iso, tgt_iso)]                        # out-of-context domain distance
        if "src_lp" in E:
            lp = np.nan_to_num(np.stack([E["src_lp"], E["tgt_lp"]], 1), nan=-10.0)
            cols += [lp[:, 0], lp[:, 1], lp[:, 0] - lp[:, 1]]
        X = np.column_stack(cols)
        return csr_matrix(StandardScaler().fit(X[fit]).transform(X))
    block.__name__ = f"signals[{model}]"
    return block


def combine(*blocks, weights=None):
    weights = weights or [1] * len(blocks)

    def block(rows, fit):
        return hstack([b(rows, fit) * w for b, w in zip(blocks, weights)]).tocsr()
    block.__name__ = " + ".join(f"{b.__name__}" + (f"*{w}" if w != 1 else "") for b, w in zip(blocks, weights))
    return block


HANDCRAFTED = dict_block(lex_feats, dep_feats, conc_feats, wn_feats, freq_feats)
