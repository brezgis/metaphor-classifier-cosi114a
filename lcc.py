"""Load the LCC Metaphor Datasets (Mohler et al. 2016).

Each LCC instance is a (source, target) word pair inside a sentence, scored
for metaphoricity 0-3 by human annotators (-1 = invalid syntactic relation).
Following the original project: drop instances whose mean human score is < 0,
label metaphor if mean score >= 2.

Benchmark: en_small, split 80/10/10 grouped by document. The large bundles
contain en_small, so en_large instances inherit en_small's split by document
and test sentences are kept out of training (see `load`).
"""
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
SEED = 0
LANGS = ("en", "es", "ru", "fa")


def parse(xml_path):
    rows = []
    for _, inst in ET.iterparse(xml_path):
        if inst.tag != "LmInstance":
            continue
        # Human annotations only; the large bundles also carry system scores
        # (AutomaticMetaphoricityAnnotation), which iter() does not match.
        annots = list(inst.iter("MetaphoricityAnnotation"))
        cur = inst.find("TextContent/Current")
        if not annots or cur is None:
            inst.clear()
            continue
        scores = [float(a.get("score")) for a in annots]
        mean = statistics.mean(scores)
        # Rebuild the sentence while recording character spans of the tagged words.
        text, spans = cur.text or "", {}
        for child in cur:
            start = len(text)
            text += "".join(child.itertext())
            spans[child.tag] = (start, len(text))
            text += child.tail or ""
        if mean >= 0 and "LmSource" in spans and "LmTarget" in spans:
            rows.append({
                "id": inst.get("id"),
                "docid": inst.get("docid"),
                "text": text,
                "src_span": spans["LmSource"],
                "tgt_span": spans["LmTarget"],
                "src_text": text[slice(*spans["LmSource"])],
                "tgt_text": text[slice(*spans["LmTarget"])],
                "concept": inst.get("targetConcept"),
                "chain": inst.get("chain"),
                "scores": scores,
                "annotators": [a.get("annotatorID") for a in annots],
                "label": int(mean >= 2),
            })
        inst.clear()
    return rows


def split(rows, seed=SEED):
    """80/10/10 split grouped by document, so no document spans two splits."""
    idx = np.arange(len(rows))
    groups = np.array([r["docid"] for r in rows])
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train, rest = next(outer.split(idx, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    dev, test = next(inner.split(rest, groups=groups[rest]))
    names = np.empty(len(rows), dtype=object)
    names[train], names[rest[dev]], names[rest[test]] = "train", "dev", "test"
    for r, name in zip(rows, names):
        r["split"] = name
    return rows


def _inherit_small_split(large, small):
    """en_large: reuse en_small's split by document; keep test sentences out of training."""
    doc_split = {r["docid"]: r["split"] for r in small}
    small_ids = {r["id"] for r in small}
    test_texts = {r["text"] for r in small if r["split"] == "test"}
    for r in large:
        r["in_small"] = r["id"] in small_ids
        r["split"] = doc_split.get(r["docid"], "train")
        if r["split"] == "train" and r["text"] in test_texts:
            r["split"] = "excluded"  # same sentence as a test item, in another document
    return large


def load(lang="en", size="small"):
    path = PROCESSED / f"{lang}_{size}.jsonl"
    if not path.exists():
        rows = parse(DATA / f"{lang}_{size}.xml")
        if (lang, size) == ("en", "large"):
            rows = _inherit_small_split(rows, load("en", "small"))
        else:
            rows = split(rows)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return [json.loads(line) for line in path.open()]


if __name__ == "__main__":
    for lang, size in [("en", "small"), ("en", "large"), ("es", "large"), ("ru", "large"), ("fa", "large")]:
        rows = load(lang, size)
        print(f"{lang}_{size}: {len(rows)} instances, {len({r['docid'] for r in rows})} documents")
        for s in ("train", "dev", "test", "excluded"):
            part = [r for r in rows if r["split"] == s]
            if part:
                print(f"  {s:8s} n={len(part):6d}  metaphor rate={np.mean([r['label'] for r in part]):.3f}")
