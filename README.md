# Metaphor detection on the LCC datasets

A COSI 114A (Brandeis) course project: classifying source/target word pairs in
the LCC Metaphor Datasets as metaphorical or literal, in English, Spanish, Russian
and Farsi. The write-up is in `paper/`. It works up from a sentence-level baseline
through handcrafted pair features and frozen encoders to a fine-tuned multilingual
encoder, asking how far classical models get, whether nonlinear models help, and
whether it works across languages.

**Headline (English, locked test set, F1):** sentence-level baseline 0.588 → handcrafted
pair features 0.717 → frozen mDeBERTa + MLP 0.812 → fine-tuned mDeBERTa 0.827.
The fine-tuned multilingual model also gets 0.74 (Spanish), 0.86 (Russian)
and 0.68 (Farsi).

The fine-tuned four-language model is on Hugging Face:
[brezgis/lcc-metaphor-mdeberta](https://huggingface.co/brezgis/lcc-metaphor-mdeberta).

## Data

LCC Metaphor Datasets (Mohler et al., LREC 2016; https://github.com/lcc-api/metaphor,
CC BY-NC-SA 4.0). The data isn't redistributed here; see Setup below. Each instance is a **source/target word pair inside a
sentence** (`the <LmSource>War</LmSource> on <LmTarget>Poverty</LmTarget>`) with
human metaphoricity scores 0-3. Drop pairs with mean
score < 0 (-1 = invalid relation), label metaphor if mean >= 2. Only human scores
are used; the large bundles also contain system scores.

| | pairs | metaphor rate | split |
|---|---|---|---|
| en_small (benchmark) | 8,724 | 35% | 7,016 / 842 / 866, grouped by document |
| en_large | 52,118 | 39% | contains en_small; inherits its split by document |
| es / ru / fa large | 42,165 / 22,978 / 26,685 | 26% / 38% / 33% | 80/10/10 by document |

en_small is a strict subset of en_large, so large-English training excludes
every test document and any test sentence repeated elsewhere (`lcc.py`).

**Human ceiling.** On pairs with two annotators, one annotator scored against the
other gets F1 0.66 (en), 0.60 (es), 0.82 (ru), 0.39 (fa); Cohen's κ 0.51 / 0.48
/ 0.69 / **0.12**. The Farsi labels are close to noise. The models beat
annotator-vs-annotator F1 because they learn the average annotator.

## Protocol

- The test splits were locked. All model selection used 5-fold CV over en_small
  train+dev, grouped by document (`cv.py`), or the dev splits (multilingual,
  fine-tuning). `final.py` scored every finalist on test exactly once and reports
  all of them, not only the best.
  (Exception: the sentence-level baselines, below, were scored on en_small test
  before the lock.)
- Everything learned from data (vocabularies, scalers, SVD) is fit inside each fold.
- Every CV run is logged to `results/experiments.jsonl`; final numbers are in
  `results/final.json`.

## Why sentence-level models fall short

A sentence-level model (bag-of-words, or BERT [CLS]) sees the whole sentence, but
LCC labels one word pair within it, so the model has no way to know which words
are being judged. Some sentences contain several pairs with opposite labels.
Without class weighting such models also lean literal, and frozen [CLS] is a weak
sentence vector. BoW / BERT [CLS] / the two combined score 0.496 / 0.445 / 0.513
test F1; see `legacy/` and `results/results.json`.

## What was learned

**1. Feature groups (exp1; LR, CV F1).** Which word is the source carries most of
the signal. Sentence context *hurts*.

| features | CV F1 |
|---|---|
| sentence BoW + length | 0.538 |
| frequency (2 feats) / concreteness (8) | 0.502 / 0.525 |
| dependency neighbourhood / ±3 window | 0.553 / 0.562 |
| LM signals: surprisal, MIP and similarity cosines (63) | 0.577 |
| WordNet supersenses + polysemy | 0.578 |
| char n-grams of source/target | 0.677 |
| lexical: lemmas, POS, target concept, LCC chain | 0.692 |
| all handcrafted + window + chars + LM signals | **0.713** |
| ... + sentence BoW | 0.695 |

The theory-driven features behave as the literature predicts: very concrete
sources are metaphors 50% of the time vs. 21% for very abstract ones, and
metaphorical sources are much less predictable from context (mean masked-LM
log-prob −7.2 vs. −5.2). But they're small and mostly overlap with lemma identity.

**2. Frozen encoders (exp2).** Middle layers beat the last layer (BERT: 0.751 at
L5 vs. 0.723 at L12). Adding the source word encoded *on its own* (the MIP
contrast) helps every model by 0.01–0.02; adding [CLS] or the sentence mean hurts.
Best: **mDeBERTa-v3-base L7, 0.767** (source + target + isolated source: 0.765–0.772),
ahead of BERT/RoBERTa (0.75), XLM-R base/large (0.744/0.755) and mBERT (0.73).

**3. Linear separability (exp3, exp3b).** Same features, different classifiers (CV F1):

| features (dims) | train acc, linear SVM | LR | lin. SVM | RBF SVM | MLP | boosting | RF | NB |
|---|---|---|---|---|---|---|---|---|
| lexical sparse (40k) | 0.999 | 0.672 | 0.675 | 0.646 | 0.688 | – | 0.632 | 0.684 |
| all handcrafted (49k) | 1.000 | 0.713 | 0.714 | 0.691 | – | – | 0.655 | – |
| theory only (73) | 0.726 | 0.636 | 0.637 | 0.636 | 0.636 | **0.653** | 0.592 | 0.553 |
| mDeBERTa spans (2.3k) | 1.000 | 0.772 | 0.759 | **0.782** | 0.770 | 0.755 | 0.670 | 0.698 |
| LSA(lexical) + theory + mDeBERTa | – | 0.773 | – | **0.784** | 0.772 | 0.759 | – | – |

The high-dimensional representations are linearly separable on training data,
and nonlinear models gain little (+0.01 for RBF on embeddings) or lose. Only the
small 73-feature theory space isn't separable, and there gradient boosting is the
one model that finds extra structure. LSA *hurts* the lexical features (0.647 vs.
0.672; SVD drops the rare-word identities), and in the combined matrix it adds
nothing beyond the embeddings. Random forests are the worst throughout. Naive
Bayes is a strong cheap baseline. Conclusion: the representation matters far
more than the decision boundary.

**4. More data (exp4).** Training on en_large (~47k pairs per fold vs. ~6.3k) is
the biggest lever for frozen models: handcrafted 0.713 → 0.779, mDeBERTa LR
0.772 → 0.788, + handcrafted 0.798, **MLP 0.770 → 0.804**. Nonlinearity only
pays off with data.

**5. Fine-tuning (finetune.py).** mDeBERTa-v3 with the pair marked in the text
(`<< source >>`, `[[ target ]]`), class-weighted, 3 epochs, epoch picked on dev.
Three seeds agree within ±0.003.

## Final test results

English, en_small test (n = 866), 95% bootstrap CIs:

| model | train data | test F1 | 95% CI |
|---|---|---|---|
| sentence-level [CLS] + BoW | small | 0.513 | 0.463–0.559 |
| sentence BoW + length, balanced | small | 0.588 | 0.544–0.631 |
| lexical sparse, Naive Bayes | small | 0.692 | 0.651–0.732 |
| all handcrafted, LR | small | 0.717 | 0.674–0.756 |
| mDeBERTa spans, LR | small | 0.772 | 0.734–0.809 |
| mDeBERTa spans, RBF SVM | small | 0.775 | 0.737–0.810 |
| LSA + theory + mDeBERTa, RBF SVM | small | 0.777 | 0.740–0.811 |
| all handcrafted, LR | large | 0.776 | 0.738–0.810 |
| BERT L5 spans, LR | large | 0.771 | 0.733–0.806 |
| mDeBERTa spans, LR | large | 0.809 | 0.775–0.840 |
| mDeBERTa + handcrafted, LR | large | 0.806 | 0.771–0.839 |
| **mDeBERTa spans, MLP** (best frozen) | large | **0.812** | 0.778–0.844 |
| fine-tuned mDeBERTa, English only | large | 0.833 | 0.800–0.862 |
| **fine-tuned mDeBERTa, all 4 languages** (3 seeds) | large ×4 | **0.827 ± 0.002** | ≈0.79–0.86 |

The best frozen model's CI overlaps the fine-tuned model's: with 866 test pairs,
frozen + MLP and fine-tuning are not clearly separable on English.

All languages, test F1 (fine-tuned joint: mean of 3 seeds):

| model | en | es | ru | fa |
|---|---|---|---|---|
| frozen mDeBERTa + LR, in-language | 0.801 | 0.708 | 0.833 | 0.670 |
| frozen, zero-shot from English | 0.801 | 0.663 | 0.746 | 0.649 |
| frozen, joint (4 languages) | 0.797 | 0.701 | 0.815 | 0.676 |
| fine-tuned, English only (zero-shot elsewhere) | 0.833 | 0.669 | 0.733 | 0.642 |
| **fine-tuned, joint (4 languages)** | **0.827** | **0.737** | **0.857** | **0.682** |
| *human: annotator vs. annotator* | *0.66* | *0.60* | *0.82* | *0.39* |

- Zero-shot transfer works: a model that never saw Spanish/Russian/Farsi labels
  keeps 90–97% of in-language performance.
- With frozen features, joint training doesn't help the non-English languages.
  When fine-tuning, it does: a trainable encoder can absorb the differences
  between the four annotation teams.
- Per-language scores track annotation quality (Russian has the highest κ and
  the best scores).

## Setup

Python 3.10, see `requirements.txt` (tested with torch 2.12, transformers 4.52,
scikit-learn 1.7). Then:

```
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('wordnet')"
```

**Data.** Download the LCC bundles from https://github.com/lcc-api/metaphor and
put the extracted XML files (`en_small.xml`, `en_large.xml`, `es_large.xml`,
`ru_large.xml`, `fa_large.xml`, ...) in `data/`.

**Concreteness norms.** `features.py` reads Brysbaert, Warriner & Kuperman (2014)
from `resources/brysbaert_concreteness.txt` (the tab-separated supplementary file,
with the header row `Word  Bigram  Conc.M  Conc.SD ...`).

## Reproduce

```
python3 lcc.py                                     # parse + split all bundles -> data/processed/
python3 embed.py MODEL LANG SIZE [LAYERS]          # frozen features -> features/
python3 exp1_features.py                           # feature groups
python3 exp2_embeddings.py [MODEL ...]             # encoders x layers x vectors
python3 exp3_classifiers.py [SET ...]              # classifier families + separability
python3 exp3b_svd.py                               # LSA densification
python3 exp4_more_data.py                          # train on en_large
python3 exp5_multilingual.py                       # in-language / zero-shot / joint
SEED=0 python3 finetune.py RUN LANGS [EPOCHS]      # fine-tuning reference (GPU, ~12-30 min)
python3 final.py                                   # one-time test evaluation
```

Embeddings used: bert-base-uncased, roberta-base, xlm-roberta-base/large,
bert-base-multilingual-cased, mdeberta-v3-base (en_small, all layers), plus
bert-base-uncased (en_large) and mdeberta-v3-base (all four large bundles) at a
few layers. Cached features and fine-tuned checkpoints go in `features/` (~17 GB in
total; make it a symlink to a big disk if you need to).

## Caveats and next steps

- One split. CV standard deviations are ~0.01, and test CIs are ±0.035 on English,
  so differences under ~0.02 shouldn't be over-read.
- The MLP and RBF SVM weren't tried on large-English training with handcrafted
  features (too slow on 145k sparse columns with ~4 CPU workers).
- The Farsi labels (κ 0.12) limit what any Farsi number means.
- Ideas: a concreteness regressor on multilingual embeddings for language-agnostic
  concreteness (Ljubešić et al. 2018); label smoothing with the raw 0–3 scores;
  scoring against the per-annotator labels to compare with the human ceiling directly.

## License

Code: MIT (see `LICENSE`). The LCC data is CC BY-NC-SA 4.0 and is not included;
if you use it, cite Mohler et al. (2016), *Introducing the LCC Metaphor Datasets*,
LREC 2016.
