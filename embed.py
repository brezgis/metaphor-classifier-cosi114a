"""Extract frozen transformer features for LCC instances.

    python embed.py MODEL LANG SIZE [LAYERS]     LAYERS: all | every2 | 6,8,12 (default all)

For each kept layer, saves (float16):
  cls, sent      [CLS] and mean over the sentence's tokens
  src, tgt       mean over the source / target span, in context
  src_iso, tgt_iso  the source / target words encoded on their own
                 (decontextualized; contrast with src/tgt is the MIP signal
                 used by MelBERT, Choi et al. 2021)
If the model has a masked-LM head, also saves src_lp / tgt_lp: mean log
probability of the span's sub-tokens when the span is masked out
(a selectional-preference-violation signal: metaphors use words in contexts
where they are unexpected).
"""
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from lcc import ROOT, load

MODEL, LANG, SIZE = sys.argv[1], sys.argv[2], sys.argv[3]
LAYER_SPEC = sys.argv[4] if len(sys.argv) > 4 else "all"
OUT = ROOT / "features" / f"{MODEL.replace('/', '_')}.{LANG}_{SIZE}.npz"
DEVICE = "cuda"


def span_mask(offsets, span):
    start, end = span
    # Sub-word tokens overlapping the span (special tokens have empty offsets).
    return (offsets[:, 0] < end) & (offsets[:, 1] > start) & (offsets[:, 1] > offsets[:, 0])


def masked_mean(hidden, mask):
    """hidden: (layers, seq, dim); mask: (seq,) bool -> (layers, dim), NaN if empty."""
    if not mask.any():
        return torch.full((hidden.shape[0], hidden.shape[2]), float("nan"))
    return hidden[:, mask].mean(1)


@torch.no_grad()
def main(batch_size=64, max_length=256):
    rows = load(LANG, SIZE)
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        model, info = AutoModelForMaskedLM.from_pretrained(MODEL, output_loading_info=True)
        # Some checkpoints (e.g. mDeBERTa-v3) ship without MLM head weights; a randomly
        # initialized head would give meaningless surprisal.
        has_mlm = not info["missing_keys"]
        if not has_mlm:
            model = AutoModel.from_pretrained(MODEL)
    except (ValueError, OSError):
        model, has_mlm = AutoModel.from_pretrained(MODEL), False
    model = model.to(DEVICE).eval()
    n_layers = model.config.num_hidden_layers + 1
    layers = {"all": list(range(n_layers)), "every2": list(range(0, n_layers, 2))}.get(
        LAYER_SPEC) or [int(x) for x in LAYER_SPEC.split(",")]

    def run(enc):
        out = model(**{k: v.to(DEVICE) for k, v in enc.items()}, output_hidden_states=True)
        hidden = torch.stack([out.hidden_states[l] for l in layers], 1).float().cpu()  # (B, L, seq, D)
        return hidden, (out.logits if has_mlm else None)

    feats = {k: [] for k in ("cls", "sent", "src", "tgt", "src_iso", "tgt_iso")}
    lps = {"src_lp": [], "tgt_lp": []}
    order = np.argsort([len(r["text"]) for r in rows])  # length-sorted batches pad less
    for i in range(0, len(rows), batch_size):
        batch = [rows[j] for j in order[i:i + batch_size]]
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            enc = tok([r["text"] for r in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping")
            hidden, _ = run(enc)
            masks = {k: [span_mask(offsets[j], r[f"{k}_span"]) for j, r in enumerate(batch)] for k in ("src", "tgt")}
            for j in range(len(batch)):
                h = hidden[j]
                content = (offsets[j][:, 1] > offsets[j][:, 0])
                feats["cls"].append(h[:, 0])
                feats["sent"].append(h[:, content].mean(1))
                for k in ("src", "tgt"):
                    feats[k].append(masked_mean(h, masks[k][j]))

            for k in ("src", "tgt"):
                # The word on its own, outside any sentence.
                enc_iso = tok([r[f"{k}_text"] for r in batch], return_tensors="pt", padding=True,
                              return_offsets_mapping=True)
                off_iso = enc_iso.pop("offset_mapping")
                h_iso, _ = run(enc_iso)
                for j in range(len(batch)):
                    feats[f"{k}_iso"].append(h_iso[j][:, off_iso[j][:, 1] > off_iso[j][:, 0]].mean(1))

                if has_mlm:
                    # Mask the whole span and score how expected the original sub-tokens are.
                    ids = enc["input_ids"].clone()
                    m = torch.stack(masks[k])
                    ids[m] = tok.mask_token_id
                    # Small sub-batches, and log-softmax only at masked positions: with
                    # 250k-token multilingual vocabularies the full logit tensor is huge.
                    for s in range(0, len(batch), 8):
                        logits = model(input_ids=ids[s:s + 8].to(DEVICE),
                                       attention_mask=enc["attention_mask"][s:s + 8].to(DEVICE)).logits
                        for j in range(s, min(s + 8, len(batch))):
                            if not m[j].any():
                                lps[f"{k}_lp"].append(float("nan"))
                                continue
                            gold = enc["input_ids"][j][m[j]].to(DEVICE)
                            logp = logits[j - s][m[j].to(DEVICE)].float().log_softmax(-1)
                            lps[f"{k}_lp"].append(logp.gather(-1, gold[:, None]).mean().item())
                        del logits
        if i % (batch_size * 50) == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    inverse = np.argsort(order)  # back to dataset order
    arrays = {k: torch.stack(v).numpy().astype(np.float16)[inverse] for k, v in feats.items()}
    if has_mlm:
        arrays.update({k: np.array(v, dtype=np.float32)[inverse] for k, v in lps.items()})
    missing = int(np.isnan(arrays["src"][:, 0, 0]).sum() + np.isnan(arrays["tgt"][:, 0, 0]).sum())
    np.savez(OUT, ids=[r["id"] for r in rows], layers=layers, **arrays)
    print(f"wrote {OUT}: {len(rows)} rows, layers {layers}, mlm={has_mlm}, {missing} spans lost to truncation")


if __name__ == "__main__":
    main()
