"""Reference model: fine-tune mDeBERTa-v3 with the source/target pair marked in the text.

    python finetune.py RUN_NAME TRAIN_LANGS [EPOCHS]      e.g.  python finetune.py en en   |   python finetune.py all en,es,ru,fa

The source and target are wrapped in plain-text markers ("<< >>" and "[[ ]]"), and the
classifier reads the mean of the marked source and target tokens plus [CLS]. Trains on
the large bundles' train splits; picks the epoch by dev F1 (English: the en_small dev
split, so the number is comparable to `final.py`), and saves that checkpoint. Test
data is never read here.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from lcc import ROOT, load

MODEL = "microsoft/mdeberta-v3-base"
CKPTS = ROOT / "features" / "checkpoints"  # on the data drive, via the features symlink
LR, BATCH, MAX_LEN = 2e-5, 32, 160
SEED = int(os.environ.get("SEED", 0))  # vary to check seed sensitivity


def mark(r):
    """Insert markers; return marked text and the new character spans of source and target."""
    text = r["text"]
    inserts = [(r["src_span"][0], "<< "), (r["src_span"][1], " >>"), (r["tgt_span"][0], "[[ "), (r["tgt_span"][1], " ]]")]
    out, prev, pos = "", 0, {}
    # At a shared position (adjacent spans, e.g. "Gun|-lock") the closer goes before the opener.
    for at, m in sorted(inserts, key=lambda p: (p[0], p[1] in ("<< ", "[[ "))):
        out += text[prev:at]
        pos[m] = len(out) + (len(m) if m in ("<< ", "[[ ") else 0)  # span starts after an opener
        out += m
        prev = at
    out += text[prev:]
    return out, (pos["<< "], pos[" >>"]), (pos["[[ "], pos[" ]]"])


class PairClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, 2))

    def forward(self, input_ids, attention_mask, src_mask, tgt_mask):
        hid = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pool = lambda m: (hid * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([hid[:, 0], pool(src_mask), pool(tgt_mask)], -1))


def encode(tok, rows):
    marked = [mark(r) for r in rows]
    enc = tok([m[0] for m in marked], padding=True, truncation=True, max_length=MAX_LEN,
              return_offsets_mapping=True, return_token_type_ids=False, return_tensors="pt")
    off = enc.pop("offset_mapping")
    span = lambda s: (off[..., 0] < s[1]) & (off[..., 1] > s[0]) & (off[..., 1] > off[..., 0])
    enc["src_mask"] = torch.stack([span(torch.tensor(m[1]))[i] for i, m in enumerate(marked)]).float()
    enc["tgt_mask"] = torch.stack([span(torch.tensor(m[2]))[i] for i, m in enumerate(marked)]).float()
    return enc


@torch.no_grad()
def predict(model, tok, rows):
    model.eval()
    probs = []
    for i in range(0, len(rows), 128):
        enc = {k: v.cuda() for k, v in encode(tok, rows[i:i + 128]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(model(**enc).float().softmax(-1)[:, 1].cpu())
    model.train()
    return torch.cat(probs).numpy()


def main(run, langs, epochs):
    torch.manual_seed(SEED)
    ckpt = CKPTS / run
    rng = np.random.default_rng(SEED)
    train = [r for l in langs for r in load(l, "large") if r["split"] == "train"]
    dev = {"en": [r for r in load("en", "small") if r["split"] == "dev"]}
    dev.update({l: [r for r in load(l, "large") if r["split"] == "dev"] for l in ("es", "ru", "fa")})
    y = np.array([r["label"] for r in train])
    weight = torch.tensor(len(y) / (2 * np.bincount(y)), dtype=torch.float).cuda()  # balanced classes

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = PairClassifier().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * (len(train) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    primary = langs[0] if len(langs) == 1 else "en"
    best, history = -1, []
    for epoch in range(epochs):
        order = rng.permutation(len(train))
        for b, i in enumerate(range(0, len(order) - BATCH + 1, BATCH)):
            batch = [train[j] for j in order[i:i + BATCH]]
            enc = {k: v.cuda() for k, v in encode(tok, batch).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(**enc).float(), torch.tensor([r["label"] for r in batch]).cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            if b % 300 == 0:
                print(f"  epoch {epoch} step {b} loss {loss.item():.3f}", flush=True)
        scores = {l: f1_score([r["label"] for r in rows], predict(model, tok, rows) >= 0.5) for l, rows in dev.items()}
        history.append(scores)
        print(f"epoch {epoch}: dev F1 " + "  ".join(f"{l}={s:.3f}" for l, s in scores.items()), flush=True)
        if scores[primary] > best:
            best = scores[primary]
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt / "model.pt")
    (ckpt / "history.json").write_text(json.dumps({"seed": SEED, "train_langs": langs, "primary": primary, "history": history}))
    print(f"best dev F1 ({primary}) = {best:.3f}; checkpoint in {ckpt}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]) if len(sys.argv) > 3 else 3)
