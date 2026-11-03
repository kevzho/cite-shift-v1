#!/usr/bin/env python3
"""D2 (exploratory) continued pretraining of SciBERT: JEPA vs masked-LM, same backbone and token budget.

Spec: private/preregistration/JEPA_EXPLORATORY_SPEC_2026-10-07.md (fixed before any run).

  --objective jepa   context encoder (SciBERT) -> [CLS] embedding -> 2-layer MLP predictor -> predicts the
                     TARGET encoder's [CLS] embedding of the cited title. Target encoder = EMA of the context
                     encoder (momentum cosine 0.996 -> 1.0), no gradient. Loss = smooth-L1 between
                     L2-normalised predicted and target embeddings. No token reconstruction. [CLS] pooling
                     because the downstream multi-head model (analysis/ml/10_train_distilled.py) reads [CLS].
                     Collapse is monitored (per-dimension std of normalised embeddings across the batch);
                     an optional variance/covariance regulariser (--vc-weight, default 0) exists only for use
                     if collapse is observed — any use is a logged deviation from the spec.
  --objective mlm    standard masked-LM continued pretraining on the context text (15% masking, 80/10/10).

Same for both arms: backbone allenai/scibert_scivocab_uncased, context max_len 256, steps x batch (token
budget), AdamW (lr, wd 0.01), linear warmup then linear decay, seed. Mixed precision on CUDA (bf16 if
supported, else fp16); fp32 on CPU/MPS. Resume-safe: `ckpt/last.pt` holds model, optimiser, scheduler,
EMA target, step and RNG state; re-running the same command resumes. Output backbone (save_pretrained)
in <run>/final/ — load it with `10_train_distilled.py --init <run>/final`.

Usage:
    python pretrain.py --objective jepa --seed 1 --steps 6000 --batch 64
    python pretrain.py --objective mlm  --seed 1 --steps 6000 --batch 64
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup, logging as hf_logging)

hf_logging.set_verbosity_error()
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent  # on a pod: /workspace/d2
BACKBONE = "allenai/scibert_scivocab_uncased"
DATA = ROOT / "analysis/results/ml/d2_jepa/data"
RUNS = ROOT / "analysis/results/ml/d2_jepa/runs"


def read_jsonl(p: Path, limit: int | None = None) -> list[dict]:
    out = []
    with open(p, encoding="utf-8") as h:
        for line in h:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Jepa(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = AutoModel.from_pretrained(BACKBONE)
        d = self.context.config.hidden_size
        self.predictor = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)

    @staticmethod
    def cls(enc, ids, mask):
        return enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, cids, cmask, tids, tmask):
        ctx_raw = self.cls(self.context, cids, cmask)
        pred = F.normalize(self.predictor(ctx_raw).float(), dim=-1)
        with torch.no_grad():
            self.target.eval()
            tgt = F.normalize(self.cls(self.target, tids, tmask).float(), dim=-1)
        ctx = F.normalize(ctx_raw.float(), dim=-1)   # context-encoder embedding (regularised; see vc_reg)
        return pred, tgt, ctx

    @torch.no_grad()
    def ema(self, m: float):
        for pc, pt in zip(self.context.parameters(), self.target.parameters()):
            pt.data.mul_(m).add_(pc.data, alpha=1 - m)


def vc_reg(z: torch.Tensor) -> torch.Tensor:
    """VICReg-style variance + covariance terms on the context encoder's embedding (only if --vc-weight > 0).

    z is L2-normalised; scaling by sqrt(d) makes an evenly spread unit-sphere embedding have per-dimension
    std ~ 1, which is the variance hinge's target (spec change log, 8 Oct 2026)."""
    z = z * math.sqrt(z.shape[1])
    std = torch.sqrt(z.var(0) + 1e-4)
    # Variance hinge only. The VICReg covariance term is omitted: with 16-64 examples in 768 dimensions its
    # sample-noise floor (~d/n) penalises healthy, evenly spread embeddings more than collapsed ones.
    return F.relu(1 - std).mean()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--objective", choices=["jepa", "mlm"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="gradient accumulation: forward in chunks of this size; 0 = whole batch at once")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06, help="fraction of steps")
    ap.add_argument("--max-len", type=int, default=256, help="context max length (token budget)")
    ap.add_argument("--target-max-len", type=int, default=64, help="JEPA target (cited title) max length")
    ap.add_argument("--ema-start", type=float, default=0.996)
    ap.add_argument("--vc-weight", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use only the first N pairs")
    ap.add_argument("--run-dir", type=Path, default=None)
    a = ap.parse_args()

    dev = pick_device(a.device)
    run = a.run_dir or RUNS / f"{a.objective}_seed{a.seed}"
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    if (run / "final").exists():
        raise SystemExit(f"{run}/final exists - this run is finished; refusing to overwrite")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if dev == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    tok = AutoTokenizer.from_pretrained(BACKBONE)
    train = read_jsonl(a.data / "train.jsonl", a.limit)
    val = read_jsonl(a.data / "val.jsonl", None if a.limit is None else max(16, a.limit // 10))

    if a.objective == "jepa":
        model = Jepa().to(dev)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        model = AutoModelForMaskedLM.from_pretrained(BACKBONE).to(dev)
        params = list(model.parameters())
        collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * a.steps), a.steps)
    amp_dtype = None
    if dev == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    step = 0
    order: list[int] = []
    rng = random.Random(a.seed)
    last = run / "ckpt" / "last.pt"
    if last.exists():
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); step = st["step"]; order = st["order"]
        rng.setstate(st["rng_py"]); torch.set_rng_state(st["rng_torch"].cpu())  # RNG state must be a CPU ByteTensor
        if dev == "cuda" and st.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        print(f"resumed at step {step}", flush=True)

    def next_batch():
        nonlocal order
        if len(order) < a.batch:
            fresh = list(range(len(train)))
            rng.shuffle(fresh)
            order = order + fresh
        idx, order = order[:a.batch], order[a.batch:]
        return [train[i] for i in idx]

    def enc(texts, n):
        e = tok(texts, truncation=True, max_length=n, padding=True, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    def loss_on(batch, train_mode=True):
        stats = {}
        if a.objective == "jepa":
            cids, cm = enc([b["context"] for b in batch], a.max_len)
            tids, tm = enc([b["target"] for b in batch], a.target_max_len)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred, tgt, ctx = model(cids, cm, tids, tm)
            loss = F.smooth_l1_loss(pred, tgt)
            if a.vc_weight > 0:
                loss = loss + a.vc_weight * vc_reg(ctx)
            stats["ctx_std"] = float(ctx.detach().std(0).mean())
            stats["pred_std"] = float(pred.detach().std(0).mean())
            stats["target_std"] = float(tgt.std(0).mean())
            stats["cos"] = float((pred.detach() * tgt).sum(-1).mean())
        else:
            e = tok([b["context"] for b in batch], truncation=True, max_length=a.max_len)
            col = collator([{"input_ids": x} for x in e["input_ids"]])
            ids = col["input_ids"].to(dev)
            labels = col["labels"].to(dev)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                # Vocabulary logits only at masked positions: the MLM loss is defined only there, so
                # this equals model(..., labels=...).loss but avoids a batch x len x vocab tensor.
                hidden = model.bert(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long())[0]
                sel = labels != -100
                logits = model.cls(hidden[sel])
            loss = F.cross_entropy(logits.float(), labels[sel])
        return loss, stats

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot, n = 0.0, 0
        eb = a.micro_batch or a.batch
        for i in range(0, len(val), eb):
            loss, _ = loss_on(val[i:i + eb], train_mode=False)
            tot += float(loss) * len(val[i:i + eb]); n += len(val[i:i + eb])
        model.train()
        return tot / max(1, n)

    def save_ckpt():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "order": order, "rng_py": rng.getstate(),
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if dev == "cuda" else None},
                   run / "ckpt" / "last.tmp")
        os.replace(run / "ckpt" / "last.tmp", last)

    log = open(run / "log.jsonl", "a", encoding="utf-8")
    meta = {"objective": a.objective, "seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr,
            "max_len": a.max_len, "target_max_len": a.target_max_len, "ema_start": a.ema_start,
            "vc_weight": a.vc_weight, "micro_batch": a.micro_batch or a.batch, "device": dev, "amp": str(amp_dtype), "backbone": BACKBONE,
            "n_train": len(train), "n_val": len(val), "limit": a.limit,
            "token_budget_context": a.steps * a.batch * a.max_len, "torch": torch.__version__}
    (run / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    model.train()
    t0 = time.time()
    micro = a.micro_batch or a.batch
    while step < a.steps:
        batch = next_batch()
        opt.zero_grad(set_to_none=True)
        # Gradient accumulation: the optimizer step still sees the full batch (a.batch); chunks of
        # `micro` are forwarded separately, each loss weighted by its share of the batch. Exact for
        # the JEPA smooth-L1 mean; for MLM it weights by examples rather than masked tokens (noted
        # in config.json). With --micro-batch unset this is the original single-pass update.
        loss_sum, stats = 0.0, {}
        for i in range(0, len(batch), micro):
            chunk = batch[i:i + micro]
            loss_c, stats = loss_on(chunk)
            scaler.scale(loss_c * (len(chunk) / len(batch))).backward()
            loss_sum += float(loss_c) * len(chunk) / len(batch)
        loss = torch.tensor(loss_sum)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        if a.objective == "jepa":
            m = 1.0 - (1.0 - a.ema_start) * (math.cos(math.pi * step / a.steps) + 1) / 2
            model.ema(m)
            stats["ema_m"] = round(m, 6)
        if step % a.log_every == 0 or step == 1 or step == a.steps:
            rec = {"step": step, "loss": round(float(loss), 6), "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1), **{k: round(v, 6) for k, v in stats.items()}}
            if step % a.eval_every == 0 or step == a.steps:
                rec["val_loss"] = round(evaluate(), 6)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
        if step % a.ckpt_every == 0 or step == a.steps:
            save_ckpt()

    final = run / "final"
    final.mkdir()
    backbone = model.context if a.objective == "jepa" else model
    backbone.save_pretrained(final)
    tok.save_pretrained(final)
    meta["wall_s"] = round(time.time() - t0, 1)
    meta["finished_step"] = step
    (final / "pretrain_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved backbone to {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
