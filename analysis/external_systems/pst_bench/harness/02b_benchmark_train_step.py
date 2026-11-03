"""Micro-benchmark: seconds per training sample for the official SciBERT recipe (seq 512) on this machine.
Used only to estimate the cost of a faithful re-training; does not produce a model."""
import time, sys, torch
from transformers import BertForSequenceClassification
dev = sys.argv[1]; bs = int(sys.argv[2]); steps = int(sys.argv[3])
m = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=2).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=5e-5)
x = torch.randint(100, 30000, (bs, 512), device=dev); y = torch.zeros(bs, dtype=torch.long, device=dev)
ts = []
for i in range(steps + 1):
    t = time.time(); out = m(x, labels=y); out.loss.backward(); opt.step(); opt.zero_grad()
    if dev == "mps": torch.mps.synchronize()
    ts.append(time.time() - t)
per = sum(ts[1:]) / steps / bs
print(f"device={dev} batch={bs} steps={steps} sec/sample={per:.3f} -> est. min/epoch(7645 train)={per*7645/60:.1f}")
