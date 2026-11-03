"""Step 2: fine-tune SciBERT with the OFFICIAL bert.train() from THUDM/paper-source-trace (commit d0896c6).
Deviations (all logged, none change the model recipe):
  * device: the official code picks cuda-else-cpu; here torch.cuda.is_available is patched so that
    torch.device("cuda") maps to Apple MPS. Hyper-parameters (lr 5e-5, bs 16, max_len 512, 20 epochs,
    patience 5, class-weighted CE, warmup 0.1) are untouched.
  * optional env PST_MAX_EPOCHS caps NUM_TRAIN_EPOCHS if set (default: official 20).
  * seeds fixed at 42.
"""
import sys, os, random, re, types
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper-source-trace")
sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch
np.random.seed(42); random.seed(42); torch.manual_seed(42)

if torch.backends.mps.is_available():
    _orig_device = torch.device
    torch.cuda.is_available = lambda: True
    class _Dev:  # map "cuda" -> "mps"
        def __new__(cls, spec, *a, **k):
            if isinstance(spec, str) and spec.startswith("cuda"):
                spec = "mps"
            return _orig_device(spec, *a, **k)
    import bert
    bert.torch = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    bert.torch.device = _Dev
else:
    import bert

max_ep = os.environ.get("PST_MAX_EPOCHS")
if max_ep:
    import inspect, textwrap
    src = inspect.getsource(bert.train).replace("NUM_TRAIN_EPOCHS = 20", f"NUM_TRAIN_EPOCHS = {int(max_ep)}")
    exec(compile(textwrap.dedent(src), bert.__file__, "exec"), bert.__dict__)
bert.train(model_name="scibert")
