"""Load the OFFICIAL allenai/scicite model archive (Cohan et al., NAACL 2019) with allennlp 0.8.0 on Python 3.11.

allennlp 0.8.0 targets Python 3.6 / torch 1.0 / spaCy 2.0. It is installed here with --no-deps on top of a
modern stack. The only changes are import-time shims for modules that no longer exist in modern libraries and
that the SciCite inference path never calls (coref metric, constituency-parser tag map, collections ABC aliases).
No allennlp or scicite source file is modified.
"""
import os as _os
from pathlib import Path as _Path
_CITATION_VENVS = _os.environ.get("CITATION_VENVS", _os.path.expanduser("~/.cache/citation_corpus/venvs"))
import collections
import collections.abc
import os
import random
import sys
import types

VENV = _CITATION_VENVS + "/scicite_local"
ART = os.path.join(VENV, "artefacts")
SCICITE_REPO = os.path.join(ART, "scicite")          # git clone, commit b33a8cd
MODEL_DIR = os.path.join(ART, "scicite_model")       # extracted scicite.tar.gz (sha256 4eeef0f0...)
OVERRIDES = '{"model":{"data_format":"","predict_mode":true}}'
N_THREADS = 4
SEED = 13

# ---- shims (import-time only) ----
for _n in ["Iterable", "Mapping", "MutableMapping", "Sequence", "Callable"]:
    setattr(collections, _n, getattr(collections.abc, _n))
_m = types.ModuleType("sklearn.utils.linear_assignment_")
_m.linear_assignment = None  # used only by allennlp's coref metric
sys.modules[_m.__name__] = _m
_t = types.ModuleType("spacy.lang.en.tag_map")
_t.TAG_MAP = {}  # used only by allennlp's constituency-parser predictor
sys.modules[_t.__name__] = _t
sys.path.insert(0, SCICITE_REPO)

import logging  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)  # silence allennlp INFO parameter dumps
import numpy as np  # noqa: E402
import torch  # noqa: E402


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(N_THREADS)
    torch.use_deterministic_algorithms(True)


def load_predictor():
    seed_all()
    from allennlp.models.archival import load_archive
    from allennlp.predictors.predictor import Predictor
    import scicite.models.scaffold_bilstm_attention_classifier  # noqa: F401  (registers model)
    import scicite.dataset_readers.citation_data_reader_scicite  # noqa: F401 (registers reader)
    import scicite.predictors.predictor_scicite  # noqa: F401  (registers predictor)
    arch = load_archive(MODEL_DIR, cuda_device=-1, overrides=OVERRIDES)
    pred = Predictor.from_archive(arch, "predictor_scicite")
    pred._model.eval()  # disables ELMo dropout (0.5) and GRU dropout (0.3)
    return pred


def classify(pred, text, citing="x", cited="y", idx=0):
    """One citance -> official predictor output (batch size 1, as `allennlp predict` defaults to).

    Returns (official_prediction, model_prediction, probs, passes_sentence_filter).
    official_prediction is '' when the official predictor's is_sentence() filter rejects the text; model_prediction
    is the model's argmax regardless of that filter (forward pass on the same instance).
    """
    from scicite.helper import is_sentence
    inp = {"string": text, "citingPaperId": citing, "citedPaperId": cited, "sectionName": None,
           "label": None, "excerpt_index": idx}
    passes = len(text) >= 5 and is_sentence(text)
    with torch.no_grad():
        if passes:
            out = pred.predict_json(inp)
        else:
            inst = pred._dataset_reader.text_to_instance(citation_text=text, intent=None, citing_paper_id=citing,
                                                         cited_paper_id=cited, citation_excerpt_index=idx)
            o = pred._model.forward_on_instance(inst)
            out = {"prediction": o["prediction"], "probabilities": o["probabilities"]}
    probs = [float(p) for p in out["probabilities"]]
    return (out["prediction"] if passes else ""), out["prediction"], probs, passes


LABELS = ["background", "method", "result"]  # vocabulary/labels.txt order
