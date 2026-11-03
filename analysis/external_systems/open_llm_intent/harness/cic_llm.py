"""Local runner for the released fine-tuned citation-intent LLMs of Koloveas et al. (TPDL 2025).

Paper : Koloveas, Chatzopoulos, Vergoulis, Tryfonopoulos. "Can LLMs Predict Citation Intent? An Experimental
        Analysis of In-context Learning and Fine-tuning on Open LLMs". TPDL 2025 (LNCS, doi 10.1007/978-3-032-05409-8_13),
        arXiv:2502.14561.
Code  : https://github.com/athenarc/CitationIntentOpenLLM (GPL-2.0)
Model : sknow-lab/Qwen2.5-14B-CIC-SciCite (Apache-2.0), GGUF Q4_K_M quant by mradermacher.

Prompt = the paper's fine-tuned *zero-shot, SP3, QT2 (qa-multiple-choice)* evaluation configuration, reproduced
verbatim from citation_intent_classification_experiments.py::get_predictions and experimental-configs/system_prompts.json
(key 'scicite3' / 'acl-arc3'), rendered with the chat template embedded in the GGUF (Qwen2.5 ChatML) -- the same
path LM Studio used in the paper. Decoding: greedy argmax over the raw logits, max 15 new tokens (paper: max_tokens=15,
temperature 0.0). Label cleanup = the paper's clean_labels() logic.

Everything runs in-process on this machine (llama.cpp via llama-cpp-python, Metal). No network calls.
"""
import ctypes
import json
import os
import re
import string
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import markers  # noqa: E402  vendored copy of analysis/external_systems/aclarc_multicite/markers.py

CACHE = os.path.expanduser("~/.cache/citation_corpus/venvs/open_llm_intent")
REPO = f"{CACHE}/artefacts/CitationIntentOpenLLM"
MODELS = {
    "scicite": f"{CACHE}/models/Qwen2.5-14B-CIC-SciCite.Q4_K_M.gguf",
    "acl-arc": f"{CACHE}/models/Qwen2.5-14B-CIC-ACLARC.Q4_K_M.gguf",
}
CLASS_LABELS = {  # experimental-configs/experiments_cfg.json
    "scicite": ["background information", "method", "results comparison"],
    "acl-arc": ["BACKGROUND", "MOTIVATION", "USES", "EXTENDS", "COMPARES_CONTRASTS", "FUTURE"],
}
SEED = 42
N_CTX = 2048
MAX_NEW = 15
TAG = "@@CITATION@@"


def system_prompt(dataset):
    sp = json.load(open(f"{REPO}/experimental-configs/system_prompts.json"))
    return sp[f"{dataset}3"]


def form_multiple_choice_prompt(class_labels):  # verbatim from the paper's code
    return '\n'.join([f"{letter}) {label}" for letter, label in zip(string.ascii_lowercase, class_labels)])


def user_prompt(sentence, dataset):  # template_qa in get_predictions (note the space before \n, verbatim)
    template_qa = "{sentence}\n### Question: Which is the most likely intent for this citation? \n{options}\n### Answer:"
    return template_qa.format(sentence=sentence, options=form_multiple_choice_prompt(CLASS_LABELS[dataset]))


def clean_label(text, labels):  # paper's clean_labels().check_and_replace, on the lower().strip()-ed completion
    letter_to_label = {letter: label for letter, label in zip(string.ascii_lowercase, labels)}
    stripped_text = text.strip().lower().rstrip(')')
    if stripped_text in letter_to_label:
        return letter_to_label[stripped_text]
    count_labels = {label: text.lower().count(label.lower()) for label in labels}
    if sum(count_labels.values()) == 1:
        return next(label for label, count in count_labels.items() if count == 1)
    return None


# ---------------------------------------------------------------- target-citation marking
def _sub_number(seg, n):
    m = re.search(rf"(?<!\d){n}(?!\d)", seg)
    return None if m is None else seg[:m.start()] + TAG + seg[m.end():]


def tag_edge(citances, corpus):
    """Replace the target citation marker with @@CITATION@@ (the paper's input convention: SciCite/ACL-ARC
    inputs have the target citation span replaced by the tag). Target resolution reuses the deterministic rule of
    markers.mark_edge (intersection / single_group / ambiguous / no_marker). Returns (tagged_list, resolution_list)."""
    per = [markers.find_groups(c, corpus) for c in citances]
    target = None
    if len(citances) >= 2:
        sets = [set().union(*[g[2] for g in gs]) if gs else set() for gs in per]
        inter = set.intersection(*sets) if sets else set()
        if len(inter) == 1:
            target = next(iter(inter))
        elif len(inter) > 1:
            target = inter
    out, res = [], []
    for c, gs in zip(citances, per):
        if not gs:
            out.append(c); res.append("no_marker"); continue
        if isinstance(target, int):
            hit = [g for g in gs if target in g[2]]
            if hit:
                s, e, nums = hit[0]
                if len(nums) == 1:
                    out.append(c[:s] + TAG + c[e:])
                else:  # tag only the target number inside a multi-number group: "[8,46]" -> "[@@CITATION@@,46]"
                    sub = _sub_number(c[s:e], target)
                    out.append(c[:s] + (sub if sub is not None else TAG) + c[e:])
                res.append("intersection"); continue
        cand = gs
        if isinstance(target, set):
            cand = [g for g in gs if g[2] & target] or gs
        t = c
        for s, e, _ in sorted(cand, reverse=True):  # whole group(s) -> tag
            t = t[:s] + TAG + t[e:]
        out.append(t)
        res.append("single_group" if len(gs) == 1 else "ambiguous")  # same naming as markers.mark_edge
    return out, res


# ---------------------------------------------------------------- model
class CICModel:
    def __init__(self, dataset, n_threads=4, verbose=False):
        import llama_cpp
        self.lc = llama_cpp
        self.dataset = dataset
        self.labels = CLASS_LABELS[dataset]
        self.path = MODELS[dataset]
        self.llm = llama_cpp.Llama(model_path=self.path, n_ctx=N_CTX, n_batch=512, n_ubatch=512,
                                   n_gpu_layers=-1, seed=SEED, n_threads=n_threads, n_threads_batch=n_threads,
                                   logits_all=False, verbose=verbose)
        self.n_vocab = self.llm.n_vocab()
        self.tmpl = self.llm.metadata.get("tokenizer.chat_template")
        self.sys = system_prompt(dataset)
        # render the chat template once with a sentinel so that the system block is an exact token prefix
        full = self.render("\x00SENTINEL\x00")
        self.prefix_text = full[: full.index("\x00SENTINEL\x00")]
        self.prefix_tokens = self.llm.tokenize(self.prefix_text.encode(), add_bos=False, special=True)
        self.eos = {self.llm.token_eos()}
        for t in ("<|im_end|>", "<|endoftext|>"):
            ids = self.llm.tokenize(t.encode(), add_bos=False, special=True)
            if len(ids) == 1:
                self.eos.add(ids[0])
        # first token of each label (for a first-step probability readout; label strings start the reply)
        self.label_first_tok = {l: self.llm.tokenize(l.encode(), add_bos=False, special=False)[0] for l in self.labels}
        self._prefix_ready = False

    def render(self, user_text):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        msgs = [{"role": "system", "content": self.sys}, {"role": "user", "content": user_text}]
        return env.from_string(self.tmpl).render(messages=msgs, add_generation_prompt=True,
                                                 bos_token="", eos_token="")

    def _logits(self):
        ptr = self.lc.llama_get_logits_ith(self.llm._ctx.ctx, -1)
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).astype(np.float64).copy()

    def _eval_prefix(self):
        # the system block is always evaluated as its own batch, so every item sees an identical KV prefix
        self.llm.reset()
        self.llm._ctx.kv_cache_clear() if hasattr(self.llm._ctx, "kv_cache_clear") else None
        self.llm.eval(self.prefix_tokens)
        self._prefix_ready = True

    def classify(self, sentence):
        u = user_prompt(sentence, self.dataset)
        full = self.render(u)
        assert full.startswith(self.prefix_text)
        suffix_tokens = self.llm.tokenize(full[len(self.prefix_text):].encode(), add_bos=False, special=True)
        if not self._prefix_ready:
            self._eval_prefix()
        self.llm.n_tokens = len(self.prefix_tokens)  # rewind to the cached system prefix (eval() drops later KV)
        self.llm.eval(suffix_tokens)
        gen, first_probs = [], None
        for step in range(MAX_NEW):
            lg = self._logits()
            if step == 0:
                p = np.exp(lg - lg.max()); p /= p.sum()
                first_probs = {l: float(p[t]) for l, t in self.label_first_tok.items()}
            tok = int(np.argmax(lg))
            if tok in self.eos:
                break
            gen.append(tok)
            self.llm.eval([tok])
        raw = self.llm.detokenize(gen).decode("utf-8", errors="replace")
        pred = raw.lower().strip()  # get_predictions: completion.lower().strip()
        return {"raw_completion": raw, "predicted_label": clean_label(pred, self.labels),
                "first_token_probs": first_probs, "n_prompt_tokens": len(self.prefix_tokens) + len(suffix_tokens),
                "prompt_text": full}

    def close(self):
        self.llm.close() if hasattr(self.llm, "close") else None
        del self.llm


def sha256(path, buf=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
