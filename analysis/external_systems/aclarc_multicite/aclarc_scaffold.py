"""Pure-PyTorch inference port of the released ACL-ARC citation-intent model of
Cohan, Ammar, van Zuylen & Cady (NAACL 2019), "Structural Scaffolds for Citation Intent Classification
in Scientific Publications" (allenai/scicite @ b33a8cd0df5743c9d483b1d0fdce8ec62bc8f135, Apache-2.0).

Archive: https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/models/aclarc.tar.gz
         (712,967,746 B, sha256 d144c771ca248e7472b7d982705abee8f0a9963b49f8edf13f216493a76c7b9f)
The archive's config.json / weights.th / vocabulary are loaded unchanged. The original code needs
Python 3.6 + AllenNLP 0.8; this file re-implements ONLY the inference graph, line-for-line from
allennlp v0.8.0 (modules/elmo.py, elmo_lstm.py, lstm_cell_with_projection.py, highway.py,
scalar_mix.py, basic_text_field_embedder.py, data/token_indexers/elmo_indexer.py) and
scicite/models/scaffold_bilstm_attention_classifier.py:

  tokens (spaCy tokenizer, not lower-cased)
   -> [ELMo PubMed 2x4096_512 (char-CNN + 2 highway + proj 512 + 2-layer biLSTM w/ projection,
       cell clip 3, proj clip 3, skip conn.), scalar mix w/ layer norm, bos/eos removed]  (1024)
      ++ [frozen GloVe-6B-100d as stored in weights.th]                                 (100)
      (BasicTextFieldEmbedder concatenates in sorted key order: 'elmo' then 'tokens')
   -> BiLSTM(1124 -> 2x50) -> multiplicative attention (vector 100x1) -> FF 100-20(relu)-6
   -> softmax over labels [Background, Uses, CompareOrContrast, Motivation, Extends, Future]

Known deviations from the original runtime (documented in README):
  * AllenNLP's ELMo biLSTM is *stateful* across batches; here every instance starts from zero
    state (= the state of the first instance in a fresh AllenNLP process). Needed for determinism.
  * spaCy 3.7 tokenizer rules (spacy.blank('en')) instead of spaCy 2.0 en_core_web_sm.
  * batch size 1 (as the AllenNLP predictor's forward_on_instance); scalar-mix layer norm is
    batch-dependent, so batching would change outputs.
"""
import json, os
import torch
import torch.nn as nn
import torch.nn.functional as F

LABELS_FILE = "vocabulary/labels.txt"


# ---------------- ELMo character mapper (allennlp 0.8 ELMoCharacterMapper) ----------------
MAXW = 50
BOS_C, EOS_C, BOW_C, EOW_C, PAD_C = 256, 257, 258, 259, 260


def _bos_eos(c):
    ids = [PAD_C] * MAXW
    ids[0], ids[1], ids[2] = BOW_C, c, EOW_C
    return [x + 1 for x in ids]


BOS_CHARS = torch.tensor(_bos_eos(BOS_C))
EOS_CHARS = torch.tensor(_bos_eos(EOS_C))


def word_to_char_ids(word):
    enc = word.encode("utf-8", "ignore")[: MAXW - 2]
    ids = [PAD_C] * MAXW
    ids[0] = BOW_C
    for k, b in enumerate(enc, start=1):
        ids[k] = b
    ids[len(enc) + 1] = EOW_C
    return [x + 1 for x in ids]


class LstmCellWithProjection(nn.Module):
    def __init__(self, input_size, hidden_size, cell_size, go_forward, cell_clip, proj_clip):
        super().__init__()
        self.hidden_size, self.cell_size, self.go_forward = hidden_size, cell_size, go_forward
        self.cell_clip, self.proj_clip = cell_clip, proj_clip
        self.input_linearity = nn.Linear(input_size, 4 * cell_size, bias=False)
        self.state_linearity = nn.Linear(hidden_size, 4 * cell_size, bias=True)
        self.state_projection = nn.Linear(cell_size, hidden_size, bias=False)

    def forward(self, inputs):  # inputs (1, T, in); batch size 1, no padding
        T = inputs.size(1)
        c = self.cell_size
        out = inputs.new_zeros(1, T, self.hidden_size)
        mem = inputs.new_zeros(1, c)
        st = inputs.new_zeros(1, self.hidden_size)
        PI = self.input_linearity(inputs)  # all timesteps at once (same math, one matmul)
        for t in range(T):
            idx = t if self.go_forward else T - t - 1
            pi = PI[:, idx]
            ps = self.state_linearity(st)
            ig = torch.sigmoid(pi[:, 0:c] + ps[:, 0:c])
            fg = torch.sigmoid(pi[:, c:2 * c] + ps[:, c:2 * c])
            mi = torch.tanh(pi[:, 2 * c:3 * c] + ps[:, 2 * c:3 * c])
            og = torch.sigmoid(pi[:, 3 * c:4 * c] + ps[:, 3 * c:4 * c])
            mem = ig * mi + fg * mem
            if self.cell_clip:
                mem = torch.clamp(mem, -self.cell_clip, self.cell_clip)
            h = self.state_projection(og * torch.tanh(mem))
            if self.proj_clip:
                h = torch.clamp(h, -self.proj_clip, self.proj_clip)
            st = h
            out[:, idx] = h
        return out


class ElmoLstm(nn.Module):
    def __init__(self, opt):
        super().__init__()
        lo = opt["lstm"]
        self.n_layers = lo["n_layers"]
        for i in range(self.n_layers):
            inp = lo["projection_dim"]
            setattr(self, f"forward_layer_{i}", LstmCellWithProjection(inp, lo["projection_dim"], lo["dim"], True, lo["cell_clip"], lo["proj_clip"]))
            setattr(self, f"backward_layer_{i}", LstmCellWithProjection(inp, lo["projection_dim"], lo["dim"], False, lo["cell_clip"], lo["proj_clip"]))

    def forward(self, x):
        fwd = bwd = x
        outs = []
        for i in range(self.n_layers):
            fc, bc = fwd, bwd
            fwd = getattr(self, f"forward_layer_{i}")(fwd)
            bwd = getattr(self, f"backward_layer_{i}")(bwd)
            if i != 0:
                fwd = fwd + fc
                bwd = bwd + bc
            outs.append(torch.cat([fwd, bwd], -1))
        return outs


class Highway(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self._layers = nn.ModuleList([nn.Linear(dim, 2 * dim) for _ in range(n)])

    def forward(self, x):
        for layer in self._layers:
            p = layer(x)
            nl, g = p.chunk(2, dim=-1)
            nl = F.relu(nl)
            g = torch.sigmoid(g)
            x = g * x + (1 - g) * nl
        return x


class CharEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        co = opt["char_cnn"]
        self._char_embedding_weights = nn.Parameter(torch.zeros(co["n_characters"], co["embedding"]["dim"]))
        self.n_conv = len(co["filters"])
        for i, (w, n) in enumerate(co["filters"]):
            setattr(self, f"char_conv_{i}", nn.Conv1d(co["embedding"]["dim"], n, kernel_size=w, bias=True))
        nf = sum(n for _, n in co["filters"])
        self._highways = Highway(nf, co["n_highway"])
        self._projection = nn.Linear(nf, opt["lstm"]["projection_dim"], bias=True)

    def forward(self, char_ids):  # (1, T, 50), already with bos/eos rows
        e = F.embedding(char_ids.view(-1, MAXW), self._char_embedding_weights).transpose(1, 2)
        convs = []
        for i in range(self.n_conv):
            c = getattr(self, f"char_conv_{i}")(e)
            c, _ = torch.max(c, dim=-1)
            convs.append(F.relu(c))
        t = torch.cat(convs, -1)
        t = self._highways(t)
        t = self._projection(t)
        return t.view(char_ids.size(0), char_ids.size(1), -1)


class Elmo(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self._token_embedder = CharEncoder(opt)
        self._elmo_lstm = ElmoLstm(opt)
        self.scalar_parameters = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(opt["lstm"]["n_layers"] + 1)])
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, char_ids):  # (1, T, 50) without bos/eos
        ids = torch.cat([BOS_CHARS.view(1, 1, -1), char_ids, EOS_CHARS.view(1, 1, -1)], 1)
        tok = self._token_embedder(ids)
        layers = [torch.cat([tok, tok], -1)] + self._elmo_lstm(tok)
        w = F.softmax(torch.cat(list(self.scalar_parameters)), dim=0)
        mixed = 0
        for k, t in enumerate(layers):  # layer norm over all (unmasked) elements, as ScalarMix
            n = t.numel()
            mean = t.sum() / n
            var = ((t - mean) ** 2).sum() / n
            mixed = mixed + w[k] * ((t - mean) / torch.sqrt(var + 1e-12))
        return (self.gamma * mixed)[:, 1:-1]


class ScaffoldModel(nn.Module):
    def __init__(self, opt, vocab_size):
        super().__init__()
        self.elmo = Elmo(opt)
        self.glove = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(1124, 50, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = nn.Parameter(torch.zeros(100, 1))
        self.ff0 = nn.Linear(100, 20)
        self.ff1 = nn.Linear(20, 6)

    def forward(self, char_ids, tok_ids):
        emb = torch.cat([self.elmo(char_ids), self.glove(tok_ids)], -1)  # sorted keys: elmo, tokens
        enc, _ = self.lstm(emb)
        score = torch.matmul(enc, self.attention).squeeze(-1)
        attn = F.softmax(score, dim=-1).unsqueeze(-1)
        pooled = (enc * attn).sum(dim=-2)
        return self.ff1(F.relu(self.ff0(pooled)))


def _map_state_dict(sd):
    P = "text_field_embedder.token_embedder_elmo._elmo."
    out = {}
    for k, v in sd.items():
        if k.startswith(P + "_elmo_lstm._token_embedder."):
            out["elmo._token_embedder." + k[len(P + "_elmo_lstm._token_embedder."):]] = v
        elif k.startswith(P + "_elmo_lstm._elmo_lstm."):
            out["elmo._elmo_lstm." + k[len(P + "_elmo_lstm._elmo_lstm."):]] = v
        elif k.startswith(P + "scalar_mix_0.scalar_parameters."):
            out["elmo.scalar_parameters." + k.rsplit(".", 1)[1]] = v
        elif k == P + "scalar_mix_0.gamma":
            out["elmo.gamma"] = v
        elif k == "text_field_embedder.token_embedder_tokens.weight":
            out["glove.weight"] = v
        elif k.startswith("citation_text_encoder._module."):
            out["lstm." + k[len("citation_text_encoder._module."):]] = v
        elif k == "attention_seq2seq.attention":
            out["attention"] = v
        elif k.startswith("classifier_feedforward._linear_layers."):
            i, p = k[len("classifier_feedforward._linear_layers."):].split(".")
            out[f"ff{i}.{p}"] = v
        # classifier_feedforward_2/_3 are the scaffold heads (section title, cite-worthiness): unused at inference
    return out


class AclArcClassifier:
    def __init__(self, archive_dir):
        opt = json.load(open(os.path.join(archive_dir, "fta/model.elmo_text_field_embedder.elmo.options_file")))
        toks = open(os.path.join(archive_dir, "vocabulary/tokens.txt"), encoding="utf-8").read().split("\n")
        if toks and toks[-1] == "":
            toks = toks[:-1]
        self.tok2id = {t: i + 1 for i, t in enumerate(toks)}  # 0 = @@PADDING@@, 1 = @@UNKNOWN@@
        self.labels = [l for l in open(os.path.join(archive_dir, LABELS_FILE)).read().split("\n") if l]
        sd = torch.load(os.path.join(archive_dir, "weights.th"), map_location="cpu", weights_only=False)
        self.model = ScaffoldModel(opt, sd["text_field_embedder.token_embedder_tokens.weight"].shape[0])
        missing, unexpected = self.model.load_state_dict(_map_state_dict(sd), strict=True), None
        self.model.eval()
        import spacy
        self.nlp = spacy.blank("en")

    def tokenize(self, text):
        return [t.text for t in self.nlp(text) if not t.is_space]

    @torch.no_grad()
    def predict(self, text):
        toks = self.tokenize(text)
        if not toks:
            return None
        char_ids = torch.tensor([[word_to_char_ids(t) for t in toks]])
        tok_ids = torch.tensor([[self.tok2id.get(t, 1) for t in toks]])
        logits = self.model(char_ids, tok_ids)[0]
        probs = F.softmax(logits, dim=-1)
        return {"tokens": toks, "logits": logits.tolist(), "probs": dict(zip(self.labels, probs.tolist())),
                "label": self.labels[int(torch.argmax(probs))]}
