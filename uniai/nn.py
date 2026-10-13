"""Neural network building blocks and a GPT-style language model."""

import numpy as np

from .tensor import Tensor, cross_entropy, dropout, embedding, layer_norm, no_grad


class Module:
    training = True

    def parameters(self):
        """All trainable tensors, deduplicated (weight tying shares tensors)."""
        out, seen = [], set()
        for _, p in self.named_parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    def named_parameters(self, prefix=""):
        for name, v in vars(self).items():
            full = f"{prefix}{name}"
            if isinstance(v, Tensor) and v.requires_grad:
                yield full, v
            elif isinstance(v, Module):
                yield from v.named_parameters(full + ".")
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        yield from m.named_parameters(f"{full}.{i}.")

    def modules(self):
        yield self
        for v in vars(self).values():
            if isinstance(v, Module):
                yield from v.modules()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        yield from m.modules()

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        return {k: v.data.copy() for k, v in self.named_parameters()}

    def load_state_dict(self, state):
        for k, v in self.named_parameters():
            if k not in state:
                raise KeyError(f"missing parameter {k!r} in checkpoint")
            if state[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {state[k].shape} vs {v.shape}")
            v.data[...] = state[k]

    def num_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _param(arr):
    return Tensor(np.asarray(arr, dtype=np.float32), requires_grad=True)


class Linear(Module):
    def __init__(self, n_in, n_out, bias=True, std=0.02):
        self.weight = _param(np.random.randn(n_in, n_out) * std)
        self.bias = _param(np.zeros(n_out)) if bias else None

    def forward(self, x):
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, n, dim, std=0.02):
        self.weight = _param(np.random.randn(n, dim) * std)

    def forward(self, idx):
        return embedding(self.weight, idx)


class LayerNorm(Module):
    def __init__(self, dim):
        self.weight = _param(np.ones(dim))
        self.bias = _param(np.zeros(dim))

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias)


_MASKS = {}


def _causal_mask(T):
    if T not in _MASKS:
        _MASKS[T] = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
    return _MASKS[T]


class CausalSelfAttention(Module):
    def __init__(self, cfg):
        assert cfg["n_embd"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.dropout = cfg["dropout"]
        self.qkv = Linear(cfg["n_embd"], 3 * cfg["n_embd"])
        # Residual projections get a smaller init so the stream stays stable
        # as depth grows (GPT-2 trick).
        self.proj = Linear(cfg["n_embd"], cfg["n_embd"],
                           std=0.02 / np.sqrt(2 * cfg["n_layer"]))

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.n_head, C // self.n_head
        qkv = self.qkv(x).reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)  # 3,B,H,T,D
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(D))           # B,H,T,T
        att = (att + _causal_mask(T)).softmax(axis=-1)
        att = dropout(att, self.dropout, self.training)
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return dropout(self.proj(y), self.dropout, self.training)


class MLP(Module):
    def __init__(self, cfg):
        C = cfg["n_embd"]
        self.fc = Linear(C, 4 * C)
        self.proj = Linear(4 * C, C, std=0.02 / np.sqrt(2 * cfg["n_layer"]))
        self.dropout = cfg["dropout"]

    def forward(self, x):
        return dropout(self.proj(self.fc(x).gelu()), self.dropout, self.training)


class Block(Module):
    """Pre-norm transformer block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, cfg):
        self.ln1 = LayerNorm(cfg["n_embd"])
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg["n_embd"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


DEFAULT_CONFIG = dict(
    vocab_size=512,
    block_size=128,
    n_layer=4,
    n_head=4,
    n_embd=128,
    dropout=0.0,
)


class GPT(Module):
    def __init__(self, **cfg):
        self.cfg = {**DEFAULT_CONFIG, **cfg}
        c = self.cfg
        self.wte = Embedding(c["vocab_size"], c["n_embd"])
        self.wpe = Embedding(c["block_size"], c["n_embd"], std=0.01)
        self.blocks = [Block(c) for _ in range(c["n_layer"])]
        self.ln_f = LayerNorm(c["n_embd"])
        # The output head reuses the token embedding matrix (weight tying).

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg["block_size"], "sequence longer than block_size"
        x = self.wte(idx) + self.wpe(np.arange(T))
        x = dropout(x, self.cfg["dropout"], self.training)
        for block in self.blocks:
            x = block(x)
        logits = self.ln_f(x) @ self.wte.weight.T
        loss = cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, top_p=None,
                 rng=None, stream=None):
        """Autoregressive sampling. `idx` is a list of token ids."""
        rng = rng or np.random.default_rng()
        idx = list(idx)
        was_training = self.training
        self.eval()
        with no_grad():
            for _ in range(max_new_tokens):
                ctx = np.array([idx[-self.cfg["block_size"]:]])
                logits, _ = self(ctx)
                nxt = sample_next(logits.data[0, -1], temperature, top_k, top_p, rng)
                idx.append(nxt)
                if stream is not None:
                    stream(nxt)
        self.train(was_training)
        return idx


def sample_next(logits, temperature=0.8, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and top_k < len(logits):
        kth = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    if top_p is not None and top_p < 1.0:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        cut = order[cum - probs[order] > top_p]  # keep the smallest nucleus
        probs[cut] = 0.0
        probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
