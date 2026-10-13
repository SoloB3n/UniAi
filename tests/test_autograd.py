"""Numerical gradient checks: every op's backward must match finite differences."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from uniai.nn import GPT  # noqa: E402
from uniai.tensor import Tensor, cross_entropy, embedding, layer_norm  # noqa: E402
from uniai.tokenizer import Tokenizer  # noqa: E402

rng = np.random.default_rng(0)


def gradcheck(fn, *shapes, eps=1e-6, tol=1e-5):
    xs = [Tensor(rng.standard_normal(s), requires_grad=True) for s in shapes]
    for x in xs:
        x.data = x.data.astype(np.float64)
    fn(*xs).backward()
    for x in xs:
        if x.grad is None:
            x.grad = np.zeros_like(x.data)
        num = np.zeros_like(x.data)
        it = np.nditer(x.data, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = x.data[i]
            x.data[i] = old + eps
            hi = fn(*xs).data.sum()
            x.data[i] = old - eps
            lo = fn(*xs).data.sum()
            x.data[i] = old
            num[i] = (hi - lo) / (2 * eps)
        err = np.abs(num - x.grad).max() / max(1.0, np.abs(num).max())
        assert err < tol, f"gradient mismatch: rel err {err:.2e}"


def test_arithmetic():
    gradcheck(lambda a, b: ((a * b + a - b / (b * b + 1.0)) ** 2).sum(), (3, 4), (4,))


def test_matmul_batched():
    gradcheck(lambda a, b: (a @ b).sum(), (2, 3, 4, 5), (5, 6))
    gradcheck(lambda a, b: (a @ b.transpose(0, 2, 1)).tanh().sum(), (2, 3, 4), (2, 5, 4))


def test_reductions_and_shapes():
    gradcheck(lambda a: (a.mean(axis=1, keepdims=True) * a).reshape(6, 2).sum(axis=0).sum(),
              (3, 4))
    gradcheck(lambda a: (a[1:, ::2] * a[0, 1]).sum(), (3, 4))


def test_nonlinearities():
    gradcheck(lambda a: a.gelu().sum() + a.exp().sum() + (a * a + 1).log().sum(), (5, 3))
    gradcheck(lambda a, w: (a.softmax(axis=-1) * w).sum(), (3, 7), (3, 7))


def test_layer_norm():
    gradcheck(lambda x, g, b: (layer_norm(x, g, b) * x).sum(), (2, 3, 8), (8,), (8,))


def test_cross_entropy_and_embedding():
    targets = rng.integers(0, 6, size=(2, 5))
    gradcheck(lambda l: cross_entropy(l, targets), (2, 5, 6))
    idx = np.array([[0, 2, 2], [1, 0, 3]])
    gradcheck(lambda w: (embedding(w, idx) ** 2).sum(), (4, 3))


def test_gpt_end_to_end_gradient():
    """Spot-check gradients of a whole tiny transformer against finite differences."""
    np.random.seed(0)
    model = GPT(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8)
    params = model.parameters()
    for p in params:
        p.data = p.data.astype(np.float64)
    x = rng.integers(0, 11, size=(2, 6))
    y = rng.integers(0, 11, size=(2, 6))
    _, loss = model(x, y)
    loss.backward()
    eps = 1e-6
    for p in params:
        for _ in range(3):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p.data[i]
            p.data[i] = old + eps
            hi = model(x, y)[1].item()
            p.data[i] = old - eps
            lo = model(x, y)[1].item()
            p.data[i] = old
            num = (hi - lo) / (2 * eps)
            assert abs(num - p.grad[i]) < 1e-6 + 1e-4 * abs(num), (num, p.grad[i])


def test_tokenizer_roundtrip():
    text = "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura. " * 20
    tok = Tokenizer.train(text + "àèìòù €", vocab_size=300)
    for s in [text, "ché la diritta via era smarrita 🌌", "", "  spazi\n\nnewline"]:
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(text)) < len(text.encode()) / 2
    assert Tokenizer.from_json(tok.to_json()).encode(text) == tok.encode(text)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
