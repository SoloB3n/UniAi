"""Parity test: the C engine must match the (gradient-checked) Python engine.

Same weights + same batch -> same loss, same gradient for every parameter,
and the same parameters after one clipped AdamW step.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from uniai import cformat  # noqa: E402
from uniai.nn import GPT  # noqa: E402
from uniai.optim import AdamW, clip_grad_norm  # noqa: E402

BIN = os.path.join(ROOT, "uniai_c")


def run_case(cfg, B, T, seed):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = GPT(dropout=0.0, **cfg)
    # perturb norms/biases so their gradients are exercised non-trivially
    for name, p in model.named_parameters():
        if "ln" in name or name.endswith("bias"):
            p.data += 0.1 * np.random.randn(*p.shape).astype(np.float32)
    x = rng.integers(0, cfg["vocab_size"], size=(B, T))
    y = rng.integers(0, cfg["vocab_size"], size=(B, T))

    with tempfile.TemporaryDirectory() as d:
        mpath, bpath, opath = (os.path.join(d, n) for n in ("m.bin", "b.bin", "o.bin"))
        cformat.save_bin(mpath, model)
        with open(bpath, "wb") as f:
            f.write(np.array([B, T], dtype=np.int32).tobytes())
            f.write(x.astype(np.int32).tobytes())
            f.write(y.astype(np.int32).tobytes())
        subprocess.run([BIN, "check", mpath, bpath, opath], check=True)
        out = np.fromfile(opath, dtype=np.float32)

    n = model.num_parameters()
    c_loss, c_grads, c_norm, c_after = out[0], out[1:1 + n], out[1 + n], out[2 + n:]

    _, loss = model(x, y)
    loss.backward()
    py_grads = cformat.flatten(_GradView(model))
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    py_norm = clip_grad_norm(opt.params, 1.0)
    opt.step()
    py_after = cformat.flatten(model)

    def rel(a, b):
        return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)

    assert abs(c_loss - loss.item()) < 1e-5, (c_loss, loss.item())
    assert rel(c_grads, py_grads) < 1e-4, f"grad rel err {rel(c_grads, py_grads):.2e}"
    assert abs(c_norm - py_norm) < 1e-4 * py_norm
    assert rel(c_after, py_after) < 1e-5, f"adam rel err {rel(c_after, py_after):.2e}"
    return loss.item(), rel(c_grads, py_grads)


def test_dropout_gradcheck():
    """With dropout on, C gradients must match finite differences (mask replay)."""
    np.random.seed(3)
    rng = np.random.default_rng(3)
    model = GPT(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    for _, p in model.named_parameters():
        p.data += 0.1 * np.random.randn(*p.shape).astype(np.float32)
    B, T = 2, 12
    x = rng.integers(0, 50, size=(B, T))
    y = rng.integers(0, 50, size=(B, T))
    with tempfile.TemporaryDirectory() as d:
        mpath, bpath = os.path.join(d, "m.bin"), os.path.join(d, "b.bin")
        cformat.save_bin(mpath, model)
        with open(bpath, "wb") as f:
            f.write(np.array([B, T], dtype=np.int32).tobytes())
            f.write(x.astype(np.int32).tobytes() + y.astype(np.int32).tobytes())
        r = subprocess.run([BIN, "gradcheck", mpath, bpath, "0.3"],
                           capture_output=True, text=True)
    print("  " + r.stdout.strip().splitlines()[-1])
    assert r.returncode == 0, r.stdout


class _GradView:
    """Lets cformat.flatten read gradients instead of weights."""

    def __init__(self, model):
        self.cfg = model.cfg
        self._m = model

    def state_dict(self):
        return {k: v.grad for k, v in self._m.named_parameters()}


def test_parity_small():
    print("  loss %.5f, grad rel err %.1e" % run_case(
        dict(vocab_size=37, block_size=16, n_layer=2, n_head=2, n_embd=16), B=3, T=11, seed=1))


def test_parity_default_shape():
    print("  loss %.5f, grad rel err %.1e" % run_case(
        dict(vocab_size=300, block_size=64, n_layer=3, n_head=4, n_embd=64), B=4, T=64, seed=2))


if __name__ == "__main__":
    if not os.path.exists(BIN):
        sys.exit("build the engine first: make")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
