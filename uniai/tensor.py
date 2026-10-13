"""Reverse-mode automatic differentiation on top of NumPy.

A `Tensor` wraps an ndarray and remembers how it was computed. Calling
`.backward()` on a scalar walks that graph in reverse topological order and
accumulates gradients into every tensor that `requires_grad`.
"""

from contextlib import contextmanager

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad():
    """Disable graph construction (inference, evaluation)."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def _as_array(x):
    if isinstance(x, (np.ndarray, np.generic)):
        x = np.asarray(x)
        return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class Tensor:
    __array_priority__ = 100  # make `ndarray op Tensor` defer to Tensor

    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = _as_array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents
        self._op = _op
        self._backward = None

    # ------------------------------------------------------------ plumbing
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'}, grad={self.requires_grad})"

    def item(self):
        return float(self.data)

    def numpy(self):
        return self.data

    def zero_grad(self):
        self.grad = None

    def _accum(self, g):
        if self.grad is None:
            self.grad = g.astype(self.data.dtype, copy=True)
        else:
            self.grad += g

    @staticmethod
    def _make(data, parents, op, backward):
        """Create a result node, wiring the backward closure only if needed."""
        track = _grad_enabled and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=track, _parents=parents if track else (), _op=op)
        if track:
            out._backward = lambda: backward(out.grad)
        return out

    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() without a gradient needs a scalar output")
            grad = np.ones_like(self.data)

        # Iterative topological sort (deep graphs would overflow recursion).
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))

        self.grad = np.asarray(grad, dtype=self.data.dtype)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward()
                # Interior gradients are no longer needed; free memory early.
                if node is not self and node._parents:
                    node.grad = None

    # ---------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            if a.requires_grad:
                a._accum(_unbroadcast(g, a.shape))
            if b.requires_grad:
                b._accum(_unbroadcast(g, b.shape))

        return Tensor._make(a.data + b.data, (a, b), "add", bw)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -np.asarray(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self, other

        def bw(g):
            if a.requires_grad:
                a._accum(_unbroadcast(g * b.data, a.shape))
            if b.requires_grad:
                b._accum(_unbroadcast(g * a.data, b.shape))

        return Tensor._make(a.data * b.data, (a, b), "mul", bw)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self * other ** -1.0
        return self * (1.0 / other)

    def __rtruediv__(self, other):
        return self ** -1.0 * other

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar exponents are supported"
        a = self

        def bw(g):
            a._accum(g * p * a.data ** (p - 1))

        return Tensor._make(a.data ** p, (a,), "pow", bw)

    def __matmul__(self, other):
        a, b = self, other

        def bw(g):
            if a.requires_grad:
                a._accum(_unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape))
            if b.requires_grad:
                b._accum(_unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape))

        return Tensor._make(a.data @ b.data, (a, b), "matmul", bw)

    # ---------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        a = self

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            a._accum(np.broadcast_to(g, a.shape))

        return Tensor._make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum", bw)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.shape[i] for i in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ---------------------------------------------------------- elementwise
    def exp(self):
        a, y = self, np.exp(self.data)
        return Tensor._make(y, (a,), "exp", lambda g: a._accum(g * y))

    def log(self):
        a = self
        return Tensor._make(np.log(a.data), (a,), "log", lambda g: a._accum(g / a.data))

    def tanh(self):
        a, y = self, np.tanh(self.data)
        return Tensor._make(y, (a,), "tanh", lambda g: a._accum(g * (1 - y * y)))

    def relu(self):
        a = self
        mask = a.data > 0
        return Tensor._make(a.data * mask, (a,), "relu", lambda g: a._accum(g * mask))

    def gelu(self):
        """GELU, tanh approximation (as in GPT-2)."""
        a, x = self, self.data
        c = x.dtype.type(np.sqrt(2.0 / np.pi))
        k = x.dtype.type(0.044715)
        x2 = x * x
        t = np.tanh(c * x * (1 + k * x2))
        y = 0.5 * x * (1 + t)

        def bw(g):
            dinner = c * (1 + 3 * k * x2)
            a._accum(g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner))

        return Tensor._make(y, (a,), "gelu", bw)

    def softmax(self, axis=-1):
        a = self
        z = a.data - a.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        y = e / e.sum(axis=axis, keepdims=True)

        def bw(g):
            a._accum(y * (g - (g * y).sum(axis=axis, keepdims=True)))

        return Tensor._make(y, (a,), "softmax", bw)

    # -------------------------------------------------------------- shapes
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        a = self
        return Tensor._make(a.data.reshape(shape), (a,), "reshape",
                            lambda g: a._accum(g.reshape(a.shape)))

    def transpose(self, *axes):
        a = self
        if not axes:
            axes = tuple(reversed(range(a.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        inv = np.argsort(axes)
        return Tensor._make(a.data.transpose(axes), (a,), "transpose",
                            lambda g: a._accum(g.transpose(inv)))

    def __getitem__(self, idx):
        a = self
        fancy = any(isinstance(i, (list, np.ndarray, Tensor))
                    for i in (idx if isinstance(idx, tuple) else (idx,)))

        def bw(g):
            full = np.zeros_like(a.data)
            if fancy:
                np.add.at(full, idx, g)
            else:
                full[idx] += g
            a._accum(full)

        return Tensor._make(a.data[idx], (a,), "getitem", bw)


# ---------------------------------------------------------------- functional
def embedding(weight, idx):
    """Row lookup `weight[idx]` for an integer array `idx`."""
    idx = np.asarray(idx)

    def bw(g):
        full = np.zeros_like(weight.data)
        np.add.at(full, idx.reshape(-1), g.reshape(-1, weight.shape[-1]))
        weight._accum(full)

    return Tensor._make(weight.data[idx], (weight,), "embedding", bw)


def layer_norm(x, gamma, beta, eps=1e-5):
    """Fused layer normalisation over the last axis."""
    mu = x.data.mean(axis=-1, keepdims=True)
    xc = x.data - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    y = xhat * gamma.data + beta.data

    def bw(g):
        if gamma.requires_grad:
            gamma._accum(_unbroadcast(g * xhat, gamma.shape))
        if beta.requires_grad:
            beta._accum(_unbroadcast(g, beta.shape))
        if x.requires_grad:
            gx = g * gamma.data
            dx = (gx - gx.mean(axis=-1, keepdims=True)
                  - xhat * (gx * xhat).mean(axis=-1, keepdims=True)) * rstd
            x._accum(dx.astype(x.data.dtype))

    return Tensor._make(y, (x, gamma, beta), "layernorm", bw)


def cross_entropy(logits, targets, ignore_index=-1):
    """Mean negative log-likelihood of integer `targets` under `logits`.

    Fused log-softmax + NLL: numerically stable and a single backward step.
    """
    V = logits.shape[-1]
    flat = logits.data.reshape(-1, V)
    t = np.asarray(targets).reshape(-1)
    valid = t != ignore_index
    n = max(int(valid.sum()), 1)
    z = flat - flat.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logsumexp
    rows = np.arange(len(t))
    tt = np.where(valid, t, 0)
    loss = -(logp[rows, tt] * valid).sum() / n

    def bw(g):
        p = np.exp(logp)
        p[rows, tt] -= 1.0
        p *= valid[:, None] * (float(g) / n)
        logits._accum(p.reshape(logits.shape).astype(logits.data.dtype))

    return Tensor._make(np.asarray(loss, dtype=logits.data.dtype), (logits,), "xent", bw)


def dropout(x, p, training):
    if not training or p == 0.0:
        return x
    mask = (np.random.rand(*x.shape) >= p).astype(x.data.dtype) / (1.0 - p)
    return x * mask
