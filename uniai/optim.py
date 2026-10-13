"""Optimisation: AdamW, gradient clipping and a warmup + cosine schedule."""

import math

import numpy as np


class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0
        # Only matrices are decayed; biases, norms gains and 1-D params are not.
        self.decay = [p.data.ndim >= 2 for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for p, m, v, decay in zip(self.params, self.m, self.v, self.decay):
            if p.grad is None:
                continue
            g = p.grad
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            if decay and self.wd:
                p.data -= self.lr * self.wd * p.data
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

    def state_dict(self):
        return {"t": self.t, "m": [a.copy() for a in self.m], "v": [a.copy() for a in self.v]}

    def load_state_dict(self, s):
        self.t = s["t"]
        for dst, src in zip(self.m + self.v, s["m"] + s["v"]):
            dst[...] = src


def clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float(np.vdot(g, g)) for g in grads))
    if max_norm and total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g *= scale
    return total


def cosine_lr(step, max_lr, warmup, total, min_ratio=0.1):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
