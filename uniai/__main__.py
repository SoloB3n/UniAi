"""UniAi command line.

    python -m uniai train  --data data/testo.txt --out uniai.npz
    python -m uniai sample --ckpt uniai.npz --prompt "Nel mezzo"
    python -m uniai chat   --ckpt uniai.npz
"""

import argparse
import math
import os
import sys
import time

import numpy as np

from . import checkpoint
from .nn import GPT
from .optim import AdamW, clip_grad_norm, cosine_lr
from .tensor import no_grad
from .tokenizer import Tokenizer


def _tune_allocator():
    """Keep freed activation buffers in the heap instead of returning them to
    the OS after every step (glibc otherwise mmaps/munmaps each large array,
    and re-faulting those pages can dominate the step time)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 1 << 30)  # M_MMAP_THRESHOLD
        libc.mallopt(-1, 1 << 30)  # M_TRIM_THRESHOLD
    except (OSError, AttributeError):
        pass


def get_batch(data, batch, block, rng):
    ix = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[i:i + block] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + block] for i in ix]).astype(np.int64)
    return x, y


def estimate_loss(model, data, batch, block, iters, rng):
    model.eval()
    with no_grad():
        losses = [model(*get_batch(data, batch, block, rng))[1].item() for _ in range(iters)]
    model.train()
    return float(np.mean(losses))


def cmd_train(a):
    _tune_allocator()
    rng = np.random.default_rng(a.seed)
    np.random.seed(a.seed)
    with open(a.data, encoding="utf-8") as f:
        text = f.read()
    print(f"corpus: {a.data} ({len(text):,} characters)")

    if a.resume:
        model, tok, meta = checkpoint.load(a.resume)
        start = meta.get("step", 0)
        print(f"resuming from {a.resume} at step {start}")
    else:
        t0 = time.time()
        print(f"training BPE tokenizer -> {a.vocab} tokens")
        tok = Tokenizer.train(text, a.vocab, verbose=True)
        print(f"  done in {time.time() - t0:.1f}s")
        model = GPT(vocab_size=tok.vocab_size, block_size=a.block, n_layer=a.layers,
                    n_head=a.heads, n_embd=a.embd, dropout=a.dropout)
        start, meta = 0, {}

    ids = np.array(tok.encode(text), dtype=np.int32)
    print(f"tokens: {len(ids):,} ({len(text.encode()) / len(ids):.2f} bytes/token)")
    split = int(0.9 * len(ids))
    train, val = ids[:split], ids[split:]
    block = model.cfg["block_size"]
    print(f"model: {model.num_parameters():,} parameters | {model.cfg}")

    opt = AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best = meta.get("val_loss", math.inf)
    model.train()
    t0 = time.time()
    for step in range(start, a.steps):
        lr = cosine_lr(step, a.lr, a.warmup, a.steps)
        opt.lr = lr
        x, y = get_batch(train, a.batch, block, rng)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        norm = clip_grad_norm(opt.params, 1.0)
        opt.step()

        if step % a.log_every == 0:
            dt = (time.time() - t0) / max(1, step - start + 1)
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"grad {norm:.2f} | {dt * 1000:.0f} ms/step", flush=True)

        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            tl = estimate_loss(model, train, a.batch, block, a.eval_iters, rng)
            vl = estimate_loss(model, val, a.batch, block, a.eval_iters, rng)
            print(f"---- eval @ {step + 1}: train {tl:.4f} | val {vl:.4f} "
                  f"| val ppl {math.exp(vl):.1f}")
            if vl < best:
                best = vl
                checkpoint.save(a.out, model, tok,
                                {"step": step + 1, "val_loss": vl, "data": a.data})
                print(f"     saved {a.out}")
            sample = model.generate(tok.encode("\n"), 120, temperature=0.8, top_k=40, rng=rng)
            print("     sample: " + tok.decode(sample[1:]).replace("\n", "\n             "))
    print(f"best val loss {best:.4f}")


def _stream_printer(tok):
    """Print tokens as they arrive, holding back incomplete UTF-8 sequences."""
    buf = bytearray()

    def emit(t):
        buf.extend(tok.decode_bytes([t]))
        try:
            s = buf.decode("utf-8")
        except UnicodeDecodeError:
            return
        sys.stdout.write(s)
        sys.stdout.flush()
        buf.clear()

    return emit


def cmd_sample(a):
    model, tok, meta = checkpoint.load(a.ckpt)
    rng = np.random.default_rng(a.seed)
    prompt = a.prompt.encode().decode("unicode_escape") if a.escape else a.prompt
    sys.stdout.write(prompt)
    model.generate(tok.encode(prompt) or tok.encode("\n"), a.tokens,
                   temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, rng=rng,
                   stream=_stream_printer(tok))
    print()


def cmd_chat(a):
    model, tok, meta = checkpoint.load(a.ckpt)
    rng = np.random.default_rng(a.seed)
    print(f"UniAi — {model.num_parameters():,} parameters, trained on "
          f"{meta.get('data', '?')} (val loss {meta.get('val_loss', float('nan')):.3f})")
    print("Write the beginning of a text; UniAi continues it. Ctrl-D to quit.\n")
    while True:
        try:
            prompt = input("» ")
        except EOFError:
            print()
            return
        sys.stdout.write(prompt)
        model.generate(tok.encode(prompt or "\n"), a.tokens, temperature=a.temperature,
                       top_k=a.top_k, top_p=a.top_p, rng=rng, stream=_stream_printer(tok))
        print("\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="uniai", description="A language model built from scratch.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a tokenizer and a GPT on a text file")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="uniai.npz")
    t.add_argument("--resume", default=None, help="continue from a checkpoint")
    t.add_argument("--vocab", type=int, default=1024)
    t.add_argument("--block", type=int, default=128)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--embd", type=int, default=128)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--seed", type=int, default=1337)
    t.set_defaults(fn=cmd_train)

    for name, fn in (("sample", cmd_sample), ("chat", cmd_chat)):
        s = sub.add_parser(name, help=f"{name} from a trained checkpoint")
        s.add_argument("--ckpt", default="uniai.npz")
        s.add_argument("--tokens", type=int, default=300)
        s.add_argument("--temperature", type=float, default=0.8)
        s.add_argument("--top-k", type=int, default=50)
        s.add_argument("--top-p", type=float, default=0.95)
        s.add_argument("--seed", type=int, default=None)
        if name == "sample":
            s.add_argument("--prompt", default="\n")
            s.add_argument("--escape", action="store_true", help="interpret \\n etc. in prompt")
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    if getattr(a, "data", None) and not os.path.exists(a.data):
        p.error(f"no such file: {a.data}")
    a.fn(a)


if __name__ == "__main__":
    main()
