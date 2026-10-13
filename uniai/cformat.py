"""Binary formats shared with the C engine (csrc/uniai.c).

model.bin : int32[16] header (magic, version, V, T, L, H, C, step, has_opt,
            float val_loss) followed by float32 parameters in C order, where
            per-layer tensors are stacked over layers.
PREFIX_train.bin / PREFIX_val.bin : uint16 token ids.
PREFIX_vocab.bin : int32 V, then for each token uint8 length + raw bytes.
"""

import numpy as np

from .nn import GPT

MAGIC, VERSION = 0x41494E55, 1

# (C tensor, python name template, per-layer?)
LAYOUT = [
    ("wte", "wte.weight", False),
    ("wpe", "wpe.weight", False),
    ("ln1w", "blocks.{}.ln1.weight", True),
    ("ln1b", "blocks.{}.ln1.bias", True),
    ("qkvw", "blocks.{}.attn.qkv.weight", True),
    ("qkvb", "blocks.{}.attn.qkv.bias", True),
    ("apw", "blocks.{}.attn.proj.weight", True),
    ("apb", "blocks.{}.attn.proj.bias", True),
    ("ln2w", "blocks.{}.ln2.weight", True),
    ("ln2b", "blocks.{}.ln2.bias", True),
    ("fcw", "blocks.{}.mlp.fc.weight", True),
    ("fcb", "blocks.{}.mlp.fc.bias", True),
    ("fpw", "blocks.{}.mlp.proj.weight", True),
    ("fpb", "blocks.{}.mlp.proj.bias", True),
    ("lnfw", "ln_f.weight", False),
    ("lnfb", "ln_f.bias", False),
]


def flatten(model):
    """Python parameters -> one float32 vector in C order."""
    sd, L = model.state_dict(), model.cfg["n_layer"]
    parts = []
    for _, name, per_layer in LAYOUT:
        names = [name.format(i) for i in range(L)] if per_layer else [name]
        parts += [sd[n].astype(np.float32).ravel() for n in names]
    return np.concatenate(parts)


def unflatten(model, flat):
    """Inverse of flatten: returns a state_dict for `model`."""
    shapes = dict((k, v.shape) for k, v in model.state_dict().items())
    L, out, off = model.cfg["n_layer"], {}, 0
    for _, name, per_layer in LAYOUT:
        for n in ([name.format(i) for i in range(L)] if per_layer else [name]):
            size = int(np.prod(shapes[n]))
            out[n] = flat[off:off + size].reshape(shapes[n])
            off += size
    assert off == len(flat), "parameter count mismatch"
    return out


def save_bin(path, model, step=0, val_loss=1e30):
    c = model.cfg
    hdr = np.zeros(16, dtype=np.int32)
    hdr[:9] = [MAGIC, VERSION, c["vocab_size"], c["block_size"], c["n_layer"],
               c["n_head"], c["n_embd"], step, 0]
    hdr[9] = np.array([val_loss], dtype=np.float32).view(np.int32)[0]
    with open(path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(flatten(model).tobytes())


def load_bin(path):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(64), dtype=np.int32)
        if hdr[0] != MAGIC or hdr[1] != VERSION:
            raise ValueError(f"{path}: not a UniAi model file")
        V, T, L, H, C = (int(x) for x in hdr[2:7])
        model = GPT(vocab_size=V, block_size=T, n_layer=L, n_head=H, n_embd=C, dropout=0.0)
        n = model.num_parameters()
        flat = np.frombuffer(f.read(4 * n), dtype=np.float32)
    model.load_state_dict(unflatten(model, flat))
    meta = {"step": int(hdr[7]), "val_loss": float(hdr[9:10].view(np.float32)[0])}
    return model.eval(), meta


def save_tokens(path, ids):
    np.asarray(ids, dtype=np.uint16).tofile(path)


def save_vocab(path, tokenizer):
    with open(path, "wb") as f:
        f.write(np.int32(tokenizer.vocab_size).tobytes())
        for i in range(tokenizer.vocab_size):
            b = tokenizer.vocab[i]
            assert len(b) < 256
            f.write(bytes([len(b)]) + b)
