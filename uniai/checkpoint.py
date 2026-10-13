"""Save/load a model + tokenizer as a single .npz file."""

import json

import numpy as np

from .nn import GPT
from .tokenizer import Tokenizer


def save(path, model, tokenizer, meta=None):
    arrays = {f"param/{k}": v for k, v in model.state_dict().items()}
    arrays["config"] = np.frombuffer(json.dumps(model.cfg).encode(), dtype=np.uint8)
    arrays["tokenizer"] = np.frombuffer(tokenizer.to_json().encode(), dtype=np.uint8)
    arrays["meta"] = np.frombuffer(json.dumps(meta or {}).encode(), dtype=np.uint8)
    np.savez(path, **arrays)


def load(path):
    z = np.load(path)
    cfg = json.loads(z["config"].tobytes())
    model = GPT(**cfg)
    model.load_state_dict({k[6:]: z[k] for k in z.files if k.startswith("param/")})
    tokenizer = Tokenizer.from_json(z["tokenizer"].tobytes().decode())
    meta = json.loads(z["meta"].tobytes()) if "meta" in z.files else {}
    return model.eval(), tokenizer, meta
