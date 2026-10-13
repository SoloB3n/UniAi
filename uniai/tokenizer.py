"""Byte-level Byte Pair Encoding tokenizer, trained from raw text.

Any string round-trips losslessly: the base vocabulary is the 256 byte values,
and each learned merge fuses the most frequent adjacent pair into a new token.
Text is first split into word-like chunks so merges never cross word
boundaries, which keeps training fast and tokens meaningful.
"""

import json
import re
from collections import Counter

SPLIT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+""")


def _merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges=None):
        self.merges = {tuple(p): i for p, i in (merges or [])}
        self._build()

    def _build(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), i in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[i] = self.vocab[a] + self.vocab[b]
        self._cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        words = Counter(SPLIT.findall(text))
        seqs = {w: list(w.encode("utf-8")) for w in words}
        # Pair counts and an index pair -> words containing it, updated
        # incrementally so each merge only touches the words it affects.
        pairs, where = Counter(), {}
        for w, ids in seqs.items():
            for p in zip(ids, ids[1:]):
                pairs[p] += words[w]
                where.setdefault(p, set()).add(w)
        merges = []
        for new in range(256, vocab_size):
            if not pairs:
                break
            best, count = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < 2:
                break
            merges.append((best, new))
            for w in where.pop(best, ()):
                ids, f = seqs[w], words[w]
                for p in zip(ids, ids[1:]):
                    pairs[p] -= f
                    if pairs[p] <= 0:
                        del pairs[p]
                ids = seqs[w] = _merge(ids, best, new)
                for p in zip(ids, ids[1:]):
                    pairs[p] += f
                    where.setdefault(p, set()).add(w)
            if verbose and (new - 255) % 128 == 0:
                tok = cls(merges)
                print(f"  merge {new - 255:4d}/{vocab_size - 256}: "
                      f"{tok.vocab[new]!r} (x{count})")
        return cls(merges)

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge available, exactly like training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def decode_bytes(self, ids):
        return b"".join(self.vocab[i] for i in ids)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[list(p), i] for p, i in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])

    def to_json(self):
        return json.dumps({"merges": [[list(p), i] for p, i in self.merges.items()]})

    @classmethod
    def from_json(cls, s):
        return cls(json.loads(s)["merges"])
