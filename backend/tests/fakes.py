"""Test doubles: a deterministic embedder and a scripted chat model."""

import hashlib
import math
import re

DIM = 64


def fake_vector(text: str, dim: int = DIM) -> list[float]:
    """Hashed bag-of-words: texts that share words get similar vectors, deterministically."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    if not any(vec):
        vec[0] = 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


class FakeEmbedder:
    def __init__(self, dim: int = DIM, model: str = "fake-embedding") -> None:
        self.dim = dim
        self.model = model
        self.calls: list[list[str]] = []
        self.fail_with: Exception | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        self.calls.append(list(texts))
        return [fake_vector(t, self.dim) for t in texts]
