"""Embedding backends. All return L2-normalised float32 vectors, so cosine = dot product."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from mnemo.config import Settings


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    """Local ONNX embeddings on CPU. No API key, no per-token cost."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.dim = int(self._model.embedding_size)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        # BGE-style models use an instruction prefix for queries; fastembed applies it here.
        return _normalise(np.stack(list(self._model.query_embed([text]))))[0]


class OpenAICompatEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        base_url, api_key, _ = settings.resolved_llm()
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = settings.embedding_model
        self.dim = len(self.embed_query("dimension probe"))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(self, "dim", 0)), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=texts)
        return _normalise(np.array([d.embedding for d in response.data]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    No model download and fully reproducible, which makes it ideal for unit tests and CI.
    Texts that share words get similar vectors, which is enough to test ranking logic.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] % 2 else -1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model)
    if settings.embedder == "openai":
        return OpenAICompatEmbedder(settings)
    return HashEmbedder()
