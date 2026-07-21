"""
reid/similarity.py

Cosine similarity helpers used by the Identity Manager.
Embeddings are assumed to already be L2-normalized (reid_model.py does this),
so cosine similarity reduces to a plain dot product -- but we don't assume
that blindly; we normalize defensively here too, since a caller could pass
in a raw feature vector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _safe_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D embeddings. Returns a scalar in [-1, 1]."""
    a = _safe_normalize(np.asarray(a, dtype=np.float32))
    b = _safe_normalize(np.asarray(b, dtype=np.float32))
    return float(np.dot(a, b))


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """
    query:   (D,) or (N, D)
    gallery: (M, D)
    returns: (M,) if query is 1-D, else (N, M)
    """
    query = np.asarray(query, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)

    if gallery.size == 0:
        return np.zeros((0,) if query.ndim == 1 else (query.shape[0], 0), dtype=np.float32)

    q = _safe_normalize(query, axis=-1)
    g = _safe_normalize(gallery, axis=-1)

    if q.ndim == 1:
        return g @ q  # (M,)
    return q @ g.T  # (N, M)


def best_match(
    query: np.ndarray,
    gallery_ids: list,
    gallery_embeddings: np.ndarray,
    threshold: float,
) -> "tuple[Optional[str], float]":
    """
    Returns (best_id, best_score). best_id is None if nothing clears `threshold`.
    """
    if len(gallery_ids) == 0:
        return None, -1.0

    scores = cosine_similarity_matrix(query, gallery_embeddings)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= threshold:
        return gallery_ids[best_idx], best_score
    return None, best_score
