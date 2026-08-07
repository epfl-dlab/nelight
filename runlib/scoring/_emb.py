"""NumPy helpers matching the paper's torch F.cosine_similarity usage."""

from __future__ import annotations

import numpy as np


def as_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_vec(x, axis=None):
    a = as_np(x)
    if axis is None:
        return a
    return a.mean(axis=axis)


def cosine(a, b, eps: float = 1e-8) -> float:
    """Match ``F.cosine_similarity(a, b, dim=1, eps=eps)[0]``.

    Torch broadcasts leading dims; for NCSE, sentence emb is often ``(n, 768)``
    against entity ``(1, 768)``, and the paper code kept only index 0.
    """
    a = as_np(a).astype(np.float64, copy=False)
    b = as_np(b).astype(np.float64, copy=False)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a2, b2 = np.broadcast_arrays(a, b)
    # cosine along feature axis (last)
    dots = (a2 * b2).sum(axis=-1)
    na = np.linalg.norm(a2, axis=-1)
    nb = np.linalg.norm(b2, axis=-1)
    out = dots / np.maximum(na * nb, eps)
    return float(np.asarray(out).reshape(-1)[0])


def stack_mean(arrays) -> np.ndarray:
    """Mean of per-item mean(axis=1) vectors — ``torch.cat(...).mean(axis=0)``."""
    parts = []
    for x in arrays:
        m = mean_vec(x, axis=1)
        if m.ndim == 1:
            m = m.reshape(1, -1)
        parts.append(m)
    return np.concatenate(parts, axis=0).mean(axis=0)
