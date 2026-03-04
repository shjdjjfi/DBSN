"""Graph-OT Calibration (GOTC) for cross-modal retrieval.

This module provides inference-time score post-processing that combines
(1) Sinkhorn OT-based calibration and (2) kNN graph propagation.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _to_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={arr.shape}")
    return arr


def sinkhorn_calibrate(
    S: np.ndarray,
    r: np.ndarray | None = None,
    c: np.ndarray | None = None,
    iters: int = 20,
    eps: float = 0.05,
) -> np.ndarray:
    """Apply Sinkhorn normalization over similarities.

    Args:
        S: Similarity matrix (n_query, n_target), larger = more similar.
        r: Row marginal (length n_query). Defaults to uniform.
        c: Column marginal (length n_target). Defaults to uniform.
        iters: Number of Sinkhorn iterations.
        eps: Entropic temperature, lower means sharper transport.

    Returns:
        Calibrated score matrix S_ot as a transport-like matrix.
    """
    S = _to_float_array(S)
    n, m = S.shape
    if eps <= 0:
        raise ValueError("eps must be positive")

    if r is None:
        r = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        r = np.asarray(r, dtype=np.float64)
        r = r / (r.sum() + 1e-12)

    if c is None:
        c = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        c = np.asarray(c, dtype=np.float64)
        c = c / (c.sum() + 1e-12)

    if r.shape[0] != n or c.shape[0] != m:
        raise ValueError(f"Marginals shape mismatch: r={r.shape}, c={c.shape}, S={S.shape}")

    S_shift = S - S.max(axis=1, keepdims=True)
    K = np.exp(S_shift / eps)
    K = np.maximum(K, 1e-20)

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)

    for _ in range(max(1, iters)):
        Kv = K @ v + 1e-20
        u = r / Kv
        KTu = K.T @ u + 1e-20
        v = c / KTu

    P = (u[:, None] * K) * v[None, :]
    return P


def knn_graph_from_scores(S_ot: np.ndarray, k: int = 20, sym: bool = True) -> sparse.csr_matrix:
    """Build a kNN graph over queries from score profiles.

    Each query is represented by its calibrated score distribution over targets.
    """
    X = _to_float_array(S_ot)
    n = X.shape[0]
    if n == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)
    if k <= 0:
        raise ValueError("k must be > 0")

    k = min(k, n - 1) if n > 1 else 0
    if k == 0:
        return sparse.identity(n, format="csr", dtype=np.float64)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)

    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row = np.repeat(np.arange(n), k)
    col = idx.reshape(-1)
    data = sim[np.repeat(np.arange(n), k), col]
    data = np.maximum(data, 0.0)

    A = sparse.csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float64)

    if sym:
        A = 0.5 * (A + A.T)

    A = A + sparse.identity(n, format="csr", dtype=np.float64)

    d = np.asarray(A.sum(axis=1)).ravel()
    d_inv = 1.0 / np.maximum(d, 1e-12)
    D_inv = sparse.diags(d_inv)
    A = D_inv @ A
    return A.tocsr()


def propagate_scores(
    S_ot: np.ndarray,
    A: sparse.csr_matrix,
    steps: int = 2,
    alpha: float = 0.9,
) -> np.ndarray:
    """Propagate scores over query graph."""
    S0 = _to_float_array(S_ot)
    if not sparse.isspmatrix_csr(A):
        A = sparse.csr_matrix(A)

    if A.shape[0] != S0.shape[0] or A.shape[1] != S0.shape[0]:
        raise ValueError(f"A shape {A.shape} incompatible with S {S0.shape}")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be in [0, 1]")

    S_t = S0.copy()
    for _ in range(max(1, steps)):
        S_t = alpha * (A @ S_t) + (1.0 - alpha) * S0
    return np.asarray(S_t)


def gotc(
    S: np.ndarray,
    k: int = 20,
    outer_iters: int = 2,
    sinkhorn_iters: int = 20,
    eps: float = 0.05,
    prop_steps: int = 2,
    alpha: float = 0.9,
    sym: bool = True,
) -> np.ndarray:
    """Alternate OT calibration and graph propagation."""
    S_t = _to_float_array(S)
    for _ in range(max(1, outer_iters)):
        S_ot = sinkhorn_calibrate(S_t, iters=sinkhorn_iters, eps=eps)
        A = knn_graph_from_scores(S_ot, k=k, sym=sym)
        S_t = propagate_scores(S_ot, A, steps=prop_steps, alpha=alpha)
    return S_t
