"""Graph-OT Calibration (GOTC) for cross-modal retrieval.

This module combines Sinkhorn-style OT calibration and graph propagation
into an inference-time operator.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def sinkhorn_calibrate(
    S: np.ndarray,
    r: np.ndarray | None = None,
    c: np.ndarray | None = None,
    iters: int = 20,
    eps: float = 0.05,
) -> np.ndarray:
    """Calibrate a similarity matrix with entropy-regularized OT (Sinkhorn).

    Args:
        S: Similarity matrix of shape (n_queries, n_targets), larger is better.
        r: Source/query marginal distribution, shape (n_queries,).
        c: Target marginal distribution, shape (n_targets,).
        iters: Sinkhorn iterations.
        eps: Entropic regularization temperature.

    Returns:
        S_ot: Calibrated score matrix in the same shape as ``S``.
    """
    if S.ndim != 2:
        raise ValueError(f"S must be 2D, got shape={S.shape}")
    nq, nt = S.shape
    if r is None:
        r = np.full(nq, 1.0 / nq, dtype=np.float64)
    if c is None:
        c = np.full(nt, 1.0 / nt, dtype=np.float64)

    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if r.shape != (nq,):
        raise ValueError(f"r must have shape ({nq},), got {r.shape}")
    if c.shape != (nt,):
        raise ValueError(f"c must have shape ({nt},), got {c.shape}")
    if np.any(r <= 0) or np.any(c <= 0):
        raise ValueError("r and c must be strictly positive")

    r = r / r.sum()
    c = c / c.sum()

    # log-sum-exp style stabilization by shifting per-matrix max.
    S64 = np.asarray(S, dtype=np.float64)
    S_shift = S64 - np.max(S64)
    K = np.exp(S_shift / max(eps, 1e-8))
    K = np.maximum(K, 1e-300)

    u = np.ones(nq, dtype=np.float64)
    v = np.ones(nt, dtype=np.float64)
    for _ in range(iters):
        Kv = K @ v
        Kv = np.maximum(Kv, 1e-300)
        u = r / Kv

        KTu = K.T @ u
        KTu = np.maximum(KTu, 1e-300)
        v = c / KTu

    # Dual hubness-like offsets (following DBSN Sinkhorn calibration intuition).
    q_bias = -eps * np.log(np.maximum(u, 1e-300))
    t_bias = -eps * np.log(np.maximum(v, 1e-300))

    S_ot = S64 - q_bias[:, None] - t_bias[None, :]
    return S_ot


def knn_graph_from_scores(S_ot: np.ndarray, k: int = 20, sym: bool = True) -> sparse.csr_matrix:
    """Build a kNN graph among queries from score similarity patterns.

    We connect query i to query j if their calibrated score vectors are similar
    (dot-product in score space).

    Args:
        S_ot: Calibrated score matrix (n_queries, n_targets).
        k: Number of nearest neighbors per query.
        sym: If True, symmetrize graph with max(A, A.T).

    Returns:
        CSR adjacency matrix A of shape (n_queries, n_queries), row-normalized.
    """
    if S_ot.ndim != 2:
        raise ValueError(f"S_ot must be 2D, got shape={S_ot.shape}")

    nq = S_ot.shape[0]
    if nq == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    k_eff = min(max(1, k), max(1, nq - 1))

    # Query-query affinity from score-space similarities.
    X = np.asarray(S_ot, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    QQ = X @ X.T
    np.fill_diagonal(QQ, -np.inf)

    rows, cols, vals = [], [], []
    for i in range(nq):
        idx = np.argpartition(QQ[i], -k_eff)[-k_eff:]
        sim = QQ[i, idx]
        # positive edge weights
        w = np.exp(sim - np.max(sim))
        w = w / np.maximum(w.sum(), 1e-12)
        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
        vals.extend(w.tolist())

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(nq, nq), dtype=np.float64)

    if sym:
        A = A.maximum(A.T)

    row_sum = np.asarray(A.sum(axis=1)).ravel()
    inv = np.reciprocal(np.maximum(row_sum, 1e-12))
    A = sparse.diags(inv) @ A
    return A.tocsr()


def propagate_scores(
    S_ot: np.ndarray,
    A: sparse.csr_matrix,
    steps: int = 2,
    alpha: float = 0.9,
) -> np.ndarray:
    """Diffuse scores over a query graph.

    Iteration: S_{t+1} = alpha * A * S_t + (1-alpha) * S_ot.
    """
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()

    S_base = np.asarray(S_ot, dtype=np.float64)
    S_t = S_base.copy()
    for _ in range(steps):
        S_t = alpha * (A @ S_t) + (1.0 - alpha) * S_base
    return S_t


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
    S_t = np.asarray(S, dtype=np.float64)
    for _ in range(outer_iters):
        S_ot = sinkhorn_calibrate(S_t, iters=sinkhorn_iters, eps=eps)
        A = knn_graph_from_scores(S_ot, k=k, sym=sym)
        S_t = propagate_scores(S_ot, A, steps=prop_steps, alpha=alpha)
    return S_t
