#!/usr/bin/env python3
"""Evaluate Graph-OT Calibration (GOTC) vs baselines.

Default mode uses synthetic embeddings that induce hubness and optional pair noise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gotc.gotc import gotc, knn_graph_from_scores, propagate_scores, sinkhorn_calibrate


@dataclass
class Metrics:
    r1_i2t: float
    r5_i2t: float
    r10_i2t: float
    r1_t2i: float
    r5_t2i: float
    r10_t2i: float
    hub_skew_i2t: float
    hub_gini_i2t: float


def compute_recall(scores: np.ndarray) -> tuple[float, float, float, float, float, float]:
    n = scores.shape[0]
    gt = np.arange(n)

    # Image/query -> text/target
    order_i2t = np.argsort(-scores, axis=1)
    pos_i2t = np.argmax(order_i2t == gt[:, None], axis=1)

    # Text/target -> image/query
    order_t2i = np.argsort(-scores, axis=0)
    pos_t2i = np.argmax(order_t2i == gt[None, :], axis=0)

    def at_k(pos: np.ndarray, k: int) -> float:
        return float(np.mean(pos < k) * 100.0)

    return (
        at_k(pos_i2t, 1),
        at_k(pos_i2t, 5),
        at_k(pos_i2t, 10),
        at_k(pos_t2i, 1),
        at_k(pos_t2i, 5),
        at_k(pos_t2i, 10),
    )


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0)
    if x.sum() <= 1e-12:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def hubness_metrics(scores: np.ndarray) -> tuple[float, float]:
    # How often each target appears as top-1 for queries.
    top1_targets = np.argmax(scores, axis=1)
    counts = np.bincount(top1_targets, minlength=scores.shape[1])
    return skewness(counts), gini(counts)


def evaluate(scores: np.ndarray) -> Metrics:
    r = compute_recall(scores)
    s, g = hubness_metrics(scores)
    return Metrics(*r, s, g)


def make_synthetic_embeddings(
    n: int = 1000,
    dim: int = 256,
    hub_ratio: float = 0.1,
    noise_rate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic cross-modal embeddings with controllable hubness + noise."""
    rng = np.random.default_rng(seed)

    n_clusters = max(8, n // 25)
    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    cid = rng.integers(0, n_clusters, size=n)

    pair_id = rng.normal(size=(n, dim))
    pair_id /= np.linalg.norm(pair_id, axis=1, keepdims=True) + 1e-12

    latent = 0.75 * centers[cid] + 0.25 * pair_id
    img = latent + 0.10 * rng.normal(size=(n, dim))
    txt = latent + 0.14 * rng.normal(size=(n, dim))

    # Hubs: collapse some targets towards a few shared directions.
    n_hubs = max(1, int(n * hub_ratio))
    hub_dirs = rng.normal(size=(max(3, n_hubs // 20), dim))
    hub_dirs /= np.linalg.norm(hub_dirs, axis=1, keepdims=True) + 1e-12
    hub_idx = rng.choice(n, size=n_hubs, replace=False)
    for i, idx in enumerate(hub_idx):
        h = hub_dirs[i % hub_dirs.shape[0]]
        txt[idx] = 0.65 * h + 0.35 * txt[idx]

    # Pair noise: shuffle a fraction of targets across identities.
    m = int(n * noise_rate)
    if m > 1:
        noisy = rng.choice(n, size=m, replace=False)
        perm = noisy.copy()
        rng.shuffle(perm)
        txt[noisy] = txt[perm]

    img /= np.linalg.norm(img, axis=1, keepdims=True) + 1e-12
    txt /= np.linalg.norm(txt, axis=1, keepdims=True) + 1e-12
    return img, txt


def format_metrics(name: str, m: Metrics) -> str:
    return (
        f"{name:9s} | "
        f"{m.r1_i2t:6.2f} {m.r5_i2t:6.2f} {m.r10_i2t:6.2f} | "
        f"{m.r1_t2i:6.2f} {m.r5_t2i:6.2f} {m.r10_t2i:6.2f} | "
        f"{m.hub_skew_i2t:7.3f} {m.hub_gini_i2t:7.3f}"
    )


def run_once(args: argparse.Namespace, noise_rate: float) -> dict[str, Metrics]:
    img, txt = make_synthetic_embeddings(
        n=args.n,
        dim=args.dim,
        hub_ratio=args.hub_ratio,
        noise_rate=noise_rate,
        seed=args.seed,
    )
    S = img @ txt.T

    S_ot = sinkhorn_calibrate(S, iters=args.sinkhorn_iters, eps=args.eps)

    A_base = knn_graph_from_scores(S, k=args.k, sym=True)
    S_prop = propagate_scores(S, A_base, steps=args.prop_steps, alpha=args.alpha)

    S_gotc = gotc(
        S,
        k=args.k,
        outer_iters=args.outer_iters,
        sinkhorn_iters=args.sinkhorn_iters,
        eps=args.eps,
        prop_steps=args.prop_steps,
        alpha=args.alpha,
        sym=True,
    )

    return {
        "baseline": evaluate(S),
        "ot_only": evaluate(S_ot),
        "prop_only": evaluate(S_prop),
        "gotc": evaluate(S_gotc),
    }


def write_report(path: Path, args: argparse.Namespace, all_results: dict[float, dict[str, Metrics]]) -> None:
    lines = []
    lines.append("# GOTC Report\n")
    lines.append("## Commands\n")
    lines.append("```bash")
    lines.append(
        "python tools/gotc_eval.py "
        f"--noise_rates {' '.join(map(str, args.noise_rates))} --n {args.n} --dim {args.dim} "
        f"--k {args.k} --outer_iters {args.outer_iters} --sinkhorn_iters {args.sinkhorn_iters} "
        f"--eps {args.eps} --prop_steps {args.prop_steps} --alpha {args.alpha} --seed {args.seed}"
    )
    lines.append("```\n")

    for nr, res in all_results.items():
        lines.append(f"## Noise rate = {nr}\n")
        lines.append("method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini")
        lines.append("--------- | -------------------- | -------------------- | ----------------")
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            lines.append(format_metrics(name, res[name]))
        lines.append("")

    lines.append("## Why GOTC helps\n")
    lines.append(
        "Sinkhorn OT calibration redistributes mass across targets, reducing the tendency "
        "for a few hub targets to dominate top-1 matches (lower hubness skew/Gini)."
    )
    lines.append(
        "Graph propagation then denoises local neighborhood inconsistencies by diffusing "
        "scores over similar-query relations, which is especially useful under pair noise."
    )
    lines.append(
        "Alternating OT and propagation compounds both effects: debias global matching and "
        "repair local noisy correspondences."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hub_ratio", type=float, default=0.12)
    p.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--outer_iters", type=int, default=2)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--prop_steps", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report_path", type=Path, default=Path("report_gotc.md"))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    all_results: dict[float, dict[str, Metrics]] = {}

    for nr in args.noise_rates:
        all_results[nr] = run_once(args, noise_rate=nr)

    for nr, res in all_results.items():
        print(f"\n=== noise_rate={nr} ===")
        print("method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini")
        print("--------- | -------------------- | -------------------- | ----------------")
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            print(format_metrics(name, res[name]))

    write_report(args.report_path, args, all_results)
    print(f"\nWrote report to {args.report_path}")


if __name__ == "__main__":
    main()
