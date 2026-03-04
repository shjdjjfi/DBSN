#!/usr/bin/env python3
"""Evaluate Graph-OT Calibration (GOTC) vs baselines.

Supports:
1) toy synthetic mode (fast sanity check)
2) real embedding mode from .npy features (recommended for paper-like comparisons)
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
    order_i2t = np.argsort(-scores, axis=1)
    pos_i2t = np.argmax(order_i2t == gt[:, None], axis=1)
    order_t2i = np.argsort(-scores, axis=0)
    pos_t2i = np.argmax(order_t2i == gt[None, :], axis=0)

    def at_k(pos: np.ndarray, k: int) -> float:
        return float(np.mean(pos < k) * 100.0)

    return (at_k(pos_i2t, 1), at_k(pos_i2t, 5), at_k(pos_i2t, 10), at_k(pos_t2i, 1), at_k(pos_t2i, 5), at_k(pos_t2i, 10))


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
    top1_targets = np.argmax(scores, axis=1)
    counts = np.bincount(top1_targets, minlength=scores.shape[1])
    return skewness(counts), gini(counts)


def evaluate(scores: np.ndarray) -> Metrics:
    r = compute_recall(scores)
    s, g = hubness_metrics(scores)
    return Metrics(*r, s, g)


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def make_synthetic_embeddings(n: int, dim: int, hub_ratio: float, noise_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_clusters = max(8, n // 25)
    centers = l2norm(rng.normal(size=(n_clusters, dim)))
    cid = rng.integers(0, n_clusters, size=n)
    pair_id = l2norm(rng.normal(size=(n, dim)))

    latent = 0.75 * centers[cid] + 0.25 * pair_id
    img = latent + 0.10 * rng.normal(size=(n, dim))
    txt = latent + 0.14 * rng.normal(size=(n, dim))

    n_hubs = max(1, int(n * hub_ratio))
    hub_dirs = l2norm(rng.normal(size=(max(3, n_hubs // 20), dim)))
    hub_idx = rng.choice(n, size=n_hubs, replace=False)
    for i, idx in enumerate(hub_idx):
        h = hub_dirs[i % hub_dirs.shape[0]]
        txt[idx] = 0.65 * h + 0.35 * txt[idx]

    m = int(n * noise_rate)
    if m > 1:
        noisy = rng.choice(n, size=m, replace=False)
        perm = noisy.copy()
        rng.shuffle(perm)
        txt[noisy] = txt[perm]

    return l2norm(img), l2norm(txt)


def load_embeddings(args: argparse.Namespace, noise_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if args.image_emb_path and args.text_emb_path:
        if not args.image_emb_path.exists():
            raise FileNotFoundError(f"image embedding file not found: {args.image_emb_path}")
        if not args.text_emb_path.exists():
            raise FileNotFoundError(f"text embedding file not found: {args.text_emb_path}")
        img = np.load(args.image_emb_path)
        txt = np.load(args.text_emb_path)
        if img.ndim != 2 or txt.ndim != 2:
            raise ValueError("Embedding arrays must be 2D [N, D].")
        n = min(img.shape[0], txt.shape[0])
        img = img[:n]
        txt = txt[:n]
        # keep same noise protocol for robustness if requested
        if noise_rate > 0:
            rng = np.random.default_rng(seed)
            m = int(n * noise_rate)
            if m > 1:
                idx = rng.choice(n, size=m, replace=False)
                perm = idx.copy()
                rng.shuffle(perm)
                txt[idx] = txt[perm]
        return l2norm(img.astype(np.float64)), l2norm(txt.astype(np.float64))

    return make_synthetic_embeddings(
        n=args.n,
        dim=args.dim,
        hub_ratio=args.hub_ratio,
        noise_rate=noise_rate,
        seed=seed,
    )


def run_once(args: argparse.Namespace, noise_rate: float, seed: int) -> dict[str, Metrics]:
    img, txt = load_embeddings(args, noise_rate=noise_rate, seed=seed)
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


def agg_metric_dict(items: list[Metrics]) -> Metrics:
    arr = np.array([[m.r1_i2t, m.r5_i2t, m.r10_i2t, m.r1_t2i, m.r5_t2i, m.r10_t2i, m.hub_skew_i2t, m.hub_gini_i2t] for m in items])
    mean = arr.mean(axis=0)
    return Metrics(*[float(x) for x in mean])


def format_metrics(name: str, m: Metrics) -> str:
    return (
        f"{name:9s} | "
        f"{m.r1_i2t:6.2f} {m.r5_i2t:6.2f} {m.r10_i2t:6.2f} | "
        f"{m.r1_t2i:6.2f} {m.r5_t2i:6.2f} {m.r10_t2i:6.2f} | "
        f"{m.hub_skew_i2t:7.3f} {m.hub_gini_i2t:7.3f}"
    )


def write_report(path: Path, args: argparse.Namespace, all_results: dict[float, dict[str, Metrics]]) -> None:
    mode = "real-embeddings" if (args.image_emb_path and args.text_emb_path) else "toy-synthetic"
    lines = ["# GOTC Report\n", "## Run setup\n", f"- mode: {mode}", f"- repeats: {args.repeats}", ""]
    if mode == "real-embeddings":
        lines += [f"- image_emb_path: {args.image_emb_path}", f"- text_emb_path: {args.text_emb_path}", ""]

    for nr, res in all_results.items():
        lines.append(f"## Noise rate = {nr}\n")
        lines.append("method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini")
        lines.append("--------- | -------------------- | -------------------- | ----------------")
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            lines.append(format_metrics(name, res[name]))
        lines.append("")

    lines.append("## Why fast?\n")
    lines.append("Toy mode does not run any large backbone model. It only post-processes a similarity matrix, so runtime is seconds.")
    lines.append("For paper-level comparability, use fixed real embeddings (e.g., CLIP features) via --image_emb_path/--text_emb_path.")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--hub_ratio", type=float, default=0.12)
    p.add_argument("--noise_rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    p.add_argument("--noise_rate", type=float, default=None, help="single noise value (overrides --noise_rates)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--outer_iters", type=int, default=2)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--prop_steps", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1, help="repeat runs with different seeds and report mean")
    p.add_argument("--image_emb_path", "--img_emb_path", dest="image_emb_path", type=Path, default=None)
    p.add_argument("--text_emb_path", "--txt_emb_path", dest="text_emb_path", type=Path, default=None)
    p.add_argument("--report_path", type=Path, default=Path("report_gotc.md"))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if (args.image_emb_path is None) ^ (args.text_emb_path is None):
        raise ValueError("Please provide both --image_emb_path and --text_emb_path, or neither.")

    noise_rates = [args.noise_rate] if args.noise_rate is not None else args.noise_rates
    all_results: dict[float, dict[str, Metrics]] = {}

    for nr in noise_rates:
        runs: dict[str, list[Metrics]] = {"baseline": [], "ot_only": [], "prop_only": [], "gotc": []}
        for rid in range(args.repeats):
            res = run_once(args, noise_rate=nr, seed=args.seed + rid)
            for k, v in res.items():
                runs[k].append(v)
        all_results[nr] = {k: agg_metric_dict(vs) for k, vs in runs.items()}

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
