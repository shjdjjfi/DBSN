#!/usr/bin/env python3
"""Evaluate Graph-OT Calibration (GOTC) for cross-modal retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from gotc.gotc import gotc, knn_graph_from_scores, propagate_scores, sinkhorn_calibrate


def _as_tensor(x):
    if hasattr(x, "detach"):
        return x
    for key in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(x, key):
            v = getattr(x, key)
            if key == "last_hidden_state":
                v = v[:, 0, :]
            return v
    raise TypeError(f"Unsupported feature output type: {type(x)}")


def recall_at_k_index(S: np.ndarray, gt: np.ndarray, ks=(1, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        topk = order[:, : min(k, order.shape[1])]
        hit = (topk == gt[:, None]).any(axis=1).mean()
        out[f"R@{k}"] = float(hit)
    return out


def recall_at_k_class(S: np.ndarray, src_labels: np.ndarray, tgt_labels: np.ndarray, ks=(1, 5, 10)):
    order = np.argsort(-S, axis=1)
    out = {}
    src_labels = np.asarray(src_labels)
    tgt_labels = np.asarray(tgt_labels)
    for k in ks:
        topk = order[:, : min(k, order.shape[1])]
        pred_labels = tgt_labels[topk]
        hit = (pred_labels == src_labels[:, None]).any(axis=1).mean()
        out[f"R@{k}"] = float(hit)
    return out


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(np.maximum(x, 0))
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std() + 1e-12
    return float(np.mean(((x - mu) / sd) ** 3))


def hubness_metrics(S: np.ndarray):
    top1 = np.argmax(S, axis=1)
    cnt = np.bincount(top1, minlength=S.shape[1]).astype(np.float64)
    return {
        "top1_gini": gini(cnt),
        "top1_skewness": skewness(cnt),
        "top1_max_count": float(cnt.max()),
    }


def make_toy_embeddings(n=1000, d=256, hub_ratio=0.03, seed=0):
    rng = np.random.default_rng(seed)
    n_cluster = max(10, n // 40)

    centers = rng.normal(size=(n_cluster, d))
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    cid = rng.integers(0, n_cluster, size=n)

    latent = centers[cid] + 0.1 * rng.normal(size=(n, d))
    latent = latent / (np.linalg.norm(latent, axis=1, keepdims=True) + 1e-12)

    img = latent + 0.12 * rng.normal(size=(n, d))
    txt = latent + 0.18 * rng.normal(size=(n, d))

    n_hubs = max(1, int(n * hub_ratio))
    hub_proto = centers[rng.choice(n_cluster, size=n_hubs, replace=True)]
    for j in range(n_hubs):
        idx = rng.choice(n, size=max(8, n // (3 * n_hubs)), replace=False)
        txt[idx] = 0.55 * txt[idx] + 0.45 * hub_proto[j]

    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-12)
    txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-12)
    return img, txt


def load_clip_cifar10(n_samples=1000, batch_size=64, device="cpu"):
    import torch
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import functional as TF
    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    ds = CIFAR10(root=".cache", train=False, download=True)
    n = min(n_samples, len(ds))

    imgs = []
    texts = []
    labels = []
    for i in range(n):
        pil, label = ds[i]
        cls = ds.classes[label]
        texts.append(f"a photo of a {cls}")
        imgs.append(TF.to_pil_image(np.array(pil)))
        labels.append(int(label))

    with torch.no_grad():
        image_features = []
        text_features = []
        for i in range(0, n, batch_size):
            bi = imgs[i : i + batch_size]
            bt = texts[i : i + batch_size]

            img_inputs = processor(images=bi, return_tensors="pt", padding=True).to(device)
            txt_inputs = processor(text=bt, return_tensors="pt", padding=True, truncation=True).to(device)

            img_f = model.get_image_features(**img_inputs)
            txt_f = model.get_text_features(**txt_inputs)

            img_f = _as_tensor(img_f)
            txt_f = _as_tensor(txt_f)

            image_features.append(np.asarray(img_f.detach().cpu().numpy()))
            text_features.append(np.asarray(txt_f.detach().cpu().numpy()))

    img = np.concatenate(image_features, axis=0)
    txt = np.concatenate(text_features, axis=0)
    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-12)
    txt = txt / (np.linalg.norm(txt, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(labels, dtype=np.int64)
    return img, txt, labels


def inject_pair_noise(txt_emb: np.ndarray, noise_rate: float, seed: int = 0, tgt_labels: np.ndarray | None = None):
    n = txt_emb.shape[0]
    gt = np.arange(n)
    labels_out = None if tgt_labels is None else np.asarray(tgt_labels).copy()
    if noise_rate <= 0:
        return txt_emb, gt, labels_out

    rng = np.random.default_rng(seed)
    m = int(n * noise_rate)
    noisy_ids = rng.choice(n, size=m, replace=False)
    shuffled = noisy_ids.copy()
    rng.shuffle(shuffled)

    txt_noisy = txt_emb.copy()
    txt_noisy[noisy_ids] = txt_emb[shuffled]

    gt_noisy = gt.copy()
    gt_noisy[noisy_ids] = shuffled

    if labels_out is not None:
        labels_out[noisy_ids] = labels_out[shuffled]

    return txt_noisy, gt_noisy, labels_out


def evaluate_once(
    S: np.ndarray,
    gt_i2t: np.ndarray,
    gt_t2i: np.ndarray,
    args,
    i2t_labels: tuple[np.ndarray, np.ndarray] | None = None,
    t2i_labels: tuple[np.ndarray, np.ndarray] | None = None,
):
    results = {}

    if i2t_labels is None:
        i2t_eval = lambda M: recall_at_k_index(M, gt_i2t)
    else:
        i2t_src, i2t_tgt = i2t_labels
        i2t_eval = lambda M: recall_at_k_class(M, i2t_src, i2t_tgt)

    if t2i_labels is None:
        t2i_eval = lambda M: recall_at_k_index(M.T, gt_t2i)
    else:
        t2i_src, t2i_tgt = t2i_labels
        t2i_eval = lambda M: recall_at_k_class(M.T, t2i_src, t2i_tgt)

    S_base = S
    results["baseline"] = {
        **{f"i2t_{k}": v for k, v in i2t_eval(S_base).items()},
        **{f"t2i_{k}": v for k, v in t2i_eval(S_base).items()},
        **hubness_metrics(S_base),
    }

    S_ot = sinkhorn_calibrate(S_base, iters=args.sinkhorn_iters, eps=args.eps)
    results["ot_only"] = {
        **{f"i2t_{k}": v for k, v in i2t_eval(S_ot).items()},
        **{f"t2i_{k}": v for k, v in t2i_eval(S_ot).items()},
        **hubness_metrics(S_ot),
    }

    A_base = knn_graph_from_scores(S_base, k=args.k, sym=True)
    S_prop = propagate_scores(S_base, A_base, steps=args.prop_steps, alpha=args.alpha)
    results["prop_only"] = {
        **{f"i2t_{k}": v for k, v in i2t_eval(S_prop).items()},
        **{f"t2i_{k}": v for k, v in t2i_eval(S_prop).items()},
        **hubness_metrics(S_prop),
    }

    S_gotc = gotc(
        S_base,
        k=args.k,
        outer_iters=args.outer_iters,
        sinkhorn_iters=args.sinkhorn_iters,
        eps=args.eps,
        prop_steps=args.prop_steps,
        alpha=args.alpha,
    )
    results["gotc"] = {
        **{f"i2t_{k}": v for k, v in i2t_eval(S_gotc).items()},
        **{f"t2i_{k}": v for k, v in t2i_eval(S_gotc).items()},
        **hubness_metrics(S_gotc),
    }
    return results


def format_markdown_table(all_res):
    header = "| noise | method | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | gini | skew | max_top1 |"
    sep = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for noise, methods in all_res.items():
        for method, m in methods.items():
            rows.append(
                "| {noise:.1f} | {method} | {i2t_R1:.4f} | {i2t_R5:.4f} | {i2t_R10:.4f} | {t2i_R1:.4f} | {t2i_R5:.4f} | {t2i_R10:.4f} | {gini:.4f} | {skew:.4f} | {mx:.1f} |".format(
                    noise=noise,
                    method=method,
                    i2t_R1=m["i2t_R@1"],
                    i2t_R5=m["i2t_R@5"],
                    i2t_R10=m["i2t_R@10"],
                    t2i_R1=m["t2i_R@1"],
                    t2i_R5=m["t2i_R@5"],
                    t2i_R10=m["t2i_R@10"],
                    gini=m["top1_gini"],
                    skew=m["top1_skewness"],
                    mx=m["top1_max_count"],
                )
            )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["toy", "clip-cifar10"], default="toy")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--outer_iters", type=int, default=2)
    parser.add_argument("--sinkhorn_iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--prop_steps", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--noise_rate", type=float, default=None)
    parser.add_argument("--noise_rates", type=str, default="0,0.1,0.2,0.4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--clip_eval_level",
        choices=["index", "class"],
        default="class",
        help="For clip-cifar10 mode: index-level one-to-one or class-level retrieval hit.",
    )
    parser.add_argument("--out_json", type=str, default="gotc_results.json")
    parser.add_argument("--out_md", type=str, default="gotc_results.md")
    args = parser.parse_args()

    img_labels = None
    txt_labels = None
    if args.mode == "toy":
        img, txt = make_toy_embeddings(n=args.n_samples, d=args.dim, seed=args.seed)
    else:
        img, txt, labels = load_clip_cifar10(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            device=args.device,
        )
        img_labels = labels.copy()
        txt_labels = labels.copy()

    noise_rates = [args.noise_rate] if args.noise_rate is not None else [float(x) for x in args.noise_rates.split(",")]
    all_res = {}
    for nr in noise_rates:
        txt_noisy, gt_i2t, txt_labels_noisy = inject_pair_noise(
            txt,
            nr,
            seed=args.seed,
            tgt_labels=txt_labels,
        )
        S = img @ txt_noisy.T

        gt_t2i = np.empty_like(gt_i2t)
        gt_t2i[gt_i2t] = np.arange(len(gt_i2t))

        i2t_labels = None
        t2i_labels = None
        if args.mode == "clip-cifar10" and args.clip_eval_level == "class":
            i2t_labels = (img_labels, txt_labels_noisy)
            t2i_labels = (txt_labels_noisy, img_labels)

        all_res[float(nr)] = evaluate_once(
            S,
            gt_i2t,
            gt_t2i,
            args,
            i2t_labels=i2t_labels,
            t2i_labels=t2i_labels,
        )

    payload = {
        "metadata": {
            "mode": args.mode,
            "clip_eval_level": args.clip_eval_level if args.mode == "clip-cifar10" else "index",
            "noise_rates": noise_rates,
        },
        "results": all_res,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = format_markdown_table(all_res)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")

    print(table)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
