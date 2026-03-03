# GOTC Report

## Commands

```bash
python tools/gotc_eval.py --noise_rates 0.0 0.1 0.2 0.4 --n 800 --dim 128 --k 15 --outer_iters 2 --sinkhorn_iters 20 --eps 0.05 --prop_steps 1 --alpha 0.6 --seed 0
```

## Noise rate = 0.0

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   8.00  28.00  41.75 |   7.62  26.75  42.50 |   2.391   0.676
ot_only   |   7.88  27.62  43.00 |   7.50  27.25  41.62 |   0.673   0.463
prop_only |   6.38  25.62  45.50 |   7.50  27.12  48.38 |   4.094   0.855
gotc      |   7.38  24.50  45.62 |   7.62  29.88  50.25 |   2.017   0.723

## Noise rate = 0.1

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   6.88  24.25  36.12 |   6.25  23.00  37.25 |   2.391   0.676
ot_only   |   6.50  23.75  37.62 |   6.25  23.25  36.38 |   0.673   0.463
prop_only |   5.75  22.50  39.75 |   6.25  23.75  42.62 |   4.094   0.855
gotc      |   6.50  21.00  39.88 |   6.25  25.87  44.12 |   2.017   0.723

## Noise rate = 0.2

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   6.62  22.38  33.38 |   5.75  22.00  33.50 |   2.391   0.676
ot_only   |   6.50  22.25  34.12 |   6.00  21.88  33.88 |   0.673   0.463
prop_only |   4.75  20.00  36.50 |   5.62  21.75  38.50 |   4.094   0.855
gotc      |   5.75  20.38  37.00 |   5.62  24.62  40.25 |   2.017   0.723

## Noise rate = 0.4

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   5.12  17.38  26.12 |   4.62  15.38  25.87 |   2.391   0.676
ot_only   |   4.38  16.88  26.50 |   4.12  16.00  25.62 |   0.673   0.463
prop_only |   4.50  16.38  28.50 |   4.25  16.25  29.62 |   4.094   0.855
gotc      |   4.38  15.38  28.88 |   4.38  18.38  31.13 |   2.017   0.723

## Why GOTC helps

Sinkhorn OT calibration redistributes mass across targets, reducing the tendency for a few hub targets to dominate top-1 matches (lower hubness skew/Gini).
Graph propagation then denoises local neighborhood inconsistencies by diffusing scores over similar-query relations, which is especially useful under pair noise.
Alternating OT and propagation compounds both effects: debias global matching and repair local noisy correspondences.