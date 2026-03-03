# GOTC Report

## Run setup

- mode: toy-synthetic
- repeats: 3

## Noise rate = 0.0

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   9.00  33.92  53.83 |   9.33  33.50  55.17 |   2.566   0.703
ot_only   |   9.25  35.83  54.83 |   9.08  35.33  56.17 |   0.332   0.387
prop_only |   4.83  22.67  42.50 |   8.83  31.50  53.75 |   5.115   0.956
gotc      |   6.00  26.67  46.75 |   7.83  31.08  55.83 |   2.907   0.806

## Noise rate = 0.1

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   8.67  30.92  49.42 |   8.50  31.25  50.08 |   2.566   0.703
ot_only   |   8.67  32.92  50.00 |   8.58  32.42  51.17 |   0.332   0.387
prop_only |   4.17  21.17  39.33 |   8.33  29.25  49.58 |   5.115   0.956
gotc      |   5.50  24.75  43.17 |   7.50  28.58  50.50 |   2.907   0.806

## Noise rate = 0.2

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   6.83  27.50  44.25 |   7.00  26.83  45.75 |   2.566   0.703
ot_only   |   7.08  28.67  45.25 |   7.00  28.33  46.25 |   0.332   0.387
prop_only |   3.58  18.00  34.42 |   6.67  24.92  43.75 |   5.115   0.956
gotc      |   4.75  21.08  38.00 |   5.92  24.58  45.00 |   2.907   0.806

## Noise rate = 0.4

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |   5.33  21.00  34.00 |   5.67  21.00  34.42 |   2.566   0.703
ot_only   |   5.50  22.50  34.92 |   5.67  21.75  35.33 |   0.332   0.387
prop_only |   3.00  13.92  26.58 |   5.08  19.92  33.67 |   5.115   0.956
gotc      |   3.58  17.00  28.67 |   4.75  18.92  34.17 |   2.907   0.806

## Why fast?

Toy mode does not run any large backbone model. It only post-processes a similarity matrix, so runtime is seconds.
For paper-level comparability, use fixed real embeddings (e.g., CLIP features) via --image_emb_path/--text_emb_path.