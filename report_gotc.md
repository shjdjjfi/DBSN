# GOTC Report

## Run setup

- mode: toy-synthetic
- repeats: 2

## Noise rate = 0.0

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |  11.00  35.50  57.75 |  11.25  36.75  60.50 |   2.725   0.697
ot_only   |  12.00  40.00  61.25 |  11.50  39.25  61.75 |   0.246   0.339
prop_only |   5.25  22.75  43.50 |   9.75  35.00  57.75 |   5.040   0.947
gotc      |  11.50  34.50  56.25 |   8.75  39.00  59.50 |   3.853   0.647

## Noise rate = 0.1

method    | R@1    R@5    R@10   | R@1    R@5    R@10   | HubSkew  HubGini
--------- | -------------------- | -------------------- | ----------------
baseline  |  10.25  31.25  51.75 |  10.25  33.00  54.25 |   2.725   0.697
ot_only   |  10.50  35.75  54.75 |  10.00  35.75  55.75 |   0.246   0.339
prop_only |   4.50  20.50  39.25 |   8.75  31.00  51.75 |   5.040   0.947
gotc      |  10.75  31.00  50.00 |   7.50  35.25  53.50 |   3.853   0.647

## Why fast?

Toy mode does not run any large backbone model. It only post-processes a similarity matrix, so runtime is seconds.
For paper-level comparability, use fixed real embeddings (e.g., CLIP features) via --image_emb_path/--text_emb_path.