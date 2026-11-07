# Pooled hole speed (`HMP50`)

This folder builds the Task 1 speed inside the 90-minute gaps after the Task 2 windows. Each speed
is a 50/50 mix of two parts:

- the corrected fill (`T1`);
- a two-state speed: free-flow speed and queued speed, weighted by a queue probability.

That queue probability is first averaged over links that usually queue together (`HMP50`).
`serve_candidate.py` runs in `pipeline/task1.sh` step 7. It writes
`serving/official_state_updates.parquet`.

## Files kept byte-identical

`frozen_recipe.json` records the SHA256 of `common.py`, `structure.py`, `full_pool.py`,
`feb_pool.py` and `strength.py`. `serve_candidate.py` checks these five hashes before serving.
For this reason the five files have no added comments; this table describes them.

| File | Role in the reproduction | What it does |
|---|---|---|
| `common.py` | Imported by every file here | `load` reads the gap rows of one split: 18 five-minute steps after each window, with the earlier one-step (`T1`) and 25 % mixture (`HM25`) predictions. `strict` replaces the split statistics with those of the other day parity (odd or even days). `evaluate` scores a speed prediction against `HM25`: RMSE and relative MSE per subset, plus a 95 % bootstrap interval over panel-days. |
| `structure.py` | `topology` is called by `full_pool.py` | `topology` clusters the links of one corridor by how often they queue together. It uses the released speeds of days with one parity. A link counts as queued when its speed is at most 60 % of free-flow speed. The distance between two links is 1 minus the Jaccard overlap of their queued cells, and only pairs with at least 20 shared observations get a measured distance. Clusters come from complete linkage cut at 0.05. `run` holds the pilot comparisons (isotonic smoothing, pooling, oracle). |
| `full_pool.py` | Runs in `pipeline/task1.sh` step 7, once per split | Rebuilds the gap-model features for every eligible gap cell and checks them against the stored rows (difference below 2e-4). It predicts queue probability, free-flow speed and queued speed with the hole-mixture models, each fitted on the other day parity. It averages the queue probability over the links of each cluster at the same window and step. Output: `full_pool/<split>_predictions.parquet`. Column `adaptive` = 0.75 × `T1` + 0.25 × pooled mixture. |
| `feb_pool.py` | `cluster` is imported by `serve_candidate.py` | `cluster` applies the clustering rule of `structure.topology` to a given speed array. `run` is the February pseudo-gap check of the same models. |
| `strength.py` | Record | Compares the 25 % mixture, the 50 % mixture and the unpooled 50 % mixture on validation gaps and February pseudo-gaps. It fixed the 0.5 weight that `frozen_recipe.json` stores. |

`boundary_context.py` and `serve_candidate.py` carry their own comments.
