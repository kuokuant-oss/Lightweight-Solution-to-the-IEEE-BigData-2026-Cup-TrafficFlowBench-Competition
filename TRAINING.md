# Training

Reproducing the selected submission needs no training. Every trained model and every train-derived
table that the run reads is shipped frozen under `trafficflowbench/`, at the path where the original
programs expect it, and `checksums.txt` lists their SHA256. `run_reproduce.sh` only loads them. A few
Task 1 heads are refitted inside the run on each month's own released observations (see "Fitted during
the run"); they are part of the Task 1 method, not pre-trained models.

This file records how each frozen model was trained: data, labels, features, parameters, seeds, tree
counts and cost, and the program that trained it. Trainer paths are relative to the root of the
development repository, whose layout the package's `trafficflowbench/` folder mirrors. Trainers marked
"in package" ship under `trafficflowbench/` because the serving code imports them. Trainers marked
"in `training/`" are copied under `training/` at the same relative path, unchanged except for added
comments (for example `training/scripts/34_task1_blend.py`). Programs marked "not in package" built
intermediate tables only. The trainers are the last stages of a longer pipeline and read intermediate
tables that are not distributed (listed at the end), so they do not retrain from the raw release on
their own.

## Conventions

- **Release split.** Train = 2030-06-01 … 2031-02-28 (273 days, with ground truth). Validation = March
  2031 (public leaderboard). Private = April 2031.
- **Train split only.** Every shipped model was fitted on train-split days only. Several use February
  2031, which is part of the train split; each recipe says so.
- **Development folds inside train** (`data/processed/splits/day_split.csv`, shipped): "dev" =
  2030-06-01 … 2031-01-03, "val fold" = 2031-01-04 … 2031-02-28.
- **Units.** Speed km/h, flow veh/h (all lanes unless "per lane"), density veh/km.
- **Queued cell.** Speed at or below the link's queue cut: the released `v_cut` where given, else
  0.6 × free-flow speed, on score-eligible cells (`src/tfb/score2.py`).
- **Replicated windows.** The organiser's window selector re-run on train days: onset and ongoing
  windows at the official 360-minute spacing. Several hole and Task 2 models train on them.
- **Hole.** The 90 minutes after a Task 2 origin T (6-step horizon plus 60 minutes) that the release
  blanks on the whole corridor; its cells are Task 1 targets.
- **Libraries.** LightGBM 4.7.0; CatBoost 1.2.10 (the version is written inside the `.cbm` files);
  SciPy 1.18.1; NumPy 2.5.3; pandas 2.3.3; Python 3.12.14 (training stamps and
  `build/review_20260930_round20/ENVIRONMENT_SNAPSHOT.json`). scikit-learn for the site models only.
- **Parameters.** LightGBM parameters not listed are defaults. "Default seeds" means `seed` was not set
  (model header `seed: 0`). Every LightGBM model has `deterministic=0`; thread counts are from the
  model headers.

## Short names

| Name | Plain meaning |
|---|---|
| C1 | Base Task 1 state: the v3 stack outside holes, the G2 hole model inside them. |
| v3 stack | LightGBM that combines five classical fills of a blank cell with its observed neighbourhood. |
| k_aux, k* | k_aux: a model of link density. k*: the projection that pulls each cell's flow/speed ratio 10% toward k_aux and re-solves speed and flow. |
| G2, G2v2 (v2_F) | Hole ("gap") models, first and second version. v2_F is the G2v2 fit with arrangement F (fit Jun–Dec 2030, early stopping Jan 2031, test Feb 2031). |
| my_blocks | Co-switching blocks: runs of adjacent links that switch into queue together. |
| RLB, RL | Residual LightGBM on top of C1; RLB adds co-switching block features, RL is the version without them. |
| DK50 | Density (k) correction head, applied at projection strength 0.50. |
| CAP | CatBoost second residual learner (from the model-capacity study), blended 50% with RLB. |
| D12 stage-2 (S2_100) | Small second residual on the D12_I405 pair, served with its first 100 trees. |
| T1 (FS) | Month-specific hole speed correction, fitted on pseudo holes of the same month. |
| HM25 | Two-state (free / queued) hole speed, blended 25% with T1. |
| HMP50 | HM25 with the queue probability averaged over co-switching clusters, blended 50% with T1. |
| PRP, PR head, c_Jan | PRP: final hole speed. PR head: queue probability that reads on-ramp counts inside the hole, pooled over clusters. c_Jan: January-fitted hole speed correction with ramp counts. |
| J1, SG1 | J1: density head fitted on the CAP state. SG1: J1 applied only where speed < 0.85 × free-flow speed. |
| 0922 head | Onset hazard LightGBM frozen on 2026-09-22 (`pipeline/onset.sh` calls it "M1"; unrelated to the monotone M1 below). |
| panel_shrink25 | Onset decoder: per-corridor probability thresholds shrunk 25% local / 75% global. |
| u-models | Per-site logistic models of the chance that a bottleneck site breaks down in the next 30 minutes. |
| A1 template | For a chosen site, the links queued in at least half of that site's training onsets. |
| FE | Family ensemble of 24 conditional-logit models that pick one site per onset window. |
| A25 | Location-free ongoing LightGBM ensemble (6 members); weight 0.25 in the mix and the empty-window fallback. |
| rep2, selrec, geomselrec | rep2: corrected replicated windows and the ongoing models trained on them. selrec: with selector and recency features; geomselrec: plus queue geometry. |
| N6 | The six rep2 models refitted through February 2031 (the "N" slots of the mix). |
| M1f, M1sf | Monotone-constrained ongoing models (probe M1 / M1s) in their through-February form. |
| R20 operator, F2 operator | Two ongoing serving paths: R20 builds the selected cells; the older F2 path supplies the cells R26c inherits through the delta patches. |
| BND, LE table | BND: correction of the selector inputs for the official 360-minute spacing. LE table: expected count of rejected predecessors, from train candidates. |
| τ | Block threshold of the ongoing decode (0.45). |

## Lineage

```
TASK 1 / TASK 3  (Task 3 is scored on the Task 1 values; it has no model of its own)

masked month -> LRTC + 3 interpolations; train weekday mean -----+
blend weights (NNLS) --------------------------------------------+-> v3 stack -----+
G2 hole model + weekday profiles ------------------------ (inside holes) ---------+-> C1
C1 -> RLB residual (8 corridors) / RL (D12_I405 pair) -> k* (k_aux, 5 corridors)
   +-> DK50 head -> CAP CatBoost blend -> SG1 (J1 head where v < .85 vf) -> general cells
   +-> D12 stage-2 on RL ---------------------------------------------------> D12_I405 cells
   +-> G2v2 v2_F ------------------------------------------------------------> hole flow
         +-> T1, HM25, HMP50, PR head (fitted in the run) + c_Jan ----------> hole speed

TASK 2 ONSET  (only the step at T+30 is predicted)
queue priors + 0922 head -> panel_shrink25 thresholds (Feb 2031 calibration) -> base decisions
  D7_I10_E, D7_I10_W, D7_I210_E, D7_I405_N: my_blocks decode (block mean >= .7, plus top block)
  D12_I5_N, D12_I5_S, D7_I405_S: u-models -> FE site choice (site phase) -> A1 template >= .5
  D7_I210_W: base decisions

TASK 2 ONGOING
N6 selrec x3 + M1sf x3 -> selrec slot  (each seed: mean of the two)    BND: LE table corrects the
N6 geomselrec x3 + M1f x3 -> geom slot                                   selector inputs of N6
A25 x6 -> A slot
p = .75 * mean(selrec, geom slots) + .25 * A25 -> mean over strict block >= .45 -> queued cells
empty window -> A25 strict-block mean >= .5
F2 operator (.75 * N6 + .25 * A25 on my_blocks, >= .5) -> cells inherited through the delta patches

TASK 4: regularised non-negative least squares per corridor and month; nothing is trained.
```

## Recipes

### Task 1 / Task 3

**Blend weights** (`data/interim/m11_task1_blend/blend_weights.csv`)
- Predicts: one non-negative weight per classical fill and channel. The weighted sum is the "blend"
  input of the v3 stack.
- Data: dev-fold days of June 2030, all corridors, the target cells of the train masks. Labels: true
  speed and flow.
- Method: SciPy NNLS per channel, one weight vector for every corridor. Result: speed = .993 LRTC-TNN +
  .007 spatial interpolation; flow = .062 LRTC-TNN + .345 temporal + .577 spatial + .018 weekday mean.
- Trainer: `scripts/34_task1_blend.py` (in `training/`).

**v3 stack** (`build/task1_stack/stack_{speed,flow}.txt`, `stack_features.csv`)
- Predicts: speed and flow of each blank cell.
- Data: the dev fold minus its last 14 days, 2030-06-01 … 2030-12-20 (203 days), 10 corridors. LRTC
  is solved in 31-day chunks, as at serving. Target cells of the train masks, 25% row sample
  (random_state 20260920): 5,860,097 rows. The last 14 dev days (2030-12-21 … 2031-01-03,
  1,589,413 rows) are evaluation only. No February data.
- Labels: true speed; true flow (all lanes). No sample weights.
- Features: 71 (`src/tfb/task1stack.py`): the five fills (LRTC-TNN, temporal, spatial and two-axis
  interpolation, weekday time-of-day mean) and their blend; observed values of the same link up to
  30 minutes before and after and of up to 3 links on each side; occupancy context; lanes, free-flow
  speed, capacity.
- LightGBM l2, learning rate .05, 127 leaves, min_data_in_leaf 200, feature and bagging fraction .8
  (freq 1), seed 20260920, 900 rounds, no early stopping, all cores.
- Trainer: `scripts/113_task1_stack_probe.py --panels <10 corridors> --fit 0 --eval 14 --chunk 31
  --rounds 900 --sample 0.25 --save` (in `training/`). The command was not logged; it was rebuilt from
  the run stamp, and a rerun on 2026-11-07 reproduced both models and the feature list byte for byte.

**k_aux density model and the k\* settings** (`build/final_sprint_review_20260922/experiment_audit/state_recency/aux_density_weightedk/k_l1_weightedL_600.txt`; `build/strongest_candidate_20260922/state/transfer/selection_before_held.json`)
- Predicts: link density k = flow / max(speed, 1) of a blank cell.
- Data: D7_I10_E and D12_I5_N, train days 2030-06-01 … 2030-12-20; 1,000 target cells sampled per day
  (seed 20260922): 406,000 rows. No February data in the fit.
- Features: the 71 v3 stack features. Weights: link length, normalised to mean 1, so the absolute-error
  loss targets the vehicle count on the link.
- LightGBM regression_l1, learning rate .05, 127 leaves, min_data_in_leaf 200, feature and bagging
  fraction .8, seed 20260922, 600 rounds, 4 threads.
- k\* settings (selection file): k = 0.9 × flow/speed + 0.1 × k_aux on D7_I10_E, D7_I405_N, D7_I405_S,
  D12_I5_N and D12_I5_S (zero elsewhere), with per-corridor precision weights from calibration RMSE.
  The rule was chosen on the development calibration days 2030-12-21 … 2031-01-04 by
  `scripts/audit_strongest_state_transfer_20260922.py` (in package).
- Trainer: `build/final_sprint_review_20260922/experiment_audit/aux_density_weightedk.py` (in `training/`);
  its feature caches come from `state_recency.py` in the same folder (not in package).

**G2 hole model** (`build/review_20260924/work/g2/gap_{speed,flow}.txt`, `gap_features.json`, `<corridor>_profile_all.npz`)
- Predicts: speed and per-lane flow of every link at each of the 18 steps of a hole.
- Data: holes cut at the replicated windows of the 8 Task 2 corridors
  (`build/review_20260923/work/e2/replicated_unique.parquet`). The hole is blank; the history
  T−60 … T−5 is restored from truth, as the release republishes it. 35% cell sample outside February
  (rng seed 7). Fit 2030-06 … 2030-12 (3,623,736 rows); early stopping on January 2031 (555,562 rows);
  February 2031 is a test only (1,409,814 rows).
- Labels: true speed; true flow per lane.
- Features: 51: last observed value before and first after the hole and their distances, 20-minute
  means on both sides, ratios to the weekday profile, corridor demand level, lowest speed of the last
  hour, the straight line across the hole, queue state before and after, the same values of the links
  ±1 and ±2, time of day, lanes, free-flow speed, capacity, length, corridor code.
- Weekday × time-of-day profiles: leave-one-month-out for the training rows; the served profiles
  (`profile_all.npz`) use all 9 train months, February included.
- LightGBM regression, learning rate .05, 127 leaves, min_data_in_leaf 200, feature and bagging
  fraction .8, λ2 1, seed 1, at most 3,000 rounds with early stopping 100: speed 2,984 trees, flow
  3,000. 8 threads.
- Role: C1 inside holes, which is context for RLB and CAP; its values are overwritten by G2v2 and PRP.
- Trainer: `build/review_20260924/work/g2_train.py` (in `training/`). Rows and profiles:
  `g2_gap_data.py --panel <corridor>` (in package).

**Co-switching blocks** (`build/review_20260924_newworld/work/blocks/my_blocks.npy` and the identical `build/review_20260924_ctm/ruler/blocks/my_blocks.npy`)
- Groups adjacent links that switch into queue together.
- Data: true speeds of all 273 train days, 8 corridors (February included). For each adjacent pair,
  the share of queue onsets at one link that the neighbour matches at the same 5-minute step (the
  neighbour observed and free just before). Pairs with a share ≥ .5 join one block: 47–170 blocks per
  corridor. Deterministic.
- Used by RLB, G2v2, the onset block decode, the F2 ongoing operator and the A-block fallback.
- Trainer: `build/review_20260924_newworld/work/blocks/my_blocks.py` (in `training/`).

**RLB residual model and RL fallback** (`build/review_20260924_ctm/rl/models/RLB_final/`, `RL_final/`)
- Predicts: the error of C1 (truth − C1) in speed and in per-lane flow, at blank cells outside holes.
- Data: 2031-01-04 … 2031-01-31 and February 2031, the 8 Task 2 corridors, with C1 rebuilt on those
  days (holes at official-like windows). C1 is out of sample there: the stack fit ends 2030-12-20.
  5,952,690 rows; free-flow cells (C1 speed ≥ .85 × free-flow speed) kept at 30% (rng seed 0) with
  weight 1/0.3, leaving 1,975,322 rows (the same rows CAP uses). Uses February.
- Features: RL 106, with no link or corridor identity: the C1 speed, flow and density of the cell and
  its speed ratio to free flow; free-flow speed, lanes, length, time of day; differences to the
  observed (else C1) value at ±1 and ±2 steps and ±1 and ±2 links; neighbours' occupancy; local
  shocks; mean observed shocks in bands 4–8, 9–16, 17–32 and 33+ links away at −5, 0 and +5 minutes;
  corridor mean. RLB adds 24 block features (queued share and state of the observed block-mates, the
  link's queued-speed template): 130.
- LightGBM l2, learning rate .05, 63 leaves, min_data_in_leaf 300, feature and bagging fraction .8,
  λ2 10, 400 rounds, default seeds, 3 threads.
- Serving: RLB on the 8 Task 2 corridors; RL on D12_I405_N/S, which neither model saw in fitting.
- Trainer: `build/review_20260924_ctm/rl/fit_final.py --months 2031_01 2031_02 --variant {RLB,RL}
  --name {RLB,RL}_final` (in `training/`; it imports `rl_eval.py`, `rleval.py` and `rlfeat.py`, which
  ship). Tables: `g3x_rl.py`, `make_feats.py`, `make_blk.py` (not in package).

**G2v2 hole model v2_F** (`build/review_20260924_ctm/g2v2/models/v2_F/`, `rows/<corridor>_tmpl.npz`)
- Predicts: as G2. Its flow is the final hole flow.
- Data: holes at the corrected replicated windows with their selector predecessors (`sel_seq.py` →
  `train_windows_v2seq.parquet`), 8 corridors, a fixed 35% sample (column u, rng seed 11). Fit
  2030-06 … 2030-12, 3,598,488 rows; early stopping January 2031, 553,555 rows; February test only.
- Features: 73 = G2's 51 plus 22: window condition (queue already present or not), the selector
  features as served, block states, queue templates, distance to the nearest queued link. Templates
  are leave-one-month-out for training rows; the served `tmpl.npz` histograms cover all 9 train
  months, February included.
- LightGBM as G2 (seed 1, λ2 1); both heads ran the full 3,000 rounds (early stopping never fired);
  3 threads.
- Trainer: `build/review_20260924_ctm/g2v2/train_v2.py` (arguments recorded in
  `models/v2_F/train_info.json`: arrangement F, rows v2, features v2, seed 1, learning rate .05; in
  `training/`). Rows and templates: `data_v2.py --panel <corridor>` (in package).
- The sibling v2_J (fit Jun–Nov 2030 + Feb 2031, early stopping Dec 2030) is not shipped; its January
  predictions are the base of c_Jan.

**DK50 density head** (`build/review_20260927_structural/density/l2_uniform.txt`, `density_recipe.json`, `density/selection_before_march.json`)
- Predicts: the error of the per-lane density of the parent state on general cells. Speed and flow are
  then moved half-way (strength .5) toward the corrected density.
- Data: 2031-01-04 … 2031-01-31, 8 corridors, 2,929,441 cells. The parent is a two-fold January RLB
  (fit on one half of January, predict the other), so the labels are out of fold; no February labels
  in the fit.
- Features: 143 = the 130 RLB features + 13 density columns (parent speed, flow, density and their
  changes; neighbours' density at ±1, ±2 steps and links). Uniform weights.
- LightGBM l2, learning rate .05, 31 leaves, min_data_in_leaf 400, λ2 10, feature and bagging
  fraction .8, 200 rounds, default seeds, 3 threads.
- Variant (l2_uniform) and strength (.5) chosen on February 2031 among six pre-declared candidates.
- Trainer: `build/review_20260927_structural/density_experiment.py` (in package). January parent:
  `january_parent.py` (not in package). February tables: `matched_background.py` (in package).

**CAP CatBoost heads** (`build/model_capacity_20260927/state/C6_long_final/y_{v,q}.cbm`, `trial.json`; `state/data.json`, `state/selection.json`, `state/confirmation/report.json`)
- Predicts: the same targets as RLB. Served as DK50 + 0.5 × (CatBoost∘k\* − RLB∘k\*).
- Data: 2031-01-04 … 2031-01-31 and February 2031, 8 corridors: 5,952,690 rows, 1,975,322 after the
  30% free-cell sample (rng seed 0), free cells weighted 1/0.3. Uses February.
- Features: the 130 RLB features.
- CatBoost 1.2.10, RMSE, depth 6, symmetric trees, 4,800 iterations, learning rate .05,
  l2_leaf_reg 10, border_count 128, MVS bootstrap (subsample .8), random_seed 0, thread_count 6.
- Choices: recipe, 4,800 iterations and blend weight .5 chosen on January 25–31 with fit days January
  4–17 and stopping days January 18–24; confirmed on February with a January-only refit; final fit on
  January + February.
- Trainer: `build/model_capacity_20260927/state_train.py selected --final` after `prepare_final`,
  driven by `state_pipeline.py` (both in `training/`; the SHA256 of `state_train.py` equals
  `source_sha` in `trial.json`). Selection: `state_eval.py --freeze` (in `training/`). February check:
  `state_confirm.py` (in package).

**D12_I405 stage-2 heads** (`build/review_20260929_target092/noise_floor/d12_prod_stage2_y_{v,q}.txt`)
- Predicts: the remaining error of RL on the D12_I405 pair (no Task 2 windows; RL never fitted there).
- Data: 2031-01-04 … 2031-01-31 and February 2031 rows of D12_I405_N/S from the RL tables; target =
  truth − RL_final; free cells 30% (rng seed 0) with weight 1/0.3. Uses February.
- Features: the 106 RL features.
- LightGBM l2, learning rate .05, 31 leaves, min_data_in_leaf 300, feature and bagging fraction .8,
  λ2 10, seed 0, 2 threads, 200 rounds trained; the first 100 are served. The 200-round version was
  declared first; 100 rounds scored higher in a January-fit, February-test check (+.000043 against
  +.000028 local score).
- Trainer: `build/review_20260929_target092/noise_floor/d12_serve.py` (in package; it fits only when the
  frozen heads are absent).

**PRP January heads** (`build/review_20260929_target092/gap_structure/b3_sp_M2_janfit.txt` = c_Jan, `f1_phead_janfit.txt`)
- c_Jan predicts the in-hole speed error of G2v2 from the on-ramp and off-ramp counts inside the hole
  (ramp detectors are not blanked; a queue on the mainline throttles its on-ramps).
- Data: January 2031 holes at all corrected replicated windows of the month, 50% cell sample: 790,656
  rows. Base = v2_J, which did not see January; target = true speed − v2_J speed. v2_J was fitted with
  February, so this model depends on February through its base.
- Features: 57 = 24 hole features of G2v2 + 16 ramp-count features (off- and on-ramp counts inside the
  hole at, above and below the link, relative to the same ramps in the 30 minutes before and after the
  hole) + 17 features of the nearest on-ramps.
- LightGBM l2, learning rate .05, 31 leaves, min_data_in_leaf 500, feature and bagging fraction .8,
  λ2 10, seed 5, 500 rounds, 2 threads.
- F1 head (`f1_phead_janfit.txt`): same rows, features and parameters, binary label "queued". It is run
  by `c4_phead_transfer.py` in the reproduction, but its output does not reach the final hole speed:
  `g2_persplit.py` and `d2_official_prp.py` do not read it.
- Trainers: `gap_structure/b3_near.py` (in `training/`; rows from `b1_build.py`, not in package);
  `gap_structure/f1_phead.py` (in package).

**J1 density head and SG1 gate** (`build/review_20260930_next/state_density/cap_full_feb.txt`; `density_gate/FROZEN_TRANSFER.md`, `feb_results.json`)
- Predicts: the per-lane density error of the CAP state on general cells. SG1 applies it, with the .5
  projection, only where the CAP speed is below 0.85 × free-flow speed, that is in transition and queue.
- Data: February 2031, 8 corridors, 3,023,299 cells. The parent is a CAP fitted on January only, so it
  is out of fit on February.
- Features: 143 (the DK50 set, rebuilt on CAP values).
- LightGBM l2, learning rate .05, 31 leaves, min_data_in_leaf 400, λ2 10, feature and bagging fraction
  .8, seed 0, 200 rounds, 2 threads. Gate .85 is fixed; it was checked on the two halves of February.
- Trainer: `build/review_20260930_next/state_density/j1_pilot.py` (`fit_final`; in package).

### Fitted during the run

These Task 1 heads are fitted inside `pipeline/task1.sh` on each month's own released observations
(offline Task 1). Nothing from them is shipped.

| Head | Fitted on | Recipe | Program |
|---|---|---|---|
| LRTC-TNN fill | the month's 31-day masked tensor (unsupervised) | ρ 1e-5, θ .10 | `src/tfb/baselines.py`, called by `build/review_20260924_ctm/rl/serve_feats.py` |
| T1 hole speed correction | pseudo holes cut into the month's observed cells | LightGBM l2, lr .05, 31 leaves, min 400, λ2 10, ff/bf .8, seed 7, 300 rounds | `build/review_20260925/round3/task1_trans_serving/s6_serve.py` (`serve_lib.py`) |
| HM25 heads (free speed, queued speed, P(queued)) | same pseudo holes | same recipe | `build/review_20260927/iteration2/hole_mixture_serve.py` |
| HMP50 clusters | the month's observed queue states | queue Jaccard ≥ .95 on jointly observed cells, union ≥ 20, complete linkage; mixture weight .5 (`frozen_recipe.json`) | `build/review_20260927_research/full_pool.py`, `serve_candidate.py` |
| PR head | same pseudo holes, with in-hole ramp features | LightGBM binary, lr .05, 31 leaves, min 500, λ2 10, ff/bf .8, seed 9, 400 rounds | `build/review_20260929_target092/gap_structure/d2_official_prp.py` |

### Task 2 onset

**Queue priors** (`data/interim/m13_task2_dataset/priors/<corridor>.manfc2f86607c094380.npz`)
- Per link and 5-minute time of day: the share of eligible train days on which the link was queued,
  over the 214 days 2030-06-01 … 2030-12-31 (the date list in the 0922 head's manifest; the file tag
  is a hash of that list). Built by `src/tfb/task2infer.py::queue_prior` (in package).

**0922 onset hazard head** (`build/final_sprint_review_20260922/prior/final_feature_hazard.txt`, `final_feature_hazard_manifest.json`, `onset_features.parquet`)
- Predicts: the probability that a link is queued at T+30 in an onset window.
- Data: 2,101 replicated onset windows on train days
  (`data/interim/score_sprint_20260921/recency_chain_windows.parquet`: 360-minute spacing, history
  coverage ≥ .7, queue in the horizon); one row per window and link at T+30 (307,077 rows). Fit on
  windows dated ≤ 2030-12-31; early stopping on January 2031. February is used only by the decoder.
- Features: 75: speed, flow, occupancy and margin to the queue cut at the last history step (T−5)
  and at T; minima, means,
  trends and dispersion over the last hour; margins and trends of the links ±1, ±2, ±4 and the queue
  state of the adjacent links; on- and off-ramp flows;
  link geometry; time of day; corridor and link codes (categorical, 1,450 link codes); queue prior;
  three onset-hazard columns (fit-date onset counts per link and time of day, leave-one-day-out,
  pseudocounts 20 and 10, time kernel [.25 .5 1 .5 .25]).
- Weights: window-balanced (`src/tfb/task2oof.py::window_weights`).
- LightGBM binary, learning rate .1, 127 leaves, min_data_in_leaf 10, feature fraction 1.0, bagging .8
  every 5 iterations, λ1 8.544, λ2 .0958, max_bin 127, seed 20260922, 4 threads; at most 1,000 rounds
  with early stopping 100: 129 trees.
- Trainer: `scripts/145_onset_prior_forward.py` (case `final`, arm `feature_hazard`; in package).

**Decoder thresholds** (`build/strongest_candidate_20260922/onset/final_selection.json`, `build/final_sprint_review_20260922/fullgrid/final_feature_hazard_calib.parquet`)
- The calibration file holds the head's probabilities on all physical links of the February 2031
  replicated onset windows (217 windows, 41,658 rows). The run computes the thresholds from it: global
  threshold = best onset score on a .05–.80 grid (step .025), which gives .30; per corridor, the local
  best shrunk 25% local / 75% global in log-odds. If no link passes, the top link is marked.
- The rule (panel_shrink25) was chosen among candidate decoders on three forward folds (Oct, Nov,
  Dec–Jan 4).
- Programs: `scripts/audit_20260922_fullgrid_probs.py` (calibration probabilities; in `training/`);
  `scripts/audit_strongest_onset_decoder_20260922.py` (rule; in package);
  `scripts/audit_strongest_onset_selected_20260922.py production` (thresholds; in package, runs in the
  reproduction).

**Site u-models and A1 templates** (`build/review_20260924_ctm/ruler/onset_site/u_models_C1.0.pkl`, `events_train.parquet`)
- A u-model gives, for one candidate bottleneck site, the chance that the site region (site ±1 link)
  is queued at T+30 given that it is free from T−60 to T+25.
- Data: train caches 2030-06-01 … 2031-02-28 (273 days) of D12_I5_N, D12_I5_S, D7_I210_W and
  D7_I405_S; every day and origin T from 05:00 to 20:45 that meets the free condition. Uses February.
- Features: 12: bottleneck flow/capacity ratio at T, its means over 15, 30 and 60 minutes, its maximum
  over 15 minutes, its rise against the previous half hour, and the one-hot phase of T+30 in the
  30-minute demand cycle.
- scikit-learn StandardScaler + LogisticRegression (C = 1.0, lbfgs, max_iter 2000), one per corridor
  and site: 12 models. Missing values take the training median; a site with fewer than 5 onsets gets a
  constant rate. Deterministic.
- A1 templates (stored in the same pickle): per site, the share of training first-onset events in which
  each link within ±3 of the site was queued at onset; links ≥ .5 form the predicted extent. The events
  (`events_train.parquet`: 1,471 rows, one per queued block at the onset step, from 1,231 onset events,
  2030-06-01 … 2031-02-28) also give the eligibility rule (≥ 10 training onsets per site). The events
  and templates use 0.6 × free-flow speed as the queue cut.
- Site ranges (`pv_sites.SITES`) and bottleneck links (`site_feats.BN`) are constants in the code,
  set by hand from a bottleneck scan of the training events.
- Trainer: `build/review_20260924_ctm/ruler/onset_site/serve_official.py` (`__main__`: `fit_models`
  and `eval_template.templates`; in package). Events: `ruler/onset_site/events.py` (in `training/`).
  The pickle was written in an earlier session folder that no longer exists; that folder's copy of the
  program is byte-identical to the shipped one.

**FE site choice** (`build/review_20260925/round2/onset_joint/s6_FE_models.pkl`, `site_phase.json`)
- A conditional logit compares the candidate sites of one window: P(site j) = exp(s_j) / Σ_k exp(s_k).
  The FE score is the mean softmax of 24 configurations.
- Configurations: 4 specifications × 2 training sets × L2 penalty λ ∈ {1, 5, 20}, each fitted per
  corridor (D12_I5_N, D12_I5_S, D7_I405_S, D7_I210_W). Every specification has per-site intercepts.
  C2: site load ratios, their gaps and ranks against the other sites, and demand-step phase dummies.
  C3: C2 plus per-site time-of-day and weekend slopes. C4: the u-model logit with its gaps and ranks in
  place of the load ratios. C4b: C4 plus the phase dummies (`onset_joint/s1_specs.py`).
  Training sets: "rep" = 2,108 replicated train onset windows
  (`build/review_20260924_ctm/rep2/replicated_unique_v2.parquet`); "allpos" = every train (day, T)
  with a queued site and at least 2 sites.
- Data: `build/review_20260925/onset/train_rows_<corridor>.parquet`: train dates 2030-06-01 …
  2031-02-28, origins 02:00 … 20:45, demand profiles leave-one-month-out. Uses February.
- Site phase: for each bottleneck link, the 5-minute slot of the 30-minute cycle in which the train
  weekday flow profile jumps most (all train days).
- Fit: L-BFGS-B from zero weights, at most 2,000 iterations (`cl.py`); deterministic.
- Trainers: `onset_joint/s6_fe.py`, `onset_joint/s0b_phase.py` (in `training/`). Rows:
  `build/review_20260925/onset/site_mday.py` (not in package).

### Task 2 ongoing

Shared by the LightGBM families below: label = link queued at each of the 6 steps T+5 … T+30, on
eligible cells; binary objective, 127 leaves, learning rate .05, min_data_in_leaf 100, feature fraction
.9, bagging .8 (freq 1), λ1 0, λ2 10, max_bin 127 (`scripts/strongest_20260922_ongoing.py`); weight of a
row = 1 / (rows of its window × windows of its corridor), normalised to mean 1, so every corridor
carries the same total weight.

**A25 ensemble** (`build/review_20260924_newworld/work/feat/models/og_*.txt`)
- Six location-free members. V2: 73 features (the 76-feature ongoing set of
  `build/review_20260923/ongoing_P1/manifest.json` without link code, corridor code and queue prior):
  origin state, last-hour history, neighbours, ramps, clock. V6: V2 + 20 queue-geometry features
  (93): queue tail and head position, shockwave speed, block length and age, time for the tail or head
  to reach the link.
- Seeds 20260922, 20260929, 20260936 for each of V2 and V6. Trees: V2 697 / 693 / 592, V6 539 / 557 /
  490. Threads: 6 for `og_V2_noID_noPrior`, `og_V2s2`, `og_V6_V2geom`; 8 for the other three.
- Data: replicated ongoing windows (`e2/replicated_unique.parquet`): fit ≤ 2030-12-31 (2,352
  windows); early stopping on January 2031 (354; patience 80, at most 1,000 rounds); February 2031
  (321) only to calibrate a threshold, which came out at .5.
- Trainers: `og_exp.py` (first three) and `og_ens.py` (others) in the same folder (in package).
  Training tables: `build_train_feats.py` and the geometry builder (not in package; the tables were
  deleted after training).

**rep2 production models** (`build/review_20260924_ctm/rep2/m_R_V2{selrec,geomselrec}_s0_seed{0,1,2}.txt`)
- Role in R26c: no prediction. `f2_analysis/bias/lib.py` loads them at import to build its feature
  list, and their tree counts fix the rounds of N6, M1f and M1sf.
- Data: rep2 replicated ongoing windows: fit ≤ 2030-12-31 (2,337 windows); early stopping on January
  2031 (354; patience 80, at most 1,500 rounds); February (322) for calibration only. 25% of windows
  (rng seed 123, 765 windows) get their selector features blanked, so the model also works when they
  are unknown.
- Features: selrec 78 = V2 + 2 selector features (whether the scan origin just before T was rejected
  as unchanged; how many consecutive predecessors were) + 3 recency features (how often the link was
  queued at this time of day on up to 5 earlier days of the same month); geomselrec 98 = selrec + the
  20 geometry features.
- Seeds 20260922 + s; trees selrec 693 / 681 / 658, geomselrec 554 / 577 / 681; 4 threads.
- Trainer: `build/review_20260924_ctm/rep2/train_rep2.py` (in `training/`).

**N6 refits** (`build/review_20260929_target092/contract_audit/q4_refit/m_R_V2{selrec,geomselrec}_refit_seed{0,1,2}.txt`)
- The rep2 recipe refitted through February: 2030-06-01 … 2031-02-28, 3,013 windows, 2,582,803
  eligible rows; the same 765 windows with blanked selector features; fixed rounds = rep2 tree counts;
  no early stopping; seeds 20260922 + s; 2 threads. Uses February.
- Trainers: `q4_refit/train_refit.py 2031-02-28 refit` (selrec seed 0) and
  `q4_refit/train_refit3.py 2031-02-28 refit <seed> <variant>` for the other five, run by
  `q4_refit/run_rest.sh` (in `training/`).

**M1f and M1sf monotone models** (`build/review_20261001_fresh/mono/models/M1f_seed{0,1,2}.txt`, `M1sf_seed{0,1,2}.txt`)
- Predicts: as above. M1f fills the geometry slot and M1sf the selector-recency slot, each averaged
  50/50 per seed with the matching N6 refit.
- Data: the R20 "matched" tables built over the rep2 windows by `queue_data/prepare.py` (history
  T−60 … T−5 from train truth, as the release republishes it; origin T from the masked layer; ramps at
  T and T−5; recency over 5 earlier days). Rows ≤ 2031-01-31 (`fit_matched_base196.npy`, 2,303,064)
  plus the eligible February rows (`held_matched_base196.npy` where `target_mask`, 279,739): 2,582,803
  rows, 3,013 windows. Weights recomputed over the union with the shared formula. Uses February.
- Features: M1f 196 = the 98 geomselrec inputs, imputed with fit-only medians and z-scored
  (`queue_data/TRANSFORM_base196.json`, statistics from rows ≤ 2031-01-31), plus 98 missing-value flags.
  M1sf 156: the same without the 20 geometry inputs and their flags.
- Monotone constraints, method `intermediate`, on 29 dynamic inputs (none of them geometry):
  - predicted queue probability falls as these rise (17): speed and margin to the queue cut at the
    last history step (T−5) and at T; lowest and mean speed and lowest margin of the last hour; speed
    trends over 15 and 30 minutes; margins of the links ±1, ±2, ±4; margins of the links ±1 at T;
  - and rises as these rise (12): occupancy at T−5 and at T; occupancy trend; queued at T−5 and at T;
    queued steps in the last hour; queued neighbour upstream and downstream; upstream occupancy; the
    three recency features.
  Each model file lists exactly these 29 non-zero signs.
- Parameters: the R20 `gbdt_config` (`queue_data/INPUT_MANIFEST.json`): the shared binary recipe,
  seeds 20260922 / 20260923 / 20260924, fixed 554 / 577 / 681 rounds, no early stopping, 2 threads.
- Trainers: `build/review_20261001_fresh/mono/m1f_train.py` and `m1sf_train.py` (in `training/`).
  Both import `mono/m_train_infer.py` (in package) for the cache paths, feature names, parameters and
  `variant_cols("M1")` / `variant_cols("M1s")`, which return the 29 signs, and then pass
  `monotone_constraints` and `monotone_constraints_method="intermediate"` to LightGBM themselves. The
  run logs are `mono/log_F.txt` and `mono/log_G.txt`; the forms were fixed in
  `mono/PREREG_SELECTION.md` (addenda C and F) before training.
- `m_train_infer.py` alone does not reproduce these models. Called with the name `M1f`, it applies no
  constraint (it knows only M0–M6, M0s and M1s) and trains on the rows ≤ 2031-01-31 only.

**Strict block map** (`build/review_20260930_round20/evaluation_audit/strict_blocks_through_20310131.npy`)
- The co-switching block rule on train days 2030-06-01 … 2031-01-31 (245 days); the block map of the
  R20 decode. No February data.
- Trainer: `evaluation_audit/build_strict_blocks.py` (in `training/`).

**Input transform** (`build/review_20260930_round20/queue_data/TRANSFORM_base196.json`)
- Per input: median for imputation, finite mean and standard deviation for scaling, from rows
  ≤ 2031-01-31. Built by `queue_data/transform.py` (in package).

**BND expected-count table** (`build/review_20261001_r26/bnd/le_table.json`)
- For a window whose predecessors the official scan could only partly see, the served count of rejected
  predecessors is replaced by its expectation given the known part, tabulated on the 587,496 train
  candidate origins of `rep2/candidates_v2.parquet` (2030-06-01 … 2031-02-28). Uses February.
- Program: `bnd/bnd_legal.py::le_table` (in `training/`).

**Decode constants.** Mix .75 / .25, block threshold τ = .45, fallback threshold .5. These are
constants in the serving code; nothing is fitted.

### Task 4

Path flows are solved per corridor and month by non-negative least squares against the released link
counts with a pull of weight .05 toward the organiser's weak prior (`scripts/60_build_submission.py`).
Nothing is trained.

### Package files that train nothing used here

`build/review_20260930_round20/queue_data/train_gbdt.py`, `train_encoder.py` and
`tabm_models/train.py` train R20 variants that R26c does not use. They ship because the R20 serving
preparation checks the SHA256 of every code file listed in `queue_data/INPUT_MANIFEST.json`.

## Choices made on the validation month

No shipped model was fitted on validation or private data. Some fixed choices were selected or gated
with released March observations (and, for the Task 1 HMP50 and J1/SG1 gates, April observations):
the onset block threshold .7, the 0.01 and 10-onset site rules, the exclusion of D7_I210_W, the FE
family and the A1 threshold .5; the ongoing τ = .45, the M1f/M1sf slot recipe, the BND table variant,
the A25 members and the N6 refit; the HMP50 mixture weight; the March gates of DK50 and CAP and the
March/April gates of J1/SG1. The organiser allows the train split and March data, including the public
leaderboard, for fitting and selecting the April (private) windows; Task 1 is offline.

## Cost

Hardware: Apple M4 Mac (10 CPU cores: 4 performance, 6 efficiency; 16 GB RAM), macOS 26.6.2, CPU only.

| Model | Training time | Record |
|---|---|---|
| Blend weights | not recorded | |
| v3 stack, both heads | 1,191 s, all cores (byte-identical rerun of 2026-11-07); original run not recorded | rerun log |
| k_aux | 34 s, 4 threads | `aux_density_weightedk/run.log` |
| k\* rule selection | 649 s (with its held-out evaluation) | `transfer/results.json` |
| G2, both heads | about 9 min, 8 threads | file times; no log |
| Co-switching blocks | not recorded | |
| RLB / RL | 314 s / 209 s, 3 threads | `manifest.json` |
| G2v2 v2_F, both heads | 2,119 s, 3 threads | `train_info.json` |
| DK50 January parent | 152 s | `density/january_parent.json` |
| DK50 head | about 39 s (158 s after start; the first head ended at 119 s) | `density_experiment.log` |
| CAP, both heads | 3,102 s (speed 1,614 s, flow 1,488 s), 6 threads, 3.1 GB peak | `trial.json` |
| D12 stage-2 | 40 s, 2 threads | `d12_serve.log` |
| c_Jan | not recorded | |
| F1 head | 42 s (whole program), 1.7 GB peak | `f1_phead.log` |
| J1 head | 62 s, 2 threads, 2.7 GB peak | `full_fit.json` |
| 0922 onset head | 6.8 s, 4 threads | `prior/results.json` |
| Onset calibration probabilities | not recorded | |
| u-models, events | not recorded | |
| FE models, site phase | not recorded | |
| A25 members | not recorded | |
| rep2 production models | not recorded | |
| N6 refits | selrec 121 / 183 / 164 s, geomselrec 178 / 181 / 169 s, 2 threads, 2.7–3.9 GB peak | `train_*_refit_seed*.json` |
| M1f | 215 / 179 / 165 s, 2 threads | `mono/log_F.txt` |
| M1sf | 113 / 116 / 135 s, 2 threads | `mono/log_G.txt` |
| R20 matched tables | 269 s | `queue_data/PREPARED.json` |
| Strict block map | 22 s | `STRICT_BLOCKS_QA.json` |
| BND table | not recorded | |

The recorded times add up to about 3 hours of wall time. Building the other feature tables and the
unrecorded fits come on top.

Bit-exact retraining: LightGBM ran with `deterministic=0` and thread counts from 2 to 10. A second
run with the same thread count reproduced the model files byte for byte for the v3 stack (2026-11-07)
and for the R20 LightGBM recipe that M1f/M1sf use (`mono/models/M0_seed0.txt` equals
`queue_data/models/L0_geom_seed0.txt`). Other models were not re-trained for this check. CatBoost
needs version 1.2.10 and thread_count 6.

## Intermediate artifacts

A full retrain needs these train-derived tables, which are not distributed. Paths are in the
development repository. Each is built from the released train split only.

| Table | Used by | Built by |
|---|---|---|
| `build/final_sprint_review_20260922/experiment_audit/state_recency/{D7_I10_E,D12_I5_N}_old{0..6}.parquet` | k_aux | `experiment_audit/state_recency.py` |
| `build/review_20260923/work/e2/replicated_unique.parquet` | G2, A25 | `build/review_20260923/work/e2_windows.py` |
| `build/review_20260924/work/g2/<corridor>_rows.parquet` | G2 | `g2_gap_data.py` (in package) |
| `build/review_20260924_ctm/rl/feats/<corridor>_{2031_01,2031_02}.parquet`, `blk_<corridor>_{2031_01,2031_02}.parquet` | RLB, RL, CAP, D12 stage-2 | `rl/g3x_rl.py`, `rl/make_feats.py`, `rl/make_blk.py` |
| `build/review_20260924_ctm/g2v2/train_windows_v2seq.parquet`, `g2v2/rows/<corridor>_rows.parquet` | G2v2 | `g2v2/sel_seq.py`; `g2v2/data_v2.py` (in package) |
| `build/review_20260924_ctm/g2v2/models/v2_J/test_preds.parquet` | c_Jan, F1 | `g2v2/train_v2.py --arr J` |
| `build/review_20260927_structural/density/january_rlb_oof.npz`, `jan_rlb_fold{0,1}_y_{v,q}.txt` | DK50 | `january_parent.py` |
| `build/review_20260927_structural/density/matched/*` | DK50 strength choice, J1 | `matched_background.py` (in package) |
| `build/model_capacity_20260927/state/X_finalfit.npy`, `yw_finalfit.npz` | CAP | `state_train.py prepare_final` |
| `build/review_20260929_target092/gap_structure/b1_jan.parquet`, `b3_jan.parquet` | c_Jan, F1 | `gap_structure/b1_build.py`, `b3_near.py` |
| `build/review_20260930_next/state_density/X_cap.npy`, `meta.parquet`, `parents.npz` | J1 | `state_density/j1_pilot.py` (in package) |
| `data/interim/score_sprint_20260921/recency_chain_windows.parquet` | 0922 head | `src/tfb/task2pool_cache.py::chain_pool` via `scripts/128_train_task2_recency.py` |
| `build/review_20260924_ctm/ruler/onset_site/cache/<corridor>_train.npz` (full arrays) | u-models, events | `cache/build_cache.py <corridor> train` (in package; the shipped train caches keep only the link order) |
| `build/review_20260925/onset/train_rows_<corridor>.parquet` | FE | `build/review_20260925/onset/site_mday.py` |
| `build/review_20260924_ctm/rep2/replicated_unique_v2.parquet`, `candidates_v2.parquet`, `rep2_s0_<corridor>.parquet` | rep2, N6, FE, BND table, R20 tables | `rep2/cands_train_v2.py`, `rep2/select_train_v2.py`, `rep2/build_rep2.py` |
| `build/review_20260924_ctm/ruler2/rec_train_queue_ongoing_s0.parquet` | rep2, N6 | `rep2/build_recency.py train queue_ongoing 0` (in package) |
| `build/review_20260930_round20/queue_data/cache/{fit,held}_matched_base196.npy`, `{fit,held}_meta.npz` | M1f, M1sf | `queue_data/prepare.py`, `queue_data/transform.py` (in package) |
| A25 training tables (`queue_ongoing_<corridor>.parquet`, `geom_train_queue_ongoing.parquet`) | A25 | `work/feat/build_train_feats.py` and the geometry builder; deleted after training |

SHA256 of the tables that exist today (development repository, 2026-11-08):

```
372e50d6aec62b2412b7eb5b9c5201e8758db3e2a866f0f89494eb018063798c  build/review_20260923/work/e2/replicated_unique.parquet
580f9c89413673409a774886a2ddec23e0cb45edb0d8c0d06a1683f0ea00070d  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old0.parquet
8e59102b80c20df6646f428cfa4a042f57dd683e0d4fb0a27935eb9ea505fe23  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old1.parquet
04ea920e347071dea01b16ceffd43c932eee3b3d4986eccabbd96f28b0cb2a11  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old2.parquet
b9859615d4f1e762d0c2ba821f4ca9902f0c4de702c9a61c94b4e7863a314174  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old3.parquet
334d33ff65228b1cae309cc4a6c2a9b4885bffd1a40d27f52ebdb8b5dbe21640  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old4.parquet
e85eb797abe4a86ddf7a120e90d72b43cf0260717d031fa51bad9b2223c6015d  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old5.parquet
53af91830f811f4c5633923cf9f65b9137b074d38359ad800a8dae33eb54d56f  build/final_sprint_review_20260922/experiment_audit/state_recency/D7_I10_E_old6.parquet
4f75668826d8ec1420fbd7c846a10e65f496376bfbc75a6fba30532dae93b3a7  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old0.parquet
ad63ebc1da87f4dcdd3eb94144e2e275c26fd05568ab7a97c2f64744f9d2b9fb  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old1.parquet
e7f899c3d1954d86406c9e01772f00d9b97af11b4c17904d5ccdb84cc3ba3d8e  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old2.parquet
450dc9d623b0ad2ad0708ab43a5ea246f81c632dc6f4cc15db7fadcd387f7f47  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old3.parquet
bd9515b7989c41584bbc85bca008cda03def30f216439171392f260ae0e9bfe8  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old4.parquet
d1e02d0b65f46a363e309878067697d38ee69669e2aaada2216c793a174ab8f3  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old5.parquet
222c2bb56f6b8aa4f43ffa1d91715db9fdcc71e27fe08e270b5065271687720c  build/final_sprint_review_20260922/experiment_audit/state_recency/D12_I5_N_old6.parquet
12200d3b7da673a6e3624b7c59fb7e72519bc2032f17962c058ed6744b617e5c  build/review_20260924/work/g2/D12_I5_N_rows.parquet
f59393b3ccd3d541eaa690724bb12d496766ca3b5e28858fc8b62da55b755f6f  build/review_20260924/work/g2/D12_I5_S_rows.parquet
58ec011ba311ae2152f0546ad002bf37f5f763ae8fb66d56872aa05d2a2cb708  build/review_20260924/work/g2/D7_I10_E_rows.parquet
d8df6df6d69f565b8994d4f7387ab785f507238edb8c4dcc9bd82ebd5f052f68  build/review_20260924/work/g2/D7_I10_W_rows.parquet
38cd7a939ef651492cdd3639bf14b974744bb7398669b9a1b3ed8a6efceef0ff  build/review_20260924/work/g2/D7_I210_E_rows.parquet
61b0fb14469276107a2444b01f78126fa9c33f848fdb8a016968c1149e3b0bb5  build/review_20260924/work/g2/D7_I210_W_rows.parquet
c638d9fe5e30a47fd7d454b9a667f3dfed21f54f531c12614a41e3f7b8c95199  build/review_20260924/work/g2/D7_I405_N_rows.parquet
1eab2b11c026f8d9cba9a1c70573dce5a23043044de44ea54971a453db8c3795  build/review_20260924/work/g2/D7_I405_S_rows.parquet
118c85ee28526070bbc9a269aac49fe3f7d44d6c2d5ad133a3b105e8cfcf91b1  build/review_20260924_ctm/rl/feats/D12_I405_N_2031_01.parquet
9f727fa9f41699b0287cfc82440fc16da53838a60078c3ad3b1a26ce2f6d7f65  build/review_20260924_ctm/rl/feats/D12_I405_N_2031_02.parquet
308b1ce7159d0f777c639f44c47c85b8f6189ede1c2936b8a562e474ffad148e  build/review_20260924_ctm/rl/feats/D12_I405_S_2031_01.parquet
3bb368dcbf601a60ba40d1c3b48b979c3133038d4666048e791d31ed590205ab  build/review_20260924_ctm/rl/feats/D12_I405_S_2031_02.parquet
87d0b094fffa3c7a7af48bb7c59bf345676e21acce71ea8efdea3bd83d01000b  build/review_20260924_ctm/rl/feats/D12_I5_N_2031_01.parquet
b788179f67d0d2a0f163c286e6b410bb9aa45557909fc7e4e2cefff16e0b0438  build/review_20260924_ctm/rl/feats/D12_I5_N_2031_02.parquet
92982671b0e99d6afb733e4195caf4d0359127ebbdad5821499249e0bff2cc5b  build/review_20260924_ctm/rl/feats/D12_I5_S_2031_01.parquet
b13953a937dd3eaf220e7b14f2b9a64952cbe7686e56a2aa611293b2c42eec5c  build/review_20260924_ctm/rl/feats/D12_I5_S_2031_02.parquet
c1dc4a157fc147b18227dce4192edb8ada2324a341416b2c9a570a5c5e6153a7  build/review_20260924_ctm/rl/feats/D7_I10_E_2031_01.parquet
99f2b750f6a8f07f228f8bb8c3956dab33034cea26242f8ab501a6572f1bb24f  build/review_20260924_ctm/rl/feats/D7_I10_E_2031_02.parquet
a3cc952b25a5be5a382e9dfd2a8bffa809f23a088522262e80fdc26ae1a1d6cd  build/review_20260924_ctm/rl/feats/D7_I10_W_2031_01.parquet
cb6d09d14807fde78fd800cadab722f13cbcde393a25f39c003c03f3a03ff21a  build/review_20260924_ctm/rl/feats/D7_I10_W_2031_02.parquet
55930cff1d5c7c6f14e1bf30f2cc79f242b2d580288e2dc60938ff16571f3c23  build/review_20260924_ctm/rl/feats/D7_I210_E_2031_01.parquet
e69c3caba2ba38f70aedf21f5c9317fd7051c60860c161359f7bf32ced4b6b09  build/review_20260924_ctm/rl/feats/D7_I210_E_2031_02.parquet
be61ed1140563d83c3d834ea0bbd83d0619d79067803593fbf4b0e3aab00f4a1  build/review_20260924_ctm/rl/feats/D7_I210_W_2031_01.parquet
e43c397922e9215754c14292ab18a07357bc58cf307755aa282ae687b475cf6a  build/review_20260924_ctm/rl/feats/D7_I210_W_2031_02.parquet
7f3a796b82458d69fa35defce4b980832f3b05af1a18c7ec4a407fca38b3e245  build/review_20260924_ctm/rl/feats/D7_I405_N_2031_01.parquet
944caa79ad5d58d93770b130f0755518cd0f98f8a454c203cbfc8bf4bab1f750  build/review_20260924_ctm/rl/feats/D7_I405_N_2031_02.parquet
f42d7323c614471e8c5fb94ad91f38432d5a8c1d413997d9fef04d7b3addc12d  build/review_20260924_ctm/rl/feats/D7_I405_S_2031_01.parquet
30f82faafbef13a345e0ef49ebe123532bcc4076d59a4f9d76ccbea4b3dfa4a1  build/review_20260924_ctm/rl/feats/D7_I405_S_2031_02.parquet
cc0d0f80cd8fd4d600088757aec7fae9f4ea57787deea7809a04229528d8e2cb  build/review_20260924_ctm/rl/feats/blk_D12_I405_N_2031_01.parquet
9af1036db48a3a6caf47402f492d161a7cb7fa29927fc3173c9ecd8899031d21  build/review_20260924_ctm/rl/feats/blk_D12_I405_N_2031_02.parquet
3645d30b34719aa25936f14cbf535e44e34765f1a671e39dc9aff4e73d158e35  build/review_20260924_ctm/rl/feats/blk_D12_I405_S_2031_01.parquet
6435fe3561134fcfad30ab96c05e761109a616c2fc0f6d0c1b8ad7a40e76afc9  build/review_20260924_ctm/rl/feats/blk_D12_I405_S_2031_02.parquet
48a08347feed4aeb09c805f68a898229aed8386ff47f12c10b041c05a9be9dfd  build/review_20260924_ctm/rl/feats/blk_D12_I5_N_2031_01.parquet
f062c84e556d64adaafd66e9a7a0b59f2061866fc43dc3739893c66fe35c745c  build/review_20260924_ctm/rl/feats/blk_D12_I5_N_2031_02.parquet
a23935a6cfd5a59e734d9bad28eaa4e3119d886a9c8e722417416438bd76258e  build/review_20260924_ctm/rl/feats/blk_D12_I5_S_2031_01.parquet
f5b98d597c896145ad10129a63c01f9382d057009bf30a07c2a387c323ddc9a1  build/review_20260924_ctm/rl/feats/blk_D12_I5_S_2031_02.parquet
4cc82fd93fd0da1cb4197420c97a32737ec87f2f6ddd006fd2936d10cc3fc751  build/review_20260924_ctm/rl/feats/blk_D7_I10_E_2031_01.parquet
01b137a4153ab364828f9c364d60c74a50fbc30e7d03154a061bf327fe11361c  build/review_20260924_ctm/rl/feats/blk_D7_I10_E_2031_02.parquet
f77716f8848af8859aedbe92bb1fe0d852dee7b6126defc593af78434ac67c4a  build/review_20260924_ctm/rl/feats/blk_D7_I10_W_2031_01.parquet
2f0323c97dacefa79b1720d70d4e54ce6ecabd8ffb604578be7685ff00ec06ce  build/review_20260924_ctm/rl/feats/blk_D7_I10_W_2031_02.parquet
22efb3844f7120d9bdd1dbe00ee0b3590de9f3692746d340e7dee8eecc125004  build/review_20260924_ctm/rl/feats/blk_D7_I210_E_2031_01.parquet
6144c8f66427618cb44c143561140fca26f814c2672f2950106ae32bd6f7786e  build/review_20260924_ctm/rl/feats/blk_D7_I210_E_2031_02.parquet
13ac66fc4e297a2e0252ec234a4e80cb177ed53fc66fbdeae8e42d7e5a61ce69  build/review_20260924_ctm/rl/feats/blk_D7_I210_W_2031_01.parquet
ae1da0a243605a909b1ee86070ba90823bd45400cc6758685d6fedcd2f65f133  build/review_20260924_ctm/rl/feats/blk_D7_I210_W_2031_02.parquet
5674176cf84f0a8f43519b3d2d9ebe41e3ce8ef1f0a9bbdca53b33a5d51f436d  build/review_20260924_ctm/rl/feats/blk_D7_I405_N_2031_01.parquet
63e981950fdb18e84c5dcf1b47c6d4b2d21dc1205e40d9d0bc8b4ef219548b98  build/review_20260924_ctm/rl/feats/blk_D7_I405_N_2031_02.parquet
7b30f772317e5556aa1df406c42edd487e08ac57deae8bc05cb57ea9461014ce  build/review_20260924_ctm/rl/feats/blk_D7_I405_S_2031_01.parquet
56093b7aca52dd77cbfbdded5229dc244bb2300c46aa11988bb98f21f501ff78  build/review_20260924_ctm/rl/feats/blk_D7_I405_S_2031_02.parquet
b6cb463621e8cf54ff8bd56477408439f48f3f913fb182fabec10fe1d17becaa  build/review_20260924_ctm/g2v2/train_windows_v2seq.parquet
41137673328c438b35296f48b6b199535c56b3834e5a4821111ca06363db904e  build/review_20260924_ctm/g2v2/rows/D12_I5_N_rows.parquet
b820ac7d0b65bbfc7acdd6a4abe932abd5d13857ed9e10d4597119e2eecac75b  build/review_20260924_ctm/g2v2/rows/D12_I5_S_rows.parquet
73616ec0bc45ba2181775d3407bf8125e0c61fc9c5f969ac7c04b6483e50513b  build/review_20260924_ctm/g2v2/rows/D7_I10_E_rows.parquet
320b41aa91851f604b212824190b2a3d57a988fd15ef3144563310a5d6e3c0b1  build/review_20260924_ctm/g2v2/rows/D7_I10_W_rows.parquet
d355089133a5452dcd86166e2845f7d21825974ddc30d393f19fbb096c2f37af  build/review_20260924_ctm/g2v2/rows/D7_I210_E_rows.parquet
026240f3b7d884f57a325b2ce5efc723ec572e2d3b931a9bb70699a985627114  build/review_20260924_ctm/g2v2/rows/D7_I210_W_rows.parquet
7132efc46c5e69586832eef93947978b29ac4db022b3facdfa6e93817eda2ba5  build/review_20260924_ctm/g2v2/rows/D7_I405_N_rows.parquet
55e0932847f884a35c7e8450d12d6d20686a320a67983ad4c82fcf6de7cfffbb  build/review_20260924_ctm/g2v2/rows/D7_I405_S_rows.parquet
f12c1f61c97dd3098a48d2769b08f48e280236dc7250683d452bb11b0adcf989  build/review_20260924_ctm/g2v2/models/v2_J/test_preds.parquet
4a2abd03854a6008945f5c63accce6f287cd3690f0b138b209ed850f83768c93  build/review_20260927_structural/density/january_rlb_oof.npz
4a4faa5120b1d5185f4f9547a4aae67447cc6d524a93ec57d469e2f26eaa5cf5  build/review_20260927_structural/density/jan_rlb_fold0_y_v.txt
3ccd4718e4ad715780d4fd91dbfc951010f3fa7192883d13cff86347bfe4e20a  build/review_20260927_structural/density/jan_rlb_fold0_y_q.txt
a455d593371f01c2e2b17e23e97efd6ba548d7b9a176a1604cc1ac79f87f74c2  build/review_20260927_structural/density/jan_rlb_fold1_y_v.txt
7fd0019962a6670ec1af04fbdb00b2ca255be2c3d4d2af441f1254e40ad978e6  build/review_20260927_structural/density/jan_rlb_fold1_y_q.txt
446b779719bcce0b3fa7a50b2fea983f091640f7d35f14502bdb691beae04e17  build/review_20260927_structural/density/matched/D12_I405_N_features.parquet
0ec92bb1cd6b192b3eec60d72d4f0f3f2f8dcb58904c459e9986159ad186a1e4  build/review_20260927_structural/density/matched/D12_I405_N_parent.npz
9128304ae06bd836c40d0c8d97b4454af872211e647d90be9173070f71e98390  build/review_20260927_structural/density/matched/D12_I405_S_features.parquet
a965bf492b2e6c6c4deeb7e30bc372e8af5720f9cb0ff6c17aad43e8dc5170be  build/review_20260927_structural/density/matched/D12_I405_S_parent.npz
72c715875b040c99f08e5c2825dc535d2a394b9ebbeed8e48e232831eba55838  build/review_20260927_structural/density/matched/D12_I5_N_features.parquet
d0ad7baa069fd3053d5965364843436189ba487733c2973f20e74bd3208955f4  build/review_20260927_structural/density/matched/D12_I5_N_parent.npz
7234647a7c165d33a232d1c603d70b19e8d80fb485990afee51bc4eaacfa2e97  build/review_20260927_structural/density/matched/D12_I5_S_features.parquet
5576c8e695af3960ac36d773f34ef1e9ee3b2039d6b7a0fa87af784c0f1ea0b0  build/review_20260927_structural/density/matched/D12_I5_S_parent.npz
9697faef4afc1b080c54a29de7763329d11ad3ef01159f9b159382304ca7f7cf  build/review_20260927_structural/density/matched/D7_I10_E_features.parquet
87bc6a5baef3bc66c353b0b6ae3638080895bb76e81dc0e28bc73fd6e6e46162  build/review_20260927_structural/density/matched/D7_I10_E_parent.npz
92d5a41fcf186cfce0be9cdc985f0e988641806e4320bcd44410c47d9d81ad57  build/review_20260927_structural/density/matched/D7_I10_W_features.parquet
47f154794664228be7d4fda679fe6a226d65ff3d7c8b647a75fc5a80cdcd6e4f  build/review_20260927_structural/density/matched/D7_I10_W_parent.npz
0bb48d122c19d89fc4eda0a1896ff3abf2c5747e475234b01f35c02bb675891f  build/review_20260927_structural/density/matched/D7_I210_E_features.parquet
62770c148efffd49cf8506c1b41194739132d345d77be4c0ec058972e05470a6  build/review_20260927_structural/density/matched/D7_I210_E_parent.npz
18faa9001cc5294140400cae96d40e449e64c9cfe20c7d6b95150c38cbf7db8e  build/review_20260927_structural/density/matched/D7_I210_W_features.parquet
20f414fd167494f71072f6b6e27d991cd40833a9ba73ac96319bcfe55031f74e  build/review_20260927_structural/density/matched/D7_I210_W_parent.npz
46e99c38821e12b1eb8932e71be132db7577cf2e846eef2ccd2a617946b31d3e  build/review_20260927_structural/density/matched/D7_I405_N_features.parquet
2889d2c45fc82711a3857cffba5fd3ef8a64e816284fb305083025cab341627d  build/review_20260927_structural/density/matched/D7_I405_N_parent.npz
6ded4c924b8df7da9de27241bc4826f6660d6f32e78e6af90924013aec93ff0d  build/review_20260927_structural/density/matched/D7_I405_S_features.parquet
a8d46c4764076e0588a4516d06240dc78b8e2910192e3027b53f4de151a20b6d  build/review_20260927_structural/density/matched/D7_I405_S_parent.npz
176530a73512fe2c5c429d6926f9f119fa4e8f792382593b300a7d01c02c42cb  build/model_capacity_20260927/state/X_finalfit.npy
41e7b4781392427b5fbc02a283c8a71c42de171b5dbe0448130fddd8955fb219  build/model_capacity_20260927/state/yw_finalfit.npz
d7ac11c35644888ae2738a52de17bd4391a3a18a8223038168f2b57afbe810a4  build/review_20260929_target092/gap_structure/b1_jan.parquet
14e09fda5f19c2b43510b289db06ae2bc54bd90077d86f367d9744c49ba99e99  build/review_20260929_target092/gap_structure/b3_jan.parquet
e8e4dd84bcc5a47eaf80e338ad23c654fd76ad79a0a6d0e770408f9a500baf53  build/review_20260930_next/state_density/X_cap.npy
d4f838c0776e80e771be801e35b413723bc951e72135df5b698310bf0c58ceff  build/review_20260930_next/state_density/meta.parquet
816a9d989e459ae0d562d549b3a4b990457ce3cb91ab7cbfed74d428a8495107  build/review_20260930_next/state_density/parents.npz
45ed32e951dffae4b85b5ea80d525016ec8b256715c7c5031ee8b9f70fac87ea  data/interim/score_sprint_20260921/recency_chain_windows.parquet
f1d9591439c76b13bcfabfa96e8d9984145a20b9f6175ae2e2716e39b1856274  build/review_20260924_ctm/ruler/onset_site/cache/D12_I5_N_train.npz
8150cafa44a3baea2a463e8ae47405212ec9d57b7cbe9d583ed132675492eaed  build/review_20260924_ctm/ruler/onset_site/cache/D12_I5_S_train.npz
888b8352b71c0dce84688fa6525ca0e88a24fef413634efa8040a5a5f8a6b130  build/review_20260924_ctm/ruler/onset_site/cache/D7_I210_W_train.npz
3ba7f619931dafd2e34b47172f29f7fd18767aa3dd5dc046fa18a5d9275342df  build/review_20260924_ctm/ruler/onset_site/cache/D7_I405_S_train.npz
a205085b1bd98b5a81217c60abe0c4d12fb3a171d78f19205dc092bb415229d4  build/review_20260925/onset/train_rows_D12_I5_N.parquet
4968a8da248b4e01c0d0127ac5498dea2cb1c28beb54cc3456db0643f8a5d38d  build/review_20260925/onset/train_rows_D12_I5_S.parquet
8cddcb0cb82c81321327b959caa0d32e132c8fa9cc982898f64f7ecee75f7ef3  build/review_20260925/onset/train_rows_D7_I210_W.parquet
05ed3f837d6b9f206afc9bab825b914becbe6fffbeda8aa20b0aa8051c017809  build/review_20260925/onset/train_rows_D7_I405_S.parquet
37fe73e4893e92b1ed0294ecc0e7ae7a3c9a95e0cf65235b398c1ff8cb9e2494  build/review_20260924_ctm/rep2/replicated_unique_v2.parquet
98ef3ebdb2853b3e7d26d1ac0f63cd31f928eca96b9a538cef925b81b2e6981a  build/review_20260924_ctm/rep2/candidates_v2.parquet
79b65ff1238c034d24034623d7b57e76f5039cc5065c9b6a851cb2929c65b415  build/review_20260924_ctm/rep2/rep2_s0_D12_I5_N.parquet
4b7526e954cf8140d31d8b44a9811bf4ae01e61bb4067429a819379fc82c1c5a  build/review_20260924_ctm/rep2/rep2_s0_D12_I5_S.parquet
cf2b58fa6e4d4a064a51a9ec8d03fdcf4abb909b8ed948dad8057b7fccb86f4e  build/review_20260924_ctm/rep2/rep2_s0_D7_I10_E.parquet
1bde71328c446cf42d9d445931f5f856ef22e32a9f8e0a47e96e46b7953f956a  build/review_20260924_ctm/rep2/rep2_s0_D7_I10_W.parquet
428aced9c44748f7a8dd24fdeb09c885d9f8e132ff4c024faf97ea37c1fc386c  build/review_20260924_ctm/rep2/rep2_s0_D7_I210_E.parquet
d38fc242fd4da23f7fc120b7b8a68a30504846619f5b7bd17189a0a61f35617a  build/review_20260924_ctm/rep2/rep2_s0_D7_I210_W.parquet
b1c625beede1fb774d9b2fb1161531053d388a5d7fd7132e56c6e70a124ee8d0  build/review_20260924_ctm/rep2/rep2_s0_D7_I405_N.parquet
75801956f1910348cb798e7e0d6fdb95bba0497fd3612589d51701aefcb41658  build/review_20260924_ctm/rep2/rep2_s0_D7_I405_S.parquet
b01208d9d08c2f112936251513c6479f2c39517848d4ac67c9e338f5857d14bd  build/review_20260924_ctm/ruler2/rec_train_queue_ongoing_s0.parquet
7386daacd6d5f8453590e2aa0bd5bdc9210bfe803e16589b32c4a40f37068bc4  build/review_20260930_round20/queue_data/cache/fit_matched_base196.npy
2ceab01b3a69ed75e7d98ab3095463b2a937fd29a7fd4f169afbd280788679eb  build/review_20260930_round20/queue_data/cache/held_matched_base196.npy
3436e45a0cb2df752eb6047a1b777cfd1b52ad17b5b17a7bd2d783b733a946d6  build/review_20260930_round20/queue_data/cache/fit_meta.npz
491f425642265f8196cb88ff868d80cfc3349a3576bd93ea0e426cc3712752c6  build/review_20260930_round20/queue_data/cache/held_meta.npz
```

The two `*_matched_base196.npy` hashes are the values recorded in `TRANSFORM_base196.json`; the
others were computed on 2026-11-08. The A25 training tables no longer exist; they can be rebuilt from
`e2/replicated_unique.parquet`.

Besides these tables, a retrain needs the release's train split, a mapping for the absolute paths the
trainers hard-code (the `bin/tfb-run` path layer maps them onto a run tree), and the JSON records that
pin each run's configuration (shipped next to the models). Retrain order:

1. Blend weights → v3 stack → feature caches → k_aux; k\* rule.
2. Replicated windows → G2 rows and profiles → G2.
3. Co-switching blocks → RL tables (January, February) → RLB and RL.
4. Corrected windows → G2v2 rows and templates → v2_F and v2_J.
5. January RLB parent → DK50 head. CAP tables → CAP selection, January refit, final fit.
6. D12 stage-2. January hole rows → c_Jan and F1. CAP parent tables → J1.
7. Onset windows → queue priors and 0922 head → February calibration probabilities.
8. Full train caches → events → u-models and templates; site phase and site rows → FE.
9. A25 tables → A25. rep2 windows, recency → rep2 → N6 refits.
10. Strict blocks; R20 matched tables → transform → M1f and M1sf. BND table.
