# Task 2 truncation audit

**Question.** Take a window with forecast origin T. Delete every released observation stamped after
T and rerun Task 2. Does the window's prediction change? A Task 2 window may read only data stamped at
or before its origin. The organisers will check this at award review, and they will also run Task 2 on
a newly generated month.

**Answer.** No. The test used 12 cuts in both months and both window types. Every one of the
1,342,530 compared queue cells kept its decision. In three of the cuts every probability was compared
as well: all 5,609,904 values are identical to the full run (max |Δp| = 0). The negative controls show
that the test detects a leak. With the origin stamp T also removed, 100% of the probabilities of the
window at T change, by up to 0.328, while every earlier window stays identical. No model is fitted
at serving time.

## Method

For each cut τ (the forecast origin of a chosen window), `audit/truncate.py` builds a truncated view
of the release, and `run_reproduce.sh --task2-only` serves Task 2 on it. `audit/compare.py` then
compares, cell by cell, the windows the view still lists with the full run on the released data
(`full-2` in `evidence/reproduction_checks.md` §4, SHA256 `cf5869e2…0fc5`). `audit/run_truncation.sh`
drives the cuts listed in `audit/cuts.csv`. It writes one checkpoint per cut, `results/cut_<name>.json`,
and a line in `results/progress.log` as each finishes. A rerun skips the cuts that have a checkpoint.

**What the view removes.** It removes every row stamped after τ, in every file of the validation and
private splits that the Task 2 code reads.

| File (per corridor and split) | Read by | In the view |
|---|---|---|
| `corridors/<p>/<s>/mainline_states_masked/*/synthetic_mainline_<day>.parquet` | masked-layer caches (`cache/build_cache.py`), the R20 serving inputs (`serving_stage/prepare.py`), recency (`dense/build_recency.py`), the onset features | days after τ left out; the day of τ keeps rows stamped ≤ τ |
| `corridors/<p>/<s>/ramp_states/*/synthetic_ramp_<day>.parquet` | ramp features at T and T−1 (`prepare.py`, the CTM and newworld feature builders) | the same |
| `task2/<p>/<s>/window_history.parquet` | every feature builder (the window's T−60 … T−5) | rows stamped after τ, and the rows of windows whose origin is after τ |
| `task2/<p>/<s>/window_index.csv` | every program (the windows to serve) | windows with origin > τ removed |

A March cut also removes all of April. Unchanged:
- the training split, whose first day is read for the link order;
- the network and config files;
- the question files (`submission_key.csv`, `sample_submission.csv`, the per-task samples), which the
  code maps only to the windows that `window_index.csv` lists.

The view is made of symbolic links to the release plus truncated copies; nothing in the release is
written. `TRUNCATION.json` in each run's folder records the rows removed.

**Which windows are compared.** The splits that hold windows up to τ are served: validation for a
March cut, both splits for an April cut. Every window with origin ≤ τ is compared, in every corridor.
That includes the window at τ itself, and its day file ends at T.

**What is compared.**
- Decisions: the 0/1 queue prediction of every queue row (window, link, step) in
  `final_sources/onset.parquet` and `ongoing.parquet`.
- Probabilities (cuts c03, c07 and c12, and the controls): every continuous output that feeds the
  decisions, value by value:
  - the onset head and the decoded onset component;
  - the A ensemble;
  - the R20 parent, M1f and M1sf seed probabilities;
  - the BND re-prediction `PB`.

## Results

### Cuts (decisions)

| Cut | Window at τ | τ (UTC) | Splits run | Windows compared (onset + ongoing) | Cells compared | Cells changed | Wall time |
|---|---|---|---|---:|---:|---:|---:|
| c01 | `Q2_D7_I210_W_validation_005` (5th onset) | 2031-03-04 11:30 | validation | 52 (31 + 21) | 56,466 | 0 | 222 s |
| c02 | `Q2_D7_I405_S_validation_005` (5th onset, multi-site) | 2031-03-05 07:10 | validation | 63 (36 + 27) | 68,934 | 0 | 243 s |
| c03 | `Q2_D7_I10_E_validation_010` (10th ongoing) | 2031-03-06 13:50 | validation | 71 (39 + 32) | 76,734 | 0 | 216 s |
| c04 | `Q2_D12_I5_S_validation_010` (10th ongoing) | 2031-03-06 15:20 | validation | 72 (39 + 33) | 77,334 | 0 | 297 s |
| c05 | `Q2_D12_I5_N_validation_005` (5th onset, multi-site) | 2031-03-07 12:55 | validation | 73 (40 + 33) | 78,882 | 0 | 297 s |
| c06 | `Q2_D12_I5_N_validation_010` (last of validation) | 2031-03-12 14:10 | validation | 80 (40 + 40) | 87,000 | 0 | 320 s |
| c07 | `Q2_D7_I210_W_private_010` (10th ongoing) | 2031-04-03 12:30 | both | 114 (62 + 52) | 128,274 | 0 | 356 s |
| c08 | `Q2_D12_I5_N_private_006` (1st ongoing) | 2031-04-03 18:20 | both | 119 (63 + 56) | 132,594 | 0 | 335 s |
| c09 | `Q2_D7_I405_N_private_010` (10th ongoing) | 2031-04-04 06:20 | both | 125 (65 + 60) | 141,744 | 0 | 353 s |
| c10 | `Q2_D12_I5_S_private_005` (5th onset, multi-site) | 2031-04-05 13:25 | both | 139 (74 + 65) | 156,120 | 0 | 379 s |
| c11 | `Q2_D7_I405_S_private_005` (5th onset, multi-site) | 2031-04-07 07:10 | both | 148 (78 + 70) | 164,448 | 0 | 396 s |
| c12 | `Q2_D7_I10_W_private_010` (last of private) | 2031-04-17 15:15 | both | 160 (80 + 80) | 174,000 | 0 | 545 s |
| **Total** | **12 cuts** | | | **1,216** | **1,342,530** | **0** | |

No row is missing or extra in any cut. In c01 some corridors have no ongoing window up to τ, and in
c07 one corridor (D7_I10_W) has no April window at all.

### Cuts (probabilities)

| Cut | Probability values compared | Changed | max abs Δp | Of them, the window at τ | Changed |
|---|---:|---:|---:|---:|---:|
| c03 | 1,092,226 | 0 | 0 | 27,156 | 0 |
| c07 | 1,791,678 | 0 | 0 | 15,066 | 0 |
| c12 | 2,726,000 | 0 | 0 | 19,902 | 0 |
| **Total** | **5,609,904** | **0** | **0** | 62,124 | 0 |

The served probabilities are identical, not only the decisions. c12 compares all 160 windows of both
splits.

### Negative controls

A control is a cut that also removes the rows stamped at τ itself, from the masked layer and the
ramps (window histories end at T−5). This takes away the origin observation T of the window at τ.
For example, D12_I5_S has an observed speed at T on 39 of its 100 links.

| Control | Window at τ | Probability values of the window at τ | Changed | Share | max abs Δp | Decision cells at τ changed | Earlier windows: probabilities / decisions changed |
|---|---|---:|---:|---:|---:|---:|---:|
| n02 (onset) | `Q2_D7_I405_S_validation_005` | 218 | 218 | 100% | 0.139 | 0 of 654 | 0 / 0 |
| n04 (ongoing) | `Q2_D12_I5_S_validation_010` | 18,600 | 18,600 | 100% | 0.212 | 0 of 600 | 0 / 0 |
| n07 (ongoing) | `Q2_D7_I210_W_private_010` | 15,066 | 15,066 | 100% | 0.328 | 3 of 486 | 0 / 0 |

Per source, for the window at τ:

| Control | Source | Values | Changed | max abs Δp |
|---|---|---:|---:|---:|
| n02 | onset head, onset component | 109 + 109 | 100% | 0.139 |
| n04 | A ensemble / parent / M1f / M1sf / PB | 600 / 7,200 / 1,800 / 1,800 / 7,200 | 100% | 0.098 / 0.212 / 0.147 / 0.084 / 0.212 |
| n07 | A ensemble / parent / M1f / M1sf / PB | 486 / 5,832 / 1,458 / 1,458 / 5,832 | 100% | 0.202 / 0.328 / 0.252 / 0.229 / 0.328 |

Removing one stamp moves every probability of the window at τ, and in n07 it flips 3 decisions. Every
earlier window stays identical to the last bit. The test therefore detects a single stamp of
information. In the 12 cuts, the window at τ keeps its stamp T and loses only what came after it, and
its probabilities do not move.

A stronger onset control, n02b, removed the stamps T−5 and T. It does not run: `serve_official.official_rows`
asserts that a window's released history covers exactly T−60 … T−5 (`results/n02b_refused.log`). The
code refuses an incomplete history rather than using it. n02 already shows that the onset outputs
respond to the stamp at T.

### No fitting at serving

`audit/run_no_refit.sh` ran `onset.sh` and `ongoing.sh` on the released data with a probe
(`audit/probe/tfb_fit_probe.py`) loaded into every Python process. The probe wraps these entry points
and records each call, then passes it on:
- lightgbm `train`, `cv`, `Booster(train_set=…)` and `Booster.update`;
- every scikit-learn estimator's `fit`, `fit_transform` and `partial_fit`;
- catboost `fit`;
- scipy.optimize `minimize`, `least_squares`, `nnls`, `lsq_linear`, `curve_fit` and `root`;
- numpy `polyfit` and `lstsq`;
- torch `Optimizer.step`.

It also lists the run-tree code each process loaded. The full report is
`results/no_refit_report.md`; the run's onset and ongoing sources are byte-identical to the full
run's.

- **What ran:** 54 Python processes and 69 code files of the run tree.
- **Fitting calls recorded: none of lightgbm, scikit-learn, catboost, scipy.optimize or torch.**
  Every model is loaded from a frozen file (`lgb.Booster(model_file=…)`, pickled u-models and
  templates, frozen tables).
- **The only recorded calls are 502 `numpy.polyfit` calls, each with its internal `lstsq`:**
  - **54** are numpy's own import-time check on macOS (`_mac_os_check`), one per process. Importing
    numpy alone produces it;
  - **260** are the tail-velocity slope in `geom.py`, run in the two geometry builders: a straight
    line through the queue tail's position over the window's last 7 history stamps;
  - **188** are the speed trend in `src/tfb/task2long.py`, run in `prepare.py`: a slope over each
    link's own history stamps.

  The last two groups are features computed on the window's own past, not fitted models.
- **Fit and train calls written in the code that ran: 14 lines.** None of them ran. They are:
  - the training branches of `og_ens.py`, `og_exp.py`, `direct.py`, `m_train_infer.py` and
    `task2oof.py`;
  - the u-model fits in `serve_official.py` and `uncond.py` (their `__main__`);
  - the conditional logit `fit`/`minimize` in `onset_joint/cl.py` and `s2_ruler.py`, called only by
    the ruler functions, which the reproduction never calls;
  - the two polyfit feature lines above.

**Inputs computed per month at run time.** Each comes only from the released data of the split being
served, and the cuts above confirm that none reads past T:
- **recency** (`dense/build_recency.py real <split>`): the share of earlier days of the same split
  with a queue at the link, read from that split's masked layer, with strictly earlier dates only;
- **k_known and spacing** (`r26_serve.official_k`): from the split's `window_index.csv`, the gap to
  the previous window of the same corridor;
- **selection features** (`dense/serve_sel2.py`): replays of the official window selector up to
  T−5, from the window index, the window history and the masked-layer cache of the split;
- **the masked-layer caches and the serving inputs** (`cache/build_cache.py`,
  `serving_stage/prepare.py`).

The Task 2 hazard tables of the onset head are recomputed at each run, from the shipped
training-split features (`onset_features.parquet`, fit dates through 2030-12-31), not from the served
split.

## Package changes the audit required

A view has fewer windows than the released layout, and the window's own day file ends at its origin.
Four kinds of check in the package stopped such a run. They were relaxed without changing any value
on the released data. A default Task-2-only run after the changes gives byte-identical onset and
ongoing sources and the submitted queue rows (`evidence/reproduction_checks.md` §5). The changes are:

1. Fixed counts of the 5-window layout (43,500 rows, 40 or 80 windows, 7,250 onset rows) now follow
   `window_index.csv`, and each step is checked against the serving map that is bound to
   `submission_key.csv`.
2. `PublishedPanel.day` asserted that a masked day file holds all 288 slots of every link.
   `e_data.py` is pinned by a training manifest, so `serving_stage/prepare.py` replaces the method
   with a copy without that one assert. Absent cells stay NaN, as before, and the readers use slots
   up to T. On the organisers' data with the rows after T removed, the original assert would have
   stopped the ongoing segment.
3. A corridor without ongoing windows in a split no longer fails the per-corridor cleanup in
   `prepare.py`.
4. A corridor without windows in a split is skipped by `pipeline/onset_head_serve.py`.

`CHANGES_FROM_ORIGINAL.md` ("Task 2 truncation audit") lists the files and SHA prefixes.

## Run record

- Hardware: Apple M4, 10 cores, 16 GB RAM, CPU only; data on an exFAT drive with 512 KB clusters.
- Wall time: 4,786 s summed over the 15 recorded runs (12 cuts, 3 controls). The runs went 2–3 at a
  time, with 3 to 9 minutes per run.
- The no-refit run took 425 s.
- A view takes about 10 GB on this drive (one cluster per linked file), so each view is deleted
  after its comparison. `out_<name>/` keeps the logs, the final sources, the probability files and
  `TRUNCATION.json`.
- Sequence:
  - all 12 cuts first ran at decision level;
  - c03, c07 and c12 were then rerun with the probability comparison, together with the three
    controls;
  - c12 and n02 were run again after the drive filled up during the first attempt.

  `results/progress.log` has every start and end.

Reproduce:

```bash
audit/run_truncation.sh --release <kaggle_public> --ref <full-run out_dir> --work <new dir> --python <python3.12>
audit/run_no_refit.sh --release <kaggle_public> --out <new dir> --python <python3.12>
python audit/no_refit_report.py --out <no-refit dir> --ref <full-run out_dir>
```

`--ref` needs the full run's tree (its `run/trafficflowbench/` holds the reference probability files).
