# Reproduction checks

Target file: `submission_R26c_R21a_BND_T45.csv`, SHA256
`cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5`, 6,980,503 rows.

## 1. Full run of `run_reproduce.sh`

Run on 2026-11-08 from this repository after the comment pass, on a fresh output folder, with the
released data and the pinned environment (`runtime-versions.txt`):

```
./run_reproduce.sh --data <kaggle_public> --out <new folder> --python <python3.12 with requirements.txt>
```

Result: `submission.csv`, 6,980,503 rows, SHA256
`cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5`: **MATCH**.

| Stage | Wall time |
|---|---:|
| 1. Stage the run tree | 2 s |
| 2. Task 1 / Task 3 state | 54 min 27 s |
| 3. Task 2 onset | 50 s |
| 4. Task 2 ongoing | 4 min 55 s |
| 5. Task 4 path flows | 9 min 51 s |
| 6. Assemble and check | 9 s |
| **Total** | **70 min 14 s** |

Hardware: Apple M4, 10 cores, 16 GB RAM, CPU only. CPU time is 8,731 s user. Peak resident memory
is 5.54 GB, in the second residual learner (CatBoost) step. The output folder holds about 10 GB.

The repository path contains spaces, and the run used it as is.

## 2. Segment runs (2026-11-07)

Each segment ran from a fresh run tree built from the released data. The run tree held only the
package's code and train-derived models and tables. Each output was compared with the file that
produced the submission: SHA256 first, then exact frame equality.

**Task 1 and Task 3 (traffic state).** All six final state sources are byte-identical. These are:

- the gap model;
- two density corrections;
- the D12_I405 correction;
- the ramp-based gap speed;
- the congested-cell density refit.

Of 158 intermediate files:

- 125 are byte-identical.
- 33 differ only in the last bits of a few values: at most 4.3e-14, and 0 after rounding to the
  3 decimals the submission stores. These are per-split LightGBM fits that the pipeline retrains at
  run time, and their summation order varies from run to run.

Wall time was 63 min, with a peak memory of 5.6 GB.

**Task 2 queue onset.** Every intermediate layer and the onset source (87,000 rows) is
byte-identical. Wall time was 60 s.

**Task 2 ongoing queue.** The ongoing source is byte-identical: 87,000 rows, of which 10,028 are
queued. Of the intermediate files, 72 are equal by SHA256 and 3 more are equal by value. Wall time
was 343 s, with a peak memory of 1.85 GB.

**Task 4 path flows.** All 70,708 path flows are byte-identical. Wall time was 15 min, single-threaded.

**Assembly.** The four segment outputs together give SHA256 `cf5869e2…0fc5`: MATCH.

## 3. Training-month access

Both Task 2 segments were rerun on reduced copies of the release, and both runs gave byte-identical
outputs.

- **Ongoing queue.** The training folder held one day only: 2030-06-01. Every other train day and all
  train ramp files were removed.
- **Onset.** All 560 February 2031 train files were removed.

The Task 2 serving code reads raw training data only from the first train day (2030-06-01), and
only its link ids and mileposts, to fix the link order. Everything else it needs from the training
months comes from the shipped models and tables.

## 4. Task 2 entry for a new month (2026-11-10)

After the edits listed in `CHANGES_FROM_ORIGINAL.md` ("Task 2 on a new month"), these runs used
the released data, the pinned environment and fresh output folders.

**Full run, defaults.** `./run_reproduce.sh --data <kaggle_public> --out <new folder> --python <python3.12>`
gave `submission.csv` with 6,980,503 rows and SHA256
`cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5`: **MATCH**. Wall time was
74 min 44 s: Task 1/3 60 min, onset 36 s, ongoing 3 min 55 s, Task 4 9 min 47 s and assembly 7 s.

**Task 2 only, default splits.**

```
./run_reproduce.sh --task2-only --data <kaggle_public> --out <new folder> --python <python3.12>
```

- Wall time: 4 min 30 s (onset 39 s, ongoing 3 min 42 s).
- `final_sources/onset.parquet` (SHA256 `bd0bb6d4…980f`) and `final_sources/ongoing.parquet`
  (`4c3d75c4…36eb`) are byte-identical to the files of the full run in §1.
- All 174,000 queue rows of `submission.csv` are identical to the queue rows of the submitted file.
- Of the 90 other files the two segments write, 85 are byte-identical. The other 5 differ only as
  expected:
  - the three `pipeline/onset_*.py` copies are the edited wrappers;
  - `serving_stage/INPUT_MANIFEST.json` records the run-tree path, the hash of the edited
    `prepare.py` and its own wall time;
  - `onset/production/manifest.json` records its creation time.

**Task 2 only, one split** (`--splits private`).

- Every Task 2 step ran on the private split alone.
- The private rows of the onset and ongoing sources (43,500 each) are identical to the private rows
  of the §1 files.
- Assembly then stopped at the new coverage check: "87000 of 174,000 queue rows not covered". The
  released key also lists the validation rows, so this stop is the intended result.

**Assembly.**

- The edited `pipeline/assemble.py`, run on the §1 final sources, gives SHA256 `cf5869e2…0fc5`:
  **MATCH**.
- With one ongoing row removed, it stops: "1 of 174,000 queue rows not covered".


## 5. Task 2 truncation audit (2026-11-10)

`audit/RESULTS.md` has the method, the full tables and the commands. In short:

- **12 cuts** in March and April, at onset and ongoing windows, including the last window of each
  split. Each cut deletes every observation stamped after a window's origin τ. All 1,216 compared
  windows (1,342,530 queue cells) keep their decisions.
- **Probabilities.** In 3 cuts every probability was compared: all 5,609,904 values are identical to
  the full run (max |Δp| = 0).
- **Negative controls.** Three controls also remove the stamp at τ. Every probability of the window
  at τ changes (max |Δp| 0.139, 0.212 and 0.328), and in one control 3 decisions change. Earlier
  windows stay identical.
- **No refit.** A probe in every Python process of a Task 2 run records no call of a lightgbm,
  scikit-learn, catboost, scipy.optimize or torch fitting routine. Every model is loaded from its
  frozen file.

Four length and completeness checks stopped Task 2 on the truncated data. They were relaxed (see
`CHANGES_FROM_ORIGINAL.md`). After that change, a default `run_reproduce.sh --task2-only` on the
released data gives `onset.parquet` `bd0bb6d4…980f` and `ongoing.parquet` `4c3d75c4…36eb`. These are
byte-identical to the full run of §1 and §4, and all 174,000 queue rows equal the submitted file. Of
the 90 other files the segments write, 86 are byte-identical; the other 4 are two edited wrapper
copies and two manifests that record run paths, code hashes and times.

## 6. Final check from a clean clone, scored on Kaggle (2026-11-10)

The repository was cloned at commit `7c6ffe8` into a new folder, so that only tracked files were
present. `requirements.txt` was installed into a new Python 3.12.14 environment, and the versions
matched the pins. The documentation edits made after this run change no program.

```
./run_reproduce.sh --data <kaggle_public> --out <new folder> --python <python3.12 with requirements.txt>
```

- `submission.csv`, 6,980,503 rows, SHA256
  `cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5`: **MATCH**.
- Wall time 64 min 56 s: Task 1/3 52 min 8 s, onset 29 s, ongoing 3 min 19 s, Task 4 8 min 50 s,
  assembly 8 s. CPU time is 8,429 s user.
- The clone had no new or changed file after the run.
- The file was submitted to Kaggle as submission 58201162. It scored public **0.90115** and private
  **0.89493**, the same as the selected submission 56745207.
