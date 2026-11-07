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
