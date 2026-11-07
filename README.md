# TrafficFlowBench 2026 — Team Anton

> DRAFT 2026-11-07. Hashes, runtimes and the file tables are filled in after the cold runs.

**Learned state reconstruction with physical projections, plus site- and block-structured queue
decoding.** This is the code and the frozen models for our entry in the
[2026 IEEE Big Data Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).
It reproduces our selected final submission from the released data.

| | |
|---|---|
| Selected submission | `submission_R26c_R21a_BND_T45.csv`, Kaggle 56745207 |
| Public leaderboard | 0.90115 |
| Private leaderboard | 0.89493 |
| SHA256 of the submitted CSV | `cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5` |
| Technical report | `TrafficFlowBench_TechnicalReport_Anton.pdf` |

The system has four parts, one per task:

1. **Task 1, state reconstruction (Task 3 is scored on it).**
   - A gradient-boosted stack over five classical reconstructions, refined by a residual model.
   - Two physical density projections, so that the reconstructed speed, flow and density agree with
     the fundamental diagram.
   - A CatBoost capacity blend.
   - The 90-minute holes that the Task 2 windows blank out get their own models:
     - a boundary-conditioned hole model;
     - a two-state (free-flow / queued) speed expectation;
     - a per-split queue probability that reads the on-ramp counts inside the hole as a queue sensor.
2. **Task 2 onset.**
   - Only the step at T+30 is predicted.
   - On corridors with several bottlenecks, a conditional logit picks the queuing site within each
     window, and a train template gives the queue's extent.
   - Elsewhere, train co-switching blocks are decoded.
3. **Task 2 ongoing.**
   - LightGBM link probabilities, with physically monotone constraints on the state features.
   - The selector spacing rule of the official windows is corrected.
   - Decoding is by train blocks at threshold 0.45.
4. **Task 4.** Path flows by regularised non-negative least squares against the released link counts
   and the weak OD prior.

`METHOD.md` describes each step with pointers into the code. `TRAINING.md` documents how every
shipped model was trained. The technical report gives the evaluation and the error analysis.

## Running

Requirements:
- Python 3.12 with the pinned packages in `requirements.txt`. CatBoost 1.2.10 is required for the
  capacity models.
- The released competition data, the `kaggle_public` folder that contains `submission_key.csv`,
  `sample_submission.csv`, `corridors/`, `task1/`, `task2/` and `task4/`.
- CPU only. No GPU is used.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
./run_reproduce.sh --data /path/to/kaggle_public --out /path/to/output_dir
```

The run:
- builds a fresh run tree in the output folder;
- rebuilds every intermediate file from the released data;
- writes `submission.csv`;
- prints its SHA256 next to the expected value above.

Runtime is TODO on a 10-core Apple M-series machine with 16 GB of RAM.

### Notes

- **Task 2 is online.** Each window reads only released data stamped at or before its origin T:
  - the window's own history T−60 … T;
  - same-split days strictly before the query day;
  - the training split.

  Nothing after T is read, including ramps inside the window's own hole and the mainline after the
  hole. The trained Task 2 models were fitted on the training split only (through 2031-02-28).
- **Task 1 is offline.** Some Task 1 heads are fitted at run time on each split's own observed cells
  (pseudo-holes cut from observed data). This is part of the method and runs inside the entry script.
- **Paths.** The programs keep the absolute paths of the machine they were written on.
  `trafficflowbench/scripts/ssd_runtime.py` maps those roots onto the run tree when the program
  starts. Source files are therefore byte-identical to the versions that produced the submission
  (see `CHANGES_FROM_ORIGINAL.md` for the few edits).

## Repository layout

```
run_reproduce.sh           single entry point
requirements.txt           pinned packages
bin/tfb-run                runs one program inside the run tree, with the path mapping
pipeline/                  ordered commands per task, run-tree staging, final assembly
trafficflowbench/          programs and frozen train-only artifacts, at their original relative paths
evidence/                  records of the submitted chain and of our verification runs
METHOD.md  TRAINING.md  CHANGES_FROM_ORIGINAL.md  THIRD_PARTY_NOTICES.md  LICENSE
```

File and folder names keep the prefixes they had during development (`review_20260925/round2`,
`R11c`, `BND`, …), so that they match the recorded runs in `evidence/`. `METHOD.md` §9 explains the
names.

## Data and licence

- No competition data, and no file derived from the validation or private splits, is included.
  Every such file is rebuilt by the run.
- The shipped models were trained on the training split only.
- No external dataset, pretrained model or paid API is used. See `THIRD_PARTY_NOTICES.md` for the
  open-source libraries.
