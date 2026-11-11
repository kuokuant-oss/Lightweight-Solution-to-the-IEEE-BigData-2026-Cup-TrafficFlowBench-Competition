# TrafficFlowBench 2026 — Team Anton

**Learned state reconstruction with physical projections, plus site- and block-structured queue
decoding.** This is the code and the frozen models for our entry in the
[2026 IEEE Big Data Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).
It reproduces our selected final submission from the released data.

| | |
|---|---|
| Selected submission | `submission_R26c_R21a_BND_T45.csv`, Kaggle 56745207 |
| Public leaderboard | 0.90115 |
| Private leaderboard | 0.89493 (rank 12) |
| SHA256 of the submitted CSV | `cf5869e23d8bf063d37e6636355f66b8385be1ee7e7257a9943a16783ac00fc5` |
| Technical report | sent to the organisers with the final package |

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

`docs/METHOD_OVERVIEW.md` describes each step, with three method diagrams in `docs/figures/`.
`TRAINING.md` documents how every shipped model was trained. Every program carries docstrings that
explain its role, inputs and outputs. The technical report gives the evaluation and the error
analysis.

## Running

Requirements:
- Python 3.12 with the pinned packages in `requirements.txt`. CatBoost 1.2.10 is required for the
  capacity models.
- The released competition data, the `kaggle_public` folder that contains `submission_key.csv`,
  `sample_submission.csv`, `corridors/`, `task1/`, `task2/` and `task4/`.
- CPU only. No GPU is used.

Install the environment with pip:

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

or with [uv](https://docs.astral.sh/uv/), which also downloads Python 3.12 when it is missing:

```bash
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Then run:

```bash
./run_reproduce.sh --data /path/to/kaggle_public --out /path/to/output_dir --python .venv/bin/python
```

The run:
- builds a fresh run tree in the output folder;
- rebuilds every intermediate file from the released data;
- writes `submission.csv`;
- prints its SHA256 next to the expected value above.

A full run takes 70 to 75 minutes on an Apple M4 (10 cores, 16 GB RAM), with a peak memory of 5.5 GB.
The output folder needs about 10 GB.

### Running Task 2 on a new month

```bash
./run_reproduce.sh --task2-only --data /path/to/new_month --out /path/to/output_dir --python .venv/bin/python
```

This stages the run tree, runs `pipeline/onset.sh` and `pipeline/ongoing.sh`, and writes
`submission.csv`. Task 1 (state) and Task 4 (path flows) are skipped, because Task 2 reads none of
their outputs. Task 1 also checks the SHA256 and row counts of the released data, so it stops on any
other data. The file keeps the full submission layout: the queue rows are predicted, and the state
and odme rows are 0. The SHA256 is printed but not compared. Task 2 takes about 5 minutes (4 min 30 s measured).

What to change:

- **`--data`** is the data root: a folder with the layout of the released `kaggle_public` folder.
  It is the only path to set. The run tree links it read-only, and every program reads the data
  through that link.
- **`--splits "<a> <b> …"`**, or the environment variable `TFB_SPLITS`, is the list of Task 2
  splits to serve. The default is `"validation private"`. Every queue row of `submission_key.csv`
  must belong to a listed split. `pipeline/assemble.py` stops when a queue row has no prediction.

Assumptions about the new data:

- the same 8 corridors with the same links, under `corridors/` and `task2/`;
- the training split as released (`corridors/<corridor>/train/`, `task2/<corridor>/train/`). The
  serving code reads only the first train day (2030-06-01), for the link order. Everything else it
  needs from training comes from the shipped models and tables;
- `task2/<corridor>/<split>/window_index.csv` and `window_history.parquet` for each split, and the
  masked layer `corridors/<corridor>/<split>/mainline_states_masked/`;
- `submission_key.csv` and `sample_submission.csv` with the queue rows of those windows;
- windows of 6 steps each (T+5 … T+30). The number of windows per corridor is read from
  `window_index.csv`; a corridor may have none in a split. Every window listed in
  `window_history.parquet` has its full 12-stamp history (T−60 … T−5).

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
  starts. The programs therefore keep the code of the versions that produced the submission;
  `CHANGES_FROM_ORIGINAL.md` lists the few edits and the comment pass, which added docstrings and
  comments only.

## Repository layout

```
run_reproduce.sh           single entry point
requirements.txt           pinned packages
bin/tfb-run                runs one program inside the run tree, with the path mapping
pipeline/                  ordered commands per task, run-tree staging, final assembly
trafficflowbench/          programs and frozen train-only artifacts, at their original relative paths
training/                  the programs that trained the frozen models (record; not run)
docs/                      method overview and diagrams
evidence/                  submission record, reproduction and time-rule checks
audit/                     Task 2 truncation audit (delete everything after T, rerun, compare)
third_party/               licence texts of the included third-party code
checksums.txt              SHA256 of every file; check with `shasum -a 256 -c checksums.txt`
TRAINING.md  CHANGES_FROM_ORIGINAL.md  THIRD_PARTY_NOTICES.md  LICENSE
```

File and folder names keep the prefixes they had during development (`review_20260925/round2`,
`R11c`, `BND`, …), so that they match the recorded runs and manifests. The docstrings and the folder
READMEs explain each name.

## Data and licence

- No competition data, and no file derived from the validation or private splits, is included.
  Every such file is rebuilt by the run.
- The shipped models were trained on the training split only.
- No external dataset, pretrained model or paid API is used. See `THIRD_PARTY_NOTICES.md` for the
  open-source libraries.
