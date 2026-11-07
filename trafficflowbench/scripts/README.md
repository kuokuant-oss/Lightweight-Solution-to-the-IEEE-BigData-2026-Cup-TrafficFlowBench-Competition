# Scripts

September builders and audits (`60_build_submission.py`, `112_build_submission_v3.py`,
`145_onset_prior_forward.py`, `audit_*.py`, `build_strongest_state_transfer_serving_20260922.py`) and
the path layer that runs them inside a run tree (`ssd_runtime.py`, `portable_paths.py`,
`portable_env.py`, `ssd_bootstrap/`). Each file carries its own comments.

## File kept byte-identical

`strongest_20260922_ongoing.py` has no added comments. Its SHA256 is recorded in
`build/review_20260930_round19/queue_models/PREPARED_B.json`, the record of the ongoing-queue
training tables. The ongoing feature builders (`build/review_20260924_newworld/work/feat/` and
`build/review_20260924_ctm/ruler/feat/build_serve_feats.py`) import it.

| Function | What it does |
|---|---|
| `windows`, `all_dates`, `identities` | Read the ongoing-queue windows of the development table, list the train dates, and give each corridor and link an integer code. |
| `net_columns` | Ramp observations at the forecast time T and at T-5 min; on-ramps and off-ramps stay separate. |
| `features` | Builds and caches the per-link, per-step feature frame of the windows. |
| `priors` | Adds each link's queue prior, built from the given training dates. |
| `train`, `predict` | Fit LightGBM with per-window weights and an early-stopping period; the prior can enter as a starting offset (logit of the prior). |
| `frame`, `score`, `calibrate` | Build the official IoU frame at a threshold, average IoU over corridors, and pick the threshold with the best mean IoU on corridors with at least 10 windows over 5 dates. |
| `run_case`, `main` | Run one predeclared experiment with time-ordered fit, early-stopping, calibration and evaluation periods. |
