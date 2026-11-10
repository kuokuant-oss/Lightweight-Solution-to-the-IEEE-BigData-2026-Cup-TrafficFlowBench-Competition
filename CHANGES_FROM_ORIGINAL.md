# Changes from the original sources

Every program in `trafficflowbench/` keeps the code of the version that produced the submission, except
the files listed here. SHA prefixes are original -> package, taken before the comment pass below. New
programs written for this package live in `pipeline/` and are listed per section.

## Comments and docstrings (2026-11-08)

English module, class and function docstrings and `#` comments were added to 161 code files in
`trafficflowbench/`, `pipeline/`, `bin/`, `run_reproduce.sh` and `docs/figures/`. The pass only
inserted lines. Every original line stays unchanged and in order, and no code line was added. A
check confirmed this file by file:

- **Python:** the syntax tree with docstrings removed is identical to the uncommented version.
- **Shell:** only whole-line comments were added, none inside a here-document, and `sh -n` passes.

The full reproduction run after the comment pass gave the same submission SHA256 (see
`evidence/reproduction_checks.md`).

26 files keep their exact bytes, because their SHA256 is recorded in a training manifest that a
program checks before serving. Their explanations are in the README of their folder:

- `build/review_20260927_research/{common,structure,full_pool,feb_pool,strength}.py`
  (`frozen_recipe.json`);
- `build/review_20260930_round20/queue_data/*.py` (top level), `tabm_models/*.py` and
  `tabm_runtime/*.py` (`INPUT_MANIFEST.json`);
- `scripts/strongest_20260922_ongoing.py` (`round19/queue_models/PREPARED_B.json`).

## Training programs (2026-11-08)

`training/` holds the 23 programs that trained the frozen models and tables, at their development
paths. Each was copied unchanged and then given comments and docstrings by the same insertion-only
pass, which was checked against the original file. `training/README.md` lists the original SHA256 of
each program. The reproduction runs none of them.

## Shared (path layer and research gate)

| File (under `trafficflowbench/`) | What | Why |
|---|---|---|
| `scripts/portable_paths.py` (`aabeb251` -> `48fa4e87`) | `_tail()` tests the run tree's own root before the old roots, and `rebase_text()` rebases all roots in one regex pass with the run tree's root listed first. | The path layer maps the development machine's absolute roots onto the run tree. Without the change, a run tree placed under one of those old roots was mapped onto itself a second time (doubled path). Behaviour is unchanged for run trees anywhere else. |
| `src/tfb/gate.py` (`60bd2178` -> `a3ef25a5`) | `require()` returns immediately unless `TFB_RESEARCH_GATE=on`; `check()` is unchanged. | The research gate is a development-time check of project notes and documents; it does not look at data or models. It is off unless `TFB_RESEARCH_GATE=on`. In the reproduction, `pipeline/od.py` imports `scripts/60_build_submission.py`, which calls `require()` at import; with `TFB_RESEARCH_GATE` unset the call returns at once. The v3 stack trainer (a training record) calls it too. |


## Onset / OD

Edits to copied files (package copies only; the development tree is unchanged). Each edit is marked
`Package edit (2026-11-07)` in the source. SHA prefixes are original -> package.

| File (under `trafficflowbench/`) | What | Why |
|---|---|---|
| `build/review_20260924_ctm/ruler/onset_site/oracle_site.py` (`545ab652` -> `39a51ee9`) | The six import-time lines that load the March pseudo-val ruler (`fakeval` labels/windows, `pv_prod.parquet`, `pv_sites.parquet`) and the validation-cache link maps are replaced by `lab = w = pred0 = ps = None; LI = {}`. | Gap G8. `serve_official` -> `eval_template` -> `oracle_site` runs these loads on every import. They feed only the ruler functions (`rep`, `restrict`, `eval_template.apply`), which the reproduction never calls, and the ruler files are validation-split derived, so they are not shipped. |
| `build/review_20260925/round2/onset_joint/common.py` (`17bf8295` -> `605a0946`) | Three import-time statements set to `None`: `SITE_PV` (March pseudo-val site rows), `BLK`/`U_PV` (ruler inputs), `CH_D`/`S_D`/`FR_D` (D's score on the ruler). | Gap G8, same reason. What sv2 uses (`TPL`, `UMODELS`, `NEV`, `choices`, `MS`, `SO`) is unchanged. |
| `build/review_20260924_newworld/work/blocks/build_D_inputs.py` (`f638709f` -> `1dfa448e`) | (1) `Z` (the C3 folder) changed from a temporary folder that was deleted to `.../work/onset_site`; `os` added to the import line. (2) The ongoing block-mean loop runs only for `../feat/ogens_{tag}_prob.parquet` files that exist. | (1) Gap G2: the C3 component is recomputed by `pipeline/onset_c3.py` into `work/onset_site/` (where the preserved copy sat). (2) The ongoing half needs the ongoing segment's A-ensemble probabilities; the onset run has none, and R26c does not use `D_ongoing_*`. With those files present the loop runs as before. |
| `build/review_20260925/round2/synthesis/sv2_official_FEA1.py` (`6cb504f6` -> `6ddbd507`) | (1) The A1 (`onset_extent/onset_A1_official.parquet`) and FE (`onset_joint/real/onset_FE_<split>.parquet`) references are read, and assertions (2) and (3) run, only when those files exist. (2) After writing `onset_FEA1_official.parquet`, the program exits when the G1 parent CSV is absent instead of failing its SHA assert. | (1) Parity-only reads of validation/private-derived files; a cold run does not have them. The development run `onset-1` borrowed them and both assertions passed with 0 mismatches. Assertion (1) (D reproduction) always runs. (2) The G1-based candidate CSV is not part of the reproduction; `pipeline/assemble.py` writes the submission. |

Artifacts reduced (not byte-identical to the original, same path):

| File | What | Why |
|---|---|---|
| `build/review_20260924_ctm/cache/{D12_I5_N,D12_I5_S,D7_I210_W,D7_I405_S}_train.npz`, `build/review_20260924_ctm/ruler/onset_site/cache/{D12_I5_N,D12_I5_S,D7_I405_S}_train.npz` | Only the `links` array is kept (`np.savez(dst, links=z['links'])`; dtype and order unchanged, equal to the validation caches' `links`). Originals 70-170 MB each (405 MB for the four panels), package copies 3-8 KB. | The onset serving path reads only `z['links']` from the train caches (`serve_official.decode`, sv2 `LIt`). The full arrays are raw train values including February 2031; rebuilding them with `build_cache.py <panel> train` would read the February raw files. |

Recreated symlink: `build/review_20260925/onset/fakeHERE/cache -> ../../../review_20260924_ctm/cache` (as in the
development tree; sv2 sets `serve_official.HERE` to `fakeHERE`).

New programs (not copies; in `pipeline/`, staged into `<run>/trafficflowbench/pipeline/` by the shell
scripts because the path layer maps the workspace roots onto the run tree):
- `onset_head_serve.py`: the serving block of `scripts/145_onset_prior_forward.py` (case `final`, arm
  `feature_hazard`) with the saved model instead of a refit; imports 145 for `Hazard`, `identity`,
  `predictions`, `decode`.
- `onset_c3.py`: recomputes D's C3 component (gap G2) with `serve_official.official_rows/score_rows/decode`
  and `common.choices`, the way `sv2_official_FEA1.py` recomputes D's choice.
- `onset_sources.py`: maps `onset_FEA1_official.parquet` through `submission_key.csv` to `final_sources/onset.parquet`.
- `od.py`: imports `scripts/60_build_submission.py` unchanged, points its `OUT` at `<run>/work/od/`, calls
  `build_odme()` only (gap G10), and keys the result to `final_sources/odme.parquet`.
- `onset.sh`, `od.sh`: the ordered commands.

## Ongoing

Segment: Task 2 queue_ongoing (`pipeline/ongoing.sh`). Paths are relative to `trafficflowbench/`.
Every other file of this segment is a byte-identical copy (checked against the development file list, which is not shipped). Edits are marked
`package edit` / `package addition` in the source. SHA prefixes are original -> package.

### Edited copies

| File | What | Why |
|---|---|---|
| `build/review_20260930_round20/queue_data/serving_stage/prepare.py` (`1c8e77e4` -> `9c07c61d`) | In `submission_binding()`: removed `assert sha(PARENT)==EXPECTED_PARENT_SHA`; removed the chunk-by-chunk comparison of `sample_submission.csv` against the R16b CSV (`pi=pd.read_csv(PARENT…)`, `zip_longest`, `a.equals(b)`); the loop now iterates the sample alone and keeps its own assertion that IDs are `1..N` in order; `PARENT` dropped from the hashed `files`; manifest field `sample_R16b_all_ID_task_order_exact` now reads `'not checked (package edit: R16b binding removed)'`. 5 lines; feature code untouched. | The R16b CSV is a whole earlier submission (state + queue + OD), which this segment cannot regenerate. The read was a provenance/order check only; no value from it entered a feature or the output mapping (the mapping is built from `submission_key.csv`). |
| `build/review_20260929_target092/main/serve_refit.py` (`c2d084a1` -> `22ea0b25`) | (1) The hard-coded guard (`GUARD = 'Q2_D12_I5_N_validation_007'`, line removed) and the champion-CSV read and parity gate (`deliverables/2026-09-27/capacity/submission_HMP50_DK50_CAP.csv`; the `CHAMP` constant is left but no longer read) are replaced by the generic never-empty rule: it imports `never_empty()` from `review_20260925/make_never_empty_fallback.py`, merges that program's A-block table, and for both the production decode (`bp0 >= .5`, `d0`) and the refit decode (`bp1 >= .5`, `d1`) gives every window with no queued cell its A-block decode. The windows where it fires are printed and written to `main/never_empty_fallback.json`. (2) One added output: `main/ongoing_refitN6_d1.parquet` (submission_id, window_id, timestamp, link_id, split, d1), the refit decision on all 87,000 official ongoing rows. The mix (`.75·mean(N6) + .25·A`), `my_blocks` block mean and `>= .5` are unchanged. | No window ID may be hard-coded, since Task 2 may be run on a new month. The CAP CSV is a whole earlier submission that this segment cannot regenerate; its ongoing cells were `bp0 >= .5` (originally asserted, 0/87,000 mismatches) plus the hand-made guard cells. On this release the generic rule fires on exactly one window, `Q2_D12_I5_N_validation_007`, for both decodes, and gives the same 15 cells as the hand-made guard, so the override file and R16b bits are byte-identical (checked in development; the segment report is not shipped). R16b's ongoing cells are CAP + the override = `d1`, which the R16b bits step needs on every row. |

### New files (package additions; each replaces a hand step or a file this segment cannot read)

| File | What | Replaces |
|---|---|---|
| `build/review_20260924_newworld/work/feat/ogens_serve.py` | Serve-only part of `og_ens.py`: imports `og_ens` for `MEMBERS` (names, feature lists) and uses `og_exp.Q.predict`; averages the 6 A25 members on `serve_{realval,realpriv}` + `geom_serve_{…}` and writes `ogens_{realval,realpriv}_prob.parquet`. Models read from `models/og_*.txt`. | `og_ens.py main()`, which also loads the deleted training tables and calibrates on February 2031 train rows (not needed to serve). |
| `build/review_20260925/make_never_empty_fallback.py` | Defines `never_empty(window_id, decision, fallback)` (a window whose decision has no queued cell takes its fallback cells; the R20 operator's rule) and, run as a program, writes `candidates/a_block_fallback.parquet` for every official ongoing row of both splits: the A25 link probability averaged within (window, timestamp, `my_blocks` block), `>= .5` (the arithmetic of the ongoing half of `work/blocks/build_D_inputs.py`). | The hand-made `candidates/guard_val007.parquet` (one hard-coded window, no generator) and the `GUARD` constant. Its `Q2_D12_I5_N_validation_007` rows equal the hand-made file value for value. |
| `build/review_20261001_fresh/serve/make_r16b_bits.py` | Maps `ongoing_refitN6_d1.parquet` onto `serving_stage/<split>_SUBMISSION_MAP` rows → `R16b_ongoing_bits.npz` (int8, keys validation/private), the file `s2_decide.r16b_bits()` loads. | `r16b_bits()` reading the R16b CSV. |
| `build/review_20261001_r22/serve/make_r21a_bits.py` | Saves `decisions_C1f_<split>.npz['delta']` → `R21a_ongoing_bits.npz` (int8), the file `r22_serve.r21a_bits()` loads. | `r21a_bits()` reading the R21a CSV (which it asserts equals that delta). |
| `build/review_20261001_r26/serve/make_ongoing_source.py` | `decisions_BNDT45_<split>.npz` (`submission_id`, `delta`) → `<run>/final_sources/ongoing.parquet` (submission_id int64, queue_pred int8; validation then private). | `r26_serve_t45.py patch` (streams the R21a CSV). |

### Pipeline-level substitutions (no file edited)

- `review_20260924_ctm/serve/run_real.sh` is not run: its first lines wait for rep2 training builds and rebuild the train recency table. `ongoing.sh` runs its serve-side commands directly (`build_serve_feats.py`, `build_geom.py serve`, `dense/build_recency.py real`) with the same arguments.
- `build_serve_feats.py` writes `serve_<tag>.parquet` into its working directory (the original runs `cd`'d into `ctm/serve` and `newworld/work/feat`); under `tfb-run` the working directory is always the repo root, so `ongoing.sh` moves the file (shell `mv`, the only command not run through `tfb-run`) to the folder the original run left it in, where `ogens_serve.py` and `serve_refit.py` read it. This keeps the builder byte-identical; the alternative would be editing it to take an output path.
- `cache/build_cache.py <panel> <split>` is run for validation and private only (the selector's masked-layer cache); the train caches are not needed to serve.
- `s3_patch_csv.py`, `r22_serve.py patch` and `r26_serve*.py patch` (which stream earlier whole-submission CSVs) are not run; `pipeline/assemble.py` writes the CSV from `final_sources/`.

## Task 1/3

All edits are in package copies under `trafficflowbench/`; the live tree is unchanged. Every edited line
carries a `# package edit` comment. No edit changes a computed value: each one removes or guards a read whose only
purpose was a parity check, a research gate, or a SHA check of a whole historical CSV, or it serves a frozen model
instead of refitting it. Where a SHA check is softened, the program's own row-by-row value check on the rows it uses
stays in force (named below).

| File | What | Why |
|---|---|---|
| `build/review_20260924_ctm/rl/serve_feats.py` | (1) `champ` is read only if the champion CSV exists; the per-day parity block runs inside `if champ is not None:` (re-indented, unchanged), `ku` (k* updates) is read only for that check, and the parity report is written only when checked. (2) New module dict `STACK`: the v3 stack values of each day are kept before G2 replaces the holes, and a new side output `feats/c1all_serve_<panel>_<split>.parquet` (submission_id, stack_v, stack_q, c1_v, c1_q, hole) is written for every key row. | (1) The champion CSV and k* updates are read only to report parity with the shipped champion; in a cold run they do not exist yet (k* is computed after C1). (2) The v3 state (step 2 CSV) and the champion state (step 5 CSV) are inputs of the k* builder and of `apply_rl.py`; the side output lets `pipeline/task1_state_csv.py` rebuild them without a second LRTC pass. The serve/blk tables are byte-identical to the originals (40/40 SHA-equal). |
| `build/review_20260927/iteration2/hole_mixture_serve.py` | `assert sha(T1 CSV) == PARENT_SHA` -> prints whether the SHA matches. | The T1 parent CSV is a regenerated state-only stand-in (queue/OD rows are placeholders). The program's own check `rounded(rebuilt T1) == parent_speed` on all 128,465 hole rows stays. |
| `build/review_20260927_research/serve_candidate.py` | Same for the HM25 parent CSV SHA; the February/validation research gate on `strength/results.json` runs only if that file exists. | Own check `rounded(rebuilt HM25) == parent_speed` (== 0) stays. `strength/results.json` mixes validation and February research results and is not shipped (the validation part that downstream code reads is rebuilt by `pipeline/task1_strength_validation.py`). |
| `build/review_20260927_structural/build_candidate.py` (DK50) | March confirmation gate (`density/march_float64/results.json`) runs only if the file exists (also its SHA in the output manifest: `None` when absent); parent SHA assert -> printed status. | The March gate file is built from the validation split and is not shipped. Own check `dv == dq == 0` (parent state == RLB+k* recomposition on every non-hole row of the 8 panels) stays. |
| `build/model_capacity_20260927/build_state_candidate.py` (CAP) | Same pattern: March gate (`state/march/results.json`) only if present (manifest SHA `None` when absent); parent SHA -> printed status. The February gate (`state/confirmation/report.json`, train-derived) is shipped and still asserted. | Own check `dv == dq == 0` (parent == DK50 recomposition) stays. |
| `build/review_20260929_target092/noise_floor/d12_serve.py` | If `d12_prod_stage2_y_{v,q}.txt` exist, they are loaded instead of refitting (the original fit block is kept under `else:`); the validation self-supervised check runs only for tables that exist. | The stage-2 fit reads the January+February train feature caches; the heads are frozen artifacts. The check is research-only. Output `d12_state_updates_S2_100.parquet` is byte-identical to the original. |
| `build/review_20260930_next/state_density/j1_official.py` | `read_parent`: SHA assert -> printed status; `serve`: the March/April month gates run only for result files that exist. | The CAP and R11c parents are regenerated in the run. `parity` mode still asserts CAP == R11c on all 6,010,695 general keys before `serve` reads R11c. |
| `build/review_20260930_next/state_density/density_gate/sg1_official.py` | The month gate dict keeps only result files that exist. | March/April gate files are not shipped. The source checks (J1 updates SHA `223e4009…`, model SHA, J1 manifest) still run and pass. |

New package files (not copies of the development tree):
- `pipeline/task1.sh` — the ordered segment.
- `pipeline/task1_state_csv.py` — writes state-only stand-ins for the historical parent submission CSVs that
  programs read (v3, D, G1_FE_A1, T1, HM25, HMP50, R11c), from the keyed files regenerated in the run, in the number
  format each original builder wrote. Queue/OD rows are placeholders; they never feed Task 1.
- `pipeline/task1_strength_validation.py` — the validation part of `review_20260927_research/strength.py`
  (`strength/validation_predictions.parquet`), without its February research scoring.
- `pipeline/make_state_updates_gap_prp.py` — copy of `fixes/make_state_updates_gap_prp.py` (FIXES F2).
- `pipeline/task1_final_sources.py` — copies the six keyed sources to `<run>/final_sources/`.

## Task 2 on a new month (2026-11-10)

The organisers will rerun Task 2 on a newly generated month with the same file layout. These edits
make the run easy to point at new data and change nothing on the released data: on it, every
default reproduces the earlier outputs byte for byte (see `evidence/reproduction_checks.md` §4).
No model, threshold or decision rule changed. SHA prefixes are package before -> after.

- `run_reproduce.sh` (`c0697fe1` -> `18787b95`): `--task2-only` runs stages 1, 3, 4 and 6 and
  skips Task 1 (which checks the SHA256 and row counts of the released data) and Task 4.
  `--splits` sets `TFB_SPLITS`. `--help` prints the whole header comment.
- `pipeline/assemble.py` (`9f845056` -> `f828b3eb`): every queue row of `submission_key.csv` must
  now be covered by the onset or the ongoing source. Before, an uncovered queue row was written as
  0 without a message. `--queue-only` writes the full layout with the state and odme rows set to 0.
- `TFB_SPLITS` (default `validation private`) replaces the split names that were written into the
  Task 2 code. The newworld file tags stay `realval` / `realpriv` for the released splits (other
  splits get `real<split>`), the step-log names stay `*_val` / `*_pri`, and the counts that are
  totals over both splits are now the per-split count times the number of splits. (The per-split
  counts were then taken from `window_index.csv`; see "Task 2 truncation audit" below.) Each edited
  Python file is marked `Package edit (2026-11-10)`.

| File | SHA prefix |
|---|---|
| `pipeline/onset.sh` | `5b791fbe` -> `d822d379` |
| `pipeline/ongoing.sh` | `ebd6f2f0` -> `8b29a959` |
| `pipeline/onset_head_serve.py` | `81e3c05e` -> `62968bff` |
| `pipeline/onset_c3.py` | `8f63cb11` -> `bdf7fce0` |
| `pipeline/onset_sources.py` | `a0080995` -> `b459b497` |
| `build/review_20260924_newworld/work/blocks/build_D_inputs.py` | `79a33dfd` -> `85f56d55` |
| `build/review_20260925/make_never_empty_fallback.py` | `c17dee5c` -> `cd07bf78` |
| `build/review_20260925/round2/synthesis/sv2_official_FEA1.py` | `04a9a3f6` -> `ca7d5faf` |
| `build/review_20260929_target092/main/serve_refit.py` | `7e11bcad` -> `aae9b18d` |
| `build/review_20260930_round20/queue_data/serving_stage/prepare.py` | `9c84b298` -> `c159cb90` |
| `build/review_20261001_fresh/serve/make_r16b_bits.py` | `deafed61` -> `304d1014` |
| `build/review_20261001_fresh/serve/s1_serve_prob.py` | `92c3f582` -> `8d818a30` |
| `build/review_20261001_fresh/serve/s2_decide.py` | `908d1024` -> `4a540269` |
| `build/review_20261001_r22/serve/make_r21a_bits.py` | `45b2b41d` -> `f0f5be97` |
| `build/review_20261001_r22/serve/r22_serve.py` | `2e3e399a` -> `c3fbc549` |
| `build/review_20261001_r26/serve/make_ongoing_source.py` | `4644f504` -> `c7a7be85` |

None of these files is pinned by a training manifest. The 26 pinned files listed above keep their
bytes.

## Task 2 truncation audit (2026-11-10)

`audit/` reruns Task 2 on copies of the release with every observation after a cut removed (see
`audit/RESULTS.md`). The cut views have fewer windows than the released layout, and the window's
own day file ends at its origin. Four kinds of check stopped such a run. They are relaxed as follows;
on the released data every value and output is unchanged (`evidence/reproduction_checks.md` §5).
SHA prefixes are the previous commit -> this one.

- Fixed row and window counts of the 5-window layout now follow `window_index.csv`. Each step checks
  its rows against the serving map that `prepare.py` binds to `submission_key.csv`:
  `pipeline/onset_sources.py` (`b459b497` -> `f0830f7f`; 6 key rows per step-6 decision),
  `serving_stage/prepare.py`, `make_never_empty_fallback.py` (`cd07bf78` -> `30a9cf26`),
  `make_r16b_bits.py` (`304d1014` -> `36dc81e1`), `s1_serve_prob.py` (`8d818a30` -> `28bc912c`),
  `make_r21a_bits.py` (`f0f5be97` -> `941a5886`), `r22_serve.py` (`c3fbc549` -> `543d40d4`) and
  `make_ongoing_source.py` (`c7a7be85` -> `31b15519`).
- `PublishedPanel.day` (`queue_data/e_data.py`) asserted that a masked day file holds all 288 slots
  of every link. `e_data.py` is pinned by `queue_data/INPUT_MANIFEST.json` and keeps its bytes;
  `serving_stage/prepare.py` (`c159cb90` -> `05d6d9c8`) replaces the method with a copy that lacks
  only this assert. Rows are placed by timestamp in the links x 288 grid as before, absent cells stay
  NaN, and its readers use the slots up to T.
- A panel without ongoing windows in a split: `prepare.py` no longer fails on the per-panel cleanup.
- A panel without windows in a split: `pipeline/onset_head_serve.py` (`62968bff` -> `90e33419`) skips
  it.
