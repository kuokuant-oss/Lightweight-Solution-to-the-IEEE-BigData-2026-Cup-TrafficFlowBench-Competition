# Ongoing-queue serving inputs (`R20` operator)

`queue_data/serving_stage/prepare.py` runs in `pipeline/ongoing.sh` (step `r20_prepare`). For the 80
official ongoing-queue windows it builds the model inputs from data up to the forecast time T only:

- the released masked day at T;
- the window's own released history (T-60 min to T-5 min);
- ramp counts at T and T-5 min;
- recency features from earlier released days of the same split.

Later steps in `build/review_20261001_*` turn these inputs into queue probabilities and decisions.

## Files kept byte-identical

`queue_data/INPUT_MANIFEST.json` records the SHA256 of 18 code files that were frozen together
with the training data. `serving_stage/prepare.py` checks every one of those hashes before it runs,
so these files have no added comments. `INFERENCE_MANIFEST.json` records the same files and the
inference scripts of the original run.

| File | Role in the reproduction | What it does |
|---|---|---|
| `queue_data/common.py` | Imported by `prepare.py` | Shared settings: the 98 base feature names (from `round19/queue_models/PREPARED_B.json`), the eight corridors, the key columns, the fit period (through 2031-01-31) and the February held-out month, plus `sha` and `dump` helpers. |
| `queue_data/public_operator.py` | Imported by `prepare.py` | `prefix` copies the 12 history slots before T. `source_day` blanks every slot after T. `strict_context` loads the corridor context of `build/review_20260927_direct/direct.py`, whose `features` method builds the window features. |
| `queue_data/e_data.py` | Imported through `public_operator.py` | `PublishedPanel` reads a corridor's released days and builds the encoder input of a window (`encode`, 4 × 9 × 13 per link). Inputs up to T and train truth labels stay in separate structures. |
| `queue_data/prepare_march.py` | Imported by `prepare.py` | `legal_frame` builds the features of one window after writing the released history into the day. `source_poison` checks the time rule: it overwrites every value after T with 987654 and confirms that the features and the encoder input are unchanged. |
| `queue_data/transform.py` | Imported by `prepare.py` | `transform_block` fills missing values with the training-period median, standardises each column with the training mean and standard deviation, and appends one missing-value flag per column (98 → 196 columns, `TRANSFORM_base196.json`). |
| `queue_data/prepare.py`, `freeze_data.py`, `assemble_oof.py`, `train_gbdt.py`, `train_encoder.py`, `infer_gbdt.py`, `infer_encoder.py`, `inference_common.py`, `encoder_reader.py`, `e_architecture.py` | Record | The training run of this operator: build the training tables, freeze data and code hashes, assemble out-of-fold outputs, train the LightGBM and encoder models, run inference on the held-out month. |
| `tabm_models/model.py`, `model_embedded.py`, `train.py` | Record | Training code for TabM, a tabular neural-network ensemble, which this run evaluated. |
| `tabm_runtime/tabm.py`, `rtdl_num_embeddings.py` | Record | Third-party TabM and numerical-embedding code by Yandex Research (see `THIRD_PARTY_NOTICES.md`). |

The record files and the TabM code are listed in `INPUT_MANIFEST.json`, so they ship with the
package. The reproduction imports none of them. The pinned environment has no PyTorch, which these
files need.
