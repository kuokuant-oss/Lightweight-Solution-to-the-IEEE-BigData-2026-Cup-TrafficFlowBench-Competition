# External data, models and third-party code

## External data

None. The entry uses only the released competition package. No real-world traffic data (PeMS or
other) is used by any component of the submitted file.

## Pretrained models and paid or rate-limited APIs

None. Every model in `trafficflowbench/build/` was trained by us on the training split of the
released data.

## Third-party code included in this repository

| Code | Where | Source | Licence |
|---|---|---|---|
| LRTC-TNN tensor completion (truncated nuclear norm), copied verbatim from the reference notebook | `trafficflowbench/src/tfb/baselines.py` (`lrtc_tnn`) | Xinyu Chen, transdim, `imputer/LRTC-TNN.ipynb`, https://github.com/xinychen/transdim | MIT, Copyright (c) 2018 Xinyu Chen |
| Parts of the official scoring code (Task 2 IoU aggregation; Task 3 mixed field and conservation terms), copied for local validation and for the Task 1 physical projection | `trafficflowbench/src/tfb/score2.py`, `trafficflowbench/src/tfb/physics.py` | TrafficFlowBench organizers, public repository (pin `c88cddf`) | MIT, Copyright (c) 2026 TrafficFlowBench organizers |
| TabM 0.0.3 (`tabm.py`) and rtdl_num_embeddings 0.0.12 (`rtdl_num_embeddings.py`), copied unchanged | `trafficflowbench/build/review_20260930_round20/tabm_runtime/` | Yandex Research, https://github.com/yandex-research/tabm and https://github.com/yandex-research/rtdl-num-embeddings | TabM: Apache-2.0, Copyright 2024 Yandex LLC. rtdl_num_embeddings: MIT, Copyright (c) 2022 Authors of "On Embeddings for Numerical Features in Tabular Deep Learning" |

The TabM files ship because the ongoing-queue serving step checks the SHA256 of every code file
listed in its training manifest (`trafficflowbench/build/review_20260930_round20/README.md`). The
reproduction imports neither file, and the pinned environment has no PyTorch.

The full licence texts are in `third_party/`.

## Libraries (installed, not bundled)

| Package | Version | Licence |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause |
| pandas | 2.3.3 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| pyarrow | 25.0.1 | Apache-2.0 |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| LightGBM | 4.7.0 | MIT |
| CatBoost | 1.2.10 | Apache-2.0 |
| joblib | 1.6.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
