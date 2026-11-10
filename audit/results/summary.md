| Cut | Window at τ | τ (UTC) | Splits run | Windows compared (onset + ongoing) | Cells compared | Cells changed | Rows missing / extra | Wall time |
|---|---|---|---|---:|---:|---:|---:|---:|
| c01 | `Q2_D7_I210_W_validation_005` | 2031-03-04T11:30Z | validation | 52 (31 + 21) | 56,466 | 0 | 0 / 0 | 222 s |
| c02 | `Q2_D7_I405_S_validation_005` | 2031-03-05T07:10Z | validation | 63 (36 + 27) | 68,934 | 0 | 0 / 0 | 243 s |
| c03 | `Q2_D7_I10_E_validation_010` | 2031-03-06T13:50Z | validation | 71 (39 + 32) | 76,734 | 0 | 0 / 0 | 216 s |
| c04 | `Q2_D12_I5_S_validation_010` | 2031-03-06T15:20Z | validation | 72 (39 + 33) | 77,334 | 0 | 0 / 0 | 297 s |
| c05 | `Q2_D12_I5_N_validation_005` | 2031-03-07T12:55Z | validation | 73 (40 + 33) | 78,882 | 0 | 0 / 0 | 297 s |
| c06 | `Q2_D12_I5_N_validation_010` | 2031-03-12T14:10Z | validation | 80 (40 + 40) | 87,000 | 0 | 0 / 0 | 320 s |
| c07 | `Q2_D7_I210_W_private_010` | 2031-04-03T12:30Z | validation private | 114 (62 + 52) | 128,274 | 0 | 0 / 0 | 356 s |
| c08 | `Q2_D12_I5_N_private_006` | 2031-04-03T18:20Z | validation private | 119 (63 + 56) | 132,594 | 0 | 0 / 0 | 335 s |
| c09 | `Q2_D7_I405_N_private_010` | 2031-04-04T06:20Z | validation private | 125 (65 + 60) | 141,744 | 0 | 0 / 0 | 353 s |
| c10 | `Q2_D12_I5_S_private_005` | 2031-04-05T13:25Z | validation private | 139 (74 + 65) | 156,120 | 0 | 0 / 0 | 379 s |
| c11 | `Q2_D7_I405_S_private_005` | 2031-04-07T07:10Z | validation private | 148 (78 + 70) | 164,448 | 0 | 0 / 0 | 396 s |
| c12 | `Q2_D7_I10_W_private_010` | 2031-04-17T15:15Z | validation private | 160 (80 + 80) | 174,000 | 0 | 0 / 0 | 545 s |
| **Total** | 12 cuts | | | 1,216 | 1,342,530 | 0 | 0 / 0 | |

| Control | Window at τ | τ (UTC) | Cells of the window at τ | Of them changed | Cells of earlier windows | Of them changed | Windows that changed |
|---|---|---|---:|---:|---:|---:|---|
| n02 | `Q2_D7_I405_S_validation_005` | 2031-03-05T07:10Z | 654 | 0 | 68,280 | 0 | none |
| n04 | `Q2_D12_I5_S_validation_010` | 2031-03-06T15:20Z | 600 | 0 | 76,734 | 0 | none |
| n07 | `Q2_D7_I210_W_private_010` | 2031-04-03T12:30Z | 486 | 3 | 127,788 | 0 | `Q2_D7_I210_W_private_010` |

| Run | Window at τ | Stamps removed at τ | Probability values compared | Windows before τ: changed | max abs Δp | Window at τ: values | changed | share | max abs Δp | Decision cells changed at τ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c03 | `Q2_D7_I10_E_validation_010` | 0 | 1,092,226 | 0 | 0 | 27,156 | 0 | 0.0% | 0 | 0 of 876 |
| c07 | `Q2_D7_I210_W_private_010` | 0 | 1,791,678 | 0 | 0 | 15,066 | 0 | 0.0% | 0 | 0 of 486 |
| c12 | `Q2_D7_I10_W_private_010` | 0 | 2,726,000 | 0 | 0 | 19,902 | 0 | 0.0% | 0 | 0 of 642 |
| n02 | `Q2_D7_I405_S_validation_005` | 1 | 943,162 | 0 | 0 | 218 | 218 | 100.0% | 0.139 | 0 of 654 |
| n04 | `Q2_D12_I5_S_validation_010` | 1 | 1,110,826 | 0 | 0 | 18,600 | 18,600 | 100.0% | 0.212 | 0 of 600 |
| n07 | `Q2_D7_I210_W_private_010` | 1 | 1,791,678 | 0 | 0 | 15,066 | 15,066 | 100.0% | 0.328 | 3 of 486 |

| Run | Source | Values (window at τ) | Changed | Share | max abs Δp |
|---|---|---:|---:|---:|---:|
| n02 | onset_head | 109 | 109 | 100.0% | 0.139 |
| n02 | onset_component | 109 | 109 | 100.0% | 0.139 |
| n04 | ongoing_A | 600 | 600 | 100.0% | 0.0981 |
| n04 | ongoing_parent | 7,200 | 7,200 | 100.0% | 0.212 |
| n04 | ongoing_M1f | 1,800 | 1,800 | 100.0% | 0.147 |
| n04 | ongoing_M1sf | 1,800 | 1,800 | 100.0% | 0.0839 |
| n04 | ongoing_PB | 7,200 | 7,200 | 100.0% | 0.212 |
| n07 | ongoing_A | 486 | 486 | 100.0% | 0.202 |
| n07 | ongoing_parent | 5,832 | 5,832 | 100.0% | 0.328 |
| n07 | ongoing_M1f | 1,458 | 1,458 | 100.0% | 0.252 |
| n07 | ongoing_M1sf | 1,458 | 1,458 | 100.0% | 0.229 |
| n07 | ongoing_PB | 5,832 | 5,832 | 100.0% | 0.328 |
