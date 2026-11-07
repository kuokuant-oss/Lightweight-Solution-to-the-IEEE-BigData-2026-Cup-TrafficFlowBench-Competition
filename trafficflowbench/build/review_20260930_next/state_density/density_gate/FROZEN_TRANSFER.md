# SG1 transfer freeze

February 唯一 arm 通過：兩半 SG1−C1 total_B 為 +0.0000140646／+0.0000087115，合併 +0.0000102813，原共享日期 full-support bootstrap CI [+0.0000056137,+0.0000145534]。沒有讀取任何新 SG1 March/April 分數。

立即凍結 parent gate `round3 CAP speed < .85 × saved float32 vf`，dtype／乘法完全沿用 February；等於 threshold 時不修改。沿用 J1 full-February head SHA `37b6d4363ab0316ea9e47df7f1e436fb89f7dd6085390c4dfaaa8230227187d6`、原 .5 density projection 與 rounding。Gate 限八-panel general rows，沒有 per-split/per-panel 調整，也沒有新 fit。

三／四月唯一測試 SG1 與既有 C0/C1，必須 exact reproduce C0/C1 daily sums。每個月 SG1−C1、SG1−C0 的 macro10 total 都 ≥0；只接受這個固定規則，否則保留原 J1/R16a。全部 gate 通過才從原 C1 keyed updates 選取 parent gate=true 的 subset，old values 與 parent 仍為 R11c，另寫新檔；不修改原產物，不建新 CSV。
