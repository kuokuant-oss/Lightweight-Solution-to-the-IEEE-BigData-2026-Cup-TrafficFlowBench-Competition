# Training programs

These programs trained the frozen models and tables under `trafficflowbench/`. They are kept as the
record of the exact procedure: the reproduction runs none of them. `TRAINING.md` describes each
recipe: data, labels, features, parameters, seeds and cost.

Each file sits at the relative path it had in the development repository. Each is unchanged except
for added comments and docstrings. A check confirmed that the syntax tree with docstrings removed
equals the original's. The original SHA256 values are listed below.

The programs read intermediate tables that are not distributed (`TRAINING.md`, "Intermediate
artifacts"). They also keep the development machine's absolute root, which `bin/tfb-run` maps onto a
run tree.

| Program | Produced | SHA256 of the original |
|---|---|---|
| `scripts/34_task1_blend.py` | data/interim/m11_task1_blend/blend_weights.csv | `d8274c87c606367af714f76c4cdfe772df9762f2a94905397ccda3843808ee99` |
| `scripts/113_task1_stack_probe.py` | build/task1_stack/stack_speed.txt, stack_flow.txt, stack_features.csv | `444110580b48c30ec07af35eb62358c9bcdd2cac87e9f6a31af7b3759ad7e35b` |
| `build/final_sprint_review_20260922/experiment_audit/aux_density_weightedk.py` | build/final_sprint_review_20260922/experiment_audit/state_recency/aux_density_weightedk/k_l1_weightedL_600.txt | `7f43b429c9e30665b969a6590561be89a0a31e98479f60500aa6038b2f711726` |
| `build/review_20260924/work/g2_train.py` | build/review_20260924/work/g2/gap_speed.txt, gap_flow.txt, gap_features.json | `101465c18d789d61e0f893bb0ce076b7e932becda685e08a5d728f6d660b7cfa` |
| `build/review_20260924_newworld/work/blocks/my_blocks.py` | build/review_20260924_newworld/work/blocks/my_blocks.npy (byte-identical copy at build/review_20260924_ctm/ruler/blocks/my_blocks.npy) | `658ccec2ad20aa2904469c42ed9d137e185b7e91c98f6c8cd73f450e15dfbaaf` |
| `build/review_20260924_ctm/rl/fit_final.py` | build/review_20260924_ctm/rl/models/RLB_final/{y_v.txt,y_q.txt,feats.json,manifest.json}, models/RL_final/{y_v.txt,y_q.txt,feats.json,manifest.json} | `e2ec9d4460094a30a7a9ebe27467c99b490fc057cc0f233a590a29eca17513bf` |
| `build/review_20260924_ctm/g2v2/train_v2.py` | build/review_20260924_ctm/g2v2/models/v2_F/gap_speed.txt, gap_flow.txt, gap_features.json | `a070e47755db8713234fc46c75ff29f1b763330d8db2e8e7c697a9ae28564ec7` |
| `build/model_capacity_20260927/state_train.py` | build/model_capacity_20260927/state/C6_long_final/y_v.cbm, y_q.cbm, trial.json; build/model_capacity_20260927/state/data.json | `5984b99c6536e82e301fbdde5076bd04d05d82cf24700974f4987cab4e5734ce` |
| `build/model_capacity_20260927/state_eval.py` | build/model_capacity_20260927/state/selection.json | `0f7fa5cc2d9938cbce2cd5b833b41919ef09f0dfafb3227904ddde778851dd94` |
| `build/model_capacity_20260927/state_pipeline.py` | driver: ran state_eval.py --freeze, state_train.py (refit, prepare_final, final) and state_confirm.py that produced the CAP files above | `9404dfeab19de53d212d381619928ec6611d37506daddbaee0152625a296152a` |
| `build/review_20260929_target092/gap_structure/b3_near.py` | build/review_20260929_target092/gap_structure/b3_sp_M2_janfit.txt | `d739b0b28f1941eb979753bd800590c2c1059be506c599cff5447b042989ea11` |
| `scripts/audit_20260922_fullgrid_probs.py` | build/final_sprint_review_20260922/fullgrid/final_feature_hazard_calib.parquet | `5a54106900ce5d1d75da5d1ddd89f18538ae706c6108fc73aa2c8b0edf33699e` |
| `build/review_20260924_ctm/ruler/onset_site/events.py` | build/review_20260924_ctm/ruler/onset_site/events_train.parquet | `c54c9d16fab61c4903962fc468e08e2d6b9e66e1af73b076088cdb399064bdd1` |
| `build/review_20260925/round2/onset_joint/s6_fe.py` | build/review_20260925/round2/onset_joint/s6_FE_models.pkl | `86f8821526368eb3ee516e44b396aa12e59b2fb1fe6eb2eaa714c0e0c99b659c` |
| `build/review_20260925/round2/onset_joint/s0b_phase.py` | build/review_20260925/round2/onset_joint/site_phase.json | `aa461906c87715e387a90b6ac93283868fc8a5639077ad70040ca19e3fa8cd7a` |
| `build/review_20260924_ctm/rep2/train_rep2.py` | build/review_20260924_ctm/rep2/m_R_V2selrec_s0_seed{0,1,2}.txt, m_R_V2geomselrec_s0_seed{0,1,2}.txt | `8016d77b74f480641867da42dce5f8a2b108749d20d15083dcdca757633ea15e` |
| `build/review_20260929_target092/contract_audit/q4_refit/train_refit.py` | build/review_20260929_target092/contract_audit/q4_refit/m_R_V2selrec_refit_seed0.txt | `a2878f080fbc70bb5c941b494d81fe5f5a8a3160e0c31b5ff7dd2ae9111aed26` |
| `build/review_20260929_target092/contract_audit/q4_refit/train_refit3.py` | build/review_20260929_target092/contract_audit/q4_refit/m_R_V2selrec_refit_seed{1,2}.txt, m_R_V2geomselrec_refit_seed{0,1,2}.txt | `31bedc836feef4a5cab0ea3e4c241960b2c2d88edb13ff4135fcada46e5250ac` |
| `build/review_20260929_target092/contract_audit/q4_refit/run_rest.sh` | driver: ran train_refit3.py for the five refits above (selrec seeds 1-2, geomselrec seeds 0-2) | `2ac79d26d2ccc82754d881ff18c40cb8bd883c2f5d98e801879f3ccc4474cbd5` |
| `build/review_20261001_fresh/mono/m1f_train.py` | build/review_20261001_fresh/mono/models/M1f_seed{0,1,2}.txt | `fc04b6728f391d4bcaa4ad5df67906d8097bcf23916f2ffc0ebe871f9773a56a` |
| `build/review_20261001_fresh/mono/m1sf_train.py` | build/review_20261001_fresh/mono/models/M1sf_seed{0,1,2}.txt | `7f32521161286a26a24f113f2038e3653632b5264854f6e959784f11ddb4a478` |
| `build/review_20260930_round20/evaluation_audit/build_strict_blocks.py` | build/review_20260930_round20/evaluation_audit/strict_blocks_through_20310131.npy | `0e4f5df3c9fd91c2322ad52f2ee2ac6052cb8517f1426fd3a412be744613728f` |
| `build/review_20261001_r26/bnd/bnd_legal.py` | build/review_20261001_r26/bnd/le_table.json | `23aea04cd8211382d3955282aee35fced7b1abc3d4b3ab1918b23aa104474fc5` |
