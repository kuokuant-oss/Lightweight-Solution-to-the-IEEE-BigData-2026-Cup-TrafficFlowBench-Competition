"""Copy the six regenerated Task 1/3 state sources of a run into <run>/final_sources/ (submission_id, speed_kmh, flow_vph),
in the order pipeline/assemble.py applies them.

  python task1_final_sources.py <run_root>
"""
# Reproduction pipeline: step 11 of pipeline/task1.sh (log final_sources).
import sys
from pathlib import Path
import pandas as pd

# Argument: the run tree root; b is its trafficflowbench/build folder.
run = Path(sys.argv[1]); b = run / 'trafficflowbench/build'
# (output name, source under build/) in the order assemble.py applies them; a later source overrides
# an earlier one on the ids it lists. G2v2: gap-model hole flow; DK50: density correction; CAP:
# CatBoost residual heads; D12: stage-2 correction on the D12_I405 pair; PRP: two-state hole speeds;
# SG1: density refit on cells whose parent speed is below 85 % of the free-flow speed.
SRC = [('state_1_g2v2.parquet', 'review_20260925/task1/g2v2F_state_rows.parquet'),
       ('state_2_dk50.parquet', 'review_20260927_structural/density/serving/state_updates.parquet'),
       ('state_3_cap.parquet', 'model_capacity_20260927/state/serving/state_updates.parquet'),
       ('state_4_d12.parquet', 'review_20260929_target092/noise_floor/d12_state_updates_S2_100.parquet'),
       ('state_5_prp.parquet', 'review_20260929_target092/main/state_updates_gap_prp.parquet'),
       ('state_6_sg1.parquet', 'review_20260930_next/state_density/density_gate/state_updates_SG1_CAP_DENSITY_GATE.parquet')]
# Copy only the three state columns; each source lists every id once.
out = run / 'final_sources'; out.mkdir(exist_ok=True)
for name, rel in SRC:
    d = pd.read_parquet(b / rel, columns=['submission_id', 'speed_kmh', 'flow_vph'])
    assert d.submission_id.is_unique
    d.to_parquet(out / name, index=False)
    print(f'{name:22s} {len(d):>9,} rows  <- {rel}')
