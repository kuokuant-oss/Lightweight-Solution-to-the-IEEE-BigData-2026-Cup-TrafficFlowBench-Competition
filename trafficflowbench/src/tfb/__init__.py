"""Core helpers shared by the TrafficFlowBench programs.

The modules hold the release paths and official scoring constants
(`release`), the per-day link x time grid (`grid`), local copies of the
official Task 1 and Task 2 scorers (`score1`, `score2`), Task 3 physics
scoring (`physics`), run provenance stamps (`prov`), the research gate
(`gate`), the Task 1 bias calibration (`calib`), the Task 1 fills and stack
(`baselines`, `predict1`, `task1stack`) and the Task 2 feature, prior and
evaluation code (`task2*`). Programs import the package as `src.tfb` after
putting the `trafficflowbench/` folder on `sys.path`.
"""

