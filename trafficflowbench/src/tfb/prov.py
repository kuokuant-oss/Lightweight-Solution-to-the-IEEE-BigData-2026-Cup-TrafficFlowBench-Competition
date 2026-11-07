"""Run provenance. Every validation_v3 artifact is identified by these fields.

HANDOFF.md §7.5: a number without a base commit, a data/window manifest SHA, a
seed and a config is not evidence. This module is what makes that cheap to
satisfy, so no run can be written without them.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import release as R


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Returns the SHA-256 hex digest of a file, read in `chunk`-byte blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_frame(d: pd.DataFrame) -> str:
    """Content hash of a table, independent of row order and index."""
    cols = sorted(d.columns)
    s = d[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(s, index=False).values.tobytes()).hexdigest()


def git_state() -> dict[str, str]:
    """Returns the git state of the repository that holds REPO.

    Returns:
        dict with `commit`, `branch`, `dirty_tracked` ('yes' when a tracked
        file differs from the commit), the first 20 `tracked_changes` lines and
        the count of untracked files `n_untracked`.
    """
    def run(*args: str) -> str:
        """Runs one git command in REPO; returns its stripped stdout or an `<unavailable: ...>` note."""
        try:
            return subprocess.run(["git", *args], cwd=R.REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as exc:                                  # pragma: no cover
            return f"<unavailable: {exc}>"
    # Tracked-vs-untracked matters here: a run writes its own stamps and tables,
    # which are untracked until committed, so a plain `status --porcelain` would
    # report every run as dirty and the field would carry no information.
    # dirty_tracked is the one that answers "is the committed source what ran".
    tracked = run("status", "--porcelain", "--untracked-files=no")
    untracked = run("ls-files", "--others", "--exclude-standard")
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked": "yes" if tracked else "no",
            "tracked_changes": tracked.splitlines()[:20],
            "n_untracked": len(untracked.splitlines()) if untracked else 0}


def env_stamp() -> dict[str, str]:
    """Returns the Python, platform, numpy, pandas, scipy and lightgbm versions."""
    import numpy, scipy                                           # noqa: E401
    try:
        import lightgbm
        lgb_v = lightgbm.__version__
    except Exception:                                             # pragma: no cover
        lgb_v = "<unavailable>"
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": numpy.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "lightgbm": lgb_v}


# HARNESS_SOURCE: repository files shared by every validation_v3 Task 2 run;
# source_state() hashes them into each stamp ("<missing>" for absent files).
# CONTRACT: the written validation protocol, hashed with them.
HARNESS_SOURCE = ("src/tfb/task2pool.py", "src/tfb/task2build.py", "src/tfb/task2eval.py", "src/tfb/task2cand.py", "src/tfb/task2oof.py",
                  "src/tfb/task2feat.py", "src/tfb/score2.py", "src/tfb/release.py",
                  "src/tfb/prov.py", "scripts/63_v3_pool.py", "scripts/64_v3_features.py",
                  "scripts/65_v3_selfcheck.py", "scripts/66_v3_paired_eval.py", "scripts/68_oof_baseline.py",
                  "scripts/70_reweight_experiment.py", "src/tfb/task2abl.py")
CONTRACT = "doc/plan/validation-v3.md"


def source_state(extra: tuple[str, ...] = ()) -> dict:
    """SHA-256 of every file that decides a validation_v3 number, plus the contract.

    A stamp that records only ``git.commit`` cannot show that the promotion
    rule was not edited after the fact: a dirty tree hides both the code and
    the criteria. Hashing them into the stamp makes any later edit visible,
    and a stamp taken on a clean commit ties the whole run to that commit.

    ``extra`` carries the files specific to one candidate -- its own script and
    any module only it uses. HARNESS_SOURCE lists what every validation_v3 run
    shares; without ``extra`` a candidate's own code would never be hashed, so
    the stamp could not show which version of the candidate produced the number.
    """
    out = {}
    for rel in tuple(HARNESS_SOURCE) + tuple(extra):
        p = R.REPO / rel
        out[rel] = sha256_file(p) if p.exists() else "<missing>"
    c = R.REPO / CONTRACT
    return {"harness_source": out,
            "contract": {CONTRACT: sha256_file(c) if c.exists() else "<missing>"}}


def stamp(extra_sources: tuple[str, ...] = (), **extra) -> dict:
    """Builds the provenance record of one run.

    Args:
        extra_sources: repository-relative files specific to this run, hashed
            together with HARNESS_SOURCE.
        **extra: further fields copied into the record (e.g. kind, inputs,
            outputs).

    Returns:
        dict with `utc` (UTC time of the call), `git`, `env`, `source` and the
        extra fields.
    """
    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(), "env": env_stamp(),
            "source": source_state(extra_sources), **extra}


def write_stamp(path: str | Path, payload: dict) -> None:
    """Writes a stamp as indented JSON with sorted keys, creating the parent folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                          encoding="utf-8")
