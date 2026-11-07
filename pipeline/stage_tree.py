"""Create a fresh run tree from the package.

  python pipeline/stage_tree.py --run <run_root> --data <kaggle_public dir> [--borrow <list.txt>]

<run_root>/trafficflowbench   copy of <package>/trafficflowbench (code and train-only artifacts)
<run_root>/tfb_competition_data/kaggle_public -> --data (symlink; the release is only read)

--borrow is for development only: a text file of workspace-relative paths (one per line) that
are COPIED from the development workspace into the run tree, to stand in for upstream outputs
that another pipeline segment will regenerate. They are listed in <run_root>/BORROWED.txt so a
run that used them can never be mistaken for a cold run.
"""
import argparse, hashlib, os, shutil, sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DEV_WORKSPACE = Path(os.environ.get("TFB_DEV_WORKSPACE", "/Volumes/X9-2TB/trafficflow"))


def ignore(_dir, names):
    return [n for n in names if n.startswith("._") or n in ("__pycache__", ".DS_Store")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--borrow")
    a = ap.parse_args()
    run = Path(a.run).absolute()
    if run.exists() and any(run.iterdir()):
        sys.exit(f"refusing to reuse a non-empty run tree: {run}")
    data = Path(a.data).absolute()
    for need in ("submission_key.csv", "sample_submission.csv", "corridors"):
        if not (data / need).exists():
            sys.exit(f"--data does not look like the released kaggle_public folder (missing {need})")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PKG / "trafficflowbench", run / "trafficflowbench", ignore=ignore, symlinks=True)
    (run / "tfb_competition_data").mkdir()
    (run / "tfb_competition_data" / "kaggle_public").symlink_to(data, target_is_directory=True)
    if a.borrow:
        lines = [l.strip() for l in Path(a.borrow).read_text().splitlines() if l.strip() and not l.startswith("#")]
        rec = []
        for rel in lines:
            src, dst = DEV_WORKSPACE / rel, run / rel
            if not src.exists():
                sys.exit(f"borrow source missing: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
                rec.append(f"{rel}/  (directory)")
            else:
                shutil.copy2(src, dst)
                rec.append(f"{rel}  {hashlib.sha256(dst.read_bytes()).hexdigest()}")
        (run / "BORROWED.txt").write_text("\n".join(rec) + "\n")
        print(f"borrowed {len(rec)} paths (development run, not cold)")
    print(f"run tree ready: {run}")


if __name__ == "__main__":
    main()
