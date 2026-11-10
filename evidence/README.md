# Evidence

These are the records of the selected submission and of our checks. They let a reader verify the
result without rerunning the pipeline. No file derived from the validation or private splits is
included.

| File | Contents |
|---|---|
| `submission.json` | Kaggle record of the selected submission: reference 56745207, file name, time, public and private score, SHA256. |
| `reproduction_checks.md` | Full run of `run_reproduce.sh` (time, peak memory, SHA256), the earlier segment runs compared file by file with the files that produced the submission, the checks on which training files the serving code reads, and the runs of the Task-2-only entry for a new month. |
| `runtime-versions.txt` | Python, package versions and hardware of the verified runs. |
| `check_comments_only.py` | Proof that the comment pass changed no code. It compares every code file with the uncommented first commit. |

To run the comment check from the repository root:

```bash
python3 evidence/check_comments_only.py . --base "$(git rev-list --max-parents=0 HEAD)"
```
