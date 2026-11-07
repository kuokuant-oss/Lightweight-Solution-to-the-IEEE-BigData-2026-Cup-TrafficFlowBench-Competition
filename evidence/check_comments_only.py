"""Checks that edits to code files add only comments and docstrings.

Compares every tracked code file in a git work tree with its version at a
base revision (default HEAD) and fails when an edit could change what the
code does.

Rules:
  * Python files: the syntax tree with docstrings removed is identical, and
    every original line is kept unchanged and in order (edits only insert
    lines). The shebang and coding lines stay on top.
  * Shell files: edits only insert blank or whole-line comment lines, none
    inside a here-document, and the file passes `sh -n`.
  * Pinned files (their committed SHA256 is recorded in another tracked
    file, so a program verifies their bytes at run time) stay byte-identical.

Usage:
  python3 check_comments_only.py <repo_dir> [--base REV] [file ...]

In this repository the uncommented code is the first commit:
  python3 evidence/check_comments_only.py . --base "$(git rev-list --max-parents=0 HEAD)"

Exit status is 0 when every checked file passes, 1 otherwise.
"""

import ast
import difflib
import hashlib
import os
import re
import subprocess
import sys

DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
CODING_RE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")
HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
BASE = "HEAD"


def git(repo, *args):
    """Runs a git command in `repo` and returns its raw stdout."""
    return subprocess.run(["git", "-C", repo, *args], check=True,
                          capture_output=True).stdout


def code_files(repo):
    """Lists tracked Python and shell files (symbolic links excluded)."""
    out = []
    for f in git(repo, "ls-files", "-z").decode().split("\0"):
        if not f or os.path.islink(os.path.join(repo, f)):
            continue
        if f.endswith((".py", ".sh")) or f == "bin/tfb-run":
            out.append(f)
    return out


def pinned_files(repo, files):
    """Returns files whose committed SHA256 appears in another tracked file."""
    texts = []
    for f in git(repo, "ls-files", "-z").decode().split("\0"):
        p = os.path.join(repo, f)
        if (f and f != "checksums.txt" and not os.path.islink(p)
                and f.endswith((".json", ".py", ".txt", ".csv", ".md"))
                and os.path.getsize(p) < 50_000_000):
            texts.append(git(repo, "show", f"{BASE}:{f}").decode("utf-8", "ignore"))
    blob = "\n".join(texts)
    pinned = set()
    for f in files:
        h = hashlib.sha256(git(repo, "show", f"{BASE}:{f}")).hexdigest()
        if h in blob:
            pinned.add(f)
    return pinned


def strip_docstrings(tree):
    """Removes the docstring statement of the module, each class and function."""
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:]
    return tree


def only_insertions(old_lines, new_lines):
    """Returns the inserted line indices (in new) or None if any line changed."""
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    inserted = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag != "insert":
            return None
        inserted.extend(range(j1, j2))
    return inserted


def check_python(old, new):
    """Returns a list of problems for one Python file (empty when it passes)."""
    problems = []
    try:
        t_old = strip_docstrings(ast.parse(old))
        t_new = strip_docstrings(ast.parse(new))
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    if ast.dump(t_old) != ast.dump(t_new):
        problems.append("syntax tree differs after removing docstrings")
    old_lines, new_lines = old.splitlines(), new.splitlines()
    if only_insertions(old_lines, new_lines) is None:
        problems.append("an original line was changed or removed")
    if old_lines and old_lines[0].startswith(b"#!") and new_lines[:1] != old_lines[:1]:
        problems.append("shebang line moved or changed")
    old_cookie = [l for l in old_lines[:2] if CODING_RE.match(l)]
    if old_cookie and old_cookie != [l for l in new_lines[:2] if CODING_RE.match(l)]:
        problems.append("coding line moved or changed")
    return problems


def heredoc_bodies(lines):
    """Returns the set of line indices that sit inside a here-document."""
    inside, i = set(), 0
    while i < len(lines):
        m = HEREDOC_RE.search(lines[i])
        if m and not lines[i].lstrip().startswith("#"):
            end = m.group(2)
            j = i + 1
            while j < len(lines) and lines[j].strip() != end:
                inside.add(j)
                j += 1
            i = j
        i += 1
    return inside


def check_shell(path, old, new):
    """Returns a list of problems for one shell file (empty when it passes)."""
    problems = []
    old_lines = old.decode().splitlines()
    new_lines = new.decode().splitlines()
    inserted = only_insertions(old_lines, new_lines)
    if inserted is None:
        return ["an original line was changed or removed"]
    body = heredoc_bodies(new_lines)
    for j in inserted:
        line = new_lines[j]
        if line.strip() and not line.lstrip().startswith("#"):
            problems.append(f"inserted non-comment line {j + 1}: {line[:60]}")
        if j in body:
            problems.append(f"inserted line {j + 1} sits inside a here-document")
        if j == 0 and old_lines and old_lines[0].startswith("#!"):
            problems.append("line inserted above the shebang")
    r = subprocess.run(["sh", "-n", path], capture_output=True, text=True)
    if r.returncode:
        problems.append(f"sh -n failed: {r.stderr.strip()}")
    return problems


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    repo = os.path.abspath(argv[1])
    rest = argv[2:]
    global BASE
    if rest[:1] == ["--base"]:
        BASE, rest = rest[1], rest[2:]
    files = rest or code_files(repo)
    pinned = pinned_files(repo, code_files(repo))
    n_bad = n_changed = 0
    for f in files:
        old = git(repo, "show", f"{BASE}:{f}")
        with open(os.path.join(repo, f), "rb") as fh:
            new = fh.read()
        if old == new:
            continue
        n_changed += 1
        if f in pinned:
            problems = ["pinned file: its SHA256 is checked at run time"]
        elif f.endswith(".py"):
            problems = check_python(old, new)
        else:
            problems = check_shell(os.path.join(repo, f), old, new)
        for p in problems:
            print(f"FAIL {f}: {p}")
        n_bad += bool(problems)
    print(f"checked {len(files)} files, {n_changed} edited, {n_bad} failing, "
          f"{len(pinned)} pinned")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
