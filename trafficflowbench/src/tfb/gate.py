"""submission_compact checkpoint 的 baseline 研究 Gate。"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTES = REPO / "doc" / "reference" / "notes"
PLAN = REPO / "doc" / "plan"

TARGETS = ("task1", "task2", "task3", "task4")
CATEGORIES = ("名次解法", "文獻", "強代碼", "破題")
SOURCE_MARKS = ("[官方]", "[文獻]", "[選手]", "[本地]")
MIN_SOURCES_PER_NOTE = 3
COVERAGE_HEADING = "## 步驟 1 的四類資料 × 四個目標"

CONTAMINATED_TOKENS = (
    "ablation_queue_zero",
    "calib_official_t1",
    "baseline_v1.csv",
    "baseline_v2.csv",
    "0.57849",
    "0.74175",
    "0.73993",
    "0.62880",
)


class GateFailure(RuntimeError):
    """研究 Gate 未通過。"""


def _tables(text: str) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for line in text.splitlines():
        value = line.strip()
        if value.startswith("|") and value.endswith("|"):
            cells = [cell.strip() for cell in value.strip("|").split("|")]
            if all(set(cell) <= set("-: ") for cell in cells):
                continue
            cur.append(cells)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _check_coverage(fail: list[str]) -> None:
    readme = NOTES / "README.md"
    if not readme.is_file():
        fail.append("步驟 1:notes/README.md 不存在")
        return
    text = readme.read_text(encoding="utf-8")
    if COVERAGE_HEADING not in text:
        fail.append(f"步驟 1:缺少「{COVERAGE_HEADING}」")
        return
    tables = _tables(text.split(COVERAGE_HEADING, 1)[1])
    if not tables:
        fail.append("步驟 1:覆蓋表不存在")
        return
    head, *rows = tables[0]
    for category in CATEGORIES:
        if not any(category in cell for cell in head):
            fail.append(f"步驟 1:覆蓋表缺少 {category}")
    names = {row[0].lower() for row in rows if row}
    for target in TARGETS:
        if not any(target in name for name in names):
            fail.append(f"步驟 1:覆蓋表缺少 {target}")
    for row in rows:
        for i, cell in enumerate(row[1:], start=1):
            if not cell or cell in {"-", "—", "?", "TBD"}:
                column = head[i] if i < len(head) else f"第 {i} 欄"
                fail.append(f"步驟 1:{row[0]} 的 {column} 留白")


def _check_notes(fail: list[str]) -> None:
    files = sorted(path for path in NOTES.glob("*.md")
                   if path.name != "README.md" and not path.name.startswith("._"))
    if not files:
        fail.append("步驟 2:notes 目錄沒有筆記")
        return
    for path in files:
        text = path.read_text(encoding="utf-8")
        count = len(re.findall(r"https?://", text)) + sum(text.count(mark) for mark in SOURCE_MARKS)
        if count < MIN_SOURCES_PER_NOTE:
            fail.append(f"步驟 2:{path.name} 只有 {count} 個來源標註")


def _check_plan(fail: list[str]) -> None:
    plans = sorted(path for path in PLAN.glob("baseline*.md") if not path.name.startswith("._"))
    if not plans:
        fail.append("步驟 3:缺少 doc/plan/baseline*.md")
        return
    note_ids = {path.stem.split("-")[0] for path in NOTES.glob("*.md")}
    for path in plans:
        method_rows = 0
        for table in _tables(path.read_text(encoding="utf-8")):
            head, *rows = table
            source_i = next((i for i, cell in enumerate(head) if "來源" in cell), None)
            analysis_i = next((i for i, cell in enumerate(head) if "分析" in cell), None)
            if source_i is None:
                continue
            if analysis_i is None:
                fail.append(f"步驟 3:{path.name} 的方法表缺少分析欄")
                continue
            for row in rows:
                if max(source_i, analysis_i) >= len(row):
                    continue
                method_rows += 1
                cited = re.findall(r"notes/(\d{2})", row[source_i])
                if not cited:
                    fail.append(f"步驟 3:{path.name}「{row[0]}」沒有 notes/NN 來源")
                for note_id in cited:
                    if note_id not in note_ids:
                        fail.append(f"步驟 3:{path.name}「{row[0]}」引用不存在的 notes/{note_id}")
                if not re.search(r"\d", row[analysis_i]):
                    fail.append(f"步驟 3:{path.name}「{row[0]}」沒有已量數字")
        if method_rows == 0:
            fail.append(f"步驟 3:{path.name} 沒有來源／分析方法表")


def _check_contamination(fail: list[str]) -> None:
    here = Path(__file__).resolve()
    for directory in ("doc", "scripts", "src"):
        for path in sorted((REPO / directory).rglob("*")):
            if path.resolve() == here or path.name.startswith("._"):
                continue
            if path.suffix not in {".md", ".py"} or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            hits = sorted(token for token in CONTAMINATED_TOKENS if token in text)
            if hits:
                fail.append(f"污染隔離:{path.relative_to(REPO)} 命中 {hits}")


def check() -> list[str]:
    fail: list[str] = []
    _check_coverage(fail)
    _check_notes(fail)
    _check_plan(fail)
    _check_contamination(fail)
    return fail


def require() -> None:
    # 2026-11-07 使用者授權關閉此閘門（原話：「我 現在授權關掉此閘門」），供最終套件復現 v3 stack 等舊建檔程式。
    # 閘門檢查的是 doc/scripts/src 的文件與字串，不是資料或模型；11-07 的 3 筆命中是存檔副本與血統文件的誤判。
    # 預設關閉；設 TFB_RESEARCH_GATE=on 可恢復原本的檢查。check() 不變，仍可單獨呼叫。
    import os
    if os.environ.get("TFB_RESEARCH_GATE", "off").lower() != "on":
        return
    fail = check()
    if fail:
        detail = "\n".join(f"  ✗ {item}" for item in fail)
        raise GateFailure(f"研究 Gate 未通過，不能訓練模型：\n{detail}")
