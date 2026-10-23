"""Run with: python -m pytest tests -q   (or: python tests/test_basic.py)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import parse_output  # noqa: E402
from sql_guard import execute_safe  # noqa: E402


def blocked(sql):
    try:
        execute_safe(sql)
    except Exception:
        return True
    return False


def test_guard_blocks_writes_and_multi_statements():
    for bad in ["DROP TABLE campaigns", "DELETE FROM line_items", "SELECT 1; DROP TABLE campaigns",
                "UPDATE campaigns SET total_budget = 0", "PRAGMA table_info(campaigns)",
                "ATTACH DATABASE 'x.db' AS x", "WITH t AS (SELECT 1) DELETE FROM campaigns"]:
        assert blocked(bad), bad


def test_guard_allows_select_and_caps_rows():
    cols, rows = execute_safe("SELECT * FROM daily_performance", max_rows=5)
    assert len(rows) == 5 and "report_date" in cols
    cols, rows = execute_safe("WITH t AS (SELECT 1 AS x) SELECT x FROM t")
    assert rows == [(1,)]


def test_parse_output():
    assert parse_output("CLARIFY: Which metric?") == ("clarify", "Which metric?")
    assert parse_output("```sql\nSELECT 1;\n```") == ("sql", "SELECT 1")


def test_all_gold_sql_runs():
    for split in ("val", "test"):
        for line in open(f"data/{split}.jsonl"):
            r = json.loads(line)
            if r["kind"] != "ambiguous":
                execute_safe(r["target"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
