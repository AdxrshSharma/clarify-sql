"""Safe SQL execution: read-only connection + SQLite authorizer + timeout.

Defence in depth, because an LLM wrote this SQL:
1. Only a single SELECT/WITH statement is accepted.
2. The database file is opened read-only (mode=ro) with PRAGMA query_only.
3. A SQLite authorizer denies every action except SELECT/READ/FUNCTION/RECURSIVE.
4. A progress handler aborts queries that run longer than `timeout_s`.
"""
import re
import sqlite3
import time

from common import DB_PATH

# Numeric action codes (stable across Python versions)
SQLITE_READ, SQLITE_SELECT, SQLITE_FUNCTION, SQLITE_RECURSIVE = 20, 21, 31, 33
_ALLOWED = {SQLITE_READ, SQLITE_SELECT, SQLITE_FUNCTION, SQLITE_RECURSIVE}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def execute_safe(sql: str, db_path=DB_PATH, max_rows: int = 200, timeout_s: float = 5.0):
    sql = sql.strip().rstrip(";").strip()
    if ";" in sql:
        raise ValueError("Multiple statements are not allowed")
    if not re.match(r"(?is)^(select|with)\b", sql):
        raise ValueError("Only SELECT queries are allowed")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.execute("PRAGMA query_only = ON")
        con.set_authorizer(_authorizer)
        start = time.time()
        con.set_progress_handler(lambda: 1 if time.time() - start > timeout_s else 0, 10_000)
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows)
        cols = [d[0] for d in cur.description]
        return cols, rows
    finally:
        con.close()
