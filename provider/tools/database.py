"""
Read-only SQL query tool over a small seeded dataset.

Two independent layers of write protection, deliberately redundant:
  1. Statement allow-listing: only a single SELECT statement is accepted at
     all (checked before touching the database). Blocks multi-statement
     injection like "SELECT 1; DROP TABLE products;".
  2. SQLite opened in true read-only mode via a `file:...?mode=ro` URI, so
     even a write statement that slipped past (1) would fail at the
     connection level, not just be "policy-disallowed".

Neither layer alone is bulletproof against a sufficiently creative SQL
dialect trick; the two together are the actual guarantee.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.db"

# Anything containing these (case-insensitive, word-boundary) is rejected
# outright regardless of position in the string.
_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|TRIGGER)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(Exception):
    pass


def _validate_readonly_sql(sql: str) -> str:
    stripped = sql.strip().rstrip(";").strip()

    if ";" in stripped:
        raise UnsafeQueryError("Multiple statements are not allowed.")

    if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT statements are allowed.")

    if _FORBIDDEN_KEYWORDS.search(stripped):
        raise UnsafeQueryError(
            "Query contains a keyword that is not allowed in read-only mode."
        )

    return stripped


def init_db(db_path: Path | str = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL
            )
            """
        )
        cur = conn.execute("SELECT COUNT(*) FROM products")
        if cur.fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO products (id, name, category, price) VALUES (?, ?, ?, ?)",
                [
                    (1, "Wireless Mouse", "electronics", 24.99),
                    (2, "Standing Desk", "furniture", 349.00),
                    (3, "Notebook (pack of 3)", "office", 6.50),
                    (4, "Mechanical Keyboard", "electronics", 89.99),
                    (5, "Desk Lamp", "furniture", 22.00),
                ],
            )
        conn.commit()


def query(sql: str, db_path: Path | str = DB_PATH) -> dict:
    """
    Raises UnsafeQueryError if the SQL isn't a plain single SELECT.
    Raises sqlite3.Error for genuine SQL errors (unknown column, syntax, etc).
    """
    safe_sql = _validate_readonly_sql(sql)

    ro_uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(ro_uri, uri=True)
    try:
        cur = conn.execute(safe_sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [list(row) for row in cur.fetchall()]
    finally:
        conn.close()

    return {"columns": columns, "rows": rows, "row_count": len(rows)}
