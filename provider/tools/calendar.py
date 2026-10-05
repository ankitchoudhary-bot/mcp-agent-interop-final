"""
Toy calendar tool backed by SQLite. Real state, real file on disk -- this is
deliberately not an in-memory fake, so restart-persistence and concurrent
access are real concerns, not simulated ones.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "calendar.db"


def init_db(db_path: Path | str = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL
            )
            """
        )
        conn.commit()


@contextmanager
def _connection(db_path: Path | str = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def list_events(start_date: str, end_date: str, db_path: Path | str = DB_PATH) -> list[dict]:
    """
    start_date/end_date are 'YYYY-MM-DD'. Returns events whose start falls
    within [start_date 00:00, end_date 23:59:59], inclusive.
    """
    range_start = f"{start_date}T00:00:00"
    range_end = f"{end_date}T23:59:59"
    with _connection(db_path) as conn:
        cur = conn.execute(
            "SELECT id, title, start, end FROM events "
            "WHERE start >= ? AND start <= ? ORDER BY start ASC",
            (range_start, range_end),
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "title": r[1], "start": r[2], "end": r[3]} for r in rows
    ]


def create_event(title: str, start: str, end: str, db_path: Path | str = DB_PATH) -> dict:
    """
    start/end are ISO 8601 datetimes. Raises ValueError if end <= start --
    this is a domain validation the tool owns, distinct from schema
    validation (which only checks shape, not business rules).
    """
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    if end_dt <= start_dt:
        raise ValueError("Event end must be after event start.")

    with _connection(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO events (title, start, end) VALUES (?, ?, ?)",
            (title, start, end),
        )
        conn.commit()
        event_id = cur.lastrowid

    return {"id": event_id, "title": title, "start": start, "end": end}
