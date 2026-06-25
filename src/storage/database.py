"""
storage/database.py
===================
SQLite persistence. Stores raw uploaded commentary and the parsed delivery
records so analyses are reproducible and auditable. The schema mirrors the
parser output exactly — we persist facts, not derived/guessed values.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from config import DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    format      TEXT,
    raw_text    TEXT
);

CREATE TABLE IF NOT EXISTS deliveries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL,
    seq             INTEGER,
    over_num        INTEGER,
    ball_num        INTEGER,
    over_str        TEXT,
    bowler          TEXT,
    batter          TEXT,
    batter_runs     INTEGER,
    extras_total    INTEGER,
    wides           INTEGER,
    no_balls        INTEGER,
    byes            INTEGER,
    leg_byes        INTEGER,
    total_runs      INTEGER,
    is_wicket       INTEGER,
    dismissal_type  TEXT,
    dismissed_batter TEXT,
    fielders        TEXT,
    is_four         INTEGER,
    is_six          INTEGER,
    is_boundary     INTEGER,
    is_legal        INTEGER,
    ball_faced      INTEGER,
    is_dot          INTEGER,
    bowler_runs     INTEGER,
    event_label     TEXT,
    raw_text        TEXT,
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries (match_id);
"""


class Database:
    def __init__(self, path: str = DATABASE_PATH):
        self.path = path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def save_match(self, name: str, df: pd.DataFrame, raw_text: str,
                   fmt: str = "") -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO matches (name, uploaded_at, format, raw_text) "
                "VALUES (?, ?, ?, ?)",
                (name, datetime.utcnow().isoformat(), fmt, raw_text))
            match_id = cur.lastrowid
            if not df.empty:
                cols = [c for c in df.columns if c in {
                    "seq", "over_num", "ball_num", "over_str", "bowler", "batter",
                    "batter_runs", "extras_total", "wides", "no_balls", "byes",
                    "leg_byes", "total_runs", "is_wicket", "dismissal_type",
                    "dismissed_batter", "fielders", "is_four", "is_six",
                    "is_boundary", "is_legal", "ball_faced", "is_dot",
                    "bowler_runs", "event_label", "raw_text"}]
                rows = []
                for _, r in df.iterrows():
                    rows.append([match_id] + [
                        int(r[c]) if isinstance(r[c], (bool,)) else r[c]
                        for c in cols])
                placeholders = ", ".join(["?"] * (len(cols) + 1))
                conn.executemany(
                    f"INSERT INTO deliveries (match_id, {', '.join(cols)}) "
                    f"VALUES ({placeholders})", rows)
            return match_id

    def list_matches(self) -> List[Dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT match_id, name, uploaded_at, format FROM matches "
                "ORDER BY match_id DESC").fetchall()]

    def load_deliveries(self, match_id: int) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM deliveries WHERE match_id = ? ORDER BY seq",
                conn, params=(match_id,))

    def delete_match(self, match_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM deliveries WHERE match_id = ?", (match_id,))
            conn.execute("DELETE FROM matches WHERE match_id = ?", (match_id,))
