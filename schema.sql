-- ===========================================================================
-- AI Cricket Intelligence System — SQLite database schema
-- ===========================================================================
-- This file documents the persistence layer used by src/storage/database.py.
-- The application creates these tables automatically on first run; this file
-- is provided for reference and for manual database initialisation:
--
--     sqlite3 cricket_intelligence.db < schema.sql
--
-- Design notes:
--   * `matches`    : one row per uploaded innings/commentary file.
--   * `deliveries` : one row per parsed ball, storing the structured fields
--                    produced by the deterministic commentary parser. Boolean
--                    flags are stored as INTEGER (0/1) per SQLite convention.
--   * Every analytic the app displays can be recomputed from `deliveries`,
--     which is what keeps results auditable and free of fabrication.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,          -- user-supplied or file-derived label
    uploaded_at TEXT NOT NULL,          -- ISO-8601 UTC timestamp
    format      TEXT,                   -- detected format, e.g. 'T20' / 'ODI'
    raw_text    TEXT                    -- original commentary, for re-parsing
);

CREATE TABLE IF NOT EXISTS deliveries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id         INTEGER NOT NULL,
    seq              INTEGER,           -- global delivery sequence (0-based)
    over_num         INTEGER,           -- completed-over index (0-based)
    ball_num         INTEGER,           -- ball within the over
    over_str         TEXT,              -- original "<over>.<ball>" token
    bowler           TEXT,
    batter           TEXT,              -- striker
    batter_runs      INTEGER,           -- runs off the bat
    extras_total     INTEGER,           -- wides + no-balls + byes + leg-byes
    wides            INTEGER,
    no_balls         INTEGER,
    byes             INTEGER,
    leg_byes         INTEGER,
    total_runs       INTEGER,           -- batter_runs + extras_total
    is_wicket        INTEGER,           -- 0/1
    dismissal_type   TEXT,
    dismissed_batter TEXT,
    fielders         TEXT,              -- comma-separated, if named
    is_four          INTEGER,           -- 0/1, explicit boundary only
    is_six           INTEGER,           -- 0/1, explicit boundary only
    is_boundary      INTEGER,           -- 0/1
    is_legal         INTEGER,           -- 0/1, not a wide/no-ball
    ball_faced       INTEGER,           -- 0/1, counts toward batter balls
    is_dot           INTEGER,           -- 0/1
    bowler_runs      INTEGER,           -- runs charged to the bowler
    event_label      TEXT,              -- e.g. 'FOUR', 'SIX', 'WICKET'
    raw_text         TEXT,              -- original commentary line
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries (match_id);
