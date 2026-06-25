"""
analytics/batting.py and bowling logic  (Modules 3 & 4)
=======================================================
All figures are aggregated directly from the parsed delivery DataFrame produced
by the commentary parser. Nothing here estimates, predicts, or invents a value.
If the DataFrame is empty, every function returns empty structures so the UI can
render "Cannot determine from provided commentary." instead of a fabricated stat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


# --------------------------------------------------------------------------- #
# Module 3 — Batting analytics
# --------------------------------------------------------------------------- #
def batting_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Per-batter scorecard. Rows ordered by batting appearance order."""
    if df.empty:
        return pd.DataFrame()

    # A batter "faces" a delivery when ball_faced is True (excludes wides).
    faced = df[df["ball_faced"]].copy()

    rows = []
    first_seen = df.groupby("batter")["seq"].min().to_dict()
    for batter in sorted(df["batter"].unique(), key=lambda b: first_seen[b]):
        bf = faced[faced["batter"] == batter]
        all_b = df[df["batter"] == batter]
        runs = int(bf["batter_runs"].sum())
        balls = int(len(bf))
        fours = int(bf["is_four"].sum())
        sixes = int(bf["is_six"].sum())
        dots = int(((bf["batter_runs"] == 0) & (bf["extras_total"] == 0)).sum())
        singles = int((bf["batter_runs"] == 1).sum())
        doubles = int((bf["batter_runs"] == 2).sum())
        triples = int((bf["batter_runs"] == 3).sum())

        out_rows = all_b[all_b["is_wicket"]]
        dismissed = bool(len(out_rows))
        dismissal = out_rows.iloc[0]["dismissal_type"] if dismissed else "not out"

        rows.append({
            "Batter": batter,
            "Runs": runs,
            "Balls": balls,
            "4s": fours,
            "6s": sixes,
            "Dots": dots,
            "1s": singles,
            "2s": doubles,
            "3s": triples,
            "Strike Rate": round(_safe_div(runs * 100, balls), 2),
            "Dismissal": dismissal,
            "Out": dismissed,
        })
    return pd.DataFrame(rows)


def run_progression(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative team runs after each delivery (for run-progression charts)."""
    if df.empty:
        return pd.DataFrame()
    prog = df.sort_values("seq").copy()
    prog["cumulative_runs"] = prog["total_runs"].cumsum()
    prog["cumulative_wickets"] = prog["is_wicket"].cumsum()
    prog["legal_ball_no"] = prog["is_legal"].cumsum()
    return prog[["seq", "over_str", "legal_ball_no", "total_runs",
                 "cumulative_runs", "cumulative_wickets", "event_label", "raw_text"]]



