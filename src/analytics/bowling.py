"""
analytics/bowling.py  (Module 4: Bowling Analytics)
===================================================
Per-bowler figures aggregated directly from parsed deliveries.

Scoring conventions (see parser docstring):
  * runs charged to bowler = batter_runs + wides + no-balls (NOT byes/leg-byes)
  * a run-out is NOT credited to the bowler
  * only legal balls (not wides/no-balls) advance the bowler's over count
"""

from __future__ import annotations

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def bowling_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    first_seen = df.groupby("bowler")["seq"].min().to_dict()
    for bowler in sorted(df["bowler"].unique(), key=lambda b: first_seen[b]):
        bdf = df[df["bowler"] == bowler]
        legal = bdf[bdf["is_legal"]]
        legal_balls = int(len(legal))
        runs_conceded = int(bdf["bowler_runs"].sum())
        wkts = int(((bdf["is_wicket"]) & (bdf["dismissal_type"] != "run out")).sum())
        dots = int((legal["total_runs"] == 0).sum())
        boundaries = int(legal["is_boundary"].sum())
        overs_decimal = _safe_div(legal_balls, 6)

        maidens = 0
        for _, grp in bdf.groupby("over_num"):
            ov_legal = grp[grp["is_legal"]]
            if len(ov_legal) == 6 and int(grp["bowler_runs"].sum()) == 0:
                maidens += 1

        rows.append({
            "Bowler": bowler,
            "Overs": f"{legal_balls // 6}.{legal_balls % 6}",
            "Balls": legal_balls,
            "Maidens": maidens,
            "Runs": runs_conceded,
            "Wickets": wkts,
            "Economy": round(_safe_div(runs_conceded, overs_decimal), 2),
            "Dot %": round(_safe_div(dots * 100, legal_balls), 1),
            "Boundary %": round(_safe_div(boundaries * 100, legal_balls), 1),
        })
    return pd.DataFrame(rows)


def over_by_over(df: pd.DataFrame) -> pd.DataFrame:
    """Runs and wickets per over (for over-by-over performance charts)."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("over_num").agg(
        runs=("total_runs", "sum"),
        wickets=("is_wicket", "sum"),
        bowler=("bowler", "first"),
    ).reset_index()
    g["runs"] = g["runs"].astype(int)
    g["wickets"] = g["wickets"].astype(int)
    return g
