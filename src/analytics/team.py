"""
analytics/team.py  (Module 5: Team Analytics)
=============================================
Team-level aggregates: total score, run rate, wickets, partnership breakdown and
phase-wise (powerplay / middle / death) scoring.

PARTNERSHIPS — an honest note on inference
------------------------------------------
Standard commentary lines name only the striker, not the non-striker. We
therefore cannot always name *both* partners with certainty. A partnership here
is defined deterministically as "all runs scored between two consecutive wicket
events" and is attributed to the set of batters observed facing during that span.
This is stated plainly in the UI rather than guessing a non-striker's identity.
"""

from __future__ import annotations

import pandas as pd

from config import FORMAT_OVERS, PHASE_DEFINITIONS


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def detect_format(df: pd.DataFrame) -> str:
    """Infer match format from the highest over number actually present."""
    if df.empty:
        return "UNKNOWN"
    max_over = int(df["over_num"].max())
    if max_over <= 19:
        return "T20"
    if max_over <= 49:
        return "ODI"
    return "TEST"


def team_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    legal_balls = int(df["is_legal"].sum())
    total = int(df["total_runs"].sum())
    wkts = int(df["is_wicket"].sum())
    overs_decimal = _safe_div(legal_balls, 6)
    return {
        "total_runs": total,
        "wickets": wkts,
        "legal_balls": legal_balls,
        "overs": f"{legal_balls // 6}.{legal_balls % 6}",
        "run_rate": round(_safe_div(total, overs_decimal), 2),
        "extras": int(df["extras_total"].sum()),
        "boundaries": int(df["is_boundary"].sum()),
        "dot_balls": int((df["is_legal"] & (df["total_runs"] == 0)).sum()),
        "format": detect_format(df),
    }


def partnerships(df: pd.DataFrame) -> pd.DataFrame:
    """Runs scored between consecutive wickets (wicket boundaries)."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    current = {"runs": 0, "balls": 0, "batters": set(), "start": None, "end": None}
    wicket_no = 0
    for _, r in df.sort_values("seq").iterrows():
        if current["start"] is None:
            current["start"] = r["over_str"]
        current["runs"] += int(r["total_runs"])
        current["balls"] += int(bool(r["ball_faced"]))
        current["batters"].add(r["batter"])
        current["end"] = r["over_str"]
        if r["is_wicket"]:
            wicket_no += 1
            rows.append({
                "Wicket": wicket_no,
                "Runs": current["runs"],
                "Balls": current["balls"],
                "Batters involved": ", ".join(sorted(current["batters"])),
                "From": current["start"],
                "To": current["end"],
                "Ended by": f"{r['dismissal_type']} ({r['batter']})",
            })
            current = {"runs": 0, "balls": 0, "batters": set(), "start": None, "end": None}

    # Unbroken final partnership (if innings did not end on a wicket).
    if current["balls"] > 0 or current["runs"] > 0:
        rows.append({
            "Wicket": "unbroken",
            "Runs": current["runs"],
            "Balls": current["balls"],
            "Batters involved": ", ".join(sorted(current["batters"])),
            "From": current["start"],
            "To": current["end"],
            "Ended by": "not out",
        })
    return pd.DataFrame(rows)


def phase_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Powerplay / middle / death scoring. Only applied to T20 and ODI."""
    if df.empty:
        return pd.DataFrame()
    fmt = detect_format(df)
    if fmt not in PHASE_DEFINITIONS:
        return pd.DataFrame()

    phases = PHASE_DEFINITIONS[fmt]
    spec = [
        ("Powerplay", phases.powerplay),
        ("Middle Overs", phases.middle),
        ("Death Overs", phases.death),
    ]
    rows = []
    for name, (lo, hi) in spec:
        seg = df[(df["over_num"] >= lo) & (df["over_num"] <= hi)]
        if seg.empty:
            continue
        legal = int(seg["is_legal"].sum())
        runs = int(seg["total_runs"].sum())
        rows.append({
            "Phase": name,
            "Overs": f"{lo}-{hi}",
            "Runs": runs,
            "Wickets": int(seg["is_wicket"].sum()),
            "Balls": legal,
            "Run Rate": round(_safe_div(runs * 6, legal), 2),
            "Boundaries": int(seg["is_boundary"].sum()),
        })
    return pd.DataFrame(rows)
