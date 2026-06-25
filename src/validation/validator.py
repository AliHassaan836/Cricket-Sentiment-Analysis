"""
validation/validator.py  (Module 14: Hallucination Prevention)
==============================================================
A validation gate that cross-checks every derived statistic against the raw
parsed data before the UI displays it. If any internal inconsistency is found,
the UI shows "Data inconsistency detected." next to the affected figure rather
than presenting a number that cannot be reconciled.

Checks performed
----------------
1. Batting runs + extras  ==  team total runs.
2. Bowler runs + byes + leg-byes  ==  team total runs.
3. Sum of dismissed batters  ==  team wickets.
4. Partnership runs + extras  ==  team total (partnerships partition the innings).
5. No delivery has negative runs.
6. Every parsed delivery maps to an over/ball/bowler/batter.
7. Report any unparsed commentary lines (potential data loss).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.analytics.batting import batting_scorecard
from src.analytics.bowling import bowling_scorecard
from src.analytics.team import team_summary, partnerships


def validate(df: pd.DataFrame, unparsed_lines: List[Dict] | None = None) -> Dict:
    unparsed_lines = unparsed_lines or []
    report = {"passed": True, "checks": [], "warnings": []}

    def check(name: str, condition: bool, detail: str = ""):
        report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            report["passed"] = False

    if df.empty:
        report["passed"] = False
        report["checks"].append({"name": "data present", "passed": False,
                                 "detail": "No deliveries parsed."})
        return report

    ts = team_summary(df)
    total = ts["total_runs"]
    extras = ts["extras"]

    # 1. Batting runs + extras == total
    bat = batting_scorecard(df)
    bat_runs = int(bat["Runs"].sum()) if not bat.empty else 0
    check("Batting runs + extras = team total",
          bat_runs + extras == total,
          f"{bat_runs} + {extras} vs {total}")

    # 2. Bowler runs + byes + leg-byes == total
    bowler_runs = int(df["bowler_runs"].sum())
    byes_lb = int(df["byes"].sum() + df["leg_byes"].sum())
    check("Bowler runs + byes/leg-byes = team total",
          bowler_runs + byes_lb == total,
          f"{bowler_runs} + {byes_lb} vs {total}")

    # 3. Dismissed batters == wickets
    dismissed = int(bat["Out"].sum()) if "Out" in bat else int(df["is_wicket"].sum())
    check("Dismissed batters = wickets",
          dismissed == ts["wickets"],
          f"{dismissed} vs {ts['wickets']}")

    # 4. Partnerships partition the innings
    pship = partnerships(df)
    pruns = int(pship["Runs"].sum()) if not pship.empty else 0
    check("Partnership runs = team total",
          pruns == total,
          f"{pruns} vs {total}")

    # 5. No negative runs
    check("No negative run values",
          bool((df["total_runs"] >= 0).all()),
          "")

    # 6. Structural completeness
    complete = df[["over_str", "bowler", "batter"]].notna().all(axis=1).all()
    check("Every delivery has over/bowler/batter", bool(complete), "")

    # 7. Unparsed lines warning (non-fatal but surfaced)
    if unparsed_lines:
        report["warnings"].append(
            f"{len(unparsed_lines)} commentary line(s) could not be parsed and "
            f"were excluded. Review them to ensure no data was lost.")
        report["unparsed_lines"] = unparsed_lines

    return report


def guard(value, df: pd.DataFrame, unparsed_lines=None, placeholder="Data inconsistency detected."):
    """Return `value` only if validation passes; else the placeholder string."""
    rep = validate(df, unparsed_lines)
    return value if rep["passed"] else placeholder
