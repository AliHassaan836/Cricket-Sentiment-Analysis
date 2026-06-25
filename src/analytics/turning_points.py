"""
analytics/turning_points.py  (Module 7: Turning Point Detection)
================================================================
A turning point is an over (or wicket) after which the match's trajectory
measurably changed. We quantify "trajectory" with the cumulative run rate before
and after the over, plus wicket impact. Every turning point cites the commentary
and the exact run-rate numbers that justify it — no narrative without evidence.

Scoring (transparent):
    swing_magnitude = |RR_after_over - RR_before_over|
    a turning point is flagged when:
        * an over caused a run-rate swing >= config.run_rate_swing_delta, OR
        * an over conceded >= high_scoring_over_runs, OR
        * an over contained a wicket that ended a partnership >= big_partnership_runs
Turning points are ranked by a combined importance score.
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from config import EVENT_THRESHOLDS as ET
from src.analytics.team import partnerships


def _rr(df: pd.DataFrame) -> float:
    legal = int(df["is_legal"].sum())
    runs = int(df["total_runs"].sum())
    return round(runs * 6 / legal, 2) if legal else 0.0


def detect_turning_points(df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    if df.empty:
        return []

    pship = partnerships(df)
    overs = sorted(df["over_num"].unique())
    candidates: List[Dict] = []

    for ov in overs:
        before = df[df["over_num"] < ov]
        upto = df[df["over_num"] <= ov]
        seg = df[df["over_num"] == ov]

        rr_before = _rr(before)
        rr_after = _rr(upto)
        swing = round(rr_after - rr_before, 2)
        over_runs = int(seg["total_runs"].sum())
        over_wkts = int(seg["is_wicket"].sum())

        # Avoid cold-start artifacts: a run-rate "swing" is only meaningful once
        # there is a real baseline (at least one full over already bowled).
        baseline_balls = int(before["is_legal"].sum())
        swing_valid = baseline_balls >= 6

        # Partnership impact: did a wicket here end a sizeable stand?
        ended_big = 0
        for _, r in seg[seg["is_wicket"]].iterrows():
            ended = pship[pship["To"] == r["over_str"]]
            if len(ended):
                ended_big = max(ended_big, int(ended.iloc[0]["Runs"]))

        flagged = (
            (swing_valid and abs(swing) >= ET.run_rate_swing_delta)
            or over_runs >= ET.high_scoring_over_runs
            or ended_big >= ET.big_partnership_runs
        )
        if not flagged:
            continue

        # Importance score (documented, deterministic).
        importance = (
            (abs(swing) * 10 if swing_valid else 0)
            + over_runs * 1.0
            + over_wkts * 12
            + (ended_big * 0.5 if ended_big else 0)
        )

        reasons = []
        if swing_valid and abs(swing) >= ET.run_rate_swing_delta:
            direction = "increased" if swing > 0 else "dropped"
            reasons.append(
                f"Run rate {direction} from {rr_before} to {rr_after} "
                f"(swing {swing:+}).")
        if over_runs >= ET.high_scoring_over_runs:
            reasons.append(f"{over_runs} runs scored in the over.")
        if over_wkts:
            reasons.append(f"{over_wkts} wicket(s) fell.")
        if ended_big >= ET.big_partnership_runs:
            reasons.append(f"A {ended_big}-run partnership ended.")

        candidates.append({
            "over": f"Over {int(ov) + 1}",
            "over_num": int(ov),
            "title": f"{over_runs} runs, {over_wkts} wicket(s)",
            "importance": round(importance, 1),
            "rr_before": rr_before,
            "rr_after": rr_after,
            "swing": swing,
            "reason": " ".join(reasons),
            "evidence": seg["raw_text"].tolist(),
        })

    candidates.sort(key=lambda c: c["importance"], reverse=True)
    return candidates[:top_n]
