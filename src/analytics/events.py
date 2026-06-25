"""
analytics/events.py  (Module 6: Important Event Detection)
==========================================================
Detects notable in-match events and attaches *evidence* to each one. Every event
references the actual commentary line(s) that triggered it, so nothing is a
black box. Thresholds come from config.EVENT_THRESHOLDS.
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from config import EVENT_THRESHOLDS as ET
from src.analytics.team import partnerships


def _cumulative_runrate(df: pd.DataFrame, upto_seq: int) -> float:
    seg = df[df["seq"] <= upto_seq]
    legal = int(seg["is_legal"].sum())
    runs = int(seg["total_runs"].sum())
    return round(runs * 6 / legal, 2) if legal else 0.0


def detect_events(df: pd.DataFrame) -> List[Dict]:
    """Return a list of structured event dicts with evidence."""
    if df.empty:
        return []

    events: List[Dict] = []

    # --- Wickets -------------------------------------------------------- #
    pship = partnerships(df)
    # Map a wicket sequence to the partnership it ended.
    for idx, r in df[df["is_wicket"]].iterrows():
        batter = r["batter"]
        batter_runs = int(df[(df["batter"] == batter) & (df["ball_faced"])]["batter_runs"].sum())
        # partnership ending at this wicket
        ended = pship[pship["To"] == r["over_str"]]
        pruns = int(ended.iloc[0]["Runs"]) if len(ended) else None
        evidence = [r["raw_text"]]
        reason = [f"{batter} dismissed ({r['dismissal_type']}) for {batter_runs} run(s)."]
        if pruns is not None:
            reason.append(f"Ended a stand worth {pruns} run(s).")
        events.append({
            "type": "Wicket",
            "over": r["over_str"],
            "title": f"{batter} dismissed",
            "reason": " ".join(reason),
            "evidence": evidence,
            "seq": int(r["seq"]),
        })

    # --- High-scoring / expensive overs --------------------------------- #
    over_runs = df.groupby("over_num").agg(
        runs=("total_runs", "sum"),
        bowler=("bowler", "first"),
        last_seq=("seq", "max"),
    ).reset_index()
    for _, o in over_runs.iterrows():
        if int(o["runs"]) >= ET.high_scoring_over_runs:
            lines = df[df["over_num"] == o["over_num"]]["raw_text"].tolist()
            events.append({
                "type": "High-scoring over",
                "over": f"Over {int(o['over_num']) + 1}",
                "title": f"{int(o['runs'])} runs conceded by {o['bowler']}",
                "reason": f"Over went for {int(o['runs'])} runs "
                          f"(threshold {ET.high_scoring_over_runs}+).",
                "evidence": lines,
                "seq": int(o["last_seq"]),
            })

    # --- Big partnerships ----------------------------------------------- #
    for _, p in pship.iterrows():
        if int(p["Runs"]) >= ET.big_partnership_runs:
            events.append({
                "type": "Big partnership",
                "over": f"{p['From']} - {p['To']}",
                "title": f"{int(p['Runs'])}-run stand",
                "reason": f"Partnership of {int(p['Runs'])} runs "
                          f"({p['Batters involved']}).",
                "evidence": [f"Spanned {p['From']} to {p['To']}, "
                             f"ended by {p['Ended by']}."],
                "seq": -1,
            })

    # --- Collapses ------------------------------------------------------ #
    wicket_seqs = df[df["is_wicket"]][["seq", "over_str", "raw_text"]].reset_index(drop=True)
    legal_index = df.set_index("seq")["is_legal"].cumsum().to_dict()
    for i in range(len(wicket_seqs) - ET.collapse_wickets + 1):
        first = wicket_seqs.iloc[i]
        last = wicket_seqs.iloc[i + ET.collapse_wickets - 1]
        balls_between = legal_index[last["seq"]] - legal_index[first["seq"]]
        if balls_between <= ET.collapse_within_balls:
            events.append({
                "type": "Collapse",
                "over": f"{first['over_str']} - {last['over_str']}",
                "title": f"{ET.collapse_wickets} wickets in {balls_between} balls",
                "reason": f"{ET.collapse_wickets} wickets fell within "
                          f"{balls_between} legal balls.",
                "evidence": [first["raw_text"], last["raw_text"]],
                "seq": int(last["seq"]),
            })

    # De-duplicate collapses that overlap heavily (keep earliest of each window)
    events.sort(key=lambda e: (e["seq"] if e["seq"] >= 0 else 1e9))
    return events
