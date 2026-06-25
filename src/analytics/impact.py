"""
analytics/impact.py  (Module 8: Impact Player Analysis)
=======================================================
A fully transparent, reproducible impact score. There is NO randomness and NO
black box. The score for each player is a weighted sum of components that are
each computed from parsed data, then min-max normalised to 0-100 across all
players in the match.

FORMULA (weights live in config.IMPACT_WEIGHTS so they are auditable)
---------------------------------------------------------------------
Batting:
    bat_raw = runs * run_value
            + boundaries * boundary_bonus
            + sixes * six_extra_bonus
            + (SR - strike_rate_pivot) * strike_rate_value * (balls / 100)   [if balls >= min_balls]

Bowling:
    bowl_raw = wickets * wicket_value
             + (economy_pivot - economy) * economy_value * overs             [if overs >= min_overs]
             + maidens * maiden_bonus

Fielding:
    field_raw = catches * catch_value
              + runouts * runout_value
              + stumpings * stumping_value

    raw_total = bat_raw + bowl_raw + field_raw
    impact_score = 100 * (raw_total - min_raw) / (max_raw - min_raw)   [match-relative]

The component breakdown is returned alongside the score so the UI can show
exactly *why* a player ranks where they do.
"""

from __future__ import annotations

from typing import List, Dict

import pandas as pd

from config import IMPACT_WEIGHTS as W
from src.analytics.batting import batting_scorecard
from src.analytics.bowling import bowling_scorecard


def _safe_div(a, b):
    return a / b if b else 0.0


def _fielding_counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Count catches / run-outs / stumpings credited to named fielders."""
    counts: Dict[str, Dict[str, int]] = {}
    for _, r in df[df["is_wicket"]].iterrows():
        fielder = (r.get("fielders") or "").strip()
        if not fielder:
            continue
        dtype = (r.get("dismissal_type") or "").lower()
        bucket = counts.setdefault(fielder, {"catches": 0, "runouts": 0, "stumpings": 0})
        if "stump" in dtype:
            bucket["stumpings"] += 1
        elif "run out" in dtype:
            bucket["runouts"] += 1
        elif "caught" in dtype or "c " in dtype:
            bucket["catches"] += 1
    return counts


def compute_impact(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    bat = batting_scorecard(df).set_index("Batter") if not df.empty else pd.DataFrame()
    bowl = bowling_scorecard(df).set_index("Bowler") if not df.empty else pd.DataFrame()
    field = _fielding_counts(df)

    players = set(df["batter"]) | set(df["bowler"]) | set(field.keys())
    records = []

    for p in players:
        breakdown = []
        bat_raw = bowl_raw = field_raw = 0.0

        # Batting component
        if p in bat.index:
            row = bat.loc[p]
            runs = int(row["Runs"]); balls = int(row["Balls"])
            fours = int(row["4s"]); sixes = int(row["6s"])
            sr = float(row["Strike Rate"])
            boundaries = fours + sixes
            bat_raw += runs * W.run_value
            bat_raw += boundaries * W.boundary_bonus
            bat_raw += sixes * W.six_extra_bonus
            sr_term = 0.0
            if balls >= W.min_balls_for_sr:
                sr_term = (sr - W.strike_rate_pivot) * W.strike_rate_value * (balls / 100.0)
                bat_raw += sr_term
            if runs or balls:
                breakdown.append(
                    f"Bat: {runs} runs off {balls} balls (SR {sr}), "
                    f"{boundaries} boundaries -> {round(bat_raw, 1)} pts")

        # Bowling component
        if p in bowl.index:
            row = bowl.loc[p]
            wkts = int(row["Wickets"]); runs_c = int(row["Runs"])
            balls = int(row["Balls"]); maidens = int(row["Maidens"])
            econ = float(row["Economy"]); overs = balls / 6.0
            bowl_raw += wkts * W.wicket_value
            econ_term = 0.0
            if overs >= W.min_overs_for_econ:
                econ_term = (W.economy_pivot - econ) * W.economy_value * overs
                bowl_raw += econ_term
            bowl_raw += maidens * W.maiden_bonus
            if balls:
                breakdown.append(
                    f"Bowl: {wkts} wkt(s), {runs_c} runs in {row['Overs']} ov "
                    f"(econ {econ}) -> {round(bowl_raw, 1)} pts")

        # Fielding component
        if p in field:
            f = field[p]
            field_raw += f["catches"] * W.catch_value
            field_raw += f["runouts"] * W.runout_value
            field_raw += f["stumpings"] * W.stumping_value
            if any(f.values()):
                breakdown.append(
                    f"Field: {f['catches']}c {f['runouts']}ro {f['stumpings']}st "
                    f"-> {round(field_raw, 1)} pts")

        raw_total = bat_raw + bowl_raw + field_raw
        records.append({
            "Player": p,
            "raw_total": round(raw_total, 2),
            "bat_raw": round(bat_raw, 2),
            "bowl_raw": round(bowl_raw, 2),
            "field_raw": round(field_raw, 2),
            "breakdown": breakdown,
        })

    res = pd.DataFrame(records)
    if res.empty:
        return res

    # Match-relative min-max normalisation to 0-100.
    lo, hi = res["raw_total"].min(), res["raw_total"].max()
    if hi > lo:
        res["Impact Score"] = (100 * (res["raw_total"] - lo) / (hi - lo)).round(1)
    else:
        res["Impact Score"] = 100.0
    res = res.sort_values("Impact Score", ascending=False).reset_index(drop=True)
    return res
