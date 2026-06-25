"""
summarization/summarizer.py  (Module 9: Match Summarization)
============================================================
Generates a short summary, a detailed summary, and a narrative "match story".

ANTI-HALLUCINATION DESIGN
-------------------------
Summaries are GROUNDED: they are built from a structured "fact sheet" computed by
the analytics engine. We use transformer models (BART/T5) only to *rephrase* the
grounded fact sheet into fluent prose — never to generate facts. Because the
input to the model is itself only verified numbers, the model has nothing to
invent. A deterministic template summary is always produced as the baseline and
as a fallback when transformers are unavailable.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from config import NLP_CONFIG, UNKNOWN
from src.analytics import team as team_mod
from src.analytics.batting import batting_scorecard
from src.analytics.bowling import bowling_scorecard
from src.analytics.turning_points import detect_turning_points

try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:                       # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

_SUMMARIZER = None


def _load_summarizer():
    global _SUMMARIZER
    if not (_TRANSFORMERS_AVAILABLE and NLP_CONFIG.enable_transformers):
        return None
    if _SUMMARIZER is None:
        try:
            _SUMMARIZER = hf_pipeline("summarization", model=NLP_CONFIG.summarizer_model)
        except Exception:
            _SUMMARIZER = None
    return _SUMMARIZER


def build_fact_sheet(df: pd.DataFrame) -> Dict:
    """Verified facts used as the ONLY source material for summaries."""
    if df.empty:
        return {}
    ts = team_mod.team_summary(df)
    bat = batting_scorecard(df)
    bowl = bowling_scorecard(df)
    top_bat = bat.sort_values("Runs", ascending=False).head(3)
    top_bowl = bowl.sort_values("Wickets", ascending=False).head(3)
    tps = detect_turning_points(df, top_n=2)
    return {
        "team": ts,
        "top_batters": top_bat.to_dict("records"),
        "top_bowlers": top_bowl.to_dict("records"),
        "turning_points": tps,
    }


def _facts_to_text(facts: Dict) -> str:
    """Flatten the fact sheet into grounded source sentences for the model."""
    if not facts:
        return ""
    t = facts["team"]
    parts = [
        f"The batting side scored {t['total_runs']} runs for {t['wickets']} "
        f"wickets in {t['overs']} overs at a run rate of {t['run_rate']}.",
        f"There were {t['boundaries']} boundaries and {t['extras']} extras.",
    ]
    for b in facts["top_batters"]:
        if b["Runs"] > 0:
            parts.append(
                f"{b['Batter']} made {b['Runs']} runs off {b['Balls']} balls "
                f"(strike rate {b['Strike Rate']}).")
    for b in facts["top_bowlers"]:
        if b["Wickets"] > 0:
            parts.append(
                f"{b['Bowler']} took {b['Wickets']} wickets for {b['Runs']} runs "
                f"in {b['Overs']} overs.")
    for tp in facts["turning_points"]:
        parts.append(f"A turning point came in {tp['over']}: {tp['reason']}")
    return " ".join(parts)


def template_short_summary(facts: Dict) -> str:
    if not facts:
        return UNKNOWN
    t = facts["team"]
    lead_bat = facts["top_batters"][0] if facts["top_batters"] else None
    lead_bowl = facts["top_bowlers"][0] if facts["top_bowlers"] else None
    s = (f"The batting side posted {t['total_runs']}/{t['wickets']} in "
         f"{t['overs']} overs (run rate {t['run_rate']}).")
    if lead_bat and lead_bat["Runs"] > 0:
        s += (f" {lead_bat['Batter']} top-scored with {lead_bat['Runs']} "
              f"({lead_bat['Balls']} balls).")
    if lead_bowl and lead_bowl["Wickets"] > 0:
        s += (f" {lead_bowl['Bowler']} was the pick of the bowlers with "
              f"{lead_bowl['Wickets']}/{lead_bowl['Runs']}.")
    return s


def template_detailed_summary(facts: Dict) -> str:
    if not facts:
        return UNKNOWN
    t = facts["team"]
    lines = [template_short_summary(facts)]
    bats = [b for b in facts["top_batters"] if b["Runs"] > 0]
    if bats:
        frag = ", ".join(
            f"{b['Batter']} {b['Runs']} ({b['Balls']})" for b in bats)
        lines.append(f"Leading run-scorers: {frag}.")
    bowls = [b for b in facts["top_bowlers"] if b["Wickets"] > 0]
    if bowls:
        frag = ", ".join(
            f"{b['Bowler']} {b['Wickets']}/{b['Runs']}" for b in bowls)
        lines.append(f"Wicket-takers: {frag}.")
    lines.append(
        f"The innings featured {t['boundaries']} boundaries, "
        f"{t['dot_balls']} dot balls and {t['extras']} extras.")
    for tp in facts["turning_points"]:
        lines.append(f"Turning point — {tp['over']}: {tp['reason']}")
    return " ".join(lines)


def match_story(df: pd.DataFrame, facts: Dict) -> str:
    """A phase-by-phase narrative grounded in phase analysis."""
    if not facts:
        return UNKNOWN
    phases = team_mod.phase_analysis(df)
    t = facts["team"]
    story = [f"This was a {t['format']} innings."]
    for _, p in phases.iterrows():
        story.append(
            f"In the {p['Phase'].lower()} (overs {p['Overs']}), the side scored "
            f"{p['Runs']} runs for the loss of {p['Wickets']} wicket(s) at "
            f"{p['Run Rate']} per over.")
    story.append(
        f"They finished on {t['total_runs']}/{t['wickets']} after {t['overs']} overs.")
    return " ".join(story)


def transformer_summary(facts: Dict) -> str:
    """Rephrase grounded facts with a transformer, or fall back to template."""
    src = _facts_to_text(facts)
    if not src:
        return UNKNOWN
    pipe = _load_summarizer()
    if pipe is None:
        return template_detailed_summary(facts)
    try:
        out = pipe(src,
                   max_length=NLP_CONFIG.summarizer_max_len,
                   min_length=NLP_CONFIG.summarizer_min_len,
                   do_sample=False)
        return out[0]["summary_text"].strip()
    except Exception:
        return template_detailed_summary(facts)


def generate_summaries(df: pd.DataFrame, use_transformer: bool = False) -> Dict[str, str]:
    facts = build_fact_sheet(df)
    detailed = (transformer_summary(facts) if use_transformer
                else template_detailed_summary(facts))
    return {
        "short": template_short_summary(facts),
        "detailed": detailed,
        "story": match_story(df, facts),
        "backend": "transformer" if (use_transformer and _load_summarizer()) else "template",
    }
