"""
tests/test_batting.py
=====================
Unit tests for batting analytics (Module 3).

The fixture is a tiny, fully hand-computable innings so that every figure in the
scorecard (runs, balls faced, boundaries, dot balls, strike rate, dismissal)
can be checked by inspection.

Run with:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.commentary_parser import parse_commentary
from src.analytics.batting import batting_scorecard, run_progression


FIXTURE = """0.1 Starc to Kohli, no run
0.2 Starc to Kohli, FOUR
0.3 Starc to Kohli, 1 run
0.4 Starc to Rohit, 2 runs
0.5 Starc to Rohit, SIX
0.6 Starc to Rohit, no run"""


def _card():
    df = parse_commentary(FIXTURE)
    card = batting_scorecard(df)
    return card.set_index("Batter")


def test_kohli_line():
    c = _card().loc["Kohli"]
    assert c["Runs"] == 5          # 0 + 4 + 1
    assert c["Balls"] == 3
    assert c["4s"] == 1
    assert c["6s"] == 0
    assert c["Dots"] == 1          # the leading "no run"
    # Strike rate = 5/3 * 100 = 166.67
    assert round(c["Strike Rate"], 2) == 166.67
    assert c["Out"] in (False, 0)  # not dismissed in this fixture


def test_rohit_line():
    c = _card().loc["Rohit"]
    assert c["Runs"] == 8          # 2 + 6 + 0
    assert c["Balls"] == 3
    assert c["6s"] == 1
    assert c["2s"] == 1
    assert c["Dots"] == 1


def test_total_runs_reconcile():
    df = parse_commentary(FIXTURE)
    card = batting_scorecard(df)
    # Sum of batter runs equals total runs off the bat in the fixture (13).
    assert card["Runs"].sum() == 13
    assert int(df["batter_runs"].sum()) == 13


def test_run_progression_monotonic():
    df = parse_commentary(FIXTURE)
    prog = run_progression(df)
    cum = list(prog["cumulative_runs"])
    # Cumulative score must never decrease.
    assert all(b >= a for a, b in zip(cum, cum[1:]))
    # Final cumulative total equals the full innings total (13 off the bat).
    assert cum[-1] == 13


def test_empty_input_is_safe():
    import pandas as pd
    empty = parse_commentary("")
    card = batting_scorecard(empty)
    assert isinstance(card, pd.DataFrame)
    assert card.empty
