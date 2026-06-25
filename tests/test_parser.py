"""
tests/test_parser.py
====================
Unit tests for the deterministic commentary parser (Module 1).

These tests pin down the structural contract the rest of the analytics engine
depends on: correct extraction of runs, extras, wickets, dismissal types,
fielders, boundary flags, and legal-ball / ball-faced accounting. Every
expected value below is hand-computable from the fixture commentary, which is
what makes the parser auditable.

Run with:  pytest -q
"""

import os
import sys

import pandas as pd

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser.commentary_parser import CommentaryParser, parse_commentary


FIXTURE = """0.1 Starc to Kohli, no run
0.2 Starc to Kohli, FOUR
0.3 Starc to Kohli, 1 run
0.4 Starc to Rohit, 2 runs
0.5 Starc to Rohit, SIX
0.6 Starc to Rohit, no run
1.1 Cummins to Kohli, wide
1.1 Cummins to Kohli, OUT, caught by Warner
1.2 Cummins to Rohit, no ball
1.2 Cummins to Rohit, 1 run
1.3 Cummins to Rohit, OUT, run out"""


def _df():
    return parse_commentary(FIXTURE)


def test_all_lines_parsed():
    parser = CommentaryParser()
    parser.parse_text(FIXTURE)
    # 11 commentary lines, all should parse cleanly.
    assert len(parser.to_dataframe()) == 11
    assert parser.unparsed_lines == []


def test_over_and_ball_extraction():
    df = _df()
    first = df.iloc[0]
    assert first["over_str"] == "0.1"
    assert first["over_num"] == 0
    assert first["ball_num"] == 1
    assert first["bowler"] == "Starc"
    assert first["batter"] == "Kohli"


def test_runs_off_the_bat():
    df = _df()
    # Kohli: 0 + 4 + 1 (first over) then his wicket ball scores 0.
    kohli = df[df["batter"] == "Kohli"]["batter_runs"].sum()
    assert kohli == 5
    # Rohit: 2 + 6 + 0 + 1 = 9 across his deliveries.
    rohit = df[df["batter"] == "Rohit"]["batter_runs"].sum()
    assert rohit == 9


def test_boundaries_flagged_only_when_explicit():
    df = _df()
    # Exactly one FOUR and one SIX in the fixture.
    assert int(df["is_four"].sum()) == 1
    assert int(df["is_six"].sum()) == 1
    assert int(df["is_boundary"].sum()) == 2
    # The "2 runs" delivery must NOT be treated as a boundary.
    two = df[df["over_str"] == "0.4"].iloc[0]
    assert bool(two["is_boundary"]) is False


def test_extras_wide_and_no_ball():
    df = _df()
    assert int(df["wides"].sum()) == 1
    assert int(df["no_balls"].sum()) == 1
    # A wide is neither legal nor faced; a no-ball is not legal.
    wide = df[df["event_label"] == "WIDE"].iloc[0]
    assert bool(wide["is_legal"]) is False
    assert bool(wide["ball_faced"]) is False
    no_ball = df[df["event_label"] == "NO_BALL"].iloc[0]
    assert bool(no_ball["is_legal"]) is False


def test_wickets_and_dismissals():
    df = _df()
    assert int(df["is_wicket"].sum()) == 2
    caught = df[df["dismissal_type"] == "caught"].iloc[0]
    assert caught["dismissed_batter"] == "Kohli"
    assert caught["fielders"] == "Warner"
    run_out = df[df["dismissal_type"] == "run out"].iloc[0]
    assert run_out["dismissed_batter"] == "Rohit"


def test_bowler_runs_accounting():
    df = _df()
    # bowler_runs = batter_runs + wides + no_balls (byes/leg-byes excluded).
    # Total here has no byes/leg-byes, so bowler_runs == total_runs.
    assert int(df["bowler_runs"].sum()) == int(df["total_runs"].sum())
    # Wide and no-ball each add one run to the bowler.
    wide = df[df["event_label"] == "WIDE"].iloc[0]
    assert int(wide["bowler_runs"]) == 1


def test_to_dataframe_is_dataframe():
    df = _df()
    assert isinstance(df, pd.DataFrame)
    assert "raw_text" in df.columns
