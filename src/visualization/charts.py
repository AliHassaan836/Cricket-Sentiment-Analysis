"""
visualization/charts.py  (Modules 12 & 13: Visual Analytics)
============================================================
Plotly figure builders with a professional dark-themed palette.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.analytics.batting import run_progression, batting_scorecard
from src.analytics.bowling import bowling_scorecard, over_by_over
from src.analytics.team import partnerships, phase_analysis
from src.nlp.sentiment import momentum_curve

# ── Professional high-contrast palette ──────────────────────────────────────
_TEMPLATE = "plotly_dark"
_BG       = "#0E1117"
_CARD_BG  = "#161B22"
_ACCENT   = "#00B4D8"
_GREEN    = "#00C853"
_RED      = "#FF3D3D"
_GOLD     = "#FFD700"
_PURPLE   = "#BB86FC"
_ORANGE   = "#FF9100"
_CYAN     = "#00E5FF"
_WHITE    = "#E6EDF3"
_GRID     = "#21262D"

_LAYOUT = dict(
    template=_TEMPLATE,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=_CARD_BG,
    font=dict(color=_WHITE, family="Inter, system-ui, sans-serif"),
    xaxis=dict(gridcolor=_GRID, zeroline=False),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    margin=dict(l=50, r=20, t=50, b=40),
    colorway=[_ACCENT, _GREEN, _ORANGE, _PURPLE, _RED, _GOLD, _CYAN],
)


def _styled(fig: go.Figure, **kw) -> go.Figure:
    fig.update_layout(**{**_LAYOUT, **kw})
    return fig


def _empty(msg: str = "Insufficient data for this chart.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#8B949E"))
    return _styled(fig, height=280, xaxis=dict(visible=False), yaxis=dict(visible=False))


# ── Batting ─────────────────────────────────────────────────────────────────

def run_progression_chart(df: pd.DataFrame) -> go.Figure:
    prog = run_progression(df)
    if prog.empty:
        return _empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prog["legal_ball_no"], y=prog["cumulative_runs"],
        mode="lines", name="Runs",
        line=dict(width=3, color=_ACCENT),
        fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"))
    wkts = prog[prog["event_label"] == "WICKET"]
    fig.add_trace(go.Scatter(
        x=wkts["legal_ball_no"], y=wkts["cumulative_runs"],
        mode="markers", name="Wicket",
        marker=dict(color=_RED, size=12, symbol="x", line=dict(width=2, color=_RED)),
        text=wkts["raw_text"], hoverinfo="text+y"))
    return _styled(fig, title="Run Progression",
                   xaxis_title="Legal Ball", yaxis_title="Cumulative Runs", height=400)


def wicket_timeline_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty()
    prog = run_progression(df)
    wkts = prog[prog["event_label"] == "WICKET"]
    if wkts.empty:
        return _empty("No wickets fell.")
    fig = go.Figure(go.Scatter(
        x=wkts["over_str"], y=wkts["cumulative_runs"], mode="markers+text",
        text=[f"W{i+1}" for i in range(len(wkts))], textposition="top center",
        textfont=dict(color=_WHITE, size=11),
        marker=dict(size=14, color=_RED, line=dict(width=1, color="#fff"))))
    return _styled(fig, title="Fall of Wickets",
                   xaxis_title="Over", yaxis_title="Score at Fall", height=350)


def partnership_chart(df: pd.DataFrame) -> go.Figure:
    p = partnerships(df)
    if p.empty:
        return _empty()
    labels = [f"{r['Batters involved']}" for _, r in p.iterrows()]
    fig = go.Figure(go.Bar(
        x=p["Runs"], y=labels, orientation="h",
        marker=dict(color=_GREEN, line=dict(width=0)),
        text=p["Runs"], textposition="auto",
        textfont=dict(color=_WHITE, size=12)))
    return _styled(fig, title="Partnerships",
                   xaxis_title="Runs", height=max(300, 55 * len(p)))


# ── Momentum & Pressure ────────────────────────────────────────────────────

def momentum_chart(df: pd.DataFrame) -> go.Figure:
    mc = momentum_curve(df)
    if mc.empty:
        return _empty()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc["legal_ball_no"], y=mc["momentum"],
        mode="lines", name="Momentum", fill="tozeroy",
        line=dict(color=_PURPLE, width=2),
        fillcolor="rgba(187,134,252,0.10)"))
    fig.add_hline(y=0, line_dash="dot", line_color="#484F58")
    return _styled(fig, title="Match Momentum",
                   xaxis_title="Legal Ball", yaxis_title="Momentum Index", height=340)


def pressure_chart(df: pd.DataFrame) -> go.Figure:
    mc = momentum_curve(df)
    if mc.empty:
        return _empty()
    fig = go.Figure(go.Scatter(
        x=mc["legal_ball_no"], y=mc["pressure"],
        mode="lines", fill="tozeroy",
        line=dict(color=_ORANGE, width=2),
        fillcolor="rgba(255,145,0,0.10)"))
    return _styled(fig, title="Pressure Index",
                   xaxis_title="Legal Ball", yaxis_title="Pressure", height=320)


# ── Batter Charts ──────────────────────────────────────────────────────────

def batter_comparison_chart(df: pd.DataFrame) -> go.Figure:
    bat = batting_scorecard(df)
    if bat.empty:
        return _empty()
    fig = go.Figure(go.Bar(
        x=bat["Batter"], y=bat["Runs"], text=bat["Runs"], textposition="auto",
        marker=dict(
            color=bat["Strike Rate"],
            colorscale=[[0, "#1B3A4B"], [0.5, _ACCENT], [1, _CYAN]],
            showscale=True, colorbar=dict(title="SR", thickness=12)),
        textfont=dict(color=_WHITE)))
    return _styled(fig, title="Runs Scored (coloured by Strike Rate)", height=380)


def strike_rate_chart(df: pd.DataFrame) -> go.Figure:
    bat = batting_scorecard(df)
    if bat.empty:
        return _empty()
    bat = bat[bat["Balls"] > 0]
    fig = go.Figure(go.Bar(
        x=bat["Batter"], y=bat["Strike Rate"],
        text=bat["Strike Rate"].round(1), textposition="auto",
        marker=dict(color=_CYAN, line=dict(width=0)),
        textfont=dict(color=_WHITE)))
    return _styled(fig, title="Strike Rates", height=350)


# ── Bowler Charts ──────────────────────────────────────────────────────────

def bowler_comparison_chart(df: pd.DataFrame) -> go.Figure:
    bowl = bowling_scorecard(df)
    if bowl.empty:
        return _empty()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bowl["Bowler"], y=bowl["Wickets"], name="Wickets",
                         marker_color=_GREEN))
    fig.add_trace(go.Bar(x=bowl["Bowler"], y=bowl["Economy"], name="Economy",
                         marker_color=_ORANGE))
    return _styled(fig, title="Bowler Comparison", barmode="group", height=380)


def economy_chart(df: pd.DataFrame) -> go.Figure:
    bowl = bowling_scorecard(df)
    if bowl.empty:
        return _empty()
    fig = go.Figure(go.Bar(
        x=bowl["Bowler"], y=bowl["Economy"],
        text=bowl["Economy"].round(2), textposition="auto",
        marker=dict(
            color=bowl["Economy"],
            colorscale=[[0, _GREEN], [0.5, _GOLD], [1, _RED]],
            showscale=True, colorbar=dict(title="Econ", thickness=12)),
        textfont=dict(color=_WHITE)))
    return _styled(fig, title="Bowling Economy", height=350)


# ── Team Charts ────────────────────────────────────────────────────────────

def team_scoring_chart(df: pd.DataFrame) -> go.Figure:
    obo = over_by_over(df)
    if obo.empty:
        return _empty()
    obo = obo.copy()
    obo["over_label"] = (obo["over_num"] + 1).astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=obo["over_label"], y=obo["runs"], name="Runs/over",
                         marker=dict(color=_ACCENT, line=dict(width=0))))
    wkt_overs = obo[obo["wickets"] > 0]
    fig.add_trace(go.Scatter(
        x=wkt_overs["over_label"], y=wkt_overs["runs"],
        mode="markers", name="Wicket",
        marker=dict(color=_RED, size=13, symbol="x", line=dict(width=2, color=_RED))))
    return _styled(fig, title="Runs per Over",
                   xaxis_title="Over", yaxis_title="Runs", height=360)


def phase_chart(df: pd.DataFrame) -> go.Figure:
    ph = phase_analysis(df)
    if ph.empty:
        return _empty("Phase analysis not available for this format.")
    fig = go.Figure(go.Bar(
        x=ph["Phase"], y=ph["Runs"], text=ph["Runs"], textposition="auto",
        marker=dict(
            color=ph["Run Rate"],
            colorscale=[[0, "#1B3A4B"], [0.5, _GREEN], [1, _CYAN]],
            showscale=True, colorbar=dict(title="RR", thickness=12)),
        textfont=dict(color=_WHITE)))
    return _styled(fig, title="Scoring by Phase", height=350)
