"""
app.py — AI Cricket Intelligence System
========================================
Streamlit dashboard for cricket match analytics.
Run with:  streamlit run app.py
"""

from __future__ import annotations
import os
import streamlit as st

from config import UNKNOWN
from src.parser.commentary_parser import CommentaryParser
from src.analytics import batting, bowling, team, events, turning_points, impact
from src.validation.validator import validate
from src.nlp.pipeline import extract_entities
from src.nlp.sentiment import momentum_curve
from src.summarization.summarizer import generate_summaries
from src.visualization import charts
from src.report.exporter import (dataframe_to_csv_bytes, build_pdf_report,
                                 build_scorecard_csv)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_PATH = os.path.join(_APP_DIR, "sample_data", "sample_commentary.txt")
_SAMPLE_INDvNZ = os.path.join(_APP_DIR, "sample_data", "ind_vs_nz.txt")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Intelligence",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --accent: #00B4D8;
    --accent-dim: rgba(0,180,216,0.12);
    --green: #00C853;
    --red: #FF3D3D;
    --gold: #FFD700;
    --purple: #BB86FC;
    --orange: #FF9100;
    --card-bg: #161B22;
    --border: #21262D;
    --text-primary: #E6EDF3;
    --text-muted: #8B949E;
}

/* Clean up default streamlit look */
.stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.main .block-container { padding-top: 1.5rem; max-width: 1300px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span { color: var(--text-muted); }

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}

/* Dataframes */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.02em;
}
.stTabs [aria-selected="true"] {
    border-bottom-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: var(--card-bg) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: var(--accent) !important;
    color: #0D1117 !important;
}

/* Expander */
.streamlit-expanderHeader { font-weight: 600; font-size: 0.9rem; }

/* Containers with border */
div[data-testid="stExpander"],
div.stAlert {
    border-radius: 8px !important;
}

/* Validation banner */
.validation-pass {
    background: rgba(0,200,83,0.08);
    border-left: 4px solid var(--green);
    padding: 10px 16px;
    border-radius: 6px;
    color: var(--green);
    font-weight: 500;
    margin: 8px 0 16px 0;
}
.validation-fail {
    background: rgba(255,61,61,0.08);
    border-left: 4px solid var(--red);
    padding: 10px 16px;
    border-radius: 6px;
    color: var(--red);
    font-weight: 500;
    margin: 8px 0 16px 0;
}

/* Section titles */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 20px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--border);
    display: inline-block;
}

/* Summary card */
.summary-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin: 10px 0 16px 0;
    line-height: 1.7;
    color: var(--text-primary);
    font-size: 0.95rem;
}

/* Impact hero */
.impact-hero {
    background: linear-gradient(135deg, #161B22 0%, #1a2332 100%);
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    margin: 10px 0;
}
.impact-hero .trophy { font-size: 2.4rem; margin-bottom: 4px; }
.impact-hero .name {
    font-size: 1.5rem; font-weight: 700;
    color: var(--gold); margin-bottom: 2px;
}
.impact-hero .score {
    font-size: 1.1rem; color: var(--text-muted);
}

/* Turning point card */
.tp-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 8px 0;
}
.tp-card .tp-header {
    font-weight: 700; font-size: 1rem;
    color: var(--accent); margin-bottom: 4px;
}
.tp-card .tp-importance {
    float: right; background: rgba(0,180,216,0.15);
    color: var(--accent); padding: 2px 10px;
    border-radius: 12px; font-size: 0.82rem; font-weight: 600;
}
.tp-card .tp-reason {
    color: var(--text-muted); font-size: 0.9rem;
    margin-top: 4px;
}

/* Hero header */
.hero-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 4px;
}
.hero-header .logo { font-size: 2rem; }
.hero-header .title {
    font-size: 1.5rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.hero-header .badge {
    background: var(--accent); color: #0D1117;
    font-size: 0.65rem; font-weight: 700;
    padding: 2px 8px; border-radius: 4px;
    text-transform: uppercase; letter-spacing: 0.08em;
    vertical-align: middle; margin-left: 6px;
}

/* Hide hamburger menu / footer */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── State management ────────────────────────────────────────────────────────

def _init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("raw_text", "")
    st.session_state.setdefault("unparsed", [])


def _load_commentary(text: str):
    parser = CommentaryParser()
    parser.parse_text(text)
    st.session_state.df = parser.to_dataframe()
    st.session_state.raw_text = text
    st.session_state.unparsed = parser.unparsed_lines


# ── Sidebar ─────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="hero-header">'
            '<span class="logo">🏏</span>'
            '<span class="title">Cricket Intel<span class="badge">AI</span></span>'
            '</div>',
            unsafe_allow_html=True)
        st.caption("Upload ball-by-ball commentary to generate instant analytics.")
        st.markdown("---")

        uploaded = st.file_uploader("Upload Commentary (.txt)", type=["txt"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            _load_commentary(uploaded.read().decode("utf-8", errors="ignore"))

        st.markdown("")
        st.markdown("**Or try a sample:**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏏 Sample T20", use_container_width=True):
                try:
                    with open(_SAMPLE_PATH, encoding="utf-8") as fh:
                        _load_commentary(fh.read())
                except FileNotFoundError:
                    st.error("Sample file not found.")
        with c2:
            if st.button("🇮🇳 IND vs NZ", use_container_width=True):
                try:
                    with open(_SAMPLE_INDvNZ, encoding="utf-8") as fh:
                        _load_commentary(fh.read())
                except FileNotFoundError:
                    st.error("Sample file not found.")


# ── Tab: Overview ───────────────────────────────────────────────────────────

def tab_overview(df):
    rep = validate(df, st.session_state.unparsed)
    ts = team.team_summary(df)

    # Score metrics — top row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{ts['total_runs']}/{ts['wickets']}")
    c2.metric("Overs", ts["overs"])
    c3.metric("Run Rate", ts["run_rate"])
    c4.metric("Format", ts["format"])

    c5, c6, c7 = st.columns(3)
    c5.metric("Boundaries", ts["boundaries"])
    c6.metric("Extras", ts["extras"])
    c7.metric("Dot Balls", ts["dot_balls"])

    # Validation banner (clean, non-academic)
    if rep["passed"]:
        st.markdown('<div class="validation-pass">✓ All statistics verified and consistent</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="validation-fail">⚠ Some statistics could not be fully reconciled</div>',
                    unsafe_allow_html=True)
    for w in rep.get("warnings", []):
        st.warning(w)

    # Match summary
    st.markdown('<div class="section-title">Match Summary</div>', unsafe_allow_html=True)
    summaries = generate_summaries(df, use_transformer=False)
    st.markdown(f'<div class="summary-card">{summaries["short"]}</div>', unsafe_allow_html=True)

    with st.expander("Detailed Summary"):
        st.write(summaries["detailed"])
    with st.expander("Match Story"):
        st.write(summaries["story"])

    # Scorecard
    st.markdown('<div class="section-title">Scorecard</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("BATTING")
        st.dataframe(batting.batting_scorecard(df), use_container_width=True, hide_index=True)
    with cc2:
        st.caption("BOWLING")
        st.dataframe(bowling.bowling_scorecard(df), use_container_width=True, hide_index=True)


# ── Tab: Batting ────────────────────────────────────────────────────────────

def tab_batting(df):
    st.dataframe(batting.batting_scorecard(df), use_container_width=True, hide_index=True)
    st.plotly_chart(charts.run_progression_chart(df), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.batter_comparison_chart(df), use_container_width=True)
    c2.plotly_chart(charts.strike_rate_chart(df), use_container_width=True)


# ── Tab: Bowling ────────────────────────────────────────────────────────────

def tab_bowling(df):
    st.dataframe(bowling.bowling_scorecard(df), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.economy_chart(df), use_container_width=True)
    c2.plotly_chart(charts.bowler_comparison_chart(df), use_container_width=True)
    st.plotly_chart(charts.team_scoring_chart(df), use_container_width=True)


# ── Tab: Team ───────────────────────────────────────────────────────────────

def tab_team(df):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Partnerships</div>', unsafe_allow_html=True)
        st.dataframe(team.partnerships(df), use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-title">Phase Breakdown</div>', unsafe_allow_html=True)
        ph = team.phase_analysis(df)
        if ph.empty:
            st.info("Phase analysis available for T20 / ODI.")
        else:
            st.dataframe(ph, use_container_width=True, hide_index=True)
    st.plotly_chart(charts.partnership_chart(df), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.phase_chart(df), use_container_width=True)
    c4.plotly_chart(charts.wicket_timeline_chart(df), use_container_width=True)


# ── Tab: Insights ───────────────────────────────────────────────────────────

def tab_insights(df):
    ents = extract_entities(df, st.session_state.raw_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Batters", len(ents["batters"]))
    c2.metric("Bowlers", len(ents["bowlers"]))
    c3.metric("Fielders", len(ents["fielders"]))

    with st.expander("Key Players Detected"):
        for role in ("batters", "bowlers", "fielders"):
            if ents[role]:
                st.markdown(f"**{role.title()}:** {', '.join(sorted(ents[role]))}")

    st.markdown('<div class="section-title">Momentum</div>', unsafe_allow_html=True)
    st.plotly_chart(charts.momentum_chart(df), use_container_width=True)
    st.plotly_chart(charts.pressure_chart(df), use_container_width=True)


# ── Tab: Turning Points ────────────────────────────────────────────────────

def tab_turning(df):
    tps = turning_points.detect_turning_points(df, top_n=6)
    if not tps:
        st.info("No significant turning points detected.")
        return
    for tp in tps:
        st.markdown(
            f'<div class="tp-card">'
            f'<span class="tp-importance">{tp["importance"]}</span>'
            f'<div class="tp-header">{tp["over"]} — {tp["title"]}</div>'
            f'<div class="tp-reason">{tp["reason"]}</div>'
            f'</div>',
            unsafe_allow_html=True)
        with st.expander("View commentary evidence"):
            for line in tp["evidence"]:
                st.code(line, language=None)


# ── Tab: Impact Player ─────────────────────────────────────────────────────

def tab_impact(df):
    imp = impact.compute_impact(df)
    if imp.empty:
        st.info(UNKNOWN)
        return
    top = imp.iloc[0]

    st.markdown(
        f'<div class="impact-hero">'
        f'<div class="trophy">🏆</div>'
        f'<div class="name">{top["Player"]}</div>'
        f'<div class="score">Impact Score: {top["Impact Score"]}</div>'
        f'</div>',
        unsafe_allow_html=True)

    with st.expander("Performance Breakdown"):
        for b in top["breakdown"]:
            st.write(f"• {b}")

    st.markdown('<div class="section-title">Full Rankings</div>', unsafe_allow_html=True)
    display_cols = ["Player", "Impact Score", "bat_raw", "bowl_raw", "field_raw"]
    display_df = imp[display_cols].copy()
    display_df.columns = ["Player", "Impact", "Batting", "Bowling", "Fielding"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Tab: Export ─────────────────────────────────────────────────────────────

def tab_export(df):
    bat = batting.batting_scorecard(df)
    bowl = bowling.bowling_scorecard(df)
    summaries = generate_summaries(df, use_transformer=False)
    tps = turning_points.detect_turning_points(df, top_n=6)
    imp = impact.compute_impact(df)

    st.markdown('<div class="section-title">Download Reports</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📊  Scorecard CSV", build_scorecard_csv(bat, bowl),
                           "scorecard.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("📋  Ball-by-ball CSV", dataframe_to_csv_bytes(df),
                           "deliveries.csv", "text/csv", use_container_width=True)
    with c3:
        sections = {
            "Match Summary": summaries["detailed"],
            "Match Story": summaries["story"],
            "Turning Points": "\n".join(
                f"- {t['over']}: {t['reason']}" for t in tps) or UNKNOWN,
            "Most Impactful Player": (
                f"{imp.iloc[0]['Player']} (Impact {imp.iloc[0]['Impact Score']})\n" +
                "\n".join(imp.iloc[0]["breakdown"])) if not imp.empty else UNKNOWN,
        }
        pdf_bytes = build_pdf_report("Cricket Intelligence Report", sections)
        fname = "match_report.pdf" if pdf_bytes[:4] == b"%PDF" else "match_report.txt"
        st.download_button("📄  Full Report PDF", pdf_bytes, fname,
                           use_container_width=True)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    sidebar()

    df = st.session_state.df
    if df is None or df.empty:
        st.markdown("")
        st.markdown(
            '<div style="text-align:center; padding:80px 20px;">'
            '<div style="font-size:3.5rem; margin-bottom:12px;">🏏</div>'
            '<div style="font-size:1.5rem; font-weight:700; color:#E6EDF3; margin-bottom:8px;">'
            'Cricket Intelligence System</div>'
            '<div style="color:#8B949E; max-width:480px; margin:0 auto; line-height:1.6;">'
            'Upload a ball-by-ball commentary file or load a sample match '
            'from the sidebar to get started.</div>'
            '</div>',
            unsafe_allow_html=True)
        return

    # Innings selector for multi-innings files
    if "innings" in df.columns and df["innings"].nunique() > 1:
        innings_opts = ["Both Innings"] + [
            f"Innings {i}" for i in sorted(df["innings"].unique())]
        choice = st.selectbox("Select innings", innings_opts,
                              label_visibility="collapsed")
        if choice != "Both Innings":
            inn_num = int(choice.split()[-1])
            df = df[df["innings"] == inn_num].reset_index(drop=True)

    tabs = st.tabs([
        "Overview", "Batting", "Bowling", "Team",
        "Insights", "Turning Points", "Impact Player", "Export"])
    with tabs[0]: tab_overview(df)
    with tabs[1]: tab_batting(df)
    with tabs[2]: tab_bowling(df)
    with tabs[3]: tab_team(df)
    with tabs[4]: tab_insights(df)
    with tabs[5]: tab_turning(df)
    with tabs[6]: tab_impact(df)
    with tabs[7]: tab_export(df)


if __name__ == "__main__":
    main()
