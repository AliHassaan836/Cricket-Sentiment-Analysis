# 🏏 AI Cricket Intelligence System

An NLP-driven Streamlit application that turns raw **ball-by-ball cricket
commentary** into accurate, explainable match analytics: scorecards, bowling
and team analysis, turning points, impact-player rankings, AI summaries,
sentiment/momentum curves, interactive visualizations, and exportable reports.

> **Accuracy first.** Every number the app shows is computed directly from the
> parsed commentary by a deterministic engine and is checked by a validation
> layer before display. The system never invents scores, wickets, players,
> partnerships, or statistics. When a value cannot be derived from the input it
> shows: **"Cannot determine from provided commentary."**

---

## What it does

Give it a commentary file like:

```
0.1 Shaheen to Rohit, no run
0.2 Shaheen to Rohit, FOUR
0.3 Shaheen to Rohit, 1 run
0.4 Shaheen to Gill, OUT, caught behind
```

…and it produces a full dashboard:

- **Match Overview** – validated score, run rate, format detection, AI summary.
- **Batting** – per-batter runs, balls, strike rate, 4s/6s, dot balls, dismissal.
- **Bowling** – overs, runs, wickets, economy, maidens, dot %, boundary %.
- **Team** – totals, partnerships, powerplay/middle/death phase breakdown.
- **NLP Insights** – entities, important events with supporting evidence.
- **Turning Points** – ranked momentum swings, each with commentary evidence.
- **Impact Player** – transparent weighted scoring with a per-player breakdown.
- **Export** – CSV scorecards and a PDF match report.

---

## How accuracy is guaranteed

The design separates **facts** from **language**:

1. **Deterministic parser + analytics engine** (Pandas/NumPy) computes every
   statistic from the structured deliveries. This layer is fully unit-tested.
2. **NLP / transformer layer** (spaCy, NLTK, HuggingFace) only *rephrases or
   classifies* facts that already exist. Summaries are built from a verified
   "fact sheet"; the transformer is allowed to reword it, never to add numbers.
3. **Validation gate** (Module 14) cross-checks totals — batting runs + extras
   vs. team total, dismissed batters vs. wickets, partnership sums, etc. If a
   check fails, the UI reports **"Data inconsistency detected."** instead of
   guessing.

The heavy NLP packages are **optional**. If they (or their models) are not
installed, the app automatically falls back to rule-based tokenization, NER, and
template summaries, and tells you which backend is active. The deterministic
analytics are identical either way.

---

## Architecture

```
AI Cricket Intelligence System
│
├── app.py                     Streamlit UI (tabbed dashboard)
├── config.py                  Central thresholds, weights, phase definitions
│
├── src/
│   ├── parser/                Module 1  – commentary -> structured deliveries
│   ├── nlp/                   Module 2/10 – cleaning, NER, sentiment, momentum
│   ├── analytics/             Modules 3-8 – batting, bowling, team, events,
│   │                                        turning points, impact
│   ├── summarization/         Module 9  – grounded summaries & match story
│   ├── validation/            Module 14 – hallucination-prevention gate
│   ├── visualization/         Modules 12/13 – Plotly chart builders
│   ├── storage/               SQLite persistence (matches + deliveries)
│   └── report/                Module 15 – CSV / PDF export
│
├── tests/                     pytest unit tests (parser, batting, bowling, TP)
├── sample_data/               sample_commentary.txt
├── schema.sql                 Database schema (mirrors storage layer)
├── requirements.txt
└── INSTALL.md
```

### Module map

| Module | File | Responsibility |
|-------|------|----------------|
| 1  | `src/parser/commentary_parser.py` | Parse commentary into structured deliveries |
| 2  | `src/nlp/pipeline.py` | Text cleaning, tokenization, entity extraction |
| 3  | `src/analytics/batting.py` | Batting scorecard, run progression |
| 4  | `src/analytics/bowling.py` | Bowling scorecard, over-by-over |
| 5  | `src/analytics/team.py` | Totals, partnerships, phase analysis, format |
| 6  | `src/analytics/events.py` | Important-event detection with evidence |
| 7  | `src/analytics/turning_points.py` | Turning-point detection & explanations |
| 8  | `src/analytics/impact.py` | Explainable impact-player scoring |
| 9  | `src/summarization/summarizer.py` | Short / detailed summaries, match story |
| 10 | `src/nlp/sentiment.py` | Sentiment, momentum & pressure curves |
| 12/13 | `src/visualization/charts.py` | Plotly visualizations |
| 14 | `src/validation/validator.py` | Cross-check validation gate |
| 15 | `src/report/exporter.py` | CSV / PDF report export |

---

## Scoring & detection conventions

These rules are applied consistently and documented in code:

- `batter_runs` = runs off the bat; `extras` = wides + no-balls + byes + leg-byes.
- A **legal ball** is anything that is not a wide or no-ball.
- A **boundary** is flagged only when the commentary explicitly says FOUR/SIX/
  boundary — never inferred from a numeric "4 runs" (all-run fours exist).
- `bowler_runs` = batter runs + wides + no-balls (byes and leg-byes excluded).
- **Run-outs are not credited to the bowler.**
- Impact score is a transparent weighted sum (see `config.py` →
  `IMPACT_WEIGHTS`), min-max normalized to 0–100, with a per-player breakdown.

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints, click **"Load sample innings"** in
the sidebar (or upload your own `.txt`), and explore the tabs. Full setup
details, including optional model downloads, are in **INSTALL.md**.

### Tests

```bash
pytest -q
```

---

## Input format

One delivery per line:

```
<over>.<ball> <Bowler> to <Batter>, <description>
```

The parser understands runs (`no run`, `1 run`, `2 runs`…), boundaries
(`FOUR`, `SIX`), extras (`wide`, `no ball`, `byes`, `leg byes`), and dismissals
(`OUT, caught by …`, `bowled`, `lbw`, `run out`, `stumped`). Lines it cannot
parse are reported rather than silently dropped.
