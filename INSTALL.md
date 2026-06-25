# Installation Guide — AI Cricket Intelligence System

This guide covers a minimal install (deterministic analytics + visualizations +
export) and a full install (adds transformer summaries, spaCy NER, NLTK).

---

## 1. Prerequisites

- **Python 3.10 – 3.12** (3.12 recommended)
- `pip`
- ~2 GB free disk space if you install the optional transformer models

Check your version:

```bash
python --version
```

---

## 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

### Option A — Full install (all NLP features)

```bash
pip install -r requirements.txt
```

Then download the language models used by the optional NLP layer:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

Transformer summarization models (e.g. DistilBART) download automatically the
first time you enable the transformer summarizer in the UI.

### Option B — Lightweight install (no large models)

The app is fully functional without the heavy NLP stack. Open
`requirements.txt`, comment out the **OPTIONAL** group (`spacy`, `nltk`,
`transformers`, `torch`, `sentencepiece`), then:

```bash
pip install -r requirements.txt
```

In this mode the app uses rule-based tokenization/NER and template-based
summaries. The sidebar shows which backends are active. All numeric analytics
are identical to the full install.

---

## 4. Run the application

```bash
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>). Open it, then:

1. In the sidebar, click **"Load sample innings"** to try the bundled example,
   or use **"Commentary file"** to upload your own `.txt`.
2. Explore the dashboard tabs.
3. Use the **Export** tab to download CSV scorecards or a PDF report.

---

## 5. Run the tests (optional)

```bash
pip install pytest        # already included in requirements.txt
pytest -q
```

You should see all tests pass. They verify the parser and the batting, bowling,
and turning-point analytics against hand-computed fixtures.

---

## 6. Database

The app persists uploaded matches to a local SQLite file (path configured in
`config.py` → `DATABASE_PATH`). The schema is created automatically on first
run; it is also documented in `schema.sql`, which you can apply manually:

```bash
sqlite3 cricket_intelligence.db < schema.sql
```

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'plotly'`** — Plotly is a core
  dependency. Re-run `pip install -r requirements.txt`.
- **spaCy model not found** — run `python -m spacy download en_core_web_sm`.
  Without it the app falls back to rule-based NER (this is expected, not an
  error).
- **Slow first transformer summary** — the model is downloading. Subsequent
  runs are cached. If you don't want this, simply leave the transformer toggle
  off; template summaries are instant and equally accurate.
- **PDF export shows a `.txt` file instead** — `fpdf2` isn't installed; the app
  falls back to a plain-text report. Install it with `pip install fpdf2`.
- **Network-restricted environment** — transformer/model downloads require
  internet access. Use the lightweight install (Option B); everything else
  works offline.
