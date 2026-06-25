"""
nlp/pipeline.py  (Module 2: NLP Pipeline)
=========================================
Text cleaning, tokenization, and Named Entity Recognition.

IMPORTANT — grounding
---------------------
The authoritative source of player/team entities is the *parser*, which already
knows the bowler and batter of every delivery from the line structure. spaCy NER
is used as a supplementary cross-check and to surface fielder names. If spaCy or
NLTK are unavailable, the pipeline falls back to deterministic, rule-based
extraction so the application always runs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

import pandas as pd

from config import NLP_CONFIG

# ---- Optional dependencies (graceful degradation) ------------------------- #
try:
    import spacy  # type: ignore
    _SPACY_AVAILABLE = True
except Exception:                       # pragma: no cover
    _SPACY_AVAILABLE = False

try:
    import nltk  # type: ignore
    from nltk.tokenize import word_tokenize  # type: ignore
    _NLTK_AVAILABLE = True
except Exception:                       # pragma: no cover
    _NLTK_AVAILABLE = False


_NLP = None


def _load_spacy():
    global _NLP
    if not _SPACY_AVAILABLE:
        return None
    if _NLP is None:
        try:
            _NLP = spacy.load(NLP_CONFIG.spacy_model)
        except Exception:
            try:
                from spacy.cli import download  # type: ignore
                download(NLP_CONFIG.spacy_model)
                _NLP = spacy.load(NLP_CONFIG.spacy_model)
            except Exception:
                _NLP = None
    return _NLP


def clean_text(text: str) -> str:
    """Normalise commentary text: collapse whitespace, strip stray symbols."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text; uses NLTK if present, else a simple regex tokenizer."""
    if _NLTK_AVAILABLE:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z']+|\d+|[^\sA-Za-z\d]", text)


def extract_entities(df: pd.DataFrame, raw_text: str = "") -> Dict[str, List[str]]:
    """
    Return entities grounded in the parsed data, augmented by spaCy where
    available. The parsed-data entities are authoritative.
    """
    entities: Dict[str, Set[str]] = {
        "batters": set(),
        "bowlers": set(),
        "fielders": set(),
        "persons_detected": set(),
    }
    if not df.empty:
        entities["batters"].update(df["batter"].dropna().unique().tolist())
        entities["bowlers"].update(df["bowler"].dropna().unique().tolist())
        fielders = df[df["fielders"].astype(bool)]["fielders"].unique().tolist()
        entities["fielders"].update([f for f in fielders if f])

    nlp = _load_spacy()
    if nlp is not None and raw_text:
        try:
            doc = nlp(raw_text[:100000])  # cap to keep it fast
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["persons_detected"].add(ent.text.strip())
        except Exception:
            pass

    return {k: sorted(v) for k, v in entities.items()}


def pipeline_status() -> Dict[str, bool]:
    """Report which NLP backends are active (shown in the UI for transparency)."""
    return {
        "spacy": _SPACY_AVAILABLE and _load_spacy() is not None,
        "nltk": _NLTK_AVAILABLE,
    }
