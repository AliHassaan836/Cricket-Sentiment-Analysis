"""
nlp/sentiment.py  (Module 10: Sentiment & Momentum Analysis)
============================================================
Computes per-delivery sentiment (from the batting side's perspective), a rolling
momentum curve, and a pressure timeline.

Two backends:
  * Rule-based (always available): deterministic mapping from event type to a
    sentiment value (config.EVENT_SENTIMENT). This is the default and is fully
    explainable.
  * Transformer (optional): a DistilBERT sentiment model applied to the raw
    commentary text, used only if `transformers` + a model are available.

The momentum curve is a rolling weighted sum over a fixed window of legal balls,
so swings are tied directly to what happened on the field.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import EVENT_SENTIMENT, MOMENTUM_WINDOW_BALLS, NLP_CONFIG

try:
    from transformers import pipeline as hf_pipeline  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:                       # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

_SENTIMENT_PIPE = None


def _load_transformer_sentiment():
    global _SENTIMENT_PIPE
    if not (_TRANSFORMERS_AVAILABLE and NLP_CONFIG.enable_transformers):
        return None
    if _SENTIMENT_PIPE is None:
        try:
            _SENTIMENT_PIPE = hf_pipeline(
                "sentiment-analysis", model=NLP_CONFIG.sentiment_model)
        except Exception:
            _SENTIMENT_PIPE = None
    return _SENTIMENT_PIPE


def rule_based_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic sentiment value to each delivery."""
    if df.empty:
        return df
    out = df.copy()
    out["sentiment"] = out["event_label"].map(EVENT_SENTIMENT).fillna(0.0)
    out["sentiment_label"] = np.where(
        out["sentiment"] > 0.1, "Positive",
        np.where(out["sentiment"] < -0.1, "Negative", "Neutral"))
    return out


def transformer_sentiment(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Optional abstractive sentiment over raw text. Returns None if unavailable."""
    pipe = _load_transformer_sentiment()
    if pipe is None or df.empty:
        return None
    out = df.copy()
    try:
        results = pipe(out["raw_text"].tolist(), truncation=True)
        score = []
        for r in results:
            s = r["score"] if r["label"].upper().startswith("POS") else -r["score"]
            score.append(round(s, 3))
        out["sentiment"] = score
        out["sentiment_label"] = np.where(
            out["sentiment"] > 0.1, "Positive",
            np.where(out["sentiment"] < -0.1, "Negative", "Neutral"))
        return out
    except Exception:
        return None


def compute_sentiment(df: pd.DataFrame, prefer_transformer: bool = False) -> pd.DataFrame:
    """Pick a backend and return the df with sentiment columns + the backend used."""
    if df.empty:
        return df
    if prefer_transformer:
        t = transformer_sentiment(df)
        if t is not None:
            t.attrs["sentiment_backend"] = "transformer"
            return t
    out = rule_based_sentiment(df)
    out.attrs["sentiment_backend"] = "rule-based"
    return out


def momentum_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling momentum over MOMENTUM_WINDOW_BALLS legal balls.
    Positive = momentum with the batting side.
    """
    if df.empty:
        return pd.DataFrame()
    s = compute_sentiment(df)
    legal = s[s["is_legal"]].sort_values("seq").copy()
    legal["legal_ball_no"] = range(1, len(legal) + 1)
    legal["momentum"] = (
        legal["sentiment"].rolling(MOMENTUM_WINDOW_BALLS, min_periods=1).mean().round(3)
    )
    # Pressure = rolling fraction of dot balls + wickets (higher = more pressure
    # on the batting side). Deterministic and evidence-tied.
    legal["pressure_event"] = ((legal["total_runs"] == 0) | (legal["is_wicket"])).astype(int)
    legal["pressure"] = (
        legal["pressure_event"].rolling(MOMENTUM_WINDOW_BALLS, min_periods=1).mean().round(3)
    )
    return legal[["legal_ball_no", "over_str", "event_label",
                  "sentiment", "momentum", "pressure", "raw_text"]]
