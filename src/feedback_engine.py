"""
src/feedback_engine.py
Feedback collection, storage, and analytics.

Storage: CSV file (FEEDBACK_DB path from config).
Analytics: VADER sentiment on comments, aggregation by asset type.
Architecture is future-ready for model retraining triggers.
"""

from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from src.config import FEEDBACK_DB

log = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    session_id:   str
    timestamp:    str
    company:      str
    industry:     str
    tone:         str
    asset_type:   str    # "logo" | "slogan" | "palette" | "campaign" | "overall"
    rating:       int    # 1–5
    comment:      str
    sentiment:    str    # "positive" | "neutral" | "negative"
    sentiment_score: float


def _vader_sentiment(text: str) -> tuple[str, float]:
    """VADER sentiment analysis. Returns (label, compound_score)."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        try:
            sia = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:   label = "positive"
        elif score <= -0.05: label = "negative"
        else:               label = "neutral"
        return label, round(score, 4)
    except Exception:
        # Fallback: simple keyword sentiment
        pos = sum(w in text.lower() for w in ["great","love","excellent","perfect","amazing","good"])
        neg = sum(w in text.lower() for w in ["bad","poor","terrible","awful","hate","wrong"])
        if pos > neg:   return "positive",  0.5
        if neg > pos:   return "negative", -0.5
        return "neutral", 0.0


def save_feedback(
    session_id: str,
    company:    str,
    industry:   str,
    tone:       str,
    asset_type: str,
    rating:     int,
    comment:    str = "",
) -> FeedbackEntry:
    """Append a feedback entry to the CSV store."""
    label, score = _vader_sentiment(comment) if comment else ("neutral", 0.0)
    entry = FeedbackEntry(
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat(),
        company=company,
        industry=industry,
        tone=tone,
        asset_type=asset_type,
        rating=rating,
        comment=comment,
        sentiment=label,
        sentiment_score=score,
    )
    df_new = pd.DataFrame([asdict(entry)])
    try:
        if Path(FEEDBACK_DB).exists():
            existing = pd.read_csv(FEEDBACK_DB)
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            combined = df_new
        Path(FEEDBACK_DB).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(FEEDBACK_DB, index=False)
        log.info("Feedback saved: %s rating=%d", asset_type, rating)
    except Exception as e:
        log.warning("Could not save feedback: %s", e)
    return entry


def load_feedback() -> pd.DataFrame:
    """Load all feedback. Returns empty DataFrame if file absent."""
    if not Path(FEEDBACK_DB).exists():
        return pd.DataFrame(columns=[
            "session_id","timestamp","company","industry","tone",
            "asset_type","rating","comment","sentiment","sentiment_score"
        ])
    try:
        return pd.read_csv(FEEDBACK_DB, parse_dates=["timestamp"])
    except Exception as e:
        log.warning("Could not load feedback: %s", e)
        return pd.DataFrame()


def compute_feedback_summary(df: pd.DataFrame) -> dict:
    """
    Aggregate feedback into a summary for the analytics dashboard.
    Returns dict with per-asset averages, sentiment breakdown, and top issues.
    """
    if df.empty:
        return {"total": 0, "avg_rating": None, "by_asset": {}, "sentiment": {}, "top_issues": []}

    total        = len(df)
    avg_rating   = round(df["rating"].mean(), 2)
    by_asset     = df.groupby("asset_type")["rating"].agg(["mean","count"]).round(2).to_dict("index")
    sentiment_ct = df["sentiment"].value_counts().to_dict()

    # Identify top negative comment themes (simple keyword frequency)
    neg_comments = df[df["sentiment"]=="negative"]["comment"].dropna().tolist()
    word_freq: dict[str,int] = {}
    stop = {"the","a","is","it","in","of","and","to","was","i","my","this","not","very","for","with"}
    for c in neg_comments:
        for w in c.lower().split():
            w = w.strip(".,!?")
            if len(w)>3 and w not in stop:
                word_freq[w] = word_freq.get(w,0)+1
    top_issues = sorted(word_freq, key=word_freq.get, reverse=True)[:5]

    return {
        "total":      total,
        "avg_rating": avg_rating,
        "by_asset":   by_asset,
        "sentiment":  sentiment_ct,
        "top_issues": top_issues,
    }


def retraining_readiness(df: pd.DataFrame) -> dict:
    """
    Signal whether the feedback volume justifies model retraining.
    Returns readiness dict for display in the UI.
    """
    if df.empty:
        return {"ready": False, "reason": "No feedback collected yet.", "count": 0}
    n = len(df)
    avg = df["rating"].mean() if n else 0
    low_rated = len(df[df["rating"]<=2])

    if n >= 50 and low_rated/n >= 0.20:
        return {"ready": True,  "reason": f"{low_rated} low ratings detected — retraining recommended.", "count": n}
    if n >= 100:
        return {"ready": True,  "reason": f"Sufficient volume ({n} entries) for retraining.", "count": n}
    return {"ready": False, "reason": f"Need more feedback ({n}/50 collected).", "count": n}
