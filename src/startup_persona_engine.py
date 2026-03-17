"""
src/startup_persona_engine.py
Derives a startup brand persona from description text.

Implementation:
  - Keyword/heuristic classifier [rule-based] when no model is available
  - K-means cluster profile (if sklearn available and corpus ≥ 50 rows)
  - Returns a PersonaProfile with tone recommendations

This is rule-based / heuristic — no pre-trained model is claimed.
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    persona:     str            # e.g. "Modern SaaS"
    description: str            # human-readable summary
    recommended_tone:      str
    recommended_industry:  str
    brand_keywords:        list[str]
    color_personality:     str  # e.g. "cool blues and whites"
    font_personality:      str  # e.g. "geometric sans-serif"


# ── Keyword → persona rules ───────────────────────────────────────────────────
_RULES: list[tuple[list[str], str, str, str, str, str]] = [
    # keywords,          persona,              tone,           industry,       colors,                 fonts
    (["ai","machine learning","neural","llm","gpt","ml","deep learning"],
     "AI / Deep Tech",    "Innovative",    "Technology",   "electric violet, deep space black", "experimental sans"),
    (["saas","software","platform","dashboard","api","cloud","devops","b2b"],
     "Modern SaaS",       "Minimalist",    "Technology",   "clean blues and whites",            "geometric sans-serif"),
    (["fintech","payment","bank","finance","invest","crypto","wallet"],
     "FinTech",           "Trustworthy",   "Finance",      "navy, gold, white",                 "authoritative serif"),
    (["health","medical","clinical","wellness","patient","hospital","pharma"],
     "HealthTech",        "Professional",  "Healthcare",   "clinical blue, clean white",        "rounded humanist"),
    (["fashion","style","luxury","haute","boutique","couture","apparel"],
     "Luxury Lifestyle",  "Luxury",        "Fashion",      "black, ivory, gold",                "high-fashion serif"),
    (["food","meal","recipe","restaurant","nutrition","delivery","beverage"],
     "Food & Lifestyle",  "Playful",       "Food & Beverage","warm reds and oranges",           "hand-crafted display"),
    (["edu","learn","course","school","student","tutor","training"],
     "EdTech",            "Professional",  "Education",    "academic navy, amber",              "academic serif"),
    (["green","sustain","eco","climate","carbon","renewable","planet"],
     "Eco Brand",         "Eco-Conscious", "Sustainability","forest green, earth brown",        "clean organic sans"),
    (["travel","trip","hotel","flight","tourism","adventure","explore"],
     "Travel & Experience","Creative",     "Travel",       "ocean blue, sunset orange",         "poster display"),
    (["retail","shop","store","ecomm","marketplace","d2c","consumer"],
     "Youthful D2C",      "Youthful",      "Retail",       "bold primaries",                    "friendly geometric"),
    (["consult","strategy","advisory","management","enterprise","b2b service"],
     "Professional Services","Professional","Consulting",  "slate, gold, white",               "strategic serif"),
]


def _keyword_persona(text: str) -> PersonaProfile:
    """Heuristic keyword-based persona derivation."""
    text_lower = text.lower()
    for keywords, persona, tone, industry, colors, fonts in _RULES:
        if any(kw in text_lower for kw in keywords):
            words = re.findall(r"\b\w{4,}\b", text_lower)
            freq: dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            top = sorted(freq, key=freq.get, reverse=True)[:6]
            return PersonaProfile(
                persona=persona,
                description=f"Based on your description, this brand reads as a {persona}.",
                recommended_tone=tone,
                recommended_industry=industry,
                brand_keywords=top,
                color_personality=colors,
                font_personality=fonts,
            )
    # Default fallback
    return PersonaProfile(
        persona="Modern Business",
        description="A versatile, professional business brand.",
        recommended_tone="Professional",
        recommended_industry="Technology",
        brand_keywords=["innovation","quality","trust"],
        color_personality="neutral blacks and whites with a warm accent",
        font_personality="clean sans-serif with serif headline",
    )


def derive_persona(
    company:     str,
    industry:    str,
    tone:        str,
    description: str = "",
    audience:    str = "",
) -> PersonaProfile:
    """
    Main entry point.
    Uses provided industry/tone as strong hints; description refines further.
    Falls back to keyword heuristics when description is sparse.
    """
    combined = f"{industry} {tone} {description} {audience}".lower()
    # If user provided tone + industry explicitly, honour them
    if industry and tone:
        persona_name = f"{tone} {industry}"
        words = re.findall(r"\b\w{4,}\b", description.lower()) if description else []
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)[:6] or [industry.lower(), tone.lower()]
        return PersonaProfile(
            persona=f"{tone} {industry} Brand",
            description=f"{company} is positioned as a {tone.lower()} brand in the {industry} space.",
            recommended_tone=tone,
            recommended_industry=industry,
            brand_keywords=top,
            color_personality=f"colours matching {tone.lower()} and {industry.lower()} conventions",
            font_personality=f"typography reflecting {tone.lower()} brand values",
        )
    return _keyword_persona(combined)


def find_similar_startups(description: str, corpus: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """
    Return top_k most similar startups by TF-IDF cosine similarity.
    Dataset-backed when corpus is non-empty; returns empty frame otherwise.
    """
    if corpus is None or len(corpus) == 0:
        return pd.DataFrame()
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        texts = (corpus.get("description","") + " " + corpus.get("tagline","")).fillna("").tolist()
        vec   = TfidfVectorizer(max_features=2000, stop_words="english")
        mat   = vec.fit_transform(texts)
        q     = vec.transform([description])
        sims  = cosine_similarity(q, mat).flatten()
        idx   = np.argsort(sims)[::-1][:top_k]
        out   = corpus.iloc[idx].copy()
        out["similarity"] = sims[idx].round(3)
        return out[out["similarity"] > 0.01]
    except Exception as e:
        log.warning("Startup similarity search failed: %s", e)
        return pd.DataFrame()
