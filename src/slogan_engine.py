"""
src/slogan_engine.py
Slogan / tagline generation pipeline.

Pipeline (honest description):
  1. Template-based generation using brand inputs          [rule-based]
  2. TF-IDF retrieval from slogan corpus                   [dataset-backed]
  3. Gemini API enhancement / refinement                   [API — optional]
     If no API key, falls back to template + retrieval only.

Output: list of SloganCandidate(text, tone, source, confidence)
"""

from __future__ import annotations
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SloganCandidate:
    text:       str
    tone:       str
    source:     str          # "template" | "retrieval" | "gemini"
    confidence: float        # 0–1
    rationale:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class _TFIDFRetriever:
    """Lightweight TF-IDF retriever for slogan inspiration."""

    def __init__(self, corpus: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._df = corpus.copy()
        docs = (corpus["Company"].fillna("") + " " + corpus["Slogan"].fillna("")).tolist()
        self._vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
        self._mat = self._vec.fit_transform(docs)
        log.debug("TFIDF corpus size: %d", len(docs))

    def query(self, text: str, top_k: int = 5) -> list[str]:
        from sklearn.metrics.pairwise import cosine_similarity
        q   = self._vec.transform([text])
        sim = cosine_similarity(q, self._mat).flatten()
        idx = np.argsort(sim)[::-1][:top_k]
        return [self._df.iloc[i]["Slogan"] for i in idx if sim[i] > 0.01]


_retriever: Optional[_TFIDFRetriever] = None


def init_retriever(corpus: pd.DataFrame) -> None:
    """Call once at app startup with the loaded slogan dataframe."""
    global _retriever
    try:
        _retriever = _TFIDFRetriever(corpus)
    except Exception as e:
        log.warning("Could not build TFIDF retriever: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Template generation
# ─────────────────────────────────────────────────────────────────────────────

def _template_slogans(company: str, industry: str, tone: str,
                       audience: str, n: int = 3) -> list[SloganCandidate]:
    from src.config import SLOGAN_TEMPLATES

    templates = SLOGAN_TEMPLATES.get(tone, SLOGAN_TEMPLATES["Professional"])
    out: list[SloganCandidate] = []
    for t in templates[:n]:
        text = (t.replace("{company}",  company)
                  .replace("{industry}", industry.lower())
                  .replace("{audience}", audience.lower() if audience else "everyone"))
        out.append(SloganCandidate(
            text=text, tone=tone, source="template",
            confidence=0.65,
            rationale=f"Generated from '{tone}' template pattern."
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval-based inspiration
# ─────────────────────────────────────────────────────────────────────────────

def _retrieval_slogans(query: str, tone: str,
                        n: int = 3) -> list[SloganCandidate]:
    global _retriever
    if _retriever is None:
        return []
    retrieved = _retriever.query(query, top_k=n)
    return [
        SloganCandidate(
            text=s, tone=tone, source="retrieval",
            confidence=0.55,
            rationale="Semantically similar to your brand profile in the corpus."
        )
        for s in retrieved
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini enhancement
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_slogans(company: str, industry: str, tone: str,
                    audience: str, description: str,
                    seeds: list[str], n: int = 4) -> list[SloganCandidate]:
    """Call Gemini to generate and refine slogans. Returns [] on any error."""
    from src.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS
    if not GEMINI_API_KEY:
        return []

    seed_block = "\n".join(f"- {s}" for s in seeds[:3]) if seeds else "(none)"
    prompt = f"""You are a world-class brand copywriter.

Brand: {company}
Industry: {industry}
Tone: {tone}
Target audience: {audience or 'general'}
Description: {description or 'N/A'}

Seed slogans for inspiration:
{seed_block}

Task: Generate exactly {n} unique, memorable brand slogans/taglines.
Requirements:
- Each must be 3–10 words
- Match the '{tone}' brand tone exactly
- Be original (do NOT copy seeds verbatim)
- No clichés ("next level", "game changer", "journey")

Return JSON array only:
[{{"text":"...", "rationale":"one-sentence reason this fits"}}]"""

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        for model_name in GEMINI_FALLBACK_MODELS:
            try:
                model  = genai.GenerativeModel(model_name)
                result = model.generate_content(prompt)
                raw    = result.text.strip()
                # Strip markdown fences
                raw = re.sub(r"^```[a-zA-Z]*\n?","", raw)
                raw = re.sub(r"\n?```$","", raw).strip()
                import json
                items = json.loads(raw)
                return [
                    SloganCandidate(
                        text=item.get("text",""), tone=tone, source="gemini",
                        confidence=0.88,
                        rationale=item.get("rationale","Gemini-generated.")
                    )
                    for item in items
                    if isinstance(item, dict) and item.get("text","").strip()
                ]
            except Exception:
                continue
    except ImportError:
        log.warning("google-generativeai not installed; skipping Gemini slogans.")
    except Exception as e:
        log.warning("Gemini slogan generation failed: %s", e)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_slogans(
    company:     str,
    industry:    str,
    tone:        str,
    audience:    str  = "",
    description: str  = "",
    n_total:     int  = 5,
) -> list[SloganCandidate]:
    """
    Main entry point. Returns a deduplicated list of slogan candidates.
    Source mix depends on what is available:
      - Always: templates (rule-based)
      - If corpus loaded: retrieval (dataset-backed)
      - If GEMINI_API_KEY set: Gemini (API-generated)
    """
    query  = f"{company} {industry} {tone} {audience} {description}"

    tmpl   = _template_slogans(company, industry, tone, audience, n=3)
    retr   = _retrieval_slogans(query, tone, n=3)
    seeds  = [c.text for c in tmpl[:2]] + [c.text for c in retr[:1]]
    gem    = _gemini_slogans(company, industry, tone, audience, description,
                              seeds, n=n_total)

    # Merge, prefer gemini > template > retrieval; deduplicate by text
    seen: set[str] = set()
    merged: list[SloganCandidate] = []
    for c in gem + tmpl + retr:
        key = c.text.lower().strip()
        if key and key not in seen:
            seen.add(key)
            merged.append(c)
        if len(merged) >= n_total:
            break

    return merged[:n_total]
