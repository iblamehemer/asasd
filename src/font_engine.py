"""
src/font_engine.py
Font recommendation engine.

Implementation:
  Mapping-based [rule-based] — no font image dataset exists to train a CNN.
  Clearly documented. Returns Google Fonts pairings with rationale.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FontRecommendation:
    heading:    str
    body:       str
    accent:     str
    style:      str
    rationale:  str
    heading_url: str = ""
    body_url:    str = ""
    accent_url:  str = ""
    pairing_score: float = 0.0   # aesthetic harmony 0–1


def _google_font_url(font: str) -> str:
    encoded = font.replace(" ", "+")
    return f"https://fonts.google.com/specimen/{encoded}"


def recommend_fonts(industry: str, tone: str) -> FontRecommendation:
    """
    Return font pairing for an (industry, tone) pair.
    Falls back to sensible defaults if the combination is not mapped.

    Source: rule-based mapping in src/config.py — no ML model.
    """
    from src.config import FONT_MAP

    # Try exact match, then tone-only, then industry-only, then default
    key = f"{industry}-{tone}"
    data = (FONT_MAP.get(key)
            or FONT_MAP.get(f"Technology-{tone}")
            or FONT_MAP.get(f"{industry}-Professional")
            or FONT_MAP["_default"])

    # Pairing score heuristic: exact match = 0.95, fallback tiers drop by 0.1
    if key in FONT_MAP:
        score = 0.95
    elif f"Technology-{tone}" in FONT_MAP:
        score = 0.80
    else:
        score = 0.70

    return FontRecommendation(
        heading=data["heading"],
        body=data["body"],
        accent=data["accent"],
        style=data["style"],
        rationale=data["rationale"],
        heading_url=_google_font_url(data["heading"]),
        body_url=_google_font_url(data["body"]),
        accent_url=_google_font_url(data["accent"]),
        pairing_score=score,
    )


def css_import_block(rec: FontRecommendation) -> str:
    """Return a @import CSS line for embedding Google Fonts."""
    def _slug(f): return f.replace(" ", "+")
    return (
        f"@import url('https://fonts.googleapis.com/css2?"
        f"family={_slug(rec.heading)}:wght@300;400;500&"
        f"family={_slug(rec.body)}:wght@300;400;500&"
        f"family={_slug(rec.accent)}:wght@300;400');"
    )
