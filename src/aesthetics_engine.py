"""
src/aesthetics_engine.py
Brand Aesthetics Engine — scores the harmony between all brand assets.

Implementation: rule-based scoring with weighted sub-scores [heuristic].
No ML training involved; clearly documented.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AestheticsScore:
    overall:       float          # 0–100
    breakdown:     dict[str,float]  # sub-scores
    grade:         str            # A+, A, B+, B, C, D
    improvements:  list[str]
    strengths:     list[str]


def _palette_tone_match(palette: list[dict], tone: str) -> float:
    """
    Check if the palette's primary colour hue agrees with the tone.
    Rule-based heuristic.
    """
    if not palette:
        return 50.0
    primary_hex = palette[0].get("hex","#000000").lstrip("#")
    try:
        r = int(primary_hex[0:2],16)
        g = int(primary_hex[2:4],16)
        b = int(primary_hex[4:6],16)
    except (ValueError, IndexError):
        return 50.0

    dark   = (r+g+b) < 200
    warm   = r > g and r > b
    cool   = b > r and b > g
    green_ = g > r and g > b

    tone_lower = tone.lower()
    if tone_lower in ("luxury","professional","minimalist") and dark:
        return 92.0
    if tone_lower in ("bold","innovative") and cool:
        return 88.0
    if tone_lower in ("playful","youthful") and warm:
        return 86.0
    if tone_lower == "eco-conscious" and green_:
        return 94.0
    if tone_lower == "trustworthy" and cool:
        return 90.0
    return 65.0


def _font_tone_match(font_style: str, tone: str) -> float:
    """Match font category to tone expectation."""
    matrix: dict[str,list[str]] = {
        "Minimalist":   ["geometric sans-serif","experimental sans","clean organic"],
        "Bold":         ["display sans","bold impact","condensed display"],
        "Luxury":       ["high fashion serif","estate serif","editorial contrast serif"],
        "Professional": ["authoritative serif","strategic serif","academic serif"],
        "Playful":      ["hand-crafted display","friendly display","rounded humanist"],
        "Innovative":   ["experimental sans","geometric sans-serif"],
        "Trustworthy":  ["classic serif + humanist sans","authoritative serif"],
        "Creative":     ["editorial contrast serif","poster display"],
        "Eco-Conscious":["clean organic","rounded humanist"],
        "Youthful":     ["geometric friendly","friendly display"],
    }
    targets = matrix.get(tone, [])
    font_l  = font_style.lower()
    for t in targets:
        if any(word in font_l for word in t.split()):
            return 90.0
    return 65.0


def _slogan_tone_match(slogans: list[str], tone: str) -> float:
    """
    Keyword sentiment check on slogan list.
    Heuristic — not ML-based.
    """
    if not slogans:
        return 55.0
    tone_keywords: dict[str,list[str]] = {
        "Bold":         ["dominate","win","power","bold","strong","lead","no limits"],
        "Luxury":       ["art","craft","elevate","rare","curated","prestige","excellence"],
        "Playful":      ["fun","joy","smile","play","happy","delight","love","adventure"],
        "Minimalist":   ["simply","less","clean","pure","just","nothing more","clarity"],
        "Professional": ["precision","trust","excellence","quality","delivered","built"],
        "Eco-Conscious":["green","planet","sustain","earth","nature","future","clean"],
        "Innovative":   ["tomorrow","future","rethink","pioneer","beyond","new","first"],
        "Trustworthy":  ["promise","integrity","count","reliable","honest","secure"],
        "Youthful":     ["vibe","fresh","born","young","energy","now","next"],
        "Creative":     ["imagine","colour","create","breathe","dream","wonder"],
    }
    keywords = tone_keywords.get(tone, [])
    all_text = " ".join(s.lower() for s in slogans)
    hits     = sum(kw in all_text for kw in keywords)
    return min(60 + hits*10, 95)


def compute_brand_score(
    tone:       str,
    industry:   str,
    palette:    list[dict],
    font_style: str,
    slogans:    list[str],
    logo_style: str = "",
) -> AestheticsScore:
    """
    Compute an overall Brand Consistency Score from 0–100.

    Sub-scores (weighted):
      palette_tone_match   30%
      font_tone_match      25%
      slogan_tone_match    25%
      logo_style_match     20%
    """
    from src.config import HARMONY_WEIGHTS

    p_score = _palette_tone_match(palette, tone)
    f_score = _font_tone_match(font_style, tone)
    s_score = _slogan_tone_match(slogans, tone)

    # Logo style match
    style_map: dict[str,list[str]] = {
        "Luxury":       ["emblem","lettermark"],
        "Minimalist":   ["minimalist","wordmark"],
        "Bold":         ["geometric","lettermark"],
        "Professional": ["lettermark","emblem"],
        "Playful":      ["geometric"],
        "Creative":     ["emblem","minimalist"],
    }
    expected = style_map.get(tone, [])
    l_score  = 90.0 if any(e in logo_style.lower() for e in expected) else 65.0

    w = HARMONY_WEIGHTS
    overall = (p_score * w["palette_tone_match"]
             + f_score * w["font_tone_match"]
             + s_score * w["slogan_tone_match"]
             + l_score * w["logo_style_match"])
    overall = round(min(overall, 100), 1)

    if overall >= 88:  grade = "A+"
    elif overall >= 80: grade = "A"
    elif overall >= 72: grade = "B+"
    elif overall >= 64: grade = "B"
    elif overall >= 55: grade = "C"
    else:               grade = "D"

    strengths:    list[str] = []
    improvements: list[str] = []

    if p_score >= 85: strengths.append("Colour palette aligns strongly with brand tone.")
    else:             improvements.append("Consider a palette more closely matched to your tone.")
    if f_score >= 85: strengths.append("Typography reflects the intended brand personality.")
    else:             improvements.append("Explore font pairings more specific to your tone.")
    if s_score >= 75: strengths.append("Slogan language reinforces the desired brand voice.")
    else:             improvements.append("Revise slogans to include more on-tone vocabulary.")
    if l_score >= 85: strengths.append("Logo style is consistent with brand positioning.")
    else:             improvements.append("Choose a logo style that better matches your brand tone.")

    return AestheticsScore(
        overall=overall,
        breakdown={
            "Palette–Tone Match":   round(p_score,1),
            "Font–Tone Match":      round(f_score,1),
            "Slogan–Tone Match":    round(s_score,1),
            "Logo–Tone Match":      round(l_score,1),
        },
        grade=grade,
        improvements=improvements,
        strengths=strengths,
    )
