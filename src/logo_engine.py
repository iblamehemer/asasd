"""
src/logo_engine.py
Logo concept engine.

IMPORTANT: No logo image dataset was uploaded or used for training.
This engine generates SVG logo concepts using:
  - Geometric shape composition [rule-based]
  - Initial-based lettermarks
  - Icon mark suggestions
  - Style variants mapped from brand tone

A real CNN classifier would require a labelled logo image dataset
(e.g., LLD-icon or SVG-Logo-3M) — those are not available here.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass


@dataclass
class LogoConcept:
    name:        str
    description: str
    svg:         str
    style:       str    # "minimalist" | "geometric" | "lettermark" | "emblem" | "wordmark"
    tone_match:  float  # 0–1


# ─────────────────────────────────────────────────────────────────────────────
# SVG helpers
# ─────────────────────────────────────────────────────────────────────────────

def _initials(company: str, n: int = 2) -> str:
    words = re.findall(r"[A-Za-z]+", company)
    return "".join(w[0].upper() for w in words[:n]) or company[:2].upper()


def _first_word(company: str) -> str:
    return re.findall(r"[A-Za-z]+", company)[0] if re.findall(r"[A-Za-z]+", company) else company


def _svg_lettermark(company: str, primary: str, secondary: str,
                     font: str = "serif") -> str:
    init  = _initials(company, 2)
    w, h  = 200, 200
    cx, cy= w//2, h//2
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <rect width="{w}" height="{h}" fill="none"/>
  <rect x="20" y="20" width="{w-40}" height="{h-40}" rx="8"
        fill="{primary}" stroke="none"/>
  <text x="{cx}" y="{cy+22}" text-anchor="middle"
        font-family="{font}, Georgia, serif"
        font-size="72" font-weight="300"
        letter-spacing="-2" fill="{secondary}">{init}</text>
</svg>"""


def _svg_geometric_circle(company: str, primary: str, accent: str) -> str:
    init = _initials(company, 1)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect width="200" height="200" fill="none"/>
  <circle cx="100" cy="100" r="80" fill="{primary}"/>
  <circle cx="100" cy="100" r="60" fill="none" stroke="{accent}" stroke-width="1.5"/>
  <text x="100" y="118" text-anchor="middle"
        font-family="'DM Sans', sans-serif" font-size="60"
        font-weight="300" fill="{accent}">{init}</text>
</svg>"""


def _svg_wordmark(company: str, primary: str, font: str = "serif") -> str:
    name = _first_word(company).upper()
    spacing = max(4, 14 - len(name))
    w = max(300, len(name) * 28 + 60)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 100" width="{w}" height="100">
  <rect width="{w}" height="100" fill="none"/>
  <text x="{w//2}" y="65" text-anchor="middle"
        font-family="{font}, 'Playfair Display', Georgia, serif"
        font-size="42" font-weight="400"
        letter-spacing="{spacing}" fill="{primary}">{name}</text>
  <line x1="{w//2-60}" y1="75" x2="{w//2+60}" y2="75"
        stroke="{primary}" stroke-width="0.8" opacity="0.5"/>
</svg>"""


def _svg_emblem(company: str, primary: str, accent: str) -> str:
    init = _initials(company, 2)
    pts  = []
    for i in range(6):
        ang = math.radians(60*i - 30)
        pts.append(f"{100+75*math.cos(ang):.1f},{100+75*math.sin(ang):.1f}")
    polygon = " ".join(pts)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect width="200" height="200" fill="none"/>
  <polygon points="{polygon}" fill="none" stroke="{primary}" stroke-width="1.5"/>
  <polygon points="{polygon}" fill="{primary}" opacity="0.07"/>
  <text x="100" y="108" text-anchor="middle"
        font-family="'Playfair Display', Georgia, serif"
        font-size="40" font-weight="400" letter-spacing="3"
        fill="{primary}">{init}</text>
</svg>"""


def _svg_minimal_mark(company: str, primary: str, accent: str) -> str:
    """Minimalist mark: thin square bracket + initial."""
    init = _initials(company, 1)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" width="200" height="200">
  <rect width="200" height="200" fill="none"/>
  <!-- bracket left -->
  <polyline points="70,55 55,55 55,145 70,145"
            fill="none" stroke="{accent}" stroke-width="1.5"/>
  <!-- bracket right -->
  <polyline points="130,55 145,55 145,145 130,145"
            fill="none" stroke="{accent}" stroke-width="1.5"/>
  <text x="100" y="115" text-anchor="middle"
        font-family="'DM Mono', monospace" font-size="54"
        font-weight="300" fill="{primary}">{init}</text>
</svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tone → style mapping
# ─────────────────────────────────────────────────────────────────────────────

_TONE_STYLES: dict[str, str] = {
    "Minimalist":   "minimalist",
    "Bold":         "geometric",
    "Luxury":       "emblem",
    "Playful":      "geometric",
    "Professional": "lettermark",
    "Innovative":   "minimalist",
    "Trustworthy":  "lettermark",
    "Creative":     "emblem",
    "Eco-Conscious":"wordmark",
    "Youthful":     "geometric",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_logo_concepts(
    company:  str,
    tone:     str,
    palette:  list[dict],
) -> list[LogoConcept]:
    """
    Generate 4 SVG logo concepts for the brand.
    Uses rule-based SVG composition — no CNN or logo dataset.
    """
    primary   = palette[0]["hex"] if len(palette) > 0 else "#0A0A08"
    secondary = palette[3]["hex"] if len(palette) > 3 else "#F5F2EB"
    accent    = palette[2]["hex"] if len(palette) > 2 else "#C8A94A"

    preferred = _TONE_STYLES.get(tone, "lettermark")

    concepts = [
        LogoConcept(
            name="Lettermark",
            description=f"Bold initial lettermark on solid {tone.lower()} background. Clean, scalable, instantly recognisable.",
            svg=_svg_lettermark(company, primary, secondary),
            style="lettermark",
            tone_match=0.95 if preferred=="lettermark" else 0.75,
        ),
        LogoConcept(
            name="Geometric Mark",
            description="Circular badge with concentric detail — works at any size from favicon to billboard.",
            svg=_svg_geometric_circle(company, primary, accent),
            style="geometric",
            tone_match=0.95 if preferred=="geometric" else 0.75,
        ),
        LogoConcept(
            name="Wordmark",
            description="Pure typographic mark — wide tracking, hairline rule. Suits brands where the name IS the identity.",
            svg=_svg_wordmark(company, primary),
            style="wordmark",
            tone_match=0.85 if preferred=="wordmark" else 0.65,
        ),
        LogoConcept(
            name="Emblem / Crest",
            description="Hexagonal frame with letterform — heritage, craft, and authority.",
            svg=_svg_emblem(company, primary, accent),
            style="emblem",
            tone_match=0.95 if preferred=="emblem" else 0.70,
        ),
        LogoConcept(
            name="Minimal Bracket Mark",
            description="Editorial bracket device — signals precision, technology, and considered design.",
            svg=_svg_minimal_mark(company, primary, accent),
            style="minimalist",
            tone_match=0.95 if preferred=="minimalist" else 0.70,
        ),
    ]
    # Sort: preferred style first
    concepts.sort(key=lambda c: -c.tone_match)
    return concepts
