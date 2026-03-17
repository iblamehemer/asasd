"""
src/palette_engine.py
Colour palette engine.

Implementation:
  - Industry + tone mapping [rule-based]
  - Logo pixel extraction with KMeans [optional, if PIL/sklearn available]
  - Returns PaletteResult with hex codes, roles, and harmony score

No real logo dataset was used to train this engine.
The KMeans extraction is run on user-uploaded images at runtime only.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PaletteResult:
    swatches:       list[dict]  # [{role, hex, name, meaning}]
    harmony_score:  float       # 0–1
    source:         str         # "mapping" | "kmeans" | "hybrid"
    recommendation: str


def _hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


def _contrast_ratio(h1: str, h2: str) -> float:
    """WCAG relative luminance contrast ratio."""
    def lum(h):
        rgb = [c/255 for c in _hex_to_rgb(h)]
        rgb = [c/12.92 if c<=0.03928 else ((c+0.055)/1.055)**2.4 for c in rgb]
        return 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
    l1, l2 = sorted([lum(h1), lum(h2)], reverse=True)
    return (l1+0.05) / (l2+0.05)


def _harmony_score(swatches: list[dict]) -> float:
    """
    Simple harmony heuristic:
    - Rewards high contrast between primary and background
    - Rewards moderate saturation variety
    Returns 0–1.
    """
    try:
        hexes = [s["hex"] for s in swatches if s.get("hex")]
        if len(hexes) < 2:
            return 0.5
        primary = hexes[0]
        bg      = hexes[-1]  # last = background
        cr      = _contrast_ratio(primary, bg)
        # WCAG AA = 4.5, AAA = 7
        contrast_score = min(cr / 7.0, 1.0)
        # RGB spread (variety)
        rgbs    = [_hex_to_rgb(h) for h in hexes]
        spreads = [max(c)-min(c) for c in zip(*rgbs)]
        variety_score = min(sum(spreads) / (255*3), 1.0)
        return round(0.6*contrast_score + 0.4*variety_score, 3)
    except Exception:
        return 0.70


def recommend_palette(industry: str, tone: str) -> PaletteResult:
    """
    Return a 5-colour branded palette for (industry, tone).
    Source: rule-based mapping.
    """
    from src.config import PALETTE_MAP

    key     = f"{industry}-{tone}"
    swatches= (PALETTE_MAP.get(key)
               or PALETTE_MAP.get(f"{industry}-Professional")
               or PALETTE_MAP.get(f"Technology-{tone}")
               or PALETTE_MAP["_default"])

    score = _harmony_score(swatches)

    if score >= 0.75:
        rec = "Excellent harmony — this palette communicates consistently across all brand touchpoints."
    elif score >= 0.55:
        rec = "Good harmony — consider increasing contrast between primary and background for better accessibility."
    else:
        rec = "Moderate harmony — the palette may benefit from a stronger accent colour to create visual hierarchy."

    return PaletteResult(swatches=swatches, harmony_score=score,
                          source="mapping", recommendation=rec)


def extract_palette_from_image(image_bytes: bytes, n_colors: int = 5) -> PaletteResult:
    """
    K-Means colour extraction from an uploaded logo or image.
    Source: KMeans on pixel data [ML — run at runtime, no pre-trained model].
    Falls back gracefully if PIL/sklearn not available.
    """
    try:
        from PIL import Image
        from sklearn.cluster import KMeans
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((150,150))  # resize for speed
        pixels = np.array(img).reshape(-1,3).astype(float)

        # Remove near-white and near-black pixels (likely background)
        mask = ~(np.all(pixels > 240, axis=1) | np.all(pixels < 15, axis=1))
        pixels = pixels[mask]
        if len(pixels) < n_colors * 5:
            pixels = np.array(img).reshape(-1,3).astype(float)

        km = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
        km.fit(pixels)
        centers = km.cluster_centers_.astype(int)

        roles  = ["primary","secondary","accent","neutral","background"]
        names  = ["Dominant","Secondary","Accent","Supporting","Base"]
        swatches = []
        for i, (rgb, role, name) in enumerate(zip(centers, roles, names)):
            h = "#{:02x}{:02x}{:02x}".format(*rgb)
            swatches.append({"role":role,"hex":h,"name":name,"meaning":"Extracted from brand image."})

        score = _harmony_score(swatches)
        return PaletteResult(swatches=swatches, harmony_score=score,
                              source="kmeans",
                              recommendation="Palette extracted from your uploaded image via K-Means clustering.")
    except ImportError as e:
        log.warning("PIL or sklearn not available for image extraction: %s", e)
    except Exception as e:
        log.warning("Image palette extraction failed: %s", e)
    return recommend_palette("Technology", "Minimalist")  # safe fallback
