"""
src/multilingual_engine.py
Multilingual translation engine.

Priority:
  1. Gemini API          — best quality, tone-aware  [optional API]
  2. deep_translator     — free offline library       [optional library]
  3. Structured fallback — curated sample phrases     [rule-based]

All sources clearly labelled in output.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class Translation:
    language:     str
    text:         str
    source:       str     # "gemini" | "deep_translator" | "fallback"
    confidence:   str     # "high" | "medium" | "low"
    note:         str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini translation
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_translate(text: str, target_lang: str, tone: str) -> str | None:
    from src.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai, json
        genai.configure(api_key=GEMINI_API_KEY)
        prompt = (f"Translate the following brand slogan or caption into {target_lang}.\n"
                  f"Preserve the '{tone}' brand tone. Return only the translation, no explanation.\n\n"
                  f"Text: {text}")
        for mn in GEMINI_FALLBACK_MODELS[:3]:
            try:
                result = genai.GenerativeModel(mn).generate_content(prompt)
                t = result.text.strip().strip('"\'')
                if t:
                    return t
            except Exception:
                continue
    except Exception as e:
        log.warning("Gemini translation failed (%s): %s", target_lang, e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# deep-translator fallback
# ─────────────────────────────────────────────────────────────────────────────

_LANG_CODES: dict[str,str] = {
    "Hindi":"hi","Spanish":"es","French":"fr","German":"de",
    "Gujarati":"gu","Portuguese":"pt","Arabic":"ar","Japanese":"ja",
}

def _deep_translate(text: str, target_lang: str) -> str | None:
    code = _LANG_CODES.get(target_lang)
    if not code:
        return None
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target=code).translate(text)
    except Exception as e:
        log.warning("deep_translator failed (%s): %s", target_lang, e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Curated sample translations (fallback) — clearly synthetic
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLES: dict[str, dict[str,str]] = {
    "Hindi":      {
        "hello":  "नमस्ते",
        "default":"हम आपके व्यवसाय को बेहतर बनाने के लिए यहाँ हैं।",
    },
    "Spanish":    {
        "hello":  "Hola",
        "default":"Estamos aquí para hacer crecer tu marca.",
    },
    "French":     {
        "hello":  "Bonjour",
        "default":"Nous sommes là pour développer votre marque.",
    },
    "German":     {
        "hello":  "Hallo",
        "default":"Wir sind hier, um Ihre Marke zu stärken.",
    },
    "Gujarati":   {
        "hello":  "નમસ્તે",
        "default":"અમે તમારા બ્રાન્ડને વધુ સારું બનાવવા અહીં છીએ.",
    },
    "Portuguese": {
        "hello":  "Olá",
        "default":"Estamos aqui para crescer sua marca.",
    },
    "Arabic":     {
        "hello":  "مرحبا",
        "default":"نحن هنا لتعزيز علامتك التجارية.",
    },
    "Japanese":   {
        "hello":  "こんにちは",
        "default":"私たちはあなたのブランドを成長させるためにここにいます。",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def translate_text(text: str, languages: list[str], tone: str = "Professional") -> list[Translation]:
    """
    Translate `text` into each language in `languages`.
    Uses Gemini → deep_translator → curated fallback in that order.
    """
    results: list[Translation] = []
    for lang in languages:
        # 1. Gemini
        t = _gemini_translate(text, lang, tone)
        if t:
            results.append(Translation(lang, t, "gemini", "high"))
            continue
        # 2. deep_translator
        t = _deep_translate(text, lang)
        if t:
            results.append(Translation(lang, t, "deep_translator", "medium"))
            continue
        # 3. Fallback
        fallback = _SAMPLES.get(lang, {}).get("default", f"[{lang} translation unavailable]")
        results.append(Translation(
            lang, fallback, "fallback", "low",
            note="Sample translation — install deep-translator or add Gemini API key for accurate output."
        ))
    return results


def translate_batch(
    texts:     list[str],
    languages: list[str],
    tone:      str = "Professional",
) -> dict[str, list[Translation]]:
    """
    Translate multiple texts. Returns {text: [Translation, …]}.
    """
    return {t: translate_text(t, languages, tone) for t in texts}
