"""
src/branding_logic.py
Content / Social Campaign Studio — caption, hashtag, CTA generation.

Sources:
  - Platform-specific templates [rule-based]
  - Gemini API enhancement [optional API]
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ContentPack:
    platform:   str
    caption:    str
    hashtags:   list[str]
    cta:        str
    strategy:   str
    tone_note:  str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.caption) + len(" ".join(self.hashtags))


# ─────────────────────────────────────────────────────────────────────────────
# Platform character limits
# ─────────────────────────────────────────────────────────────────────────────
PLATFORM_LIMITS = {
    "Instagram": 2200, "Facebook": 63206, "Twitter / X": 280,
    "LinkedIn": 3000,  "YouTube": 5000,   "TikTok": 2200,
}

# ─────────────────────────────────────────────────────────────────────────────
# Template-based generation [rule-based]
# ─────────────────────────────────────────────────────────────────────────────

_CAPTION_TEMPLATES: dict[str, dict[str, str]] = {
    "Instagram": {
        "Awareness":   "Meet {company} — {tagline} ✨\n\n{description_short}\n\nTag someone who needs this. 👇",
        "Conversion":  "🚀 {company}: {tagline}\n\nHere's why we're different:\n▸ {benefit1}\n▸ {benefit2}\n▸ {benefit3}\n\nLink in bio → try it free today.",
        "Retention":   "You chose {company}. Here's what's next. 🙌\n\n{description_short}\n\nShare your story with #{hashtag1}",
    },
    "Facebook": {
        "Awareness":   "{company}: {tagline}\n\nWe built {company} because {description_short}.\n\nLearn more at the link below.",
        "Conversion":  "🎯 Limited time — discover why {company} is the {industry} choice for {audience}.\n\n{description_short}\n\nClick below to get started.",
        "Retention":   "To our community — thank you. {company} exists because of you.\n\n{description_short}\n\nTell us your experience in the comments.",
    },
    "Twitter / X": {
        "Awareness":   "{company}: {tagline}. Here's why that matters in {industry}. 🧵",
        "Conversion":  "Why choose {company}?\n✓ {benefit1}\n✓ {benefit2}\n\n{tagline}",
        "Retention":   "Still here with {company}? You should be. {tagline} 🎯",
    },
    "LinkedIn": {
        "Awareness":   "We built {company} to solve a real problem in {industry}.\n\n{description_short}\n\nHere's what we've learned — and where we're heading.",
        "Conversion":  "{company} helps {audience} achieve measurable results.\n\n{description_short}\n\nInterested? Let's connect.",
        "Retention":   "At {company}, we believe long-term partnerships drive the best outcomes.\n\n{description_short}\n\nWhat keeps you coming back? Share below.",
    },
}

_HASHTAG_SETS: dict[str, list[str]] = {
    "Technology":      ["#Tech","#SaaS","#Innovation","#AI","#DigitalTransformation"],
    "Finance":         ["#FinTech","#Finance","#Investment","#Money","#WealthManagement"],
    "Healthcare":      ["#HealthTech","#Healthcare","#Wellness","#MedTech","#DigitalHealth"],
    "Retail":          ["#Retail","#Shopping","#Ecommerce","#D2C","#BrandNew"],
    "Education":       ["#EdTech","#Learning","#Education","#Skills","#Growth"],
    "Food & Beverage": ["#FoodTech","#FoodLover","#Foodie","#HealthyEating","#Gourmet"],
    "Fashion":         ["#Fashion","#Style","#OOTD","#Luxury","#Design"],
    "Travel":          ["#Travel","#Adventure","#Wanderlust","#Explore","#TravelTips"],
    "Sustainability":  ["#Sustainability","#EcoFriendly","#GreenTech","#ClimateAction","#NetZero"],
    "Consulting":      ["#Consulting","#Strategy","#BusinessGrowth","#Leadership","#B2B"],
}

_CTA_BY_TYPE: dict[str, str] = {
    "Awareness":   "Learn more about {company}.",
    "Conversion":  "Start free — no credit card needed.",
    "Retention":   "Log in and see what's new.",
    "Launch":      "Be first. Join the waitlist.",
    "Seasonal":    "Limited time only — shop now.",
    "Remarketing": "You left something behind — come back.",
}

_STRATEGY_NOTES: dict[str, str] = {
    "Instagram": "Post between 9–11am or 6–9pm local time. Use 5–8 hashtags. First line must hook in 3 words.",
    "Facebook":  "Longer-form content performs well. Include a direct question to drive comments. Boost high-performing organic posts.",
    "Twitter / X":"Lead with the insight, not the product. Threads outperform single tweets 3×. Engage replies within 1 hour.",
    "LinkedIn":  "Personal stories outperform company posts. Use a 3-line hook, then expand. Tag relevant connections (not cold).",
    "YouTube":   "First 30 seconds determine retention. Include keyword in title and first 200 chars of description.",
    "TikTok":    "Native, unpolished creative outperforms produced video. Sound-on is critical — use trending audio.",
    "Email":     "Subject line is everything — A/B test 2 variants. Plain-text often outperforms HTML. Send Tuesday–Thursday 10am.",
    "Google Ads":"Match landing page to ad copy exactly. Broad match + smart bidding for discovery. Exact match for high-intent terms.",
}


def _truncate_description(text: str, words: int = 20) -> str:
    w = text.split()
    return " ".join(w[:words]) + ("…" if len(w) > words else "")


def _generate_benefits(company: str, industry: str, tone: str) -> tuple[str,str,str]:
    benefit_map = {
        "Technology":   ("AI-powered workflows","Enterprise-grade security","Instant setup — zero code"),
        "Finance":      ("Transparent fee structure","Real-time portfolio insights","Regulated and secure"),
        "Healthcare":   ("Clinically validated","HIPAA-compliant","Instant access to care"),
        "Retail":       ("Free returns","Same-day dispatch","Personalised recommendations"),
        "Education":    ("Learn at your own pace","Accredited certification","1:1 mentor support"),
        "Sustainability":("Verified carbon offset","Ethical supply chain","B-Corp certified"),
    }
    return benefit_map.get(industry, ("Quality you can trust","Results in days","Expert support"))


def generate_content(
    company:       str,
    industry:      str,
    tone:          str,
    tagline:       str,
    description:   str,
    audience:      str,
    campaign_type: str,
    platforms:     list[str],
) -> list[ContentPack]:
    """Generate platform-specific content packs. Returns one pack per platform."""
    packs: list[ContentPack] = []
    b1, b2, b3 = _generate_benefits(company, industry, tone)
    slug        = re.sub(r"[^a-zA-Z0-9]","", company)
    desc_short  = _truncate_description(description)

    industry_tags = _HASHTAG_SETS.get(industry, ["#Brand","#Business","#Growth"])
    tone_tags     = {"Bold":["#BoldBrand"],"Luxury":["#LuxuryBrand"],
                     "Playful":["#Fun","#Creative"]}.get(tone, ["#Innovation"])
    base_tags     = industry_tags + tone_tags + [f"#{slug}"]

    def _fill(template: str) -> str:
        return (template
            .replace("{company}",    company)
            .replace("{tagline}",    tagline or f"Powering {industry} forward")
            .replace("{description_short}", desc_short)
            .replace("{industry}",   industry.lower())
            .replace("{audience}",   audience or "professionals")
            .replace("{benefit1}",   b1)
            .replace("{benefit2}",   b2)
            .replace("{benefit3}",   b3)
            .replace("{hashtag1}",   slug)
        )

    for plat in platforms:
        plat_templates = _CAPTION_TEMPLATES.get(plat, _CAPTION_TEMPLATES["Facebook"])
        ctype          = campaign_type if campaign_type in plat_templates else "Awareness"
        caption        = _fill(plat_templates.get(ctype, list(plat_templates.values())[0]))
        cta            = _fill(_CTA_BY_TYPE.get(campaign_type, "Learn more."))
        strategy       = _STRATEGY_NOTES.get(plat, "Tailor content to platform norms.")
        limit          = PLATFORM_LIMITS.get(plat, 2200)

        if len(caption) > limit:
            caption = caption[:limit-3] + "…"

        packs.append(ContentPack(
            platform=plat, caption=caption, hashtags=base_tags[:8],
            cta=cta, strategy=strategy,
            tone_note=f"Written in a {tone.lower()} voice for {audience or 'your audience'}.",
        ))
    return packs


# ─────────────────────────────────────────────────────────────────────────────
# Gemini-enhanced content (optional)
# ─────────────────────────────────────────────────────────────────────────────

def enhance_content_with_gemini(pack: ContentPack, tone: str) -> ContentPack:
    """
    Optionally refine a ContentPack caption using Gemini API.
    Falls back to original pack on any failure.
    """
    from src.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS
    if not GEMINI_API_KEY:
        return pack
    try:
        import google.generativeai as genai
        import json
        genai.configure(api_key=GEMINI_API_KEY)
        prompt = f"""You are an expert social media copywriter.

Platform: {pack.platform}
Brand tone: {tone}
Draft caption:
\"\"\"
{pack.caption}
\"\"\"

Task: Rewrite this caption to be more engaging and on-brand.
Keep it within platform limits. Use tone: {tone}.
Return JSON: {{"caption": "...", "cta": "..."}}"""

        for model_name in GEMINI_FALLBACK_MODELS[:3]:
            try:
                model  = genai.GenerativeModel(model_name)
                result = model.generate_content(prompt)
                raw    = result.text.strip().lstrip("```json").rstrip("```").strip()
                data   = json.loads(raw)
                pack.caption    = data.get("caption", pack.caption)
                pack.cta        = data.get("cta",     pack.cta)
                pack.tone_note += " (Gemini-refined)"
                return pack
            except Exception:
                continue
    except Exception as e:
        log.warning("Gemini content enhancement failed: %s", e)
    return pack
