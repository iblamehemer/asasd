"""
src/config.py
NovaTech AI — Centralised configuration, paths, and look-up mappings.
All downstream modules import from here; nothing is hard-coded elsewhere.
"""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATASETS    = ROOT / "datasets"
RAW         = DATASETS / "raw"
PROCESSED   = DATASETS / "processed"
MODELS_DIR  = ROOT / "models"
ASSETS      = ROOT / "assets"
TEMP        = ASSETS / "temp"
EXPORTS     = ASSETS / "sample_exports"
LOGOS_DIR   = ASSETS / "sample_logos"
FEEDBACK_DB = ROOT / "feedback_log.csv"

# ── Dataset file names ────────────────────────────────────────────────────────
SLOGANS_RAW     = RAW / "sloganlist.csv"
STARTUPS_RAW    = RAW / "startups.csv"
MARKETING_RAW   = RAW / "marketing_campaign_dataset.csv"

SLOGANS_CLEAN   = PROCESSED / "cleaned_slogans.csv"
STARTUPS_CLEAN  = PROCESSED / "cleaned_startups.csv"
MARKETING_CLEAN = PROCESSED / "cleaned_marketing.csv"
PERSONAS_FILE   = PROCESSED / "startup_personas.csv"
CAMPAIGN_FEATS  = PROCESSED / "campaign_features.csv"

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"
GEMINI_FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-pro",
]

# ── Supported options ─────────────────────────────────────────────────────────
INDUSTRIES = [
    "Technology", "Finance", "Healthcare", "Retail", "Education",
    "Food & Beverage", "Real Estate", "Fashion", "Travel", "Sustainability",
    "Media & Entertainment", "Logistics", "Consulting", "Sports & Fitness",
]

TONES = [
    "Minimalist", "Bold", "Luxury", "Playful", "Professional",
    "Innovative", "Trustworthy", "Creative", "Eco-Conscious", "Youthful",
]

PLATFORMS = [
    "Instagram", "Facebook", "Twitter / X", "LinkedIn",
    "YouTube", "TikTok", "Email", "Google Ads",
]

REGIONS = [
    "India", "USA", "Europe", "UK", "Southeast Asia",
    "Middle East", "Latin America", "Africa", "Global",
]

LANGUAGES = [
    "English", "Hindi", "Spanish", "French", "German",
    "Gujarati", "Portuguese", "Arabic", "Japanese", "Mandarin",
]

CAMPAIGN_TYPES = [
    "Awareness", "Conversion", "Retention", "Launch", "Seasonal", "Remarketing",
]

AUDIENCE_SEGMENTS = [
    "Gen Z (18–24)", "Millennials (25–34)", "Gen X (35–50)",
    "Boomers (51+)", "B2B Decision Makers", "SMB Owners", "Enterprise",
]

# ── Font psychology mappings ───────────────────────────────────────────────────
# (industry/tone) → (heading, body, accent, rationale)
FONT_MAP: dict[str, dict] = {
    "Technology-Minimalist": {
        "heading": "Space Grotesk", "body": "Inter", "accent": "DM Mono",
        "style": "geometric sans-serif",
        "rationale": "Clean geometry signals precision engineering; monospace accent for code/data credibility."
    },
    "Technology-Innovative": {
        "heading": "Syne", "body": "DM Sans", "accent": "Fira Code",
        "style": "experimental sans",
        "rationale": "Syne's irregular geometry suggests forward-thinking; DM Sans keeps body copy readable."
    },
    "Technology-Bold": {
        "heading": "Clash Display", "body": "General Sans", "accent": "JetBrains Mono",
        "style": "display sans",
        "rationale": "High-impact display weight commands attention; technical mono accent adds authenticity."
    },
    "Finance-Professional": {
        "heading": "Playfair Display", "body": "Source Serif 4", "accent": "DM Mono",
        "style": "authoritative serif",
        "rationale": "Serifs convey heritage and trust; weight contrast between headline and body creates gravitas."
    },
    "Finance-Trustworthy": {
        "heading": "EB Garamond", "body": "Lato", "accent": "Roboto Mono",
        "style": "classic serif + humanist sans",
        "rationale": "Garamond's long history signals permanence; Lato humanises the brand for accessibility."
    },
    "Healthcare-Professional": {
        "heading": "Nunito", "body": "Open Sans", "accent": "Source Code Pro",
        "style": "rounded humanist",
        "rationale": "Rounded letterforms feel approachable and caring; high x-height ensures legibility at small sizes."
    },
    "Healthcare-Trustworthy": {
        "heading": "Raleway", "body": "Merriweather", "accent": "PT Mono",
        "style": "elegant sans + workhorse serif",
        "rationale": "Thin Raleway weights communicate clinical precision; Merriweather body aids long-form reading."
    },
    "Retail-Playful": {
        "heading": "Baloo 2", "body": "Nunito Sans", "accent": "Courier Prime",
        "style": "friendly display",
        "rationale": "Bouncy curves invite browsing; high energy without losing commercial clarity."
    },
    "Retail-Bold": {
        "heading": "Bebas Neue", "body": "Barlow", "accent": "Space Mono",
        "style": "condensed display",
        "rationale": "Tight condensed caps maximise impact on packaging and banners; Barlow body keeps price-points readable."
    },
    "Fashion-Luxury": {
        "heading": "Cormorant Garamond", "body": "Jost", "accent": "DM Mono",
        "style": "high fashion serif",
        "rationale": "Cormorant's hairline serifs channel editorial luxury; extreme tracking on caps creates runway feel."
    },
    "Fashion-Creative": {
        "heading": "Bodoni Moda", "body": "Libre Franklin", "accent": "Courier New",
        "style": "editorial contrast serif",
        "rationale": "Extreme thick-thin contrast is fashion's visual signature; Courier accent adds vintage editorial tension."
    },
    "Food & Beverage-Playful": {
        "heading": "Pacifico", "body": "Quicksand", "accent": "Caveat",
        "style": "hand-crafted display",
        "rationale": "Script headlines feel hand-made and artisanal; Caveat accent mimics chalk-board menu aesthetics."
    },
    "Food & Beverage-Bold": {
        "heading": "Anton", "body": "Roboto", "accent": "Space Mono",
        "style": "bold impact",
        "rationale": "Anton's heavy weight works at large sizes on packaging; Roboto body is near-universal in legibility."
    },
    "Education-Professional": {
        "heading": "Merriweather", "body": "Source Sans 3", "accent": "IBM Plex Mono",
        "style": "academic serif",
        "rationale": "Merriweather was designed for screen reading; IBM Plex Mono adds institutional credibility."
    },
    "Education-Youthful": {
        "heading": "Poppins", "body": "Nunito", "accent": "Fira Mono",
        "style": "geometric friendly",
        "rationale": "Poppins' perfect geometric circles feel fresh and approachable to student audiences."
    },
    "Real Estate-Luxury": {
        "heading": "Libre Baskerville", "body": "Lora", "accent": "Josefin Sans",
        "style": "estate serif",
        "rationale": "Classic serif authority combined with Josefin's spaced capitals conveys aspirational living."
    },
    "Travel-Creative": {
        "heading": "Abril Fatface", "body": "Lato", "accent": "Courier Prime",
        "style": "poster display",
        "rationale": "Abril's heavy vintage weight feels like classic travel posters; Lato body is globally legible."
    },
    "Sustainability-Eco-Conscious": {
        "heading": "Josefin Sans", "body": "Karla", "accent": "Caveat",
        "style": "clean organic",
        "rationale": "Thin, spaced sans-serifs convey transparency and minimalism; Caveat accent adds hand-crafted warmth."
    },
    "Consulting-Professional": {
        "heading": "Crimson Text", "body": "Barlow", "accent": "JetBrains Mono",
        "style": "strategic serif",
        "rationale": "Crimson communicates intellectual rigour; Barlow's wide range of weights offers strategic flexibility."
    },
    "_default": {
        "heading": "Playfair Display", "body": "DM Sans", "accent": "DM Mono",
        "style": "editorial blend",
        "rationale": "A refined pairing that balances elegance with clarity, suitable for most business contexts."
    },
}


# ── Colour psychology palette map ─────────────────────────────────────────────
# Returns list of {role, hex, name, meaning}
PALETTE_MAP: dict[str, list] = {
    "Technology-Minimalist": [
        {"role":"primary",    "hex":"#0A0A0F","name":"Void Black",    "meaning":"Depth and infinite possibility"},
        {"role":"secondary",  "hex":"#1E40AF","name":"System Blue",   "meaning":"Digital intelligence and reliability"},
        {"role":"accent",     "hex":"#6EE7B7","name":"Terminal Green","meaning":"Output, execution, success"},
        {"role":"neutral",    "hex":"#F8FAFC","name":"Cloud White",   "meaning":"Clarity and openness"},
        {"role":"background", "hex":"#0F172A","name":"Deep Space",    "meaning":"Focused, immersive environment"},
    ],
    "Technology-Innovative": [
        {"role":"primary",    "hex":"#7C3AED","name":"Electric Violet","meaning":"Imagination and future-forward thinking"},
        {"role":"secondary",  "hex":"#0EA5E9","name":"Cyber Cyan",    "meaning":"Connectivity and speed"},
        {"role":"accent",     "hex":"#F59E0B","name":"Amber Signal",  "meaning":"Energy and breakthrough moments"},
        {"role":"neutral",    "hex":"#F1F5F9","name":"Silk Grey",     "meaning":"Balance and neutrality"},
        {"role":"background", "hex":"#0C0A1A","name":"Midnight",      "meaning":"Premium, focused workspace"},
    ],
    "Finance-Professional": [
        {"role":"primary",    "hex":"#1E293B","name":"Slate Navy",    "meaning":"Stability, authority, and depth"},
        {"role":"secondary",  "hex":"#C8A94A","name":"Gold Standard", "meaning":"Prosperity and measured success"},
        {"role":"accent",     "hex":"#0F766E","name":"Treasury Teal", "meaning":"Growth and precision"},
        {"role":"neutral",    "hex":"#F8F6F1","name":"Parchment",     "meaning":"Heritage and permanence"},
        {"role":"background", "hex":"#FAFAFA","name":"Paper White",   "meaning":"Transparency and trust"},
    ],
    "Finance-Trustworthy": [
        {"role":"primary",    "hex":"#1D4ED8","name":"Sovereign Blue","meaning":"Trust and institutional confidence"},
        {"role":"secondary",  "hex":"#374151","name":"Iron Grey",     "meaning":"Solidity and longevity"},
        {"role":"accent",     "hex":"#10B981","name":"Ledger Green",  "meaning":"Positive returns and growth"},
        {"role":"neutral",    "hex":"#F9FAFB","name":"Linen White",   "meaning":"Openness and honesty"},
        {"role":"background", "hex":"#EFF6FF","name":"Sky Mist",      "meaning":"Calm, considered decision-making"},
    ],
    "Healthcare-Professional": [
        {"role":"primary",    "hex":"#0284C7","name":"Clinical Blue", "meaning":"Medical precision and calm"},
        {"role":"secondary",  "hex":"#0F766E","name":"Healing Teal",  "meaning":"Health, renewal, and care"},
        {"role":"accent",     "hex":"#F59E0B","name":"Vitality Amber","meaning":"Warmth and optimism"},
        {"role":"neutral",    "hex":"#F0FDF4","name":"Clean Mint",    "meaning":"Hygiene and freshness"},
        {"role":"background", "hex":"#FFFFFF","name":"Pure White",    "meaning":"Sterility and professionalism"},
    ],
    "Fashion-Luxury": [
        {"role":"primary",    "hex":"#0A0A08","name":"Atelier Black", "meaning":"Exclusivity and ultimate sophistication"},
        {"role":"secondary",  "hex":"#C8A94A","name":"18ct Gold",     "meaning":"Prestige and refined taste"},
        {"role":"accent",     "hex":"#78350F","name":"Cognac",        "meaning":"Heritage and artisan craft"},
        {"role":"neutral",    "hex":"#F5F0E8","name":"Ivory",         "meaning":"Timeless elegance"},
        {"role":"background", "hex":"#FAFAF9","name":"Cream",         "meaning":"Understated opulence"},
    ],
    "Food & Beverage-Playful": [
        {"role":"primary",    "hex":"#DC2626","name":"Tomato Red",    "meaning":"Appetite stimulation and energy"},
        {"role":"secondary",  "hex":"#D97706","name":"Harvest Orange","meaning":"Warmth, freshness, and appetite"},
        {"role":"accent",     "hex":"#65A30D","name":"Fresh Herb",    "meaning":"Natural ingredients and vitality"},
        {"role":"neutral",    "hex":"#FEF9EE","name":"Cream Butter",  "meaning":"Warmth and comfort"},
        {"role":"background", "hex":"#FFFBF0","name":"Warm White",    "meaning":"Clean and inviting space"},
    ],
    "Sustainability-Eco-Conscious": [
        {"role":"primary",    "hex":"#166534","name":"Forest Deep",   "meaning":"Environmental commitment and growth"},
        {"role":"secondary",  "hex":"#65A30D","name":"Leaf Green",    "meaning":"Nature, renewal, and life"},
        {"role":"accent",     "hex":"#92400E","name":"Earth Brown",   "meaning":"Grounding, authenticity, soil"},
        {"role":"neutral",    "hex":"#F0FDF4","name":"Dew Mint",      "meaning":"Freshness and clean living"},
        {"role":"background", "hex":"#FAFFF5","name":"Petal White",   "meaning":"Purity and natural simplicity"},
    ],
    "Education-Professional": [
        {"role":"primary",    "hex":"#1E3A5F","name":"Academic Navy", "meaning":"Knowledge, tradition, authority"},
        {"role":"secondary",  "hex":"#B45309","name":"Amber Ink",     "meaning":"Curiosity and intellectual warmth"},
        {"role":"accent",     "hex":"#0891B2","name":"Insight Teal",  "meaning":"Clarity and discovery"},
        {"role":"neutral",    "hex":"#F8F4F0","name":"Manuscript",    "meaning":"Scholarship and craft"},
        {"role":"background", "hex":"#FDFCFB","name":"Book White",    "meaning":"A blank page, full of potential"},
    ],
    "Retail-Bold": [
        {"role":"primary",    "hex":"#111827","name":"Storefront Black","meaning":"Bold authority and presence"},
        {"role":"secondary",  "hex":"#EF4444","name":"Sale Red",      "meaning":"Urgency, deals, and excitement"},
        {"role":"accent",     "hex":"#F59E0B","name":"Price Tag Gold","meaning":"Value and reward"},
        {"role":"neutral",    "hex":"#F9FAFB","name":"Display White", "meaning":"Clean product presentation"},
        {"role":"background", "hex":"#FFFFFF","name":"Canvas White",  "meaning":"Product focus"},
    ],
    "Travel-Creative": [
        {"role":"primary",    "hex":"#0C4A6E","name":"Deep Ocean",    "meaning":"Endless horizons and adventure"},
        {"role":"secondary",  "hex":"#D97706","name":"Sunset Orange", "meaning":"New destinations and discovery"},
        {"role":"accent",     "hex":"#065F46","name":"Jungle Green",  "meaning":"Exploration and nature"},
        {"role":"neutral",    "hex":"#FFF7ED","name":"Sand",          "meaning":"Warmth, beaches, and ease"},
        {"role":"background", "hex":"#F0F9FF","name":"Sky Blue",      "meaning":"Open sky and limitless freedom"},
    ],
    "_default": [
        {"role":"primary",    "hex":"#0A0A08","name":"Deep Black",    "meaning":"Confidence and authority"},
        {"role":"secondary",  "hex":"#C8A94A","name":"Warm Gold",     "meaning":"Quality and distinction"},
        {"role":"accent",     "hex":"#0F766E","name":"Teal",          "meaning":"Balance and clarity"},
        {"role":"neutral",    "hex":"#F5F2EB","name":"Off White",     "meaning":"Openness and approachability"},
        {"role":"background", "hex":"#FAFAFA","name":"Pure White",    "meaning":"Transparency and cleanliness"},
    ],
}


# ── Slogan tone templates ─────────────────────────────────────────────────────
SLOGAN_TEMPLATES: dict[str, list] = {
    "Minimalist":     ["{company}. Simply better.", "Less noise. More {company}.", "{company}—nothing more needed."],
    "Bold":           ["Dominate with {company}.", "{company}: Built to win.", "No limits. Just {company}."],
    "Luxury":         ["The art of {company}.", "{company}. Crafted for the few.", "Elevate to {company}."],
    "Playful":        ["{company}: fun starts here.", "Why so serious? Try {company}.", "{company}—because life's short!"],
    "Professional":   ["{company}. Precision in every detail.", "Trust begins with {company}.", "Excellence, delivered by {company}."],
    "Innovative":     ["Tomorrow belongs to {company}.", "{company}: rethinking {industry}.", "The future is already {company}."],
    "Trustworthy":    ["{company}: a promise kept.", "Built on integrity. Built by {company}.", "{company}—you can count on us."],
    "Creative":       ["{company}: where ideas breathe.", "Colour outside the lines with {company}.", "Imagine more. Choose {company}."],
    "Eco-Conscious":  ["{company}: good for people, great for the planet.", "Green thinking, powered by {company}.", "Sustain the future with {company}."],
    "Youthful":       ["{company}: your vibe, your brand.", "Born different. Born {company}.", "{company}—we get it."],
}

# ── Animation presets ─────────────────────────────────────────────────────────
ANIMATION_STYLES = ["Fade In", "Slide In Left", "Slide In Up", "Typewriter", "Zoom In", "Pulse"]

# ── Supported translation languages ──────────────────────────────────────────
TRANSLATION_TARGETS = {
    "Hindi":      "hi",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Gujarati":   "gu",
    "Portuguese": "pt",
    "Arabic":     "ar",
    "Japanese":   "ja",
}

# ── Aesthetic harmony scoring weights ────────────────────────────────────────
HARMONY_WEIGHTS = {
    "palette_tone_match":  0.30,
    "font_tone_match":     0.25,
    "slogan_tone_match":   0.25,
    "logo_style_match":    0.20,
}

# ── Model file paths ──────────────────────────────────────────────────────────
MODEL_CAMPAIGN  = MODELS_DIR / "campaign_model.pkl"
MODEL_CTR       = MODELS_DIR / "ctr_model.pkl"
MODEL_ROI       = MODELS_DIR / "roi_model.pkl"
MODEL_ENGAGE    = MODELS_DIR / "engagement_model.pkl"
ENCODERS_FILE   = MODELS_DIR / "encoders.pkl"
SCALER_FILE     = MODELS_DIR / "scaler.pkl"
