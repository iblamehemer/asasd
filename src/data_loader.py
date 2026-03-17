"""
src/data_loader.py
Safe, cached loaders for every dataset.
Falls back to embedded synthetic data if files are absent so the app
always runs even on a fresh Streamlit Cloud deploy without uploaded data.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    SLOGANS_RAW, STARTUPS_RAW, MARKETING_RAW,
    SLOGANS_CLEAN, STARTUPS_CLEAN, MARKETING_CLEAN,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic fallbacks  (clearly labelled as generated, not real data)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_slogans() -> pd.DataFrame:
    """Returns a small synthetic slogan corpus when the CSV is absent."""
    data = [
        ("Nike", "Just Do It"),
        ("Apple", "Think Different"),
        ("Google", "Don't Be Evil"),
        ("Tesla", "Accelerating the World's Transition to Sustainable Energy"),
        ("Amazon", "Work Hard. Have Fun. Make History."),
        ("Microsoft", "Empower Every Person on the Planet"),
        ("Airbnb", "Belong Anywhere"),
        ("Spotify", "Music for Everyone"),
        ("Slack", "Make Work Life Simpler, More Pleasant, and More Productive"),
        ("Notion", "The connected workspace"),
        ("Figma", "Nothing great is made alone"),
        ("Stripe", "Increase the GDP of the Internet"),
        ("Linear", "Build software like great teams do"),
        ("Vercel", "Develop. Preview. Ship."),
        ("GitHub", "Where the world builds software"),
        ("OpenAI", "AI for the benefit of humanity"),
        ("Anthropic", "AI safety and research company"),
        ("HubSpot", "There's a Better Way to Grow"),
        ("Salesforce", "The Customer Company"),
        ("Zoom", "One Platform to Connect"),
    ]
    return pd.DataFrame(data, columns=["Company", "Slogan"])


def _synthetic_startups() -> pd.DataFrame:
    """Returns a small synthetic startup corpus."""
    import random
    rng = random.Random(42)
    rows = [
        ("Growlytics",   "Bengaluru", "Data that grows your business",
         "AI-powered analytics platform for e-commerce growth"),
        ("GreenLeaf",    "Amsterdam", "Sustainability, simplified",
         "Carbon footprint tracking for SMBs using IoT sensors"),
        ("MediMind",     "Boston",    "Healthcare intelligence redefined",
         "Clinical decision-support AI for hospital networks"),
        ("PayEase",      "Mumbai",    "Payments without friction",
         "UPI-first payment gateway for micro-merchants"),
        ("TalentBloom",  "Toronto",   "Hire better. Hire faster.",
         "AI resume screening and candidate ranking for HR teams"),
        ("SkyRoute",     "Dubai",     "Your journey, optimised",
         "Real-time flight and hotel bundling for corporate travel"),
        ("Craftly",      "London",    "Design for everyone",
         "No-code brand design tool powered by generative AI"),
        ("FreshBite",    "Sydney",    "Good food, no compromise",
         "D2C meal kit delivery with personalised nutrition plans"),
        ("EduPath",      "Singapore", "Learning without limits",
         "Adaptive learning platform for K-12 students"),
        ("BuildFast",    "San Francisco", "Ship your SaaS in a week",
         "Full-stack boilerplate and deployment toolkit for founders"),
    ]
    return pd.DataFrame(rows, columns=["name","city","tagline","description"])


def _synthetic_marketing() -> pd.DataFrame:
    """Returns a synthetic marketing campaign dataset of 500 rows."""
    rng = np.random.default_rng(42)
    n = 500
    companies     = ["Alpha Corp","Beta Labs","Gamma Co","Delta Inc","Epsilon Ltd"]
    ctypes        = ["Awareness","Conversion","Retention","Launch","Seasonal"]
    audiences     = ["Gen Z (18-24)","Millennials (25-34)","Gen X (35-50)",
                     "Boomers (51+)","B2B Decision Makers"]
    channels      = ["Instagram","Facebook","Twitter / X","LinkedIn",
                     "YouTube","Email","Google Ads"]
    locations     = ["India","USA","Europe","UK","Southeast Asia","Global"]
    languages     = ["English","Hindi","Spanish","French","German"]
    segments      = ["Premium","Mass Market","Niche","Enterprise","SMB"]
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    durations     = rng.integers(7, 90, n)
    impressions   = rng.integers(10_000, 2_000_000, n)
    ctr           = rng.uniform(0.005, 0.08, n)
    clicks        = (impressions * ctr).astype(int)
    conv_rate     = rng.uniform(0.01, 0.15, n)
    acq_cost      = np.round(rng.uniform(10, 500, n), 2)
    roi           = np.round(rng.uniform(-0.5, 5.0, n), 3)
    engagement    = np.round(rng.uniform(1, 10, n), 2)

    return pd.DataFrame({
        "Campaign_ID":       [f"C{i:04d}" for i in range(n)],
        "Company":           rng.choice(companies, n),
        "Campaign_Type":     rng.choice(ctypes, n),
        "Target_Audience":   rng.choice(audiences, n),
        "Duration":          durations,
        "Channel_Used":      rng.choice(channels, n),
        "Conversion_Rate":   np.round(conv_rate, 4),
        "Acquisition_Cost":  acq_cost,
        "ROI":               roi,
        "Location":          rng.choice(locations, n),
        "Language":          rng.choice(languages, n),
        "Clicks":            clicks,
        "Impressions":       impressions,
        "Engagement_Score":  engagement,
        "Customer_Segment":  rng.choice(segments, n),
        "Date":              dates,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Public loaders (Streamlit-cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_slogans() -> tuple[pd.DataFrame, bool]:
    """
    Load slogan dataset.
    Returns (df, is_real) where is_real=False means synthetic fallback was used.
    """
    # Try processed first, then raw, then synthetic
    for path in [SLOGANS_CLEAN, SLOGANS_RAW]:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                if "Company" in df.columns and "Slogan" in df.columns:
                    log.info("Loaded slogans from %s (%d rows)", path, len(df))
                    return df, True
            except Exception as e:
                log.warning("Failed to load %s: %s", path, e)
    log.warning("Slogan CSV not found — using synthetic fallback.")
    return _synthetic_slogans(), False


@st.cache_data(show_spinner=False)
def load_startups() -> tuple[pd.DataFrame, bool]:
    """Load startup dataset. Returns (df, is_real)."""
    for path in [STARTUPS_CLEAN, STARTUPS_RAW]:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                if "name" in df.columns:
                    log.info("Loaded startups from %s (%d rows)", path, len(df))
                    return df, True
            except Exception as e:
                log.warning("Failed to load %s: %s", path, e)
    log.warning("Startups CSV not found — using synthetic fallback.")
    return _synthetic_startups(), False


@st.cache_data(show_spinner=False)
def load_marketing() -> tuple[pd.DataFrame, bool]:
    """Load marketing campaign dataset. Returns (df, is_real)."""
    for path in [MARKETING_CLEAN, MARKETING_RAW]:
        if Path(path).exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                required = {"Campaign_Type", "Channel_Used", "ROI", "Engagement_Score"}
                if required.issubset(df.columns):
                    log.info("Loaded marketing from %s (%d rows)", path, len(df))
                    return df, True
            except Exception as e:
                log.warning("Failed to load %s: %s", path, e)
    log.warning("Marketing CSV not found — using synthetic fallback.")
    return _synthetic_marketing(), False
