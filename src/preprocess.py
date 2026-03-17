"""
src/preprocess.py
Clean and normalise all raw datasets for downstream use.
Run once: `python -m src.preprocess` to persist cleaned CSVs.
Also callable programmatically from training scripts.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import (
    RAW, PROCESSED,
    SLOGANS_RAW, STARTUPS_RAW, MARKETING_RAW,
    SLOGANS_CLEAN, STARTUPS_CLEAN, MARKETING_CLEAN,
    PERSONAS_FILE, CAMPAIGN_FEATS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _strip_currency(val) -> float:
    """Parse '$1,234.56' or '1234' into float. Returns NaN on failure."""
    if pd.isna(val):
        return np.nan
    cleaned = re.sub(r"[£$€₹,\s]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _clean_text(s: str) -> str:
    """Lowercase, strip extra whitespace, remove non-printable chars."""
    s = str(s).strip()
    s = re.sub(r"[^\x20-\x7E\u0900-\u097F]", " ", s)  # keep ASCII + Devanagari
    s = re.sub(r"\s+", " ", s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Slogan cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_slogans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean sloganlist.csv:
    - Drop nulls / duplicates
    - Normalise Company and Slogan text
    - Add slogan_len feature
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Company", "Slogan"])
    df["Company"] = df["Company"].apply(_clean_text).str.title()
    df["Slogan"]  = df["Slogan"].apply(_clean_text)
    df = df.drop_duplicates(subset=["Slogan"])
    df["slogan_len"] = df["Slogan"].str.split().str.len()
    df = df[df["slogan_len"] >= 2]  # remove single-word noise
    df = df.reset_index(drop=True)
    log.info("Slogans cleaned: %d rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Startup cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_startups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean startups.csv:
    - Required cols: name, city, tagline, description
    - Fill missing taglines from description
    - Normalise text
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["name","city","tagline","description"]:
        if col not in df.columns:
            df[col] = ""
    df = df.dropna(subset=["name"])
    df["name"]        = df["name"].apply(_clean_text).str.title()
    df["city"]        = df["city"].apply(_clean_text).str.title()
    df["tagline"]     = df["tagline"].apply(_clean_text)
    df["description"] = df["description"].apply(_clean_text)
    # Fill missing tagline with first 10 words of description
    mask = df["tagline"].str.len() < 3
    df.loc[mask, "tagline"] = df.loc[mask, "description"].apply(
        lambda x: " ".join(str(x).split()[:10]) + "…" if x else "")
    df = df.drop_duplicates(subset=["name"])
    df = df.reset_index(drop=True)
    log.info("Startups cleaned: %d rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Marketing campaign cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_marketing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean marketing_campaign_dataset.csv:
    - Parse Acquisition_Cost (may contain currency symbols)
    - Parse Date to datetime
    - Impute missing numerics
    - Derive CTR = Clicks / Impressions
    - Add ROI_band categorical
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Numeric cols
    for col in ["Conversion_Rate","Acquisition_Cost","ROI","Clicks","Impressions","Engagement_Score"]:
        if col in df.columns:
            df[col] = df[col].apply(_strip_currency)

    # Impute with median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]  = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek

    # Derived features
    if "Clicks" in df.columns and "Impressions" in df.columns:
        df["CTR"] = np.where(
            df["Impressions"] > 0,
            (df["Clicks"] / df["Impressions"]).round(5),
            0.0,
        )

    if "ROI" in df.columns:
        df["ROI_Band"] = pd.cut(
            df["ROI"],
            bins=[-np.inf, 0, 0.5, 1.5, 3.0, np.inf],
            labels=["Negative","Low","Medium","High","Exceptional"],
        )

    # Normalise categorical case
    for col in ["Campaign_Type","Channel_Used","Location","Language","Customer_Segment","Target_Audience"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df = df.reset_index(drop=True)
    log.info("Marketing cleaned: %d rows, %d cols", len(df), len(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Campaign feature engineering (for model training)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_campaign_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables and produce a numeric feature matrix
    for ML training. Saves to CAMPAIGN_FEATS.
    Returns the encoded dataframe.
    """
    df = df.copy()
    cat_cols = ["Campaign_Type","Channel_Used","Location","Language",
                "Customer_Segment","Target_Audience"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    encoded.columns = [c.replace(" ","_").replace("/","_") for c in encoded.columns]

    numeric = encoded.select_dtypes(include=np.number).columns.tolist()
    targets = {"ROI","Engagement_Score","CTR","Conversion_Rate"}
    features = [c for c in numeric if c not in targets and c not in
                {"Campaign_ID","Year","Month","DayOfWeek","Clicks","Impressions"}]

    out = encoded[features + [t for t in targets if t in encoded.columns]]
    log.info("Campaign features engineered: %d rows × %d cols", len(out), len(out.columns))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — run once to persist processed files
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    """Load raws, clean them, and save to datasets/processed/."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    from src.data_loader import load_slogans, load_startups, load_marketing

    slogans,  s_real = load_slogans()
    startups, st_real= load_startups()
    marketing,m_real = load_marketing()

    clean_slogans(slogans).to_csv(SLOGANS_CLEAN, index=False)
    clean_startups(startups).to_csv(STARTUPS_CLEAN, index=False)
    mk = clean_marketing(marketing)
    mk.to_csv(MARKETING_CLEAN, index=False)
    engineer_campaign_features(mk).to_csv(CAMPAIGN_FEATS, index=False)

    print("✅ Preprocessing complete.")
    print(f"   Slogans : {len(slogans)} rows  (real={s_real})")
    print(f"   Startups: {len(startups)} rows  (real={st_real})")
    print(f"   Marketing:{len(marketing)} rows (real={m_real})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
