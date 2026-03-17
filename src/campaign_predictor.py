"""
src/campaign_predictor.py
Campaign performance prediction engine.

Models trained on marketing_campaign_dataset.csv [dataset-backed]:
  - ROI prediction        → GradientBoostingRegressor (best R² in evaluation)
  - Engagement prediction → RandomForestRegressor
  - CTR prediction        → LinearRegression (interpretable baseline)

If no trained model files exist, falls back to heuristic estimates
with clear documentation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CampaignPrediction:
    roi:              float
    engagement_score: float
    ctr:              float
    conversion_rate:  float
    source:           str       # "model" | "heuristic"
    confidence:       str       # "high" | "medium" | "low"
    recommendations:  list[str]
    channel_rank:     list[tuple[str,float]]   # [(channel, score)]


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (no model file needed)
# ─────────────────────────────────────────────────────────────────────────────

_ROI_BASE: dict[str,float] = {
    "Email": 3.2, "Google Ads": 2.8, "Instagram": 2.1,
    "Facebook": 1.9, "LinkedIn": 2.4, "Twitter / X": 1.6,
    "YouTube": 2.2, "TikTok": 1.8,
}
_ENGAGE_BASE: dict[str,float] = {
    "Instagram": 7.2, "TikTok": 8.1, "YouTube": 6.9,
    "Facebook": 5.8, "Twitter / X": 5.5, "LinkedIn": 6.1,
    "Email": 4.3, "Google Ads": 4.0,
}
_CTR_BASE: dict[str,float] = {
    "Google Ads": 0.055, "Email": 0.042, "Instagram": 0.031,
    "Facebook": 0.028, "LinkedIn": 0.038, "Twitter / X": 0.022,
    "YouTube": 0.019, "TikTok": 0.025,
}
_CAMPAIGN_MULT: dict[str,float] = {
    "Conversion": 1.25, "Launch": 1.15, "Retention": 1.10,
    "Awareness": 0.90, "Seasonal": 1.05, "Remarketing": 1.20,
}
_AUDIENCE_MULT: dict[str,float] = {
    "Gen Z (18-24)": {"Instagram":1.3,"TikTok":1.5},
    "Millennials (25-34)": {"Instagram":1.2,"Facebook":1.2},
    "B2B Decision Makers": {"LinkedIn":1.5,"Email":1.3},
}
_REGION_MULT: dict[str,float] = {
    "India": 1.1, "USA": 1.0, "Europe": 0.95, "Global": 1.05,
}


def _heuristic_prediction(
    channel: str, campaign_type: str, region: str,
    audience: str, duration_days: int, budget: float,
) -> CampaignPrediction:
    norm_ch = channel.replace("Twitter/X","Twitter / X")
    roi_base    = _ROI_BASE.get(norm_ch, 2.0)
    eng_base    = _ENGAGE_BASE.get(norm_ch, 5.5)
    ctr_base    = _CTR_BASE.get(norm_ch, 0.03)
    camp_m      = _CAMPAIGN_MULT.get(campaign_type, 1.0)
    region_m    = _REGION_MULT.get(region, 1.0)

    aud_dict    = _AUDIENCE_MULT.get(audience, {})
    aud_m       = aud_dict.get(norm_ch, 1.0)

    dur_bonus   = min(1 + (duration_days - 14) * 0.004, 1.3) if duration_days > 14 else 1.0

    roi  = round(roi_base * camp_m * region_m * aud_m * dur_bonus, 3)
    eng  = round(eng_base * camp_m * aud_m, 2)
    ctr  = round(ctr_base * camp_m * region_m, 5)
    conv = round(ctr * 0.35, 5)

    recs: list[str] = []
    if dur_bonus < 1.0:
        recs.append("Extending your campaign beyond 14 days typically improves results.")
    if campaign_type == "Awareness" and roi < 2.0:
        recs.append("Awareness campaigns have lower direct ROI — pair with a Conversion campaign.")
    if audience == "B2B Decision Makers" and norm_ch not in ("LinkedIn","Email"):
        recs.append("For B2B audiences, LinkedIn and Email outperform social channels.")
    if not recs:
        recs = [f"{norm_ch} is a strong channel for your combination of {campaign_type} + {audience}."]

    # Channel ranking
    rank = sorted(_ROI_BASE.items(), key=lambda x:-x[1]*_CAMPAIGN_MULT.get(campaign_type,1.0))

    return CampaignPrediction(
        roi=roi, engagement_score=eng, ctr=ctr,
        conversion_rate=conv, source="heuristic", confidence="medium",
        recommendations=recs, channel_rank=rank[:5],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model-based prediction
# ─────────────────────────────────────────────────────────────────────────────

def _model_prediction(
    channel: str, campaign_type: str, region: str,
    audience: str, duration_days: int, budget: float, language: str,
) -> CampaignPrediction | None:
    """
    Try to load trained models and run inference.
    Returns None if models not found or inference fails.
    """
    from src.config import MODEL_ROI, MODEL_ENGAGE, MODEL_CTR, ENCODERS_FILE, SCALER_FILE
    import joblib

    for f in [MODEL_ROI, MODEL_ENGAGE, ENCODERS_FILE, SCALER_FILE]:
        if not Path(f).exists():
            return None

    try:
        model_roi   = joblib.load(MODEL_ROI)
        model_eng   = joblib.load(MODEL_ENGAGE)
        encoders    = joblib.load(ENCODERS_FILE)
        scaler      = joblib.load(SCALER_FILE)
        model_ctr   = joblib.load(MODEL_CTR) if Path(MODEL_CTR).exists() else None

        # Encode categoricals
        def safe_encode(enc, val, default=0):
            try:   return int(enc.transform([val])[0])
            except: return default

        feat = np.array([[
            safe_encode(encoders.get("Campaign_Type"), campaign_type),
            safe_encode(encoders.get("Channel_Used"),  channel),
            safe_encode(encoders.get("Location"),      region),
            safe_encode(encoders.get("Language"),      language),
            safe_encode(encoders.get("Target_Audience"),audience),
            float(duration_days),
            float(budget),
        ]])
        feat_scaled = scaler.transform(feat)

        roi  = float(model_roi.predict(feat_scaled)[0])
        eng  = float(model_eng.predict(feat_scaled)[0])
        ctr  = float(model_ctr.predict(feat_scaled)[0]) if model_ctr else roi * 0.02
        conv = ctr * 0.3

        recs = [
            f"Model predicts ROI of {roi:.2f}× — {'above' if roi>2 else 'at or below'} industry average.",
            f"Expected engagement score: {eng:.1f}/10.",
            "Optimise ad creative for the first 48 hours — early engagement boosts algorithmic reach.",
        ]
        rank = sorted(_ROI_BASE.items(), key=lambda x:-x[1])

        return CampaignPrediction(
            roi=round(roi,3), engagement_score=round(eng,2),
            ctr=round(ctr,5), conversion_rate=round(conv,5),
            source="model", confidence="high",
            recommendations=recs, channel_rank=rank[:5],
        )
    except Exception as e:
        log.warning("Model prediction failed: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_campaign(
    channel:       str  = "Instagram",
    campaign_type: str  = "Awareness",
    region:        str  = "Global",
    audience:      str  = "Millennials (25-34)",
    duration_days: int  = 30,
    budget:        float= 5000.0,
    language:      str  = "English",
) -> CampaignPrediction:
    """
    Main prediction entry point.
    Uses trained ML model if available, otherwise heuristic fallback.
    """
    model_result = _model_prediction(channel, campaign_type, region,
                                      audience, duration_days, budget, language)
    if model_result is not None:
        return model_result
    return _heuristic_prediction(channel, campaign_type, region,
                                  audience, duration_days, budget)
