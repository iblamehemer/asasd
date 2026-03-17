"""
src/feature_engineering.py
Standalone model training script.

Run: python -m src.feature_engineering

Trains and evaluates three regression models on marketing_campaign_dataset.csv:
  1. GradientBoostingRegressor → ROI prediction
  2. RandomForestRegressor     → Engagement Score prediction
  3. LinearRegression          → CTR prediction (interpretable baseline)

Saves model artifacts to models/:
  campaign_model.pkl  (GBR — best overall)
  roi_model.pkl
  engagement_model.pkl
  ctr_model.pkl
  encoders.pkl
  scaler.pkl

All training is dataset-backed when marketing CSV is available.
Falls back to synthetic data with a clear warning.
"""

from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

CAT_COLS = ["Campaign_Type","Channel_Used","Location","Language",
            "Target_Audience","Customer_Segment"]
NUM_COLS = ["Duration","Acquisition_Cost","Impressions","Clicks"]
TARGETS  = ["ROI","Engagement_Score","CTR","Conversion_Rate"]


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, dict, StandardScaler]:
    """
    Encode categoricals and scale numerics.
    Returns (X_array, encoders_dict, scaler).
    """
    df = df.copy()

    # Derive CTR if not present
    if "CTR" not in df.columns and "Clicks" in df.columns and "Impressions" in df.columns:
        df["CTR"] = np.where(df["Impressions"]>0, df["Clicks"]/df["Impressions"], 0.0)

    # Impute
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    encoders: dict[str, LabelEncoder] = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    feature_cols = CAT_COLS + [c for c in NUM_COLS if c in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values.astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, encoders, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"  {name:30s} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"model":name,"rmse":round(rmse,4),"mae":round(mae,4),"r2":round(r2,4)}


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all():
    from src.data_loader import load_marketing
    from src.preprocess  import clean_marketing
    from src.config      import MODELS_DIR

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_raw, is_real = load_marketing()
    if not is_real:
        print("⚠️  WARNING: Using synthetic marketing data. "
              "Place marketing_campaign_dataset.csv in datasets/raw/ for real training.")
    df = clean_marketing(df_raw)
    print(f"Training on {len(df)} rows (real data: {is_real})")

    X, encoders, scaler = prepare_features(df)

    results = []
    for target in ["ROI","Engagement_Score","CTR"]:
        if target not in df.columns:
            continue
        y = df[target].values.astype(float)
        # Remove rows where target is NaN after alignment
        mask  = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X_t   = X[mask]; y_t = y[mask]
        X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

        print(f"\n── Target: {target} ──")
        candidates = {
            "GradientBoostingRegressor": GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
            "RandomForestRegressor":     RandomForestRegressor(
                n_estimators=150, max_depth=6, random_state=42, n_jobs=-1),
            "Ridge":                     Ridge(alpha=1.0),
        }
        best_model, best_r2, best_name = None, -np.inf, ""
        for name, model in candidates.items():
            model.fit(X_tr, y_tr)
            res = _evaluate(name, model, X_te, y_te)
            res["target"] = target
            results.append(res)
            if res["r2"] > best_r2:
                best_r2    = res["r2"]
                best_model = model
                best_name  = name
        print(f"  → Best: {best_name} (R²={best_r2:.4f})")

        # Save best model
        suffix = target.lower().replace(" ","_")
        joblib.dump(best_model, MODELS_DIR / f"{suffix}_model.pkl")

    # Save shared artefacts
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(scaler,   MODELS_DIR / "scaler.pkl")

    # Save summary JSON
    summary = {"is_real_data": is_real, "n_rows": len(df), "results": results}
    with open(MODELS_DIR / "training_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Training complete. Artifacts saved to models/")
    print(f"   Real data: {is_real}")
    for r in results:
        print(f"   {r['target']:20s} {r['model']:35s} R²={r['r2']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
