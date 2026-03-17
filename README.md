# NovaTech AI — AI-Powered Automated Branding Assistant

> CRS AI Capstone 2025–26 · Scenario 1  
> A production-style, end-to-end branding platform built in Python and Streamlit.

---

## Problem Statement

Small and medium-sized businesses spend tens of thousands of dollars on brand agencies for assets they often receive once, use partially, and can never easily update. NovaTech AI automates the generation of complete brand identity packages — logo concepts, typography, colour palettes, slogans, animated visuals, multilingual campaign copy, and predictive campaign analytics — in a single interactive platform.

---

## Business Context

| Metric | Value |
|--------|-------|
| Target users | Founders, solo marketers, SMB owners |
| Time to complete brand kit (manual) | 2–8 weeks |
| Time with NovaTech AI | < 5 minutes |
| Languages supported | 8 |
| Campaign channels | 8 |
| Models trained | Gradient Boosting · Random Forest · Ridge |

---

## Architecture

```
User Inputs
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  src/startup_persona_engine.py  (rule-based)         │
│  → derives persona from industry + description        │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼───────────────┐
        │   Brand Identity Layer      │
        │  logo_engine.py   (SVG)     │
        │  font_engine.py   (mapping) │
        │  palette_engine.py (KMeans) │
        └────────────┬───────────────┘
                     │
        ┌────────────▼───────────────┐
        │   Content Layer             │
        │  slogan_engine.py (TF-IDF   │
        │    + templates + Gemini)    │
        │  branding_logic.py (captions│
        │    + hashtags + CTAs)       │
        └────────────┬───────────────┘
                     │
        ┌────────────▼───────────────┐
        │   Intelligence Layer        │
        │  campaign_predictor.py (ML) │
        │  multilingual_engine.py     │
        │  aesthetics_engine.py       │
        │  animation_engine.py        │
        └────────────┬───────────────┘
                     │
        ┌────────────▼───────────────┐
        │   Output Layer              │
        │  feedback_engine.py (VADER) │
        │  export_engine.py (ZIP)     │
        │  dashboard_engine.py        │
        └─────────────────────────────┘
```

---

## Module Overview

| Module | File | Source |
|--------|------|--------|
| Data Loading | `src/data_loader.py` | Dataset-backed + synthetic fallback |
| Preprocessing | `src/preprocess.py` | Dataset-backed |
| Startup Persona | `src/startup_persona_engine.py` | Rule-based (keyword heuristics) |
| Slogan Engine | `src/slogan_engine.py` | TF-IDF retrieval + templates + Gemini API |
| Font Engine | `src/font_engine.py` | Rule-based mapping |
| Palette Engine | `src/palette_engine.py` | Mapping + KMeans (runtime) |
| Logo Engine | `src/logo_engine.py` | SVG composition (rule-based) |
| Aesthetics Score | `src/aesthetics_engine.py` | Weighted rule-based heuristic |
| Campaign Predictor | `src/campaign_predictor.py` | ML (GBR/RF) + heuristic fallback |
| Content Generator | `src/branding_logic.py` | Template + optional Gemini |
| Multilingual | `src/multilingual_engine.py` | Gemini → deep-translator → fallback |
| Animation | `src/animation_engine.py` | Matplotlib FuncAnimation |
| Feedback | `src/feedback_engine.py` | VADER sentiment + CSV store |
| Export | `src/export_engine.py` | zipfile + Matplotlib |
| Dashboard | `src/dashboard_engine.py` | Plotly |

---

## Datasets Used

| Dataset | Rows | Real / Synthetic | Source |
|---------|------|-----------------|--------|
| `sloganlist.csv` | Variable | Real (when uploaded) | Uploaded |
| `startups.csv` | Variable | Real (when uploaded) | Uploaded |
| `marketing_campaign_dataset.csv` | Variable | Real (when uploaded) | Uploaded |
| Synthetic fallback | 20–500 | Generated | `src/data_loader.py` |

> **Honesty note**: If CSV files are not placed in `datasets/raw/`, the app
> automatically uses synthetic fallback data. All charts and models will work,
> but results will reflect synthetic patterns. Place real CSVs for production-grade output.

---

## Setup Instructions

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/novatech-ai.git
cd novatech-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. (Optional) Place datasets
cp /path/to/sloganlist.csv                   datasets/raw/
cp /path/to/startups.csv                     datasets/raw/
cp /path/to/marketing_campaign_dataset.csv   datasets/raw/

# 6. (Optional) Preprocess data
python -m src.preprocess

# 7. (Optional) Train ML models
python -m src.feature_engineering

# 8. Run the app
streamlit run app.py
```

### Streamlit Cloud Deployment

```
1. Push repository to GitHub (public or private)
2. Go to share.streamlit.io → New app
3. Select: repo → branch → app.py
4. In Settings → Secrets, add:
   GEMINI_API_KEY = "your_key_here"
5. Deploy
```

---

## Repository Structure

```
novatech-ai/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── config.py                   # Paths, constants, mappings
│   ├── data_loader.py              # Safe CSV loaders + fallbacks
│   ├── preprocess.py               # Data cleaning pipeline
│   ├── feature_engineering.py      # ML training script
│   ├── slogan_engine.py            # Tagline generation
│   ├── startup_persona_engine.py   # Brand persona derivation
│   ├── font_engine.py              # Font recommendation
│   ├── palette_engine.py           # Colour palette engine
│   ├── logo_engine.py              # SVG logo concept generator
│   ├── aesthetics_engine.py        # Brand consistency scoring
│   ├── campaign_predictor.py       # ML campaign prediction
│   ├── branding_logic.py           # Social content generator
│   ├── multilingual_engine.py      # Translation engine
│   ├── animation_engine.py         # GIF animation
│   ├── feedback_engine.py          # Feedback + VADER sentiment
│   ├── export_engine.py            # ZIP brand kit packager
│   └── dashboard_engine.py         # Plotly chart functions
│
├── datasets/
│   ├── raw/                        # Place uploaded CSVs here
│   └── processed/                  # Auto-generated after preprocessing
│
├── models/                         # Trained .pkl files (auto-generated)
├── assets/                         # Logo SVG, temp files, exports
├── notebooks/                      # Colab / Jupyter notebooks
├── docs/                           # Architecture, PRD, roadmap
└── deployment/                     # Streamlit Cloud config
```

---

## Sample Workflow

1. Open app → set company name, industry, tone
2. Click **Generate Logo Concepts & Font Pairing**
3. Click **Generate Slogans** → select favourite
4. Configure campaign settings → click **Predict Performance**
5. Select target languages → **Translate**
6. Click **Generate Animation**
7. **Compute Brand Consistency Score**
8. Rate your assets in Feedback
9. Click **Build & Download Brand Kit ZIP**

---

## Limitations

- Logo engine generates SVG concepts, not photorealistic designs (no logo image dataset)
- Font engine is mapping-based, not trained on font image features
- Campaign ML models achieve R² ~0.5–0.7 on synthetic data; real data will improve accuracy
- Gemini API required for highest-quality slogans, captions, and translations
- Animation export requires `Pillow` (pre-installed on Streamlit Cloud)

---

## Future Improvements

- CNN logo classifier once a labelled logo dataset is available (LLD-icon, SVG-Logo-3M)
- Font similarity CNN trained on rendered font samples
- Fine-tuned LLM on brand voice data per industry
- PostgreSQL backend for multi-user feedback
- A/B testing framework for slogan variants
- Canva / Figma API integration for design export

---

## Acknowledgments

- Gemini API by Google DeepMind
- scikit-learn, Plotly, Matplotlib, Streamlit communities
- NovaTech AI · CRS AI Capstone 2025–26
