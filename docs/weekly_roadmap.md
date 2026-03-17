# NovaTech AI — Weekly Roadmap & Implementation Checklist

## Week 1 — Problem Definition & EDA
- [x] Define problem statement and business context
- [x] Load all three CSV datasets with safe fallbacks (`src/data_loader.py`)
- [x] Clean nulls, duplicates, currency strings, dates (`src/preprocess.py`)
- [x] Derive CTR, ROI bands, feature engineering
- [x] EDA notebook outline (`notebooks/01_eda.ipynb`)
- [x] Dashboard charts: ROI by channel, engagement distribution, CTR histogram

## Week 2 — Logo Studio
- [x] 5 SVG logo concepts: Lettermark, Geometric, Wordmark, Emblem, Minimal Bracket
- [x] Tone-to-style mapping (Luxury→Emblem, Minimalist→Bracket, etc.)
- [x] Palette-responsive colours in all SVG marks
- [x] Select-and-highlight UI in app

## Week 3 — Font Recommendation Engine
- [x] 20+ industry-tone font pairings in `src/config.py`
- [x] Heading + body + accent font for each pairing
- [x] Google Fonts links
- [x] Rationale text per pairing
- [x] Fallback to `_default` for unmapped combinations

## Week 4 — Slogan / Tagline Generation
- [x] Template-based generation (10 tone templates × 3 variants each)
- [x] TF-IDF cosine similarity retrieval from slogan corpus
- [x] Gemini API enhancement (if key available)
- [x] Source labelling: template / retrieval / gemini
- [x] Confidence scores and rationale per candidate

## Week 5 — Colour Palette & Visual Harmony
- [x] 12+ industry-tone palette mappings in `src/config.py`
- [x] 5-colour palette with role, hex, name, meaning
- [x] WCAG contrast ratio harmony scoring
- [x] KMeans runtime extraction from uploaded image
- [x] Swatch display UI with role labels

## Week 6 — Animation Studio
- [x] 6 animation styles: Fade In, Slide Left, Slide Up, Typewriter, Zoom In, Pulse
- [x] Matplotlib FuncAnimation GIF export
- [x] Smoothstep easing for professional motion feel
- [x] Static PNG fallback when Pillow GIF fails
- [x] Download button for GIF

## Week 7 — Campaign Prediction
- [x] Feature engineering: encode categoricals, scale numerics
- [x] 3 models compared: GradientBoosting, RandomForest, Ridge
- [x] RMSE / MAE / R² evaluation per target
- [x] Best model saved per target (ROI, Engagement, CTR)
- [x] Heuristic fallback when no model file exists
- [x] 4 prediction metrics displayed: ROI, Engagement, CTR, Conversion Rate

## Week 8 — Multilingual Campaign Generator
- [x] 8 target languages: Hindi, Spanish, French, German, Gujarati, Portuguese, Arabic, Japanese
- [x] Priority chain: Gemini → deep-translator → curated fallback
- [x] Source + confidence label per translation
- [x] Batch translation API for multiple texts

## Week 9 — Feedback Intelligence & Model Refinement
- [x] Star rating (1–5) per asset type: logo, slogan, palette, campaign, overall
- [x] Optional comment field
- [x] VADER sentiment analysis on comments
- [x] Persistent CSV storage (`feedback_log.csv`)
- [x] Aggregated summary: avg rating, by-asset breakdown, sentiment counts
- [x] Retraining readiness signal (triggers at 50+ entries or 20%+ low ratings)

## Week 10 — Integration & Deployment
- [x] Single-page smooth-scroll Streamlit app (Waabi-inspired)
- [x] Sticky nav with anchor links to all sections
- [x] Brand Kit ZIP with 8+ asset types
- [x] Brand consistency score (0–100, grade A+→D)
- [x] AI assistant (Gemini chat with brand context)
- [x] requirements.txt, .env.example, .gitignore
- [x] README.md with full setup and deploy instructions
- [x] PRD support content
- [x] Deployment config (`deployment/streamlit_deployment.yml`)
