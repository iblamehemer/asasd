# NovaTech AI — PRD Support Content
> CRS AI Capstone 2025–26 · Scenario 1

---

## 1. Purpose

NovaTech AI exists to democratise professional brand identity creation for small and medium-sized businesses. The platform replaces a fragmented, expensive, agency-dependent process with a single AI-powered tool that produces logo concepts, typography pairings, colour palettes, taglines, social campaign copy, multilingual translations, animated brand visuals, and campaign performance predictions — all in under five minutes.

---

## 2. Objectives

| # | Objective | Metric |
|---|-----------|--------|
| O1 | Reduce brand kit generation time from weeks to minutes | < 5 min end-to-end |
| O2 | Achieve campaign prediction accuracy exceeding random baseline | R² ≥ 0.50 on held-out data |
| O3 | Support multilingual brand presence | ≥ 8 languages |
| O4 | Provide downloadable, deployment-ready brand kit | ZIP with ≥ 7 asset types |
| O5 | Enable continuous improvement through user feedback | VADER-analysed ratings stored persistently |

---

## 3. Scope

**In scope:**
- Logo concept generation (SVG, 5 styles)
- Font pairing recommendations (Google Fonts-linked)
- Colour palette generation (5 roles, harmony score)
- Slogan / tagline generation (TF-IDF + Gemini)
- Animated brand visuals (GIF, 6 styles)
- Social media captions (8 platforms)
- Campaign performance prediction (ROI, Engagement, CTR)
- Multilingual translation (8 languages)
- Brand consistency scoring
- Feedback collection and sentiment analysis
- Downloadable ZIP brand kit

**Out of scope (v1):**
- Real-time collaboration / team accounts
- Raster logo rendering (Photoshop-quality)
- E-commerce integration
- Paid media buying / campaign execution
- Mobile-native app

---

## 4. Stakeholders

| Role | Interest |
|------|----------|
| Student / Developer | Build and submit capstone |
| Course Evaluator | Assess technical depth and honesty |
| SMB Owner (end user) | Fast, quality brand kit |
| Marketing Freelancer | Client deliverable automation |
| Startup Founder | Pre-seed brand identity without agency cost |

---

## 5. Deliverables

| Deliverable | Format | Status |
|-------------|--------|--------|
| Streamlit web app | Deployed URL | ✓ Buildable |
| Source code | GitHub repository | ✓ Complete |
| Brand Kit ZIP | Downloadable in-app | ✓ Complete |
| ML models | `.pkl` files | ✓ Trained on run |
| Preprocessing pipeline | `src/preprocess.py` | ✓ Complete |
| EDA notebook | `notebooks/01_eda.ipynb` | ✓ Outline provided |
| PRD document | This file | ✓ Complete |
| Architecture diagram | `docs/architecture.md` | ✓ Inline in README |

---

## 6. Tools and Technologies

| Category | Technology | Role |
|----------|-----------|------|
| Frontend | Streamlit 1.35+ | UI + deployment |
| ML | scikit-learn | GBR, RF, Ridge for campaign prediction |
| Data | Pandas, NumPy | Preprocessing, feature engineering |
| Visualisation | Plotly, Matplotlib | Charts, animation |
| GenAI | Google Gemini API | Slogans, captions, translations |
| NLP | NLTK (VADER) | Feedback sentiment analysis |
| Image | Pillow | GIF export, palette extraction |
| Translation | deep-translator | Offline translation fallback |
| Export | zipfile | Brand kit packaging |
| Deployment | Streamlit Cloud | Public URL hosting |

---

## 7. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-end brand kit time | < 5 min | Manual timing |
| Campaign ROI prediction R² | ≥ 0.50 | `feature_engineering.py` output |
| Slogan relevance | ≥ 3 Gemini candidates per run | In-app count |
| Multilingual coverage | 8 languages | Language list |
| Brand consistency score | Displayed 0–100 | `aesthetics_engine.py` |
| Feedback collection | Persistent CSV | `feedback_engine.py` |
| Export completeness | ≥ 7 asset types in ZIP | `export_engine.py` |

---

## 8. Weekly Roadmap

| Week | Focus | Key Deliverable | Data Source |
|------|-------|-----------------|-------------|
| 1 | Problem definition, EDA | Cleaned datasets, EDA notebook | All CSVs |
| 2 | Logo studio | SVG logo engine | Rule-based |
| 3 | Font recommendation | Font pairing engine | Rule-based mapping |
| 4 | Slogan generation | TF-IDF + Gemini pipeline | sloganlist.csv |
| 5 | Colour palette | KMeans + mapping engine | Synthetic + runtime |
| 6 | Animation | Matplotlib GIF export | Rule-based |
| 7 | Campaign prediction | Trained ML models | marketing_campaign_dataset.csv |
| 8 | Multilingual | Translation pipeline | Gemini / deep-translator |
| 9 | Feedback + scoring | VADER + brand score | feedback_log.csv |
| 10 | Integration + deploy | ZIP kit + Streamlit Cloud | All sources |

---

## 9. Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Gemini API key unavailable | Medium | High | Full offline fallback for all Gemini features |
| Real datasets not uploaded | High | Medium | Synthetic fallbacks with explicit labelling |
| Pillow/matplotlib GIF fails | Low | Low | Static PNG fallback |
| ML models score low R² | Medium | Medium | Heuristic prediction fallback; models re-trainable |
| Streamlit Cloud cold start slow | Low | Low | `@st.cache_data` on all loaders |
| NLTK VADER not downloaded | Medium | Low | Auto-download in `feedback_engine.py` |

---

## 10. Evaluation Criteria

| Criterion | Weight | Evidence |
|-----------|--------|----------|
| Technical correctness | 25% | All modules syntax-verified, modular, documented |
| ML pipeline quality | 20% | 3 models compared, RMSE/R² reported, artifacts saved |
| AI/Gemini integration | 15% | Fallback chain, honest source labelling |
| UI/UX quality | 15% | Waabi-inspired single-scroll, responsive layout |
| Dataset usage | 10% | All 3 CSVs used; honest fallback when absent |
| Documentation | 10% | README, PRD, code comments throughout |
| Export completeness | 5% | ZIP with ≥ 7 asset types |

---

## 11. Final Reflection

**What worked well:**
- Modular `src/` architecture made each component independently testable
- Honest fallback design means the app runs fully without any uploaded data or API key
- Waabi-inspired UI gives the submission a professional, memorable presentation
- The brand consistency score provides a concrete, explainable output

**What would be improved in v2:**
- Train a real CNN logo classifier on LLD-icon or a scraped logo dataset
- Replace heuristic font mapping with an embedding-based similarity search on rendered font samples
- Fine-tune a smaller LLM on brand copywriting data for higher-quality offline slogans
- Add PostgreSQL for multi-session feedback aggregation

**Honest limitations acknowledged:**
- Logo engine generates SVG geometry, not pixel-perfect professional logos
- Font engine is rule-based; "recommendation" is a curated mapping, not ML
- Campaign model trained on synthetic data achieves moderate R² (≈ 0.5–0.7); real data improves this significantly

---

*NovaTech AI · CRS AI Capstone 2025–26 · Scenario 1*
