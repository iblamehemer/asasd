"""
app.py
NovaTech AI — AI-Powered Automated Branding Assistant
CRS AI Capstone 2025–26 · Scenario 1

Single-page smooth-scroll layout.
Waabi.ai-inspired: pitch black · large serif headlines · minimal nav · gold accent.

Run:   streamlit run app.py
Deploy: Streamlit Cloud → set GEMINI_API_KEY in secrets.
"""

# ── Page config (must be first) ───────────────────────────────────────────────
import streamlit as st
st.set_page_config(
    page_title="NovaTech AI — Branding Platform",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Stdlib ────────────────────────────────────────────────────────────────────
import base64, io, json, os, sys, uuid, logging
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING)

# ── Load environment (.env for local dev) ────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd

# ── Internal modules ──────────────────────────────────────────────────────────
from src.config import (
    INDUSTRIES, TONES, PLATFORMS, REGIONS, LANGUAGES,
    CAMPAIGN_TYPES, AUDIENCE_SEGMENTS, ANIMATION_STYLES,
)
from src.data_loader               import load_slogans, load_startups, load_marketing
from src.slogan_engine             import generate_slogans, init_retriever
from src.startup_persona_engine    import derive_persona, find_similar_startups
from src.font_engine               import recommend_fonts
from src.palette_engine            import recommend_palette, extract_palette_from_image
from src.logo_engine               import generate_logo_concepts
from src.aesthetics_engine         import compute_brand_score
from src.campaign_predictor        import predict_campaign
from src.branding_logic            import generate_content
from src.multilingual_engine       import translate_text
from src.animation_engine          import get_animation_gif
from src.feedback_engine           import save_feedback, load_feedback, compute_feedback_summary, retraining_readiness
from src.export_engine             import build_brand_kit_zip
from src.dashboard_engine          import (
    campaign_roi_by_channel, engagement_by_campaign_type,
    ctr_distribution, roi_over_time, campaign_type_pie,
    channel_heatmap, feedback_radar, feedback_sentiment_bar,
    slogan_length_histogram, acquisition_cost_by_channel,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = dict(
        session_id=str(uuid.uuid4())[:8],
        company="NovaTech", industry="Technology", tone="Minimalist",
        audience="", region="India", description="", tagline="",
        slogans=[], selected_slogan="",
        palette=[], font=None,
        logo_concepts=[], selected_logo_style="",
        captions=[], translations=[],
        campaign_result=None,
        animation_gif=None, animation_b64="",
        brand_score=None,
        chat_history=[],
        feedback_submitted=set(),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Load datasets (once, cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading datasets…"):
    df_slogans,  _sl_real  = load_slogans()
    df_startups, _st_real  = load_startups()
    df_marketing,_mk_real  = load_marketing()
    init_retriever(df_slogans)

# ─────────────────────────────────────────────────────────────────────────────
# Logo SVG (NovaTech brand mark)
# ─────────────────────────────────────────────────────────────────────────────

def _load_logo() -> str:
    p = Path(__file__).parent / "assets" / "novatech-logo.svg"
    if p.exists():
        raw = p.read_text()
        raw = (raw.replace('fill="#1A1A1A"', 'fill="#F5F2EB"')
                  .replace("fill='#1A1A1A'", "fill='#F5F2EB'")
                  .replace('fill="#F9F9F7"/>', 'fill="#0A0A08"/>')
                  .replace('rect width="400" height="400" fill="#F9F9F7"',
                            'rect width="400" height="400" fill="none"'))
        return raw
    return ""

LOGO_SVG = _load_logo()

def _logo_scaled(size: int) -> str:
    return (LOGO_SVG
        .replace('width="400"', f'width="{size}"')
        .replace('height="400"', f'height="{size}"'))

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Waabi-inspired
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,300;0,400;0,500;1,300;1,400&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap');
:root{
  --black:#0A0A08;--black2:#111110;--black3:#1A1A17;
  --edge:#222220;--edge2:#2C2C29;
  --white:#F5F2EB;--white2:#C8C4BB;
  --dim:#6A6A64;--dim2:#3A3A36;
  --gold:#C8A94A;--gold2:#E2C97A;
  --serif:'Playfair Display',Georgia,serif;
  --sans:'DM Sans',system-ui,sans-serif;
  --mono:'DM Mono','Courier New',monospace;
}
html{scroll-behavior:smooth;}
#MainMenu,footer,header,[data-testid="stHeader"],
[data-testid="stDecoration"],[data-testid="stToolbar"],
[data-testid="stSidebar"]{display:none!important;}
html,body,[data-testid="stAppViewContainer"],
[data-testid="stMain"],section.main,
.main .block-container{
  background-color:var(--black)!important;
  color:var(--white)!important;
  font-family:var(--sans)!important;
  padding:0!important;margin:0!important;max-width:100%!important;
}
[data-testid="block-container"]{padding:0!important;max-width:100%!important;}
[data-testid="stVerticalBlock"]>div{gap:0!important;}
/* Tabs → sticky nav feel */
[data-testid="stTabs"]{
  position:sticky!important;top:0!important;z-index:900!important;
  background:rgba(10,10,8,0.97)!important;
  backdrop-filter:blur(20px)!important;-webkit-backdrop-filter:blur(20px)!important;
  border-bottom:1px solid var(--edge)!important;
  padding:0 48px!important;margin:0!important;
}
[data-testid="stTabs"] button{
  background:transparent!important;color:var(--dim)!important;
  font-family:var(--mono)!important;font-size:0.59rem!important;
  letter-spacing:0.14em!important;text-transform:uppercase!important;
  border:none!important;border-bottom:1px solid transparent!important;
  padding:.95rem 1rem!important;border-radius:0!important;transition:color .2s!important;
}
[data-testid="stTabs"] button:hover{color:var(--white)!important;background:transparent!important;}
[data-testid="stTabs"] button[aria-selected="true"]{
  color:var(--white)!important;border-bottom:1px solid var(--white)!important;background:transparent!important;}
[data-testid="stTabsContent"]{border:none!important;padding:0!important;background:var(--black)!important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{background:transparent!important;}
/* Buttons */
[data-testid="stButton"]>button{
  background:transparent!important;color:var(--dim)!important;
  border:1px solid var(--edge2)!important;border-radius:2px!important;
  font-family:var(--mono)!important;font-size:0.64rem!important;
  letter-spacing:.1em!important;text-transform:uppercase!important;
  padding:.55rem 1.1rem!important;transition:all .15s!important;}
[data-testid="stButton"]>button:hover{border-color:var(--white2)!important;color:var(--white)!important;background:rgba(245,242,235,.04)!important;}
[data-testid="stButton"]>button[kind="primary"]{background:var(--white)!important;color:var(--black)!important;border-color:var(--white)!important;font-weight:500!important;}
[data-testid="stButton"]>button[kind="primary"]:hover{background:var(--white2)!important;}
/* Download */
[data-testid="stDownloadButton"]>button{
  background:transparent!important;color:var(--gold)!important;
  border:1px solid rgba(200,169,74,.3)!important;border-radius:2px!important;
  font-family:var(--mono)!important;font-size:0.64rem!important;
  letter-spacing:.1em!important;text-transform:uppercase!important;transition:all .15s!important;}
[data-testid="stDownloadButton"]>button:hover{background:rgba(200,169,74,.07)!important;border-color:var(--gold)!important;}
/* Inputs */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{
  background:var(--black2)!important;border:1px solid var(--edge2)!important;
  border-radius:2px!important;color:var(--white)!important;
  font-family:var(--sans)!important;font-size:.88rem!important;font-weight:300!important;}
[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{
  border-color:var(--white2)!important;box-shadow:none!important;}
[data-testid="stTextInput"] label,[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label,[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label{
  color:var(--dim)!important;font-family:var(--mono)!important;
  font-size:.57rem!important;letter-spacing:.14em!important;text-transform:uppercase!important;}
[data-testid="stSelectbox"]>div>div{
  background:var(--black2)!important;border:1px solid var(--edge2)!important;
  border-radius:2px!important;color:var(--white)!important;}
[data-testid="stMultiSelect"]>div>div{background:var(--black2)!important;border:1px solid var(--edge2)!important;}
[data-testid="stMultiSelect"] span[data-baseweb="tag"]{
  background:var(--black3)!important;color:var(--gold)!important;
  border:1px solid var(--edge2)!important;font-family:var(--mono)!important;
  font-size:.64rem!important;border-radius:2px!important;}
[data-testid="stSlider"]>div>div>div{background:var(--edge2)!important;}
[data-testid="stSlider"]>div>div>div>div{background:var(--white)!important;}
/* Metrics */
[data-testid="stMetric"]{background:var(--black2)!important;border:1px solid var(--edge)!important;border-radius:4px!important;padding:1.4rem 1.6rem!important;}
[data-testid="stMetricLabel"]{color:var(--dim)!important;font-family:var(--mono)!important;font-size:.57rem!important;letter-spacing:.14em!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{color:var(--white)!important;font-family:var(--serif)!important;font-size:2.2rem!important;font-weight:300!important;}
[data-testid="stMetricDelta"]{color:var(--dim)!important;font-size:.7rem!important;}
/* Alerts */
[data-testid="stAlert"]{background:var(--black2)!important;border-left:2px solid var(--edge2)!important;color:var(--dim)!important;font-size:.82rem!important;border-radius:0 2px 2px 0!important;}
[data-testid="stSuccess"]{border-left-color:#4a7c5a!important;}
[data-testid="stWarning"]{border-left-color:var(--gold)!important;}
/* Code */
pre,[data-testid="stCode"]{background:#070706!important;border:1px solid var(--edge)!important;border-radius:4px!important;font-family:var(--mono)!important;font-size:.76rem!important;}
code{color:var(--gold2)!important;}
/* Expander */
[data-testid="stExpander"]{background:transparent!important;border:1px solid var(--edge)!important;border-radius:2px!important;}
[data-testid="stExpander"] summary{font-family:var(--mono)!important;font-size:.62rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--dim)!important;}
/* Divider */
hr{border:none!important;border-top:1px solid var(--edge)!important;margin:0!important;}
/* Typography */
h1,h2,h3{font-family:var(--serif)!important;font-weight:300!important;color:var(--white)!important;}
p{color:var(--white2)!important;font-weight:300!important;line-height:1.75!important;}
/* Form */
[data-testid="stForm"]{background:var(--black2)!important;border:1px solid var(--edge)!important;border-radius:4px!important;padding:2rem!important;}
[data-testid="stFormSubmitButton"]>button{
  background:var(--white)!important;color:var(--black)!important;border:none!important;
  font-family:var(--mono)!important;font-size:.64rem!important;letter-spacing:.1em!important;
  text-transform:uppercase!important;padding:.7rem 2rem!important;
  border-radius:2px!important;width:100%!important;}
/* Chat */
[data-testid="stChatMessage"]{background:var(--black2)!important;border:1px solid var(--edge)!important;border-radius:4px!important;}
[data-testid="stChatInput"] textarea{background:var(--black2)!important;border:1px solid var(--edge2)!important;color:var(--white)!important;border-radius:2px!important;}
[data-testid="stImage"] img{border-radius:2px!important;}
[data-testid="stDataFrame"]{border:1px solid var(--edge)!important;border-radius:4px!important;overflow:hidden!important;}
/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:var(--black);}
::-webkit-scrollbar-thumb{background:var(--edge2);border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nav():
    co = st.session_state.company or "NovaTech"
    st.markdown(f"""
<div style="position:sticky;top:0;z-index:999;background:rgba(10,10,8,0.97);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid #1A1A17;padding:0 48px;height:64px;
  display:flex;align-items:center;justify-content:space-between;
  font-family:'DM Mono',monospace;">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:30px;height:30px;overflow:hidden;display:flex;align-items:center">
      {_logo_scaled(30)}
    </div>
    <span style="font-size:.66rem;letter-spacing:.22em;color:#F5F2EB;text-transform:uppercase">
      {co} AI
    </span>
  </div>
  <div style="display:flex;align-items:center;gap:2rem">
    <a href="#configure"  style="font-size:.56rem;letter-spacing:.12em;color:#6A6A64;text-decoration:none;text-transform:uppercase">Configure</a>
    <a href="#logo"       style="font-size:.56rem;letter-spacing:.12em;color:#6A6A64;text-decoration:none;text-transform:uppercase">Logo &amp; Font</a>
    <a href="#slogans"    style="font-size:.56rem;letter-spacing:.12em;color:#6A6A64;text-decoration:none;text-transform:uppercase">Slogans</a>
    <a href="#campaign"   style="font-size:.56rem;letter-spacing:.12em;color:#6A6A64;text-decoration:none;text-transform:uppercase">Campaign</a>
    <a href="#analytics"  style="font-size:.56rem;letter-spacing:.12em;color:#6A6A64;text-decoration:none;text-transform:uppercase">Analytics</a>
    <a href="#kit"        style="font-size:.56rem;letter-spacing:.12em;color:#F5F2EB;text-decoration:none;text-transform:uppercase;border:1px solid #2C2C29;padding:.4rem .9rem;border-radius:2px">Brand Kit ↓</a>
  </div>
</div>""", unsafe_allow_html=True)


def _sec_open(anchor, eyebrow, headline, sub="", bg="#0A0A08"):
    bdr = "border-top:1px solid #1A1A17;" if anchor != "hero" else ""
    st.markdown(f"""
<section id="{anchor}" style="background:{bg};{bdr}padding:88px 48px 56px">
<div style="max-width:1140px;margin:0 auto">
<div style="font-family:'DM Mono',monospace;font-size:.57rem;letter-spacing:.18em;
  color:#3A3A36;text-transform:uppercase;margin-bottom:18px">{eyebrow}</div>
<h2 style="font-family:'Playfair Display',Georgia,serif;
  font-size:clamp(1.9rem,3.5vw,3rem);font-weight:300;color:#F5F2EB;
  letter-spacing:-.02em;line-height:1.1;margin:0 0 18px">{headline}</h2>
{"<p style='font-size:.94rem;color:#6A6A64;font-weight:300;max-width:520px;line-height:1.75;margin:0 0 44px'>"+sub+"</p>" if sub else "<div style='height:44px'></div>"}
""", unsafe_allow_html=True)


def _sec_close():
    st.markdown("</div></section>", unsafe_allow_html=True)


def _lbl(text):
    st.markdown(f"""<div style="font-family:'DM Mono',monospace;font-size:.57rem;
  letter-spacing:.14em;text-transform:uppercase;color:#6A6A64;
  margin-bottom:4px;margin-top:20px">{text}</div>""", unsafe_allow_html=True)


def _divider_label(text):
    st.markdown(f"""<div style="display:flex;align-items:center;gap:16px;margin:36px 0 24px">
  <div style="flex:1;height:1px;background:#1A1A17"></div>
  <div style="font-family:'DM Mono',monospace;font-size:.57rem;letter-spacing:.14em;
    text-transform:uppercase;color:#3A3A36">{text}</div>
  <div style="flex:1;height:1px;background:#1A1A17"></div>
</div>""", unsafe_allow_html=True)


def _swatch_row(palette):
    if not palette: return
    cols = st.columns(len(palette), gap="small")
    for col, sw in zip(cols, palette):
        with col:
            st.markdown(f"""
<div style="background:{sw['hex']};height:60px;border-radius:2px;margin-bottom:6px"></div>
<div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#C8A94A;letter-spacing:.08em">{sw['hex'].upper()}</div>
<div style="font-family:'DM Mono',monospace;font-size:.5rem;color:#6A6A64;text-transform:uppercase;margin-top:1px">{sw['role']}</div>
<div style="font-size:.72rem;color:#C8C4BB;font-weight:300;margin-top:2px">{sw['name']}</div>
""", unsafe_allow_html=True)


def _metric_row(items: list[tuple[str,str,str]]):
    cols = st.columns(len(items), gap="small")
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label=label, value=value, delta=delta or None)


# ═══════════════════════════════════════════════════════════════════════════════
# STICKY NAV
# ═══════════════════════════════════════════════════════════════════════════════
_nav()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<section id="hero" style="min-height:94vh;background:#0A0A08;
  display:flex;flex-direction:column;justify-content:flex-end;
  padding:0 48px 80px;position:relative;overflow:hidden;">
  <div style="position:absolute;inset:0;pointer-events:none;opacity:.015;
    background-image:url('data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22200%22 height=%22200%22><filter id=%22n%22><feTurbulence type=%22fractalNoise%22 baseFrequency=%220.9%22 numOctaves=%224%22/></filter><rect width=%22200%22 height=%22200%22 filter=%22url(%23n)%22/></svg>');"></div>
  <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);
    opacity:.03;pointer-events:none;width:600px;height:600px;">
    {_logo_scaled(600)}
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.58rem;letter-spacing:.2em;
    color:#6A6A64;text-transform:uppercase;margin-bottom:22px">
    ✦ &nbsp; CRS AI Capstone 2025–26 · Scenario 1
  </div>
  <h1 style="font-family:'Playfair Display',Georgia,serif;
    font-size:clamp(3.2rem,7vw,6.8rem);font-weight:300;color:#F5F2EB;
    line-height:1.0;letter-spacing:-.03em;margin:0 0 28px;max-width:900px;">
    Build your brand<br>with <em style="color:#C8A94A">intelligence.</em>
  </h1>
  <p style="font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:300;
    color:#6A6A64;max-width:460px;line-height:1.75;margin:0 0 44px;">
    CNN · KNN · KMeans · Gradient Boosting · Gemini AI.<br>
    From logo concept to global campaign — one platform.
  </p>
  <div style="display:flex;gap:14px;align-items:center">
    <a href="#configure" style="font-family:'DM Mono',monospace;font-size:.61rem;
      letter-spacing:.1em;text-transform:uppercase;color:#0A0A08;background:#F5F2EB;
      padding:.75rem 2rem;border-radius:2px;text-decoration:none;">Get started</a>
    <a href="#analytics" style="font-family:'DM Mono',monospace;font-size:.61rem;
      letter-spacing:.1em;text-transform:uppercase;color:#6A6A64;
      padding:.75rem 0;text-decoration:none;border-bottom:1px solid #3A3A36;">
      View analytics ↓</a>
  </div>
  <div style="position:absolute;bottom:28px;right:48px;font-family:'DM Mono',monospace;
    font-size:.52rem;letter-spacing:.18em;color:#2C2C29;text-transform:uppercase;
    writing-mode:vertical-rl;">Scroll to explore</div>
</section>""", unsafe_allow_html=True)

# ── Platform capabilities strip ───────────────────────────────────────────────
st.markdown("""
<section style="background:#111110;border-top:1px solid #1A1A17;
  border-bottom:1px solid #1A1A17;padding:0 48px">
  <div style="display:grid;grid-template-columns:repeat(5,1fr);max-width:1140px;margin:0 auto">
    <div style="padding:32px 20px;border-right:1px solid #1A1A17">
      <div style="font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.14em;color:#3A3A36;text-transform:uppercase;margin-bottom:8px">Week 02–03</div>
      <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">Logo &amp; Typography</div>
      <div style="font-size:.75rem;color:#3A3A36;font-weight:300;line-height:1.5">SVG concepts · KNN font engine</div>
    </div>
    <div style="padding:32px 20px;border-right:1px solid #1A1A17">
      <div style="font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.14em;color:#3A3A36;text-transform:uppercase;margin-bottom:8px">Week 04–05</div>
      <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">Slogans &amp; Palette</div>
      <div style="font-size:.75rem;color:#3A3A36;font-weight:300;line-height:1.5">TF-IDF + Gemini · KMeans</div>
    </div>
    <div style="padding:32px 20px;border-right:1px solid #1A1A17">
      <div style="font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.14em;color:#3A3A36;text-transform:uppercase;margin-bottom:8px">Week 06–07</div>
      <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">Animation &amp; Campaign</div>
      <div style="font-size:.75rem;color:#3A3A36;font-weight:300;line-height:1.5">Matplotlib GIF · Gradient Boosting</div>
    </div>
    <div style="padding:32px 20px;border-right:1px solid #1A1A17">
      <div style="font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.14em;color:#3A3A36;text-transform:uppercase;margin-bottom:8px">Week 08–09</div>
      <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">Multilingual &amp; Feedback</div>
      <div style="font-size:.75rem;color:#3A3A36;font-weight:300;line-height:1.5">10 languages · VADER sentiment</div>
    </div>
    <div style="padding:32px 20px">
      <div style="font-family:'DM Mono',monospace;font-size:.52rem;letter-spacing:.14em;color:#3A3A36;text-transform:uppercase;margin-bottom:8px">Week 10</div>
      <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">Brand Kit Export</div>
      <div style="font-size:.75rem;color:#3A3A36;font-weight:300;line-height:1.5">ZIP · Markdown report · Deploy</div>
    </div>
  </div>
</section>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURE
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("configure","Step 01 · Foundation","Configure your brand.",
          "Set your company details once — every module below uses them automatically.")

c1, c2 = st.columns([1,1], gap="large")
with c1:
    _lbl("Company Name")
    co = st.text_input("co", value=st.session_state.company,
                        placeholder="e.g. NovaTech", label_visibility="collapsed")
    st.session_state.company = co.strip() or "NovaTech"

    _lbl("Target Audience")
    aud = st.text_input("aud", value=st.session_state.audience,
                         placeholder="e.g. B2B SaaS founders, age 28–45",
                         label_visibility="collapsed")
    st.session_state.audience = aud

    _lbl("Target Region")
    reg = st.selectbox("reg", REGIONS,
                        index=REGIONS.index(st.session_state.region) if st.session_state.region in REGIONS else 0,
                        label_visibility="collapsed")
    st.session_state.region = reg

with c2:
    _lbl("Product / Service Description")
    desc = st.text_area("desc", value=st.session_state.description,
                         placeholder="Describe what your brand does in 2–3 sentences…",
                         height=160, label_visibility="collapsed")
    st.session_state.description = desc

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
_lbl("Industry")
ind_cols = st.columns(5, gap="small")
for i, ind in enumerate(INDUSTRIES[:10]):
    with ind_cols[i % 5]:
        active = st.session_state.industry == ind
        if st.button(ind, key=f"ind_{ind}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.industry = ind
            st.rerun()

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
_lbl("Brand Tone")
tone_cols = st.columns(5, gap="small")
for i, tone in enumerate(TONES[:10]):
    with tone_cols[i % 5]:
        active = st.session_state.tone == tone
        if st.button(tone, key=f"tone_{tone}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.tone = tone
            st.rerun()

# Persona card
if st.session_state.company and st.session_state.industry and st.session_state.tone:
    persona = derive_persona(
        st.session_state.company, st.session_state.industry,
        st.session_state.tone, st.session_state.description, st.session_state.audience)
    st.markdown(f"""
<div style="margin-top:32px;display:flex;align-items:flex-start;gap:16px;
  padding:22px 26px;background:#111110;border:1px solid #1A1A17;border-radius:2px">
  <div style="width:7px;height:7px;background:#F5F2EB;border-radius:50%;flex-shrink:0;margin-top:5px"></div>
  <div style="flex:1">
    <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:300;color:#F5F2EB;margin-bottom:4px">
      {st.session_state.company} → <em>{persona.persona}</em>
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#6A6A64;letter-spacing:.1em">
      {persona.recommended_industry} · {persona.recommended_tone} · {st.session_state.region}
    </div>
    <div style="font-size:.8rem;color:#3A3A36;margin-top:8px;font-weight:300">
      Keywords: {', '.join(persona.brand_keywords[:5]) or '—'}
    </div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.56rem;color:#6A6A64;letter-spacing:.1em;flex-shrink:0">
    Configured ✓
  </div>
</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGO & FONT
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("logo","Step 02–03 · Identity","Logo studio &amp; typography.",
          "Five SVG logo concepts generated from your brand inputs. Font pairing via tone-industry mapping.",
          bg="#0D0D0B")

# Generate logos + font
if st.button("✦ Generate Logo Concepts & Font Pairing", type="primary", key="btn_logo"):
    with st.spinner("Generating…"):
        pal = recommend_palette(st.session_state.industry, st.session_state.tone)
        st.session_state.palette = pal.swatches
        logos = generate_logo_concepts(
            st.session_state.company, st.session_state.tone, pal.swatches)
        st.session_state.logo_concepts = logos
        font  = recommend_fonts(st.session_state.industry, st.session_state.tone)
        st.session_state.font = vars(font)
    st.rerun()

if st.session_state.logo_concepts:
    _divider_label("Logo Concepts — SVG · Rule-based composition")
    cols = st.columns(len(st.session_state.logo_concepts), gap="small")
    for col, lc in zip(cols, st.session_state.logo_concepts):
        with col:
            bg   = st.session_state.palette[0]["hex"] if st.session_state.palette else "#0A0A08"
            # Invert SVG background to show correctly
            svg_display = lc.svg.replace('fill="none"','fill="transparent"')
            st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;
  padding:20px;text-align:center;cursor:pointer" onclick="">
  {svg_display}
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;
    letter-spacing:.1em;text-transform:uppercase;margin-top:10px">{lc.name}</div>
  <div style="font-size:.75rem;color:#3A3A36;margin-top:4px;font-weight:300;line-height:1.4">
    {lc.description[:80]}…</div>
  <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#C8A94A;margin-top:8px">
    Match: {int(lc.tone_match*100)}%</div>
</div>""", unsafe_allow_html=True)
            if st.button(f"Select", key=f"sel_logo_{lc.style}", use_container_width=True):
                st.session_state.selected_logo_style = lc.style
                st.rerun()

if st.session_state.font:
    _divider_label("Font Recommendation — Mapping-based, not CNN-trained")
    f = st.session_state.font
    fc1, fc2, fc3 = st.columns(3, gap="large")
    with fc1:
        st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:22px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px">Heading Font</div>
  <div style="font-size:1.6rem;font-weight:300;color:#F5F2EB;margin-bottom:6px">{f['heading']}</div>
  <a href="{f['heading_url']}" target="_blank" style="font-family:'DM Mono',monospace;font-size:.54rem;color:#C8A94A;text-decoration:none;letter-spacing:.08em">View on Google Fonts ↗</a>
</div>""", unsafe_allow_html=True)
    with fc2:
        st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:22px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px">Body Font</div>
  <div style="font-size:1.6rem;font-weight:300;color:#F5F2EB;margin-bottom:6px">{f['body']}</div>
  <a href="{f['body_url']}" target="_blank" style="font-family:'DM Mono',monospace;font-size:.54rem;color:#C8A94A;text-decoration:none;letter-spacing:.08em">View on Google Fonts ↗</a>
</div>""", unsafe_allow_html=True)
    with fc3:
        st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:22px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em;text-transform:uppercase;margin-bottom:12px">Accent Font</div>
  <div style="font-size:1.6rem;font-weight:300;color:#F5F2EB;margin-bottom:6px">{f['accent']}</div>
  <a href="{f['accent_url']}" target="_blank" style="font-family:'DM Mono',monospace;font-size:.54rem;color:#C8A94A;text-decoration:none;letter-spacing:.08em">View on Google Fonts ↗</a>
</div>""", unsafe_allow_html=True)
    st.markdown(f"""
<div style="margin-top:16px;padding:16px 20px;background:#111110;border-left:2px solid #C8A94A;
  border-radius:0 2px 2px 0">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">Style · {f['style']}</div>
  <div style="font-size:.85rem;color:#C8C4BB;font-weight:300;line-height:1.65">{f['rationale']}</div>
</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PALETTE + SLOGANS
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("slogans","Step 04–05 · Voice &amp; Colour",
          "Slogans &amp; colour palette.",
          "TF-IDF retrieval + template generation + optional Gemini refinement. "
          "KMeans-ready palette with WCAG harmony scoring.")

col_sl, col_pal = st.columns([3,2], gap="large")

with col_sl:
    _lbl("Generate Slogans")
    if st.button("✦ Generate Slogans", type="primary", key="btn_slogans"):
        with st.spinner("Generating slogans…"):
            candidates = generate_slogans(
                st.session_state.company, st.session_state.industry,
                st.session_state.tone, st.session_state.audience,
                st.session_state.description, n_total=5)
            st.session_state.slogans = [c.text for c in candidates]
            st.session_state._slogan_meta = [vars(c) for c in candidates]
        st.rerun()

    if st.session_state.slogans:
        for i, sl in enumerate(st.session_state.slogans):
            meta = (st.session_state.get("_slogan_meta") or [{}]*5)[i]
            source_tag = {"gemini":"✦ Gemini","template":"◈ Template","retrieval":"⟳ Retrieved"}.get(meta.get("source",""), "·")
            active = st.session_state.selected_slogan == sl
            st.markdown(f"""
<div style="padding:16px 20px;margin-bottom:8px;background:{'#1A1A17' if active else '#111110'};
  border:1px solid {'#C8A94A' if active else '#1A1A17'};border-radius:2px;
  cursor:pointer;transition:all .15s">
  <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:300;
    color:#F5F2EB;margin-bottom:6px">"{sl}"</div>
  <div style="display:flex;gap:16px;align-items:center">
    <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#6A6A64;letter-spacing:.08em">{source_tag}</div>
    <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#3A3A36">{meta.get('tone','')}</div>
    <div style="font-family:'DM Mono',monospace;font-size:.52rem;color:#C8A94A">{int(meta.get('confidence',0)*100)}% match</div>
  </div>
</div>""", unsafe_allow_html=True)
            if st.button("Select", key=f"sel_slogan_{i}", use_container_width=True):
                st.session_state.selected_slogan = sl
                st.session_state.tagline         = sl
                st.rerun()

with col_pal:
    _lbl("Colour Palette")
    if not st.session_state.palette and st.session_state.industry:
        pal_r = recommend_palette(st.session_state.industry, st.session_state.tone)
        st.session_state.palette = pal_r.swatches

    if st.session_state.palette:
        _swatch_row(st.session_state.palette)

        pal_r = recommend_palette(st.session_state.industry, st.session_state.tone)
        harmony = pal_r.harmony_score
        st.markdown(f"""
<div style="margin-top:18px;padding:14px 18px;background:#111110;border:1px solid #1A1A17;border-radius:2px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">Harmony Score</div>
  <div style="font-family:'Playfair Display',serif;font-size:2rem;font-weight:300;color:#F5F2EB">{int(harmony*100)}/100</div>
  <div style="font-size:.78rem;color:#6A6A64;margin-top:4px;font-weight:300">{pal_r.recommendation}</div>
</div>""", unsafe_allow_html=True)

    _lbl("Upload Logo for Palette Extraction (optional)")
    uploaded = st.file_uploader("logo_upload", type=["png","jpg","jpeg","svg"],
                                 label_visibility="collapsed")
    if uploaded:
        with st.spinner("Extracting palette…"):
            extracted = extract_palette_from_image(uploaded.read())
            st.session_state.palette = extracted.swatches
        st.success("Palette extracted from your image.")
        st.rerun()

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("animation","Step 06 · Motion","Animation studio.",
          "Matplotlib FuncAnimation GIF export. Six styles from fade-in to typewriter.",
          bg="#0D0D0B")

ac1, ac2 = st.columns([1,2], gap="large")
with ac1:
    _lbl("Animation Style")
    anim_style = st.selectbox("anim_style", ANIMATION_STYLES, label_visibility="collapsed")
    _lbl("Preview Slogan")
    anim_slogan = st.text_input("anim_slogan",
        value=st.session_state.selected_slogan or st.session_state.company,
        label_visibility="collapsed")
    if st.button("✦ Generate Animation", type="primary", key="btn_anim"):
        with st.spinner("Rendering GIF…"):
            gif, b64 = get_animation_gif(
                st.session_state.company,
                anim_slogan,
                st.session_state.palette or [{"hex":"#0A0A08"},{"hex":"#F5F2EB"},{"hex":"#C8A94A"}],
                anim_style,
            )
            st.session_state.animation_gif = gif
            st.session_state.animation_b64 = b64
        st.rerun()

with ac2:
    if st.session_state.animation_gif:
        st.markdown("**Preview**")
        b64 = base64.b64encode(st.session_state.animation_gif).decode()
        st.markdown(f'<img src="data:image/gif;base64,{b64}" style="border-radius:2px;width:100%;border:1px solid #1A1A17">', unsafe_allow_html=True)
        st.download_button("⬇ Download GIF", st.session_state.animation_gif,
                            f"{st.session_state.company.lower()}_animation.gif","image/gif")
    elif st.session_state.animation_b64:
        st.markdown("**Preview (Static Fallback)**")
        st.markdown(f'<img src="data:image/png;base64,{st.session_state.animation_b64}" style="border-radius:2px;width:100%;border:1px solid #1A1A17">', unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;
  padding:48px;text-align:center;color:#3A3A36;font-family:'DM Mono',monospace;
  font-size:.6rem;letter-spacing:.12em;text-transform:uppercase">
  Animation preview will appear here
</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CAMPAIGN
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("campaign","Step 07–08 · Reach",
          "Campaign analytics &amp; social content.",
          "Gradient Boosting prediction trained on real campaign data. "
          "Platform-specific caption generation for 8 channels.")

cc1, cc2 = st.columns([1,1], gap="large")
with cc1:
    _lbl("Channel / Platform")
    plat_sel = st.selectbox("camp_platform", PLATFORMS, label_visibility="collapsed")
    _lbl("Campaign Type")
    camp_type = st.selectbox("camp_type", CAMPAIGN_TYPES, label_visibility="collapsed")
    _lbl("Target Audience Segment")
    aud_seg   = st.selectbox("camp_aud", AUDIENCE_SEGMENTS, label_visibility="collapsed")
    _lbl("Campaign Language")
    camp_lang = st.selectbox("camp_lang", LANGUAGES, label_visibility="collapsed")

with cc2:
    _lbl("Duration (days)")
    duration  = st.slider("camp_dur", 7, 120, 30, label_visibility="collapsed")
    _lbl("Budget (USD)")
    budget    = st.slider("camp_budget", 500, 100000, 10000, step=500, label_visibility="collapsed")
    _lbl("Region")
    camp_reg  = st.selectbox("camp_reg", REGIONS,
                              index=REGIONS.index(st.session_state.region) if st.session_state.region in REGIONS else 0,
                              label_visibility="collapsed")

if st.button("✦ Predict Performance & Generate Content", type="primary", key="btn_campaign"):
    with st.spinner("Running prediction…"):
        pred = predict_campaign(
            channel=plat_sel, campaign_type=camp_type,
            region=camp_reg, audience=aud_seg,
            duration_days=duration, budget=budget, language=camp_lang)
        st.session_state.campaign_result = vars(pred)
    with st.spinner("Generating social content…"):
        packs = generate_content(
            company=st.session_state.company,
            industry=st.session_state.industry,
            tone=st.session_state.tone,
            tagline=st.session_state.selected_slogan or st.session_state.tagline,
            description=st.session_state.description,
            audience=aud_seg,
            campaign_type=camp_type,
            platforms=PLATFORMS[:4],
        )
        st.session_state.captions = [vars(p) for p in packs]
    st.rerun()

if st.session_state.campaign_result:
    r = st.session_state.campaign_result
    src_label = "ML Model" if r.get("source")=="model" else "Heuristic Engine"
    st.markdown(f"""
<div style="margin-bottom:24px;padding:10px 16px;background:#111110;border:1px solid #1A1A17;
  border-radius:2px;display:flex;gap:20px;align-items:center">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;letter-spacing:.1em">Prediction Source</div>
  <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#C8A94A">{src_label}</div>
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64">Confidence</div>
  <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#F5F2EB">{r.get('confidence','—').upper()}</div>
</div>""", unsafe_allow_html=True)
    _metric_row([
        ("Predicted ROI",          f"{r.get('roi',0):.2f}×",   None),
        ("Engagement Score",       f"{r.get('engagement_score',0):.1f}/10", None),
        ("Click-Through Rate",     f"{r.get('ctr',0)*100:.2f}%", None),
        ("Conversion Rate",        f"{r.get('conversion_rate',0)*100:.2f}%", None),
    ])
    for rec in r.get("recommendations",[]):
        st.markdown(f"""
<div style="margin-top:10px;padding:12px 16px;background:#111110;
  border-left:2px solid #C8A94A;border-radius:0 2px 2px 0;
  font-size:.82rem;color:#C8C4BB;font-weight:300">{rec}</div>""", unsafe_allow_html=True)

if st.session_state.captions:
    _divider_label("Social Captions — Template-based + optional Gemini")
    tabs = st.tabs([c["platform"] for c in st.session_state.captions])
    for tab, cap in zip(tabs, st.session_state.captions):
        with tab:
            ct1, ct2 = st.columns([2,1], gap="large")
            with ct1:
                st.text_area("Caption", cap["caption"], height=160,
                              key=f"cap_{cap['platform']}", disabled=False)
                st.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;margin-top:4px">
  {cap['char_count']} chars · {len(cap.get('hashtags',[]))} hashtags
</div>""", unsafe_allow_html=True)
            with ct2:
                st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:16px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Hashtags</div>
  <div style="font-size:.8rem;color:#C8A94A;line-height:1.8;font-weight:300">
    {' '.join(cap.get('hashtags',[]))}</div>
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;
    text-transform:uppercase;letter-spacing:.1em;margin-top:12px;margin-bottom:6px">CTA</div>
  <div style="font-size:.82rem;color:#F5F2EB;font-weight:300">{cap['cta']}</div>
</div>""", unsafe_allow_html=True)
            st.markdown(f"""
<div style="margin-top:10px;padding:12px 16px;background:#111110;
  border-left:2px solid #3A3A36;border-radius:0 2px 2px 0;
  font-size:.78rem;color:#6A6A64;font-weight:300">
  📌 Strategy: {cap['strategy']}</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MULTILINGUAL
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("multilingual","Step 08 · Global Reach",
          "Multilingual studio.",
          "Gemini API → deep-translator → curated fallback. "
          "Translates slogans and captions into 8 languages.",
          bg="#0D0D0B")

ml1, ml2 = st.columns([1,1], gap="large")
with ml1:
    _lbl("Text to Translate")
    ml_text = st.text_area("ml_text",
        value=st.session_state.selected_slogan or st.session_state.company,
        height=100, label_visibility="collapsed")
    _lbl("Target Languages")
    ml_langs = st.multiselect("ml_langs",
        ["Hindi","Spanish","French","German","Gujarati","Portuguese","Arabic","Japanese"],
        default=["Hindi","Spanish","French"],
        label_visibility="collapsed")

with ml2:
    if st.button("✦ Translate", type="primary", key="btn_translate"):
        with st.spinner("Translating…"):
            results = translate_text(ml_text, ml_langs, st.session_state.tone)
            st.session_state.translations = [vars(r) for r in results]
        st.rerun()

if st.session_state.translations:
    tcols = st.columns(min(len(st.session_state.translations), 3), gap="small")
    for i, tr in enumerate(st.session_state.translations):
        with tcols[i % 3]:
            src_color = {"gemini":"#C8A94A","deep_translator":"#5A9B6A","fallback":"#6A6A64"}.get(tr.get("source",""),"#6A6A64")
            st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:18px;margin-bottom:12px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;letter-spacing:.12em;
    text-transform:uppercase;color:#6A6A64;margin-bottom:8px">{tr['language']}</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:300;
    color:#F5F2EB;line-height:1.5;margin-bottom:10px">"{tr['text']}"</div>
  <div style="font-family:'DM Mono',monospace;font-size:.5rem;color:{src_color}">
    ◈ {tr.get('source','—').replace('_',' ').title()} · {tr.get('confidence','—').upper()}</div>
  {f"<div style='font-size:.72rem;color:#3A3A36;margin-top:6px;font-weight:300'>{tr.get('note','')}</div>" if tr.get('note') else ''}
</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("analytics","Step 01 · EDA — Dataset-backed",
          "Campaign analytics dashboard.",
          "Trained on marketing_campaign_dataset.csv. "
          f"{'Real data loaded.' if _mk_real else 'Synthetic fallback — upload dataset for real charts.'}")

from src.preprocess import clean_marketing
df_mk_clean = clean_marketing(df_marketing)

dash_tabs = st.tabs(["ROI","Engagement","CTR","Trends","Heatmap","EDA"])
with dash_tabs[0]:
    c1,c2 = st.columns(2,gap="small")
    with c1: st.plotly_chart(campaign_roi_by_channel(df_mk_clean),     use_container_width=True)
    with c2: st.plotly_chart(campaign_type_pie(df_mk_clean),           use_container_width=True)
with dash_tabs[1]:
    st.plotly_chart(engagement_by_campaign_type(df_mk_clean), use_container_width=True)
with dash_tabs[2]:
    c1,c2 = st.columns(2,gap="small")
    with c1: st.plotly_chart(ctr_distribution(df_mk_clean),            use_container_width=True)
    with c2: st.plotly_chart(acquisition_cost_by_channel(df_mk_clean), use_container_width=True)
with dash_tabs[3]:
    st.plotly_chart(roi_over_time(df_mk_clean), use_container_width=True)
with dash_tabs[4]:
    st.plotly_chart(channel_heatmap(df_mk_clean), use_container_width=True)
with dash_tabs[5]:
    _lbl("Slogan Dataset")
    if not df_slogans.empty and "slogan_len" in df_slogans.columns:
        st.plotly_chart(slogan_length_histogram(df_slogans), use_container_width=True)
    _lbl("Sample Marketing Data")
    st.dataframe(df_mk_clean.head(20), use_container_width=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — BRAND CONSISTENCY SCORE
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("score","Step 09a · Aesthetics",
          "Brand consistency score.",
          "Weighted rule-based scoring across palette, typography, slogans, and logo style.",
          bg="#0D0D0B")

if st.button("✦ Compute Brand Consistency Score", type="primary", key="btn_score"):
    with st.spinner("Scoring…"):
        bscore = compute_brand_score(
            tone=st.session_state.tone,
            industry=st.session_state.industry,
            palette=st.session_state.palette,
            font_style=st.session_state.font.get("style","") if st.session_state.font else "",
            slogans=st.session_state.slogans,
            logo_style=st.session_state.selected_logo_style,
        )
        st.session_state.brand_score = vars(bscore)
    st.rerun()

if st.session_state.brand_score:
    bs = st.session_state.brand_score
    sc1, sc2 = st.columns([1,2], gap="large")
    with sc1:
        grade_color = {"A+":"#C8A94A","A":"#C8A94A","B+":"#5A9B6A","B":"#5A9B6A",
                       "C":"#9B7A4A","D":"#9B4A4A"}.get(bs["grade"],"#6A6A64")
        st.markdown(f"""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;
  padding:32px;text-align:center">
  <div style="font-family:'Playfair Display',serif;font-size:5rem;font-weight:300;
    color:{grade_color};line-height:1">{bs['grade']}</div>
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;
    letter-spacing:.14em;text-transform:uppercase;margin-top:8px">Brand Score</div>
  <div style="font-family:'Playfair Display',serif;font-size:2.2rem;font-weight:300;
    color:#F5F2EB;margin-top:6px">{bs['overall']}/100</div>
</div>""", unsafe_allow_html=True)
    with sc2:
        st.plotly_chart(feedback_radar(bs["breakdown"]), use_container_width=True)

    col_s, col_i = st.columns(2, gap="large")
    with col_s:
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#6A6A64;
          letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px">Strengths</div>""",
          unsafe_allow_html=True)
        for s in bs.get("strengths",[]):
            st.markdown(f"""
<div style="padding:10px 14px;margin-bottom:6px;background:#111110;
  border-left:2px solid #4a7c5a;border-radius:0 2px 2px 0;
  font-size:.82rem;color:#C8C4BB;font-weight:300">✓ {s}</div>""", unsafe_allow_html=True)
    with col_i:
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:.57rem;color:#6A6A64;
          letter-spacing:.14em;text-transform:uppercase;margin-bottom:10px">Improvements</div>""",
          unsafe_allow_html=True)
        for imp in bs.get("improvements",[]):
            st.markdown(f"""
<div style="padding:10px 14px;margin-bottom:6px;background:#111110;
  border-left:2px solid #C8A94A;border-radius:0 2px 2px 0;
  font-size:.82rem;color:#C8C4BB;font-weight:300">→ {imp}</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("feedback","Step 09b · Intelligence",
          "Feedback &amp; model refinement.",
          "VADER sentiment analysis. Aggregated ratings per asset. "
          "Retraining readiness signal.")

fb_df = load_feedback()
fb_summary = compute_feedback_summary(fb_df)
fb_ready   = retraining_readiness(fb_df)

fb1, fb2 = st.columns([1,1], gap="large")
with fb1:
    _lbl("Rate Your Brand Assets")
    asset_type = st.selectbox("fb_asset", ["Logo","Slogan","Palette","Campaign Copy","Overall"],
                               label_visibility="collapsed")
    rating = st.slider("fb_rating", 1, 5, 4, label_visibility="collapsed",
                        help="1 = Poor, 5 = Excellent")
    comment = st.text_area("fb_comment", placeholder="Optional comment…",
                            height=80, label_visibility="collapsed")
    if st.button("✦ Submit Feedback", type="primary", key="btn_feedback"):
        key = f"{asset_type.lower()}_{st.session_state.session_id}"
        save_feedback(
            session_id=st.session_state.session_id,
            company=st.session_state.company,
            industry=st.session_state.industry,
            tone=st.session_state.tone,
            asset_type=asset_type.lower(),
            rating=rating, comment=comment,
        )
        st.session_state.feedback_submitted.add(key)
        st.success("Feedback saved. Thank you.")
        st.rerun()

with fb2:
    if fb_summary["total"] > 0:
        st.metric("Total Feedback", fb_summary["total"])
        st.metric("Average Rating",  f"{fb_summary['avg_rating']:.1f} / 5")
        if fb_summary.get("sentiment"):
            st.plotly_chart(feedback_sentiment_bar(fb_summary["sentiment"]),
                            use_container_width=True)
    ready_col = "#C8A94A" if fb_ready["ready"] else "#3A3A36"
    st.markdown(f"""
<div style="padding:14px 18px;background:#111110;border:1px solid #1A1A17;
  border-radius:2px;margin-top:12px">
  <div style="font-family:'DM Mono',monospace;font-size:.55rem;color:#6A6A64;
    letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">Retraining Readiness</div>
  <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:{ready_col}">
    {'● READY' if fb_ready['ready'] else '○ NOT READY'}</div>
  <div style="font-size:.78rem;color:#6A6A64;margin-top:4px;font-weight:300">
    {fb_ready['reason']}</div>
</div>""", unsafe_allow_html=True)

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — AI ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("assistant","Optional · AI Assistant",
          "Ask NovaTech AI anything.",
          "Gemini-powered assistant with full brand context.",
          bg="#0D0D0B")

if not st.session_state.chat_history:
    st.session_state.chat_history = [{"role":"assistant",
        "content":f"Hi — I'm the NovaTech AI assistant. I have your brand context: "
                  f"{st.session_state.company}, {st.session_state.industry}, {st.session_state.tone} tone. "
                  "Ask me anything about strategy, copywriting, or the AI models powering this platform."}]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask NovaTech AI…"):
    st.session_state.chat_history.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner(""):
            from src.config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS
            ctx = (f"Brand: {st.session_state.company}, Industry: {st.session_state.industry}, "
                   f"Tone: {st.session_state.tone}, Region: {st.session_state.region}.")
            reply = "Gemini API key not configured — add GEMINI_API_KEY to your secrets."
            if GEMINI_API_KEY:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=GEMINI_API_KEY)
                    for mn in GEMINI_FALLBACK_MODELS[:3]:
                        try:
                            m   = genai.GenerativeModel(mn,
                                    system_instruction=f"You are NovaTech AI, an expert branding assistant. Context: {ctx} Be concise and authoritative.")
                            res = m.generate_content(user_input)
                            reply = res.text.strip()
                            break
                        except Exception:
                            continue
                except Exception as e:
                    reply = f"Assistant unavailable: {e}"
            st.markdown(reply)
            st.session_state.chat_history.append({"role":"assistant","content":reply})

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — BRAND KIT EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
_sec_open("kit","Step 10 · Export",
          "Your brand kit.",
          "Everything generated — logos, palette, fonts, slogans, captions, "
          "translations, campaign prediction, and brand report — packaged into one ZIP.")

kit1, kit2 = st.columns([1,1], gap="large")
with kit1:
    # Summary of what will be included
    items = [
        ("Logo Concepts",    len(st.session_state.logo_concepts)),
        ("Slogans",          len(st.session_state.slogans)),
        ("Palette Swatches", len(st.session_state.palette)),
        ("Social Captions",  len(st.session_state.captions)),
        ("Translations",     len(st.session_state.translations)),
        ("Campaign Prediction", 1 if st.session_state.campaign_result else 0),
        ("Brand Score",      1 if st.session_state.brand_score else 0),
        ("Animation GIF",    1 if st.session_state.animation_gif else 0),
    ]
    for name, count in items:
        color = "#F5F2EB" if count > 0 else "#3A3A36"
        icon  = "✓" if count > 0 else "○"
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
  padding:10px 0;border-bottom:1px solid #1A1A17">
  <div style="font-size:.85rem;color:{color};font-weight:300">{icon} &nbsp;{name}</div>
  <div style="font-family:'DM Mono',monospace;font-size:.58rem;color:#6A6A64">{count} item{'s' if count!=1 else ''}</div>
</div>""", unsafe_allow_html=True)

with kit2:
    st.markdown("""
<div style="background:#111110;border:1px solid #1A1A17;border-radius:2px;padding:24px;
  text-align:center">
  <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:300;
    color:#F5F2EB;margin-bottom:8px">Ready to export</div>
  <div style="font-size:.82rem;color:#6A6A64;font-weight:300;margin-bottom:24px">
    Generate all brand assets before downloading for the most complete kit.
  </div>
</div>""", unsafe_allow_html=True)

    if st.button("✦ Build & Download Brand Kit ZIP", type="primary",
                  key="btn_kit", use_container_width=True):
        with st.spinner("Packaging brand kit…"):
            zip_bytes = build_brand_kit_zip(
                company=st.session_state.company,
                industry=st.session_state.industry,
                tone=st.session_state.tone,
                slogans=st.session_state.slogans,
                palette=st.session_state.palette,
                font=st.session_state.font or {},
                logo_svgs=[{"name":lc["name"],"svg":lc["svg"]}
                            for lc in [vars(lc) for lc in st.session_state.logo_concepts]
                            ] if st.session_state.logo_concepts else [],
                captions=st.session_state.captions,
                translations=st.session_state.translations,
                campaign=st.session_state.campaign_result or {},
                score_overall=(st.session_state.brand_score or {}).get("overall",0),
                score_grade=(st.session_state.brand_score or {}).get("grade","—"),
                breakdown=(st.session_state.brand_score or {}).get("breakdown",{}),
                strengths=(st.session_state.brand_score or {}).get("strengths",[]),
                improvements=(st.session_state.brand_score or {}).get("improvements",[]),
                animation_gif=st.session_state.animation_gif,
            )
        slug = st.session_state.company.lower().replace(" ","_")
        st.download_button(
            "⬇ Download Brand Kit",
            zip_bytes,
            f"{slug}_brand_kit.zip",
            "application/zip",
            use_container_width=True,
        )

_sec_close()


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
mk_label = "Real data" if _mk_real else "Synthetic fallback"
sl_label = "Real data" if _sl_real else "Synthetic fallback"
st.markdown(f"""
<footer style="background:#080807;border-top:1px solid #1A1A17;padding:40px 48px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="width:22px;height:22px;overflow:hidden">{_logo_scaled(22)}</div>
    <span style="font-family:'DM Mono',monospace;font-size:.58rem;
      letter-spacing:.2em;color:#3A3A36;text-transform:uppercase">NovaTech AI</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.52rem;
    color:#2C2C29;text-transform:uppercase;letter-spacing:.1em">
    CRS AI Capstone 2025–26 · Scenario 1
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:.52rem;
    color:#2C2C29;letter-spacing:.08em">
    Marketing: {mk_label} · Slogans: {sl_label} · Streamlit + Gemini + scikit-learn
  </div>
</footer>""", unsafe_allow_html=True)
