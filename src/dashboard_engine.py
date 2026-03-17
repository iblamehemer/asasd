"""
src/dashboard_engine.py
Reusable Plotly chart functions for the Streamlit analytics dashboards.
All functions return go.Figure objects; callers render with st.plotly_chart().
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── Shared theme ──────────────────────────────────────────────────────────────
_DARK_LAYOUT = dict(
    paper_bgcolor="#0A0A08",
    plot_bgcolor="#111110",
    font=dict(family="DM Sans, sans-serif", color="#C8C4BB", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1A1A17", linecolor="#2C2C29", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#1A1A17", linecolor="#2C2C29", tickfont=dict(size=10)),
)
_GOLD   = "#C8A94A"
_WHITE  = "#F5F2EB"
_DIM    = "#6A6A64"
_COLORS = ["#C8A94A","#4A7C9B","#5A9B6A","#9B5A4A","#7A5A9B","#9B7A4A"]


def campaign_roi_by_channel(df: pd.DataFrame) -> go.Figure:
    """Bar chart: mean ROI grouped by channel."""
    if df.empty or "Channel_Used" not in df.columns:
        return go.Figure()
    grp = df.groupby("Channel_Used")["ROI"].mean().sort_values(ascending=True).reset_index()
    fig = go.Figure(go.Bar(
        x=grp["ROI"], y=grp["Channel_Used"], orientation="h",
        marker=dict(color=_GOLD, opacity=0.85),
        text=grp["ROI"].round(2), textposition="outside",
        textfont=dict(color=_WHITE, size=10),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Mean ROI by Channel", font=dict(color=_WHITE, size=13)),
                      showlegend=False, height=320)
    return fig


def engagement_by_campaign_type(df: pd.DataFrame) -> go.Figure:
    """Box plot: engagement score distribution by campaign type."""
    if df.empty or "Campaign_Type" not in df.columns:
        return go.Figure()
    types = df["Campaign_Type"].unique()
    fig   = go.Figure()
    for i, ct in enumerate(types):
        vals = df[df["Campaign_Type"]==ct]["Engagement_Score"].dropna()
        fig.add_trace(go.Box(y=vals, name=ct, marker_color=_COLORS[i % len(_COLORS)],
                             line_color=_COLORS[i % len(_COLORS)], boxmean="sd"))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Engagement Score by Campaign Type",
                      font=dict(color=_WHITE,size=13)), showlegend=False, height=340)
    return fig


def ctr_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of CTR values."""
    if df.empty or "CTR" not in df.columns:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=df["CTR"].clip(0, 0.15), nbinsx=40,
        marker=dict(color=_GOLD, opacity=0.8, line=dict(color="#0A0A08", width=0.5)),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Click-Through Rate Distribution",
                      font=dict(color=_WHITE,size=13)),
                      xaxis_title="CTR", yaxis_title="Count", height=300)
    return fig


def roi_over_time(df: pd.DataFrame) -> go.Figure:
    """Line chart: monthly mean ROI trend."""
    if df.empty or "Date" not in df.columns or "ROI" not in df.columns:
        return go.Figure()
    df2 = df.copy()
    df2["Month"] = pd.to_datetime(df2["Date"], errors="coerce").dt.to_period("M").astype(str)
    monthly = df2.groupby("Month")["ROI"].mean().reset_index()
    fig = go.Figure(go.Scatter(
        x=monthly["Month"], y=monthly["ROI"],
        mode="lines+markers",
        line=dict(color=_GOLD, width=1.5),
        marker=dict(color=_GOLD, size=5),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="ROI Trend Over Time",
                      font=dict(color=_WHITE,size=13)),
                      xaxis_tickangle=-35, height=300)
    return fig


def campaign_type_pie(df: pd.DataFrame) -> go.Figure:
    """Donut chart: campaign type distribution."""
    if df.empty or "Campaign_Type" not in df.columns:
        return go.Figure()
    counts = df["Campaign_Type"].value_counts().reset_index()
    counts.columns = ["type","count"]
    fig = go.Figure(go.Pie(
        labels=counts["type"], values=counts["count"],
        hole=0.55, marker=dict(colors=_COLORS[:len(counts)]),
        textinfo="label+percent", textfont=dict(size=10, color=_WHITE),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Campaign Type Mix",
                      font=dict(color=_WHITE,size=13)),
                      showlegend=False, height=300)
    return fig


def channel_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: mean ROI by Channel × Campaign Type."""
    if df.empty:
        return go.Figure()
    try:
        pivot = df.groupby(["Channel_Used","Campaign_Type"])["ROI"].mean().unstack(fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[[0,"#111110"],[0.5,"#5A3A1A"],[1,_GOLD]],
            text=pivot.values.round(2), texttemplate="%{text}",
            textfont=dict(size=9, color=_WHITE), showscale=True,
        ))
        fig.update_layout(**_DARK_LAYOUT, title=dict(text="Mean ROI: Channel × Campaign Type",
                          font=dict(color=_WHITE,size=13)), height=360)
        return fig
    except Exception:
        return go.Figure()


def feedback_radar(breakdown: dict) -> go.Figure:
    """Radar chart for brand consistency score breakdown."""
    if not breakdown:
        return go.Figure()
    categories = list(breakdown.keys())
    values     = [breakdown[k] for k in categories]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=cats_closed, fill="toself",
        line=dict(color=_GOLD, width=1.5),
        fillcolor="rgba(200,169,74,0.12)",
    ))
    fig.update_layout(
        paper_bgcolor="#0A0A08", plot_bgcolor="#0A0A08",
        font=dict(color=_WHITE, size=10),
        polar=dict(
            bgcolor="#111110",
            radialaxis=dict(visible=True, range=[0,100], tickfont=dict(color=_DIM,size=8),
                            gridcolor="#1A1A17", linecolor="#2C2C29"),
            angularaxis=dict(tickfont=dict(color=_WHITE,size=10), gridcolor="#1A1A17"),
        ),
        title=dict(text="Brand Consistency Breakdown", font=dict(color=_WHITE,size=13)),
        margin=dict(l=60,r=60,t=50,b=40), height=360,
    )
    return fig


def feedback_sentiment_bar(sentiment: dict) -> go.Figure:
    """Stacked bar chart for feedback sentiment."""
    if not sentiment:
        return go.Figure()
    color_map = {"positive": "#5A9B6A", "neutral": _DIM, "negative": "#9B4A4A"}
    fig = go.Figure()
    for label, count in sentiment.items():
        fig.add_trace(go.Bar(
            name=label.capitalize(), x=[label], y=[count],
            marker_color=color_map.get(label, _GOLD),
        ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Feedback Sentiment",
                      font=dict(color=_WHITE,size=13)),
                      barmode="group", showlegend=True, height=280)
    return fig


def slogan_length_histogram(df: pd.DataFrame) -> go.Figure:
    """EDA: distribution of slogan word lengths."""
    if df.empty or "slogan_len" not in df.columns:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=df["slogan_len"], nbinsx=20,
        marker=dict(color=_GOLD, opacity=0.8, line=dict(color="#0A0A08",width=0.5)),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Slogan Word Count Distribution",
                      font=dict(color=_WHITE,size=13)),
                      xaxis_title="Word Count", yaxis_title="Frequency", height=280)
    return fig


def acquisition_cost_by_channel(df: pd.DataFrame) -> go.Figure:
    """Bar: mean acquisition cost by channel."""
    if df.empty or "Acquisition_Cost" not in df.columns:
        return go.Figure()
    grp = df.groupby("Channel_Used")["Acquisition_Cost"].mean().sort_values().reset_index()
    fig = go.Figure(go.Bar(
        x=grp["Channel_Used"], y=grp["Acquisition_Cost"],
        marker=dict(color=_COLORS[:len(grp)]),
        text=grp["Acquisition_Cost"].round(0).astype(int),
        textposition="outside", textfont=dict(color=_WHITE,size=10),
    ))
    fig.update_layout(**_DARK_LAYOUT, title=dict(text="Mean Acquisition Cost by Channel",
                      font=dict(color=_WHITE,size=13)),
                      yaxis_title="Cost ($)", height=320, showlegend=False)
    return fig
