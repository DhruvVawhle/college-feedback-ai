"""
Overview Dashboard View for College Feedback AI.
Displays high-level KPI cards, sentiment distribution, topic analysis,
and grounded representative feedback excerpts.
"""

import streamlit as st
import pandas as pd
from ui_styles import render_header, render_kpi_card, render_sentiment_badge
from ui_charts import create_sentiment_donut_chart, create_topic_sentiment_chart
from data_manager import calculate_dataset_kpis
import openrouter_client


def render_overview_view(df: pd.DataFrame, dataset_name: str):
    """Render the executive Overview Dashboard."""
    render_header(
        title="Overview Dashboard",
        subtitle="High-level sentiment metrics, topic distribution, and executive feedback health.",
        dataset_label=dataset_name,
        record_count=len(df)
    )

    kpis = calculate_dataset_kpis(df)

    # ---------------------------------------------------------
    # 1. Executive KPI Cards
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi_card(
            label="Total Feedback",
            value=f"{kpis['total']:,}",
            badge_text="100%",
            footer_text="Total verified student records",
            variant="blue"
        )

    with col2:
        render_kpi_card(
            label="Positive Feedback",
            value=f"{kpis['positive_count']:,}",
            badge_text=kpis['positive_pct'],
            footer_text="Favorable student sentiments",
            variant="pos"
        )

    with col3:
        render_kpi_card(
            label="Negative Feedback",
            value=f"{kpis['negative_count']:,}",
            badge_text=kpis['negative_pct'],
            footer_text="Identified complaints & concerns",
            variant="neg"
        )

    with col4:
        render_kpi_card(
            label="Neutral Feedback",
            value=f"{kpis['neutral_count']:,}",
            badge_text=kpis['neutral_pct'],
            footer_text="Moderate or acceptable feedback",
            variant="neu"
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # Net Sentiment Score Banner
    net_score = kpis["net_sentiment_score"]
    health_status = kpis["sentiment_health"]
    score_color = "#10B981" if net_score >= 15 else ("#38BDF8" if net_score >= -10 else "#F43F5E")

    banner_html = f"""
    <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 22px;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 44px; height: 44px; border-radius: 10px; background: rgba(59, 130, 246, 0.12); border: 1px solid rgba(59, 130, 246, 0.25); display: flex; align-items: center; justify-content: center; font-size: 20px;">
                📈
            </div>
            <div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em;">Net Sentiment Index</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">
                    Score: <span style="color: {score_color}; font-weight: 800;">{net_score:+.1f}</span> 
                    <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 500; margin-left: 8px;">({health_status})</span>
                </div>
            </div>
        </div>
        <div style="text-align: right; font-size: 0.8rem; color: #94A3B8;">
            Formula: <code>(Positive - Negative) / Total × 100</code>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Charts Row
    # ---------------------------------------------------------
    chart_left, chart_right = st.columns([1, 1.2])

    with chart_left:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Sentiment Distribution</div>
                <div class="content-card-subtitle">Overall proportion of positive, neutral, and negative reviews</div>
            """,
            unsafe_allow_html=True
        )
        donut_fig = create_sentiment_donut_chart(df)
        st.plotly_chart(donut_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_right:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Feedback by Topic & Sentiment</div>
                <div class="content-card-subtitle">Keyword-mapped category distribution across sentiments</div>
            """,
            unsafe_allow_html=True
        )
        topic_fig = create_topic_sentiment_chart(df)
        st.plotly_chart(topic_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. Grounded Feedback Highlights
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 12px;">
            <h3 style="font-size: 1.25rem; margin-bottom: 4px;">Executive Highlights</h3>
            <p style="font-size: 0.85rem; color: #94A3B8;">Actual excerpts directly sampled from the active dataset.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    high_col1, high_col2 = st.columns(2)

    with high_col1:
        st.markdown(
            """
            <div class="content-card" style="border-top: 3px solid #10B981;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #10B981; font-size: 0.95rem;">Key Student Appreciations</div>
                    <span class="badge-pos">Positive</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        pos_samples = df[df["sentiment"] == "positive"]["feedback"].dropna().drop_duplicates().head(3).tolist()
        if pos_samples:
            for item in pos_samples:
                st.markdown(
                    f"""
                    <div class="feedback-item-card">
                        <div class="feedback-item-text">"{item}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No positive records found in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)

    with high_col2:
        st.markdown(
            """
            <div class="content-card" style="border-top: 3px solid #F43F5E;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-weight: 700; color: #F43F5E; font-size: 0.95rem;">Critical Areas for Improvement</div>
                    <span class="badge-neg">Negative</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        neg_samples = df[df["sentiment"] == "negative"]["feedback"].dropna().drop_duplicates().head(3).tolist()
        if neg_samples:
            for item in neg_samples:
                st.markdown(
                    f"""
                    <div class="feedback-item-card">
                        <div class="feedback-item-text">"{item}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No negative records found in dataset.")
        st.markdown("</div>", unsafe_allow_html=True)
