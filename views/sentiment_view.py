"""
Sentiment Analysis View for College Feedback AI.
Detailed sentiment distribution, length comparisons, and interactive
real-time single feedback sentiment prediction with probability breakdown.
"""

import streamlit as st
import pandas as pd
from ui_styles import render_header, render_sentiment_badge
from ui_charts import create_sentiment_bar_chart, create_feedback_length_chart
from ml_engine import predict_sentiment_with_probabilities
import openrouter_client


def render_sentiment_view(df: pd.DataFrame, model, model_name: str, dataset_name: str):
    """Render in-depth sentiment analytics and single text classification studio."""
    render_header(
        title="Sentiment Analysis",
        subtitle="Detailed class distributions, verbosity comparisons, and real-time sentiment testing.",
        dataset_label=dataset_name,
        record_count=len(df)
    )

    # ---------------------------------------------------------
    # 1. Distribution & Length Analysis
    # ---------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Sentiment Proportions</div>
                <div class="content-card-subtitle">Exact counts and percentages per sentiment category</div>
            """,
            unsafe_allow_html=True
        )
        bar_fig = create_sentiment_bar_chart(df)
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Feedback Verbosity & Length</div>
                <div class="content-card-subtitle">Character count distribution across sentiment types</div>
            """,
            unsafe_allow_html=True
        )
        length_fig = create_feedback_length_chart(df)
        st.plotly_chart(length_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Statistical Breakdown Table
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="margin-top: 10px; margin-bottom: 8px;">
            <h3 style="font-size: 1.15rem; margin-bottom: 4px;">Class Metrics Summary</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_copy = df.copy()
    df_copy["word_count"] = df_copy["feedback"].astype(str).str.split().str.len()
    df_copy["char_count"] = df_copy["feedback"].astype(str).str.len()

    stats_list = []
    total = len(df_copy)
    for s in ["positive", "neutral", "negative"]:
        sub = df_copy[df_copy["sentiment"] == s]
        count = len(sub)
        pct = (count / total * 100) if total > 0 else 0
        avg_words = sub["word_count"].mean() if count > 0 else 0
        avg_chars = sub["char_count"].mean() if count > 0 else 0
        stats_list.append({
            "Sentiment Class": s.capitalize(),
            "Total Entries": count,
            "Dataset Share": f"{pct:.1f}%",
            "Avg Words / Feedback": f"{avg_words:.1f}",
            "Avg Characters / Feedback": f"{avg_chars:.1f}"
        })

    stats_df = pd.DataFrame(stats_list)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. Real-Time Single Feedback Tester
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="content-card" style="border: 1px solid rgba(59, 130, 246, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                <div>
                    <div class="content-card-title">🔍 Individual Feedback Analyzer</div>
                    <div class="content-card-subtitle">
                        Test new student feedback in real-time using the trained machine learning pipeline.
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(f"Currently active classifier: **{model_name}**")

    # Quick test suggestions
    st.markdown("<span style='font-size: 0.8rem; color: #94A3B8; font-weight: 600;'>QUICK EXAMPLES:</span>", unsafe_allow_html=True)
    c_ex1, c_ex2, c_ex3 = st.columns(3)
    preset_text = None
    with c_ex1:
        if st.button("🌟 Positive Example", use_container_width=True):
            preset_text = "The faculty members are very supportive and explain difficult concepts patiently."
    with c_ex2:
        if st.button("⚠️ Negative Example", use_container_width=True):
            preset_text = "The laboratory computers are extremely slow and the internet disconnects."
    with c_ex3:
        if st.button("⚖️ Neutral Example", use_container_width=True):
            preset_text = "The college facilities and course timetable are acceptable."

    if preset_text:
        st.session_state["single_test_input"] = preset_text

    default_val = st.session_state.get("single_test_input", "")

    test_input = st.text_area(
        "Enter or paste feedback text below:",
        value=default_val,
        height=90,
        placeholder="Type a student review or click an example above..."
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("⚡ Run Sentiment Prediction", use_container_width=True, type="primary")

    if analyze_clicked:
        if test_input.strip():
            result = predict_sentiment_with_probabilities(model, test_input)
            pred = result["sentiment"]
            probs = result["probabilities"]
            detected_topic = openrouter_client.find_topic(test_input)

            badge_html = render_sentiment_badge(pred)
            topic_badge = f'<span class="badge-topic">Topic: {detected_topic.capitalize()}</span>' if detected_topic else ""

            st.markdown(
                f"""
                <div style="background: #0B0F19; border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 16px; margin-top: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.85rem; color: #94A3B8; font-weight: 600;">CLASSIFICATION RESULT</span>
                        <div style="display: flex; gap: 8px;">
                            {topic_badge}
                            {badge_html}
                        </div>
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">
                        Predicted Sentiment: <strong style="text-transform: uppercase;">{pred}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Probabilities / confidence bars
            if probs:
                st.markdown("<div style='font-size: 0.82rem; color: #94A3B8; margin-top: 12px; margin-bottom: 6px;'>CONFIDENCE DISTRIBUTION:</div>", unsafe_allow_html=True)
                p_cols = st.columns(len(probs))
                color_map = {"positive": "#10B981", "neutral": "#38BDF8", "negative": "#F43F5E"}
                for i, (c_name, p_val) in enumerate(sorted(probs.items())):
                    with p_cols[i]:
                        c_color = color_map.get(c_name.lower(), "#3B82F6")
                        st.markdown(
                            f"""
                            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 10px; text-align: center;">
                                <div style="font-size: 0.75rem; text-transform: uppercase; color: #94A3B8;">{c_name}</div>
                                <div style="font-size: 1.2rem; font-weight: 700; color: {c_color};">{p_val:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
        else:
            st.warning("Please enter some feedback text to analyze.")

    st.markdown("</div>", unsafe_allow_html=True)
