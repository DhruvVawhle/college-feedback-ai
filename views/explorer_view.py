"""
Feedback Explorer View for College Feedback AI.
Provides rich search, multi-faceted filtering, sentiment badges,
table and card stream visualizations, and CSV export.
"""

import streamlit as st
import pandas as pd
from ui_styles import render_header, render_sentiment_badge
import openrouter_client


def render_explorer_view(df: pd.DataFrame, dataset_name: str):
    """Render the searchable, filterable Feedback Explorer."""
    render_header(
        title="Feedback Explorer",
        subtitle="Search, filter, inspect, and export college feedback records.",
        dataset_label=dataset_name,
        record_count=len(df)
    )

    # ---------------------------------------------------------
    # 1. Search and Filter Bar
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="content-card" style="padding: 16px 20px 10px 20px;">
            <div class="content-card-title" style="font-size: 0.95rem; margin-bottom: 12px;">Filter Controls</div>
        """,
        unsafe_allow_html=True
    )

    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])

    with f_col1:
        search_query = st.text_input(
            "Search Feedback Text",
            placeholder="Type keyword (e.g., teacher, library, internet, clean)...",
            label_visibility="collapsed"
        )

    with f_col2:
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            options=["All Sentiments", "Positive", "Negative", "Neutral"],
            label_visibility="collapsed"
        )

    with f_col3:
        all_topics = ["All Topics"] + [t.capitalize() for t in openrouter_client.TOPICS.keys()] + ["Other"]
        topic_filter = st.selectbox(
            "Topic Filter",
            options=all_topics,
            label_visibility="collapsed"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Apply Filters
    # ---------------------------------------------------------
    filtered_df = df.copy()

    # Annotate topics
    filtered_df["topic"] = filtered_df["feedback"].apply(
        lambda t: (openrouter_client.find_topic(str(t)) or "other").capitalize()
    )
    filtered_df["char_length"] = filtered_df["feedback"].astype(str).str.len()

    # Search filter
    if search_query.strip():
        q = search_query.strip().lower()
        filtered_df = filtered_df[filtered_df["feedback"].str.lower().str.contains(q, na=False)]

    # Sentiment filter
    if sentiment_filter != "All Sentiments":
        filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter.lower()]

    # Topic filter
    if topic_filter != "All Topics":
        filtered_df = filtered_df[filtered_df["topic"] == topic_filter]

    # ---------------------------------------------------------
    # 3. Results Bar & Controls
    # ---------------------------------------------------------
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        st.markdown(
            f"""
            <div style="font-size: 0.88rem; color: #94A3B8; padding-top: 6px;">
                Showing <strong>{len(filtered_df)}</strong> of <strong>{len(df)}</strong> records 
                ({(len(filtered_df)/len(df)*100 if len(df) > 0 else 0):.1f}%)
            </div>
            """,
            unsafe_allow_html=True
        )

    with ctrl_col2:
        view_mode = st.radio(
            "View Mode",
            ["Table View", "Card Stream"],
            horizontal=True,
            label_visibility="collapsed"
        )

    with ctrl_col3:
        # Download filtered CSV
        csv_data = filtered_df[["feedback", "sentiment"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name="filtered_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. Data Display
    # ---------------------------------------------------------
    if filtered_df.empty:
        st.markdown(
            """
            <div class="status-callout status-callout-warning">
                <span>⚠️ No feedback records matched your search and filter criteria. Try broadening your keywords.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    if view_mode == "Table View":
        display_table = filtered_df[["feedback", "sentiment", "topic", "char_length"]].copy()
        display_table.columns = ["Feedback Text", "Sentiment", "Category / Topic", "Length (chars)"]
        display_table["Sentiment"] = display_table["Sentiment"].str.capitalize()

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
            height=420
        )
    else:
        # Card Stream View
        for idx, row in filtered_df.iterrows():
            sentiment = row["sentiment"]
            badge_html = render_sentiment_badge(sentiment)
            topic_str = row["topic"]

            st.markdown(
                f"""
                <div class="feedback-item-card">
                    <div class="feedback-item-text">"{row['feedback']}"</div>
                    <div class="feedback-item-meta">
                        <span class="badge-topic">Topic: {topic_str}</span>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 0.75rem; color: #64748B;">{row['char_length']} chars</span>
                            {badge_html}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
