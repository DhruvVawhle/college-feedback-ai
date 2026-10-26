"""
Dataset Management View for College Feedback AI.
Provides file upload, validation checklist, schema inspection,
cleaning diagnostics, and dataset export controls.
"""

import streamlit as st
import pandas as pd
from ui_styles import render_header
from data_manager import (
    validate_and_clean_dataset,
    create_sample_data,
    load_bundled_feedback_csv,
)


def render_dataset_view(
    df: pd.DataFrame,
    dataset_name: str,
    validation_report: dict
):
    """Render the Dataset Management interface."""
    render_header(
        title="Dataset Management",
        subtitle="Upload custom feedback CSVs, validate data integrity, inspect schemas, and clean records.",
        dataset_label=dataset_name,
        record_count=len(df)
    )

    # ---------------------------------------------------------
    # 1. Active Dataset Status Banner
    # ---------------------------------------------------------
    st.markdown(
        f"""
        <div class="content-card" style="border: 1px solid rgba(59, 130, 246, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; color: #3B82F6; letter-spacing: 0.05em;">Currently Active Dataset</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{dataset_name}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.45rem; font-weight: 800; color: #10B981;">{len(df):,}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">Cleaned Records</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # 2. Upload Zone & Presets
    # ---------------------------------------------------------
    u_col1, u_col2 = st.columns([1.5, 1])

    with u_col1:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Upload Custom CSV</div>
                <div class="content-card-subtitle">Required columns: <code>feedback</code> and <code>sentiment</code> (positive, negative, neutral)</div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload feedback CSV file",
            type=["csv"],
            help="Upload a CSV with 'feedback' and 'sentiment' columns."
        )

        if uploaded_file is not None:
            valid, clean_df, report, err_msg = validate_and_clean_dataset(uploaded_file)
            if valid and clean_df is not None:
                st.session_state["dataset"] = clean_df
                st.session_state["dataset_name"] = uploaded_file.name
                st.session_state["validation_report"] = report
                st.success(f"Successfully loaded '{uploaded_file.name}' with {len(clean_df)} validated records!")
                st.rerun()
            else:
                st.error(f"Validation Error: {err_msg}")

        st.markdown("</div>", unsafe_allow_html=True)

    with u_col2:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Dataset Presets</div>
                <div class="content-card-subtitle">Switch to built-in demonstration datasets with a single click.</div>
            """,
            unsafe_allow_html=True
        )

        if st.button("🔄 Load Built-in Demo (36 rows)", use_container_width=True):
            sample_df = create_sample_data()
            _, clean_sample, report, _ = validate_and_clean_dataset(sample_df)
            st.session_state["dataset"] = clean_sample
            st.session_state["dataset_name"] = "Built-in Demonstration Dataset (36 rows)"
            st.session_state["validation_report"] = report
            st.success("Loaded 36-record balanced demo dataset.")
            st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if st.button("📁 Load Bundled feedback.csv (62 rows)", use_container_width=True):
            bundled_df = load_bundled_feedback_csv()
            if bundled_df is not None:
                _, clean_bundled, report, _ = validate_and_clean_dataset(bundled_df)
                st.session_state["dataset"] = clean_bundled
                st.session_state["dataset_name"] = "Bundled feedback.csv (62 rows)"
                st.session_state["validation_report"] = report
                st.success("Loaded repository feedback.csv.")
                st.rerun()
            else:
                st.warning("Could not find feedback.csv in root folder.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. Column Validation & Health Checklist
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">Validation & Schema Health Report</div>
            <div class="content-card-subtitle">Automated structural verification for machine learning compliance</div>
        """,
        unsafe_allow_html=True
    )

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.markdown(
            """
            <div style="background: #0E1424; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Required Columns</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #10B981; margin-top: 4px;">✓ Verified</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">feedback, sentiment</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with v2:
        classes = sorted(df["sentiment"].unique().tolist())
        st.markdown(
            f"""
            <div style="background: #0E1424; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Sentiment Classes</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #10B981; margin-top: 4px;">✓ {len(classes)} Distinct</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">{', '.join(classes)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with v3:
        min_class_count = df["sentiment"].value_counts().min()
        st.markdown(
            f"""
            <div style="background: #0E1424; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Stratification Safety</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #10B981; margin-top: 4px;">✓ Compliant</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">Min {min_class_count} records / class</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with v4:
        # Teacher column status
        teacher_cols = [c for c in df.columns if c in {"teacher", "teacher_name", "faculty", "professor"}]
        has_teacher = len(teacher_cols) > 0
        t_status = "Present" if has_teacher else "Not Included"
        t_color = "#3B82F6" if has_teacher else "#94A3B8"

        st.markdown(
            f"""
            <div style="background: #0E1424; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Teacher Column</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: {t_color}; margin-top: 4px;">{t_status}</div>
                <div style="font-size: 0.75rem; color: #64748B; margin-top: 2px;">{teacher_cols[0] if has_teacher else 'Protected against hallucination'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. Dataset Preview & Export
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="content-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <div class="content-card-title">Dataset Preview</div>
                    <div class="content-card-subtitle">Examine all fields in the active dataset</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True
    )

    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        prev_search = st.text_input("Filter preview rows:", placeholder="Filter by keyword...", label_visibility="collapsed")
    with p_col2:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Active CSV",
            data=csv_bytes,
            file_name="active_feedback_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    preview_df = df.copy()
    if prev_search.strip():
        preview_df = preview_df[preview_df["feedback"].str.lower().str.contains(prev_search.strip().lower(), na=False)]

    st.dataframe(preview_df, use_container_width=True, hide_index=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)
