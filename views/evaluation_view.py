"""
Machine Learning Model Evaluation View for College Feedback AI.
Renders verified performance metrics, interactive Plotly confusion matrix,
per-class classification breakdown, and cross-model comparison charts.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from sklearn.pipeline import Pipeline
from ui_styles import render_header, render_kpi_card
from ui_charts import create_confusion_matrix_heatmap, create_model_comparison_chart
from ml_engine import evaluate_all_models, evaluate_single_model, MODEL_NAMES


def render_evaluation_view(
    models: Dict[str, Pipeline],
    active_model_name: str,
    X_test: pd.Series,
    y_test: pd.Series,
    labels: List[str],
    dataset_name: str
):
    """Render the Machine Learning Evaluation view."""
    render_header(
        title="ML Model Evaluation",
        subtitle="Rigorous performance validation, confusion matrix analysis, and multi-classifier comparison.",
        dataset_label=dataset_name,
        record_count=len(y_test) * 4  # total dataset size
    )

    # ---------------------------------------------------------
    # 1. Model Selector
    # ---------------------------------------------------------
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        selected_model_name = st.selectbox(
            "Select Evaluated Classifier",
            MODEL_NAMES,
            index=MODEL_NAMES.index(active_model_name) if active_model_name in MODEL_NAMES else 0
        )

    with col_info:
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; color: #94A3B8; margin-top: 4px;">
                Evaluated on holdout test set (<strong>{len(y_test)} records</strong>) with stratified 75/25 split. 
                TF-IDF: (1, 2) n-grams, max 5,000 features.
            </div>
            """,
            unsafe_allow_html=True
        )

    active_pipeline = models[selected_model_name]
    metrics = evaluate_single_model(active_pipeline, X_test, y_test, labels)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 2. Metric KPI Cards (Actual Calculated Values Only)
    # ---------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi_card(
            label="Accuracy",
            value=f"{metrics['accuracy']:.1%}",
            badge_text="Overall",
            footer_text="Exact correct predictions",
            variant="blue"
        )

    with c2:
        render_kpi_card(
            label="Macro Precision",
            value=f"{metrics['precision']:.1%}",
            badge_text="Unweighted",
            footer_text="Positive predictive value",
            variant="pos"
        )

    with c3:
        render_kpi_card(
            label="Macro Recall",
            value=f"{metrics['recall']:.1%}",
            badge_text="Sensitivity",
            footer_text="True positive detection rate",
            variant="neu"
        )

    with c4:
        render_kpi_card(
            label="Macro F1-Score",
            value=f"{metrics['f1']:.1%}",
            badge_text="Harmonic Mean",
            footer_text="Balanced accuracy metric",
            variant="pos"
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 3. Confusion Matrix & Per-Class Report
    # ---------------------------------------------------------
    cm_col, class_col = st.columns([1, 1.1])

    with cm_col:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="content-card-title">Confusion Matrix ({selected_model_name})</div>
                <div class="content-card-subtitle">Predicted vs Actual test classifications</div>
            """,
            unsafe_allow_html=True
        )
        cm_fig = create_confusion_matrix_heatmap(metrics["confusion_matrix"], labels)
        st.plotly_chart(cm_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with class_col:
        st.markdown(
            """
            <div class="content-card">
                <div class="content-card-title">Per-Class Classification Report</div>
                <div class="content-card-subtitle">Fine-grained precision, recall, and F1 across sentiment classes</div>
            """,
            unsafe_allow_html=True
        )

        rep = metrics["report"]
        class_rows = []
        for lab in labels:
            if lab in rep:
                class_rows.append({
                    "Sentiment Class": lab.capitalize(),
                    "Precision": f"{rep[lab]['precision']:.3f}",
                    "Recall": f"{rep[lab]['recall']:.3f}",
                    "F1-Score": f"{rep[lab]['f1-score']:.3f}",
                    "Support": int(rep[lab]["support"])
                })

        rep_df = pd.DataFrame(class_rows)
        st.dataframe(rep_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background: #0E1424; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px 16px; font-size: 0.8rem; color: #94A3B8;">
                <div style="font-weight: 600; color: #CBD5E1; margin-bottom: 4px;">Classifier Architecture:</div>
                <code>{str(active_pipeline.named_steps['classifier'])}</code>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 4. Multi-Model Comparison
    # ---------------------------------------------------------
    st.markdown(
        """
        <div class="content-card">
            <div class="content-card-title">Cross-Model Benchmark Comparison</div>
            <div class="content-card-subtitle">Comparative evaluation across Logistic Regression, Naive Bayes, and Linear SVM</div>
        """,
        unsafe_allow_html=True
    )

    comparison_df = evaluate_all_models(models, X_test, y_test)
    comp_fig = create_model_comparison_chart(comparison_df)
    st.plotly_chart(comp_fig, use_container_width=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    formatted_comp = comparison_df.copy()
    for col in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-score (Macro)"]:
        formatted_comp[col] = formatted_comp[col].apply(lambda v: f"{v:.3f} ({v:.1%})")

    st.dataframe(formatted_comp, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
