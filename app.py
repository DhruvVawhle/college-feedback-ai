"""
College Feedback AI - Executive SaaS Analytics & Sentiment Intelligence Platform.
Streamlit application entry point orchestrating navigation, state management,
machine learning pipelines, and OpenRouter conversational grounding.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment configuration (.env)
load_dotenv()

# Internal modules
from ui_styles import apply_custom_styles
from data_manager import (
    create_sample_data,
    load_bundled_feedback_csv,
    validate_and_clean_dataset
)
from ml_engine import (
    train_models,
    prepare_train_test_split,
    evaluate_single_model,
    MODEL_NAMES
)
from openrouter_client import (
    check_api_key_configured,
    get_configured_model
)

# Views
from views.overview_view import render_overview_view
from views.sentiment_view import render_sentiment_view
from views.explorer_view import render_explorer_view
from views.evaluation_view import render_evaluation_view
from views.assistant_view import render_assistant_view
from views.dataset_view import render_dataset_view

# -------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="College Feedback AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply executive dark SaaS design system
apply_custom_styles()


# -------------------------------------------------------------
# 2. SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "dataset" not in st.session_state:
    # Prefer bundled feedback.csv if present, otherwise built-in sample
    bundled_df = load_bundled_feedback_csv()
    if bundled_df is not None:
        st.session_state["dataset"] = bundled_df
        st.session_state["dataset_name"] = "Bundled feedback.csv (62 rows)"
        _, _, report, _ = validate_and_clean_dataset(bundled_df)
        st.session_state["validation_report"] = report
    else:
        sample_df = create_sample_data()
        st.session_state["dataset"] = sample_df
        st.session_state["dataset_name"] = "Built-in Demonstration Dataset (36 rows)"
        _, _, report, _ = validate_and_clean_dataset(sample_df)
        st.session_state["validation_report"] = report

if "active_model_name" not in st.session_state:
    st.session_state["active_model_name"] = "Logistic Regression"

if "selected_openrouter_model" not in st.session_state:
    st.session_state["selected_openrouter_model"] = get_configured_model()

df = st.session_state["dataset"]
dataset_name = st.session_state.get("dataset_name", "Current Dataset")


# -------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SETTINGS
# -------------------------------------------------------------
with st.sidebar:
    # Brand Header
    st.markdown(
        """
        <div class="sidebar-brand-container">
            <div class="sidebar-brand-icon">🎓</div>
            <div class="sidebar-brand-text">
                <span class="sidebar-brand-title">Feedback AI</span>
                <span class="sidebar-brand-subtitle">Sentiment Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Navigation Section
    st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
    nav_options = [
        "Overview Dashboard",
        "Sentiment Analysis",
        "Feedback Explorer",
        "ML Model Evaluation",
        "AI Feedback Assistant",
        "Dataset Management"
    ]
    selected_page = st.radio(
        "Navigate to",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Model Settings Section
    st.markdown('<div class="sidebar-section-title">Machine Learning</div>', unsafe_allow_html=True)
    active_model_choice = st.selectbox(
        "Active Classifier",
        MODEL_NAMES,
        index=MODEL_NAMES.index(st.session_state["active_model_name"]) if st.session_state["active_model_name"] in MODEL_NAMES else 0
    )
    st.session_state["active_model_name"] = active_model_choice

    st.markdown("---")

    # AI Integration Status
    st.markdown('<div class="sidebar-section-title">OpenRouter AI</div>', unsafe_allow_html=True)
    api_key_active = check_api_key_configured()
    if api_key_active:
        st.success("API Status: Connected")
    else:
        st.warning("API Status: Offline (Local Mode)")
        st.caption("Add `OPENROUTER_API_KEY` to `.env` to enable live LLM synthesis.")

    with st.expander("ℹ️ Privacy & Verification"):
        st.caption(
            "Only aggregated statistics and limited, anonymized excerpts are passed to the AI. "
            "Hallucination guards strictly prevent inventing faculty or course identities."
        )

    # Footer Metadata
    st.markdown(
        f"""
        <div style="padding: 12px 6px; margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.72rem; color: #64748B;">
            <div>Active Dataset: <strong>{len(df)} records</strong></div>
            <div style="margin-top: 2px;">Framework: Python & Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# 4. MODEL TRAINING & PREPARATION
# -------------------------------------------------------------
# Stratified 75/25 train/test split
X_train, X_test, y_train, y_test = prepare_train_test_split(df)

# Train and cache pipelines
models = train_models(
    tuple(X_train.tolist()),
    tuple(y_train.tolist())
)

active_model_name = st.session_state["active_model_name"]
active_model = models[active_model_name]
labels = sorted(df["sentiment"].unique().tolist())

# Precompute active model metrics for grounding & assistant
active_metrics = evaluate_single_model(active_model, X_test, y_test, labels)
eval_metrics = {
    "model_name": active_model_name,
    "accuracy": f"{active_metrics['accuracy']:.1%}",
    "f1_score": f"{active_metrics['f1']:.1%}"
}


# -------------------------------------------------------------
# 5. VIEW ROUTING
# -------------------------------------------------------------
if selected_page == "Overview Dashboard":
    render_overview_view(df, dataset_name)

elif selected_page == "Sentiment Analysis":
    render_sentiment_view(df, active_model, active_model_name, dataset_name)

elif selected_page == "Feedback Explorer":
    render_explorer_view(df, dataset_name)

elif selected_page == "ML Model Evaluation":
    render_evaluation_view(
        models=models,
        active_model_name=active_model_name,
        X_test=X_test,
        y_test=y_test,
        labels=labels,
        dataset_name=dataset_name
    )

elif selected_page == "AI Feedback Assistant":
    render_assistant_view(
        df=df,
        active_model=active_model,
        active_model_name=active_model_name,
        eval_metrics=eval_metrics,
        dataset_name=dataset_name
    )

elif selected_page == "Dataset Management":
    validation_report = st.session_state.get("validation_report", {})
    render_dataset_view(df, dataset_name, validation_report)
