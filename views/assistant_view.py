"""
AI Feedback Assistant View for College Feedback AI.
Provides an executive-grade conversational interface powered by OpenRouter API,
strictly grounded in dataset facts and local machine learning predictions.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from ui_styles import render_header
import openrouter_client
from openrouter_client import (
    check_api_key_configured,
    get_configured_model,
    generate_chat_response,
)


def render_assistant_view(
    df: pd.DataFrame,
    active_model,
    active_model_name: str,
    eval_metrics: Dict[str, Any],
    dataset_name: str
):
    """Render the AI Feedback Assistant conversation interface."""
    render_header(
        title="AI Feedback Assistant",
        subtitle="Grounded conversational intelligence for college feedback datasets and ML insights.",
        dataset_label=dataset_name,
        record_count=len(df)
    )

    api_key_active = check_api_key_configured()

    # ---------------------------------------------------------
    # 1. OpenRouter Status & Model Configuration Header
    # ---------------------------------------------------------
    top_col1, top_col2 = st.columns([2, 1])

    with top_col1:
        if api_key_active:
            st.markdown(
                """
                <div class="status-callout status-callout-success">
                    <span>🟢 <strong>OpenRouter API Connected:</strong> Live LLM reasoning enabled. Responses are grounded in verified dataset facts.</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="status-callout status-callout-warning">
                    <span>⚡ <strong>Offline Mode Active:</strong> No <code>OPENROUTER_API_KEY</code> detected. The assistant is operating using local ML classification and statistical summary heuristics.</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    with top_col2:
        env_default_model = get_configured_model()
        selected_model = st.text_input(
            "OpenRouter Model",
            value=st.session_state.get("selected_openrouter_model", env_default_model),
            help="Specify an OpenRouter model (default: openrouter/auto)"
        )
        st.session_state["selected_openrouter_model"] = selected_model

    # ---------------------------------------------------------
    # 2. Dataset Change Detection
    # ---------------------------------------------------------
    current_dataset_sig = (len(df), tuple(sorted(df["sentiment"].value_counts().items())))
    if "last_dataset_sig" in st.session_state and st.session_state.last_dataset_sig != current_dataset_sig:
        st.session_state.chat_history = [
            (
                "assistant",
                f"🔄 A new dataset has been loaded ({len(df)} feedback entries). "
                "How can I help you analyze it?"
            )
        ]
    st.session_state.last_dataset_sig = current_dataset_sig

    # Initialize chat history if absent
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            (
                "assistant",
                "Hello! I am your **College Feedback AI Assistant**. "
                "I can analyze sentiment distributions, identify student complaints, "
                "summarize key themes, evaluate teacher feedback, and suggest actionable improvements."
            )
        ]

    # ---------------------------------------------------------
    # 3. Suggested Questions Chips
    # ---------------------------------------------------------
    with st.expander("💡 Clickable Question Suggestions", expanded=False):
        st.caption("Click any suggestion to immediately send the query to the assistant:")
        s_col1, s_col2 = st.columns(2)
        selected_suggestion = None

        with s_col1:
            if st.button("📊 Overall student sentiment breakdown", use_container_width=True):
                selected_suggestion = "What is the overall student sentiment distribution?"
            if st.button("⚠️ Most frequent complaints and issues", use_container_width=True):
                selected_suggestion = "What are the most common complaints reported by students?"
            if st.button("🌟 What do students appreciate most?", use_container_width=True):
                selected_suggestion = "What do students appreciate most about the college?"
            if st.button("💡 Priority recommendations for administration", use_container_width=True):
                selected_suggestion = "What actionable improvements should the administration prioritize?"

        with s_col2:
            if st.button("⚖️ Compare positive vs negative feedback", use_container_width=True):
                selected_suggestion = "Compare the positive and negative feedback."
            if st.button("👩‍🏫 Feedback regarding faculty and teachers", use_container_width=True):
                selected_suggestion = "Which teacher received the most positive or negative feedback?"
            if st.button("📋 Executive summary of this dataset", use_container_width=True):
                selected_suggestion = "Provide an executive summary of the uploaded feedback dataset."
            if st.button("💬 Analysis of neutral student comments", use_container_width=True):
                selected_suggestion = "What are the main themes in neutral feedback?"

    # ---------------------------------------------------------
    # 4. Conversation History Container
    # ---------------------------------------------------------
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            avatar = "🎓" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.markdown(message)

    # ---------------------------------------------------------
    # 5. User Input and Execution
    # ---------------------------------------------------------
    chat_input_prompt = st.chat_input("Ask a question about the college feedback dataset...")

    active_query = None
    if chat_input_prompt and chat_input_prompt.strip():
        active_query = chat_input_prompt.strip()
    elif selected_suggestion:
        active_query = selected_suggestion.strip()

    if active_query:
        # Append and render user query
        st.session_state.chat_history.append(("user", active_query))
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_query)

        # Generate and render assistant reply
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Analyzing feedback data with AI..."):
                answer = generate_chat_response(
                    user_query=active_query,
                    dataset=df,
                    trained_clf=active_model,
                    active_clf_name=active_model_name,
                    active_eval_metrics=eval_metrics,
                    target_llm_model=selected_model,
                    chat_history=st.session_state.chat_history
                )
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))
        st.rerun()

    # ---------------------------------------------------------
    # 6. Controls Bar (Clear History)
    # ---------------------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    c_clear, c_space = st.columns([1, 4])
    with c_clear:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = [
                (
                    "assistant",
                    "Conversation history cleared. How can I assist you with the college feedback dataset today?"
                )
            ]
            st.rerun()
