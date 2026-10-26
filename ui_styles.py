"""
UI Styles and Design System for College Feedback AI SaaS Dashboard.
Provides a cohesive, executive-grade dark theme design system with subtle borders,
polished typography, KPI cards, sentiment badges, and responsive containers.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* -------------------------------------------------------------
   1. CORE DESIGN TOKENS & TYPOGRAPHY
   ------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --font-heading: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    
    --bg-main: #0B0F19;
    --bg-card: #131B2E;
    --bg-card-hover: #17223B;
    --bg-subtle: #1E293B;
    --bg-elevated: #243048;
    
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-strong: rgba(255, 255, 255, 0.14);
    
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    
    --accent-blue: #3B82F6;
    --accent-blue-hover: #2563EB;
    --accent-indigo: #6366F1;
    
    --sentiment-pos: #10B981;
    --sentiment-pos-bg: rgba(16, 185, 129, 0.12);
    --sentiment-pos-border: rgba(16, 185, 129, 0.28);
    
    --sentiment-neg: #F43F5E;
    --sentiment-neg-bg: rgba(244, 63, 94, 0.12);
    --sentiment-neg-border: rgba(244, 63, 94, 0.28);
    
    --sentiment-neu: #38BDF8;
    --sentiment-neu-bg: rgba(56, 189, 248, 0.12);
    --sentiment-neu-border: rgba(56, 189, 248, 0.28);
    
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 18px;
    
    --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    --shadow-hover: 0 8px 30px -4px rgba(0, 0, 0, 0.45);
}

/* Base application styling */
html, body, [class*="st-"] {
    font-family: var(--font-body);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-heading) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

/* Adjust page padding for clean dashboard feel */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1400px !important;
}

/* -------------------------------------------------------------
   2. SIDEBAR STYLING
   ------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background-color: #0E1424 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
}

/* Sidebar brand header */
.sidebar-brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 8px 16px 8px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}

.sidebar-brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-size: 18px;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
}

.sidebar-brand-text {
    display: flex;
    flex-direction: column;
}

.sidebar-brand-title {
    font-family: var(--font-heading);
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.sidebar-brand-subtitle {
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin: 18px 4px 8px 4px;
}

/* -------------------------------------------------------------
   3. HEADER & HERO BANNERS
   ------------------------------------------------------------- */
.dashboard-header-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 1.25rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border-subtle);
}

.dashboard-header-titles h1 {
    font-size: 1.85rem !important;
    margin-bottom: 4px !important;
    line-height: 1.2 !important;
}

.dashboard-header-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin: 0;
}

.dataset-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.dataset-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--sentiment-pos);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
}

/* -------------------------------------------------------------
   4. METRIC / KPI CARDS
   ------------------------------------------------------------- */
.kpi-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 18px 20px;
    position: relative;
    box-shadow: var(--shadow-card);
    transition: transform 0.15s ease, border-color 0.15s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 110px;
}

.kpi-card:hover {
    border-color: var(--border-strong);
    background-color: var(--bg-card-hover);
}

.kpi-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.kpi-card-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}

.kpi-card-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 9999px;
    line-height: 1.4;
}

.kpi-card-value {
    font-family: var(--font-heading);
    font-size: 1.95rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.kpi-card-footer {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
}

/* Sentiment-specific KPI variants */
.kpi-pos .kpi-card-value { color: var(--sentiment-pos); }
.kpi-pos .kpi-card-badge {
    background-color: var(--sentiment-pos-bg);
    color: var(--sentiment-pos);
    border: 1px solid var(--sentiment-pos-border);
}

.kpi-neg .kpi-card-value { color: var(--sentiment-neg); }
.kpi-neg .kpi-card-badge {
    background-color: var(--sentiment-neg-bg);
    color: var(--sentiment-neg);
    border: 1px solid var(--sentiment-neg-border);
}

.kpi-neu .kpi-card-value { color: var(--sentiment-neu); }
.kpi-neu .kpi-card-badge {
    background-color: var(--sentiment-neu-bg);
    color: var(--sentiment-neu);
    border: 1px solid var(--sentiment-neu-border);
}

.kpi-blue .kpi-card-value { color: var(--accent-blue); }
.kpi-blue .kpi-card-badge {
    background-color: rgba(59, 130, 246, 0.12);
    color: var(--accent-blue);
    border: 1px solid rgba(59, 130, 246, 0.28);
}

/* -------------------------------------------------------------
   5. CARDS & CONTAINERS
   ------------------------------------------------------------- */
.content-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 20px;
    box-shadow: var(--shadow-card);
    margin-bottom: 20px;
}

.content-card-title {
    font-family: var(--font-heading);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}

.content-card-subtitle {
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin-bottom: 16px;
}

/* -------------------------------------------------------------
   6. SENTIMENT BADGES
   ------------------------------------------------------------- */
.badge-pos {
    display: inline-flex;
    align-items: center;
    background-color: var(--sentiment-pos-bg);
    color: var(--sentiment-pos);
    border: 1px solid var(--sentiment-pos-border);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-neg {
    display: inline-flex;
    align-items: center;
    background-color: var(--sentiment-neg-bg);
    color: var(--sentiment-neg);
    border: 1px solid var(--sentiment-neg-border);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-neu {
    display: inline-flex;
    align-items: center;
    background-color: var(--sentiment-neu-bg);
    color: var(--sentiment-neu);
    border: 1px solid var(--sentiment-neu-border);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-topic {
    display: inline-flex;
    align-items: center;
    background-color: var(--bg-subtle);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 500;
}

/* -------------------------------------------------------------
   7. FEEDBACK QUOTE CARDS
   ------------------------------------------------------------- */
.feedback-item-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease;
}

.feedback-item-card:hover {
    border-color: var(--border-strong);
    background-color: var(--bg-card-hover);
}

.feedback-item-text {
    font-size: 0.92rem;
    color: var(--text-primary);
    line-height: 1.5;
    margin-bottom: 10px;
}

.feedback-item-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* -------------------------------------------------------------
   8. SYSTEM STATUS & VALIDATION CALLOUTS
   ------------------------------------------------------------- */
.status-callout {
    padding: 12px 16px;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    line-height: 1.5;
    margin-bottom: 14px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.status-callout-info {
    background-color: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.25);
    color: #93C5FD;
}

.status-callout-success {
    background-color: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #A7F3D0;
}

.status-callout-warning {
    background-color: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    color: #FDE68A;
}

.status-callout-error {
    background-color: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.25);
    color: #FECDD3;
}

/* -------------------------------------------------------------
   9. STREAMLIT WIDGET OVERRIDES & POLISH
   ------------------------------------------------------------- */
/* Form inputs */
div[data-baseweb="input"] {
    background-color: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 1px var(--accent-blue) !important;
}

/* Textarea */
div[data-baseweb="textarea"] textarea {
    background-color: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

/* Selectbox & multiselect */
div[data-baseweb="select"] {
    border-radius: var(--radius-md) !important;
}

/* Buttons */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    border-color: var(--accent-blue) !important;
    color: #FFFFFF !important;
}

/* Download buttons */
.stDownloadButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Dataframe container */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
}

/* Chat container */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 9999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.28);
}
</style>
"""


def apply_custom_styles():
    """Inject custom executive SaaS CSS styles into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, dataset_label: str = "", record_count: int = 0):
    """Render a consistent, modern dashboard header with dataset status pill."""
    status_html = ""
    if dataset_label:
        status_html = f"""
        <div class="dataset-status-pill">
            <span class="dataset-status-dot"></span>
            <span>Active Dataset: <strong>{dataset_label}</strong> ({record_count} records)</span>
        </div>
        """

    header_html = f"""
    <div class="dashboard-header-container">
        <div class="dashboard-header-titles">
            <h1>{title}</h1>
            <p class="dashboard-header-subtitle">{subtitle}</p>
        </div>
        {status_html}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    badge_text: str = "",
    footer_text: str = "",
    variant: str = "default"
):
    """
    Render a high-contrast KPI card.
    variant: 'default', 'pos', 'neg', 'neu', 'blue'
    """
    variant_class = f"kpi-{variant}" if variant != "default" else ""
    badge_html = f'<span class="kpi-card-badge">{badge_text}</span>' if badge_text else ""
    footer_html = f'<div class="kpi-card-footer">{footer_text}</div>' if footer_text else ""

    card_html = f"""
    <div class="kpi-card {variant_class}">
        <div class="kpi-card-header">
            <span class="kpi-card-label">{label}</span>
            {badge_html}
        </div>
        <div class="kpi-card-value">{value}</div>
        {footer_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_sentiment_badge(sentiment: str) -> str:
    """Return HTML string for a colored sentiment badge."""
    s = sentiment.lower().strip()
    if s == "positive":
        return '<span class="badge-pos">Positive</span>'
    elif s == "negative":
        return '<span class="badge-neg">Negative</span>'
    else:
        return '<span class="badge-neu">Neutral</span>'
