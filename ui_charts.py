"""
Interactive Plotly Visualizations for College Feedback AI.
Provides executive-grade, responsive charts with custom tooltips,
consistent brand palettes, and dark-theme transparent backgrounds.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import openrouter_client

# Harmonized color tokens
COLOR_POS = "#10B981"
COLOR_NEG = "#F43F5E"
COLOR_NEU = "#38BDF8"
COLOR_ACCENT = "#3B82F6"
COLOR_PURPLE = "#8B5CF6"
COLOR_PINK = "#EC4899"
COLOR_AMBER = "#F59E0B"

SENTIMENT_COLORS = {
    "positive": COLOR_POS,
    "negative": COLOR_NEG,
    "neutral": COLOR_NEU
}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _apply_base_layout(fig: go.Figure, height: int = 340) -> go.Figure:
    """Apply uniform transparent background, accessible typography, and compact margins."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color="#94A3B8", size=12),
        margin=dict(l=20, r=20, t=35, b=25),
        height=height,
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family=FONT_FAMILY,
            font_color="#F8FAFC",
            bordercolor="rgba(255,255,255,0.15)"
        )
    )
    return fig


def create_sentiment_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Create a sleek donut chart representing overall sentiment distribution."""
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    total = counts["count"].sum()

    colors = [SENTIMENT_COLORS.get(s.lower(), "#64748B") for s in counts["sentiment"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[s.capitalize() for s in counts["sentiment"]],
                values=counts["count"],
                hole=0.68,
                marker=dict(colors=colors, line=dict(color="#0B0F19", width=2)),
                textinfo="percent",
                textposition="inside",
                insidetextfont=dict(family=FONT_FAMILY, size=13, color="#FFFFFF"),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                sort=False
            )
        ]
    )

    fig.add_annotation(
        text=f"<b>{total}</b><br><span style='font-size:11px; color:#94A3B8;'>REVIEWS</span>",
        x=0.5,
        y=0.5,
        font=dict(family=FONT_FAMILY, size=20, color="#F8FAFC"),
        showarrow=False
    )

    _apply_base_layout(fig, height=310)
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#CBD5E1")
        ),
        margin=dict(l=10, r=10, t=10, b=40)
    )
    return fig


def create_sentiment_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Create a clean bar chart with percentages and exact counts."""
    counts = df["sentiment"].value_counts()
    categories = ["positive", "neutral", "negative"]
    labels = []
    values = []
    colors = []

    for cat in categories:
        val = counts.get(cat, 0)
        labels.append(cat.capitalize())
        values.append(val)
        colors.append(SENTIMENT_COLORS.get(cat, "#64748B"))

    total = sum(values) if sum(values) > 0 else 1
    percentages = [f"{v} ({(v/total)*100:.1f}%)" for v in values]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                text=percentages,
                textposition="auto",
                textfont=dict(family=FONT_FAMILY, size=12, color="#FFFFFF"),
                marker=dict(
                    color=colors,
                    opacity=0.9,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                    cornerradius=6
                ),
                hovertemplate="<b>%{x}</b><br>Feedback Count: %{y}<extra></extra>"
            )
        ]
    )

    _apply_base_layout(fig, height=310)
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            color="#CBD5E1",
            tickfont=dict(family=FONT_FAMILY, size=12)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            color="#94A3B8",
            title="Number of Records"
        ),
        margin=dict(l=30, r=20, t=20, b=30)
    )
    return fig


def create_topic_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """
    Extract topics from feedback and show a stacked bar chart of sentiment by topic.
    Uses openrouter_client.TOPICS for keyword matching.
    """
    records = []
    feedback_lower = df["feedback"].str.lower()

    for idx, text in enumerate(feedback_lower):
        sentiment = df["sentiment"].iloc[idx]
        topic = openrouter_client.find_topic(text) or "other"
        records.append({"topic": topic.capitalize(), "sentiment": sentiment.lower()})

    topic_df = pd.DataFrame(records)
    topic_counts = topic_df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)

    for s in ["positive", "neutral", "negative"]:
        if s not in topic_counts.columns:
            topic_counts[s] = 0

    topic_counts["total"] = topic_counts["positive"] + topic_counts["neutral"] + topic_counts["negative"]
    topic_counts = topic_counts.sort_values(by="total", ascending=True)

    fig = go.Figure()

    for s in ["positive", "neutral", "negative"]:
        fig.add_trace(
            go.Bar(
                y=topic_counts.index,
                x=topic_counts[s],
                name=s.capitalize(),
                orientation="h",
                marker=dict(color=SENTIMENT_COLORS.get(s, "#64748B")),
                hovertemplate="<b>%{y}</b><br>" + s.capitalize() + ": %{x}<extra></extra>"
            )
        )

    _apply_base_layout(fig, height=360)
    fig.update_layout(
        barmode="stack",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            title="Mentions in Feedback"
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#CBD5E1", size=11)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#CBD5E1")
        ),
        margin=dict(l=90, r=20, t=40, b=30)
    )
    return fig


def create_model_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Generate a high-contrast grouped bar chart comparing all evaluated classifiers."""
    models = comparison_df["Model"].tolist()
    
    metrics = [
        ("Accuracy", "Accuracy", COLOR_ACCENT),
        ("Precision", "Precision (Macro)", COLOR_PURPLE),
        ("Recall", "Recall (Macro)", COLOR_PINK),
        ("F1-score", "F1-score (Macro)", COLOR_POS)
    ]

    fig = go.Figure()

    for display_name, col_name, color in metrics:
        if col_name in comparison_df.columns:
            fig.add_trace(
                go.Bar(
                    name=display_name,
                    x=models,
                    y=comparison_df[col_name],
                    marker=dict(color=color, cornerradius=4),
                    text=[f"{v:.1%}" for v in comparison_df[col_name]],
                    textposition="outside",
                    textfont=dict(size=11, color="#CBD5E1"),
                    hovertemplate="<b>%{x}</b><br>" + display_name + ": %{y:.3f}<extra></extra>"
                )
            )

    _apply_base_layout(fig, height=360)
    fig.update_layout(
        barmode="group",
        yaxis=dict(
            range=[0, 1.15],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickformat=".0%",
            title="Metric Score"
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#F8FAFC", size=12, family=FONT_FAMILY)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#CBD5E1")
        ),
        margin=dict(l=30, r=20, t=40, b=30)
    )
    return fig


def create_confusion_matrix_heatmap(cm, labels: List[str]) -> go.Figure:
    """Interactive heatmap for confusion matrix with annotated cell counts."""
    label_caps = [l.capitalize() for l in labels]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=label_caps,
            y=label_caps,
            colorscale=[
                [0.0, "#0F172A"],
                [0.5, "#1E3A8A"],
                [1.0, "#3B82F6"]
            ],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        )
    )

    # Add numeric annotations
    annotations = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i][j]
            annotations.append(
                dict(
                    x=label_caps[j],
                    y=label_caps[i],
                    text=f"<b>{val}</b>",
                    font=dict(
                        family=FONT_FAMILY,
                        size=15,
                        color="#FFFFFF" if val > (cm.max() / 2) else "#94A3B8"
                    ),
                    showarrow=False
                )
            )

    _apply_base_layout(fig, height=340)
    fig.update_layout(
        annotations=annotations,
        xaxis=dict(
            title=dict(text="Predicted Sentiment", font=dict(color="#CBD5E1", size=12)),
            tickfont=dict(color="#CBD5E1", size=12),
            side="bottom"
        ),
        yaxis=dict(
            title=dict(text="Actual Sentiment", font=dict(color="#CBD5E1", size=12)),
            tickfont=dict(color="#CBD5E1", size=12),
            autorange="reversed"
        ),
        margin=dict(l=50, r=20, t=20, b=50)
    )
    return fig


def create_feedback_length_chart(df: pd.DataFrame) -> go.Figure:
    """Create a box plot showing character count distribution across sentiments."""
    df_copy = df.copy()
    df_copy["char_length"] = df_copy["feedback"].astype(str).str.len()

    fig = go.Figure()

    for s in ["positive", "neutral", "negative"]:
        subset = df_copy[df_copy["sentiment"] == s]
        if not subset.empty:
            fig.add_trace(
                go.Box(
                    y=subset["char_length"],
                    name=s.capitalize(),
                    marker_color=SENTIMENT_COLORS.get(s, "#64748B"),
                    boxmean=True,
                    hovertemplate="<b>%{x}</b><br>Length: %{y} chars<extra></extra>"
                )
            )

    _apply_base_layout(fig, height=310)
    fig.update_layout(
        yaxis=dict(
            title="Feedback Character Count",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)"
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#CBD5E1", size=12)
        ),
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=30)
    )
    return fig
