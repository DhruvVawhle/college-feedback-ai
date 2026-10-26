"""
Data Management and Validation Engine for College Feedback AI.
Handles sample dataset creation, CSV ingestion, rigorous column validation,
stratification safety checks, and dataset statistical profiling.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any, Optional


def create_sample_data() -> pd.DataFrame:
    """
    Generate the verified 36-record balanced demonstration dataset.
    Preserves exact records from original app.py.
    """
    data = {
        "feedback": [
            "The teaching quality is excellent",
            "Teachers explain concepts very clearly",
            "The library has excellent resources",
            "Laboratories are well maintained",
            "The college campus is beautiful",
            "Faculty members are very supportive",
            "The classroom environment is comfortable",
            "The practical sessions are very useful",
            "The college provides good facilities",
            "I am satisfied with the teaching methods",
            "The teachers are helpful and friendly",
            "The library collection is impressive",

            "The laboratory computers are very slow",
            "The internet connection is poor",
            "Classrooms are not properly maintained",
            "The canteen food is disappointing",
            "The teaching methods are confusing",
            "The library lacks sufficient books",
            "The washrooms are not clean",
            "The timetable changes too frequently",
            "The college infrastructure needs improvement",
            "The WiFi connection does not work properly",
            "The laboratory equipment is outdated",
            "The faculty does not explain topics clearly",

            "The college experience is average",
            "The facilities are satisfactory",
            "The teaching quality is okay",
            "The library is neither good nor bad",
            "The canteen is average",
            "The classroom facilities are acceptable",
            "The course content is adequate",
            "The college experience is normal",
            "The laboratory facilities are moderate",
            "The timetable is manageable",
            "The academic support is satisfactory",
            "The campus environment is acceptable"
        ],
        "sentiment": (
            ["positive"] * 12
            + ["negative"] * 12
            + ["neutral"] * 12
        )
    }
    return pd.DataFrame(data)


def load_bundled_feedback_csv() -> Optional[pd.DataFrame]:
    """Attempt to load the repository's bundled feedback.csv file if present."""
    if os.path.exists("feedback.csv"):
        try:
            df = pd.read_csv("feedback.csv")
            valid, cleaned_df, _, _ = validate_and_clean_dataset(df)
            if valid:
                return cleaned_df
        except Exception:
            pass
    return None


def validate_and_clean_dataset(
    source: Any
) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any], str]:
    """
    Thoroughly parse, validate, and clean an uploaded CSV or DataFrame.
    Returns: (is_valid, cleaned_df, validation_report, error_message)
    """
    report: Dict[str, Any] = {
        "raw_rows": 0,
        "clean_rows": 0,
        "empty_dropped": 0,
        "duplicates_dropped": 0,
        "invalid_sentiments_dropped": 0,
        "columns_found": [],
        "has_feedback_col": False,
        "has_sentiment_col": False,
        "has_teacher_col": False,
        "teacher_col_name": None,
        "classes_found": [],
        "class_counts": {},
        "min_class_count": 0,
        "stratification_safe": False
    }

    try:
        if isinstance(source, pd.DataFrame):
            raw_df = source.copy()
        else:
            raw_df = pd.read_csv(source)
    except Exception as e:
        return False, None, report, f"Could not read CSV file: {str(e)}"

    report["raw_rows"] = len(raw_df)
    if report["raw_rows"] == 0:
        return False, None, report, "The uploaded dataset is empty."

    # Normalize column names
    raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]
    report["columns_found"] = list(raw_df.columns)

    report["has_feedback_col"] = "feedback" in raw_df.columns
    report["has_sentiment_col"] = "sentiment" in raw_df.columns

    if not (report["has_feedback_col"] and report["has_sentiment_col"]):
        missing = []
        if not report["has_feedback_col"]:
            missing.append("'feedback'")
        if not report["has_sentiment_col"]:
            missing.append("'sentiment'")
        return False, None, report, f"Missing required column(s): {', '.join(missing)}."

    # Detect teacher/faculty column
    teacher_cols = [c for c in raw_df.columns if c in {"teacher", "teacher_name", "faculty", "professor", "instructor"}]
    if teacher_cols:
        report["has_teacher_col"] = True
        report["teacher_col_name"] = teacher_cols[0]

    # Clean text and sentiment
    df = raw_df.copy()
    df["feedback"] = df["feedback"].fillna("").astype(str).str.strip()
    df["sentiment"] = df["sentiment"].fillna("").astype(str).str.strip().str.lower()

    # Drop empty feedback
    non_empty_mask = df["feedback"] != ""
    report["empty_dropped"] = int((~non_empty_mask).sum())
    df = df[non_empty_mask]

    # Filter to allowed sentiment labels
    valid_sentiment_mask = df["sentiment"].isin(["positive", "negative", "neutral"])
    report["invalid_sentiments_dropped"] = int((~valid_sentiment_mask).sum())
    df = df[valid_sentiment_mask]

    # Drop duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["feedback", "sentiment"])
    report["duplicates_dropped"] = before_dedup - len(df)

    report["clean_rows"] = len(df)
    if report["clean_rows"] == 0:
        return False, None, report, "No valid feedback records remain after cleaning."

    report["classes_found"] = sorted(df["sentiment"].unique().tolist())
    report["class_counts"] = df["sentiment"].value_counts().to_dict()
    report["min_class_count"] = int(min(report["class_counts"].values())) if report["class_counts"] else 0

    if len(report["classes_found"]) < 2:
        return False, None, report, (
            f"The dataset contains only one sentiment class ({report['classes_found']}). "
            "At least two distinct classes ('positive', 'negative', 'neutral') are required for classification."
        )

    if report["min_class_count"] < 2:
        return False, None, report, (
            f"Class '{min(report['class_counts'], key=report['class_counts'].get)}' has fewer than 2 records. "
            "Each sentiment class requires at least 2 records for a stratified train/test split."
        )

    report["stratification_safe"] = True
    return True, df, report, ""


def calculate_dataset_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive KPI summary metrics strictly from the active dataset."""
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "positive_count": 0,
            "positive_pct": "0.0%",
            "negative_count": 0,
            "negative_pct": "0.0%",
            "neutral_count": 0,
            "neutral_pct": "0.0%",
            "net_sentiment_score": 0.0,
            "sentiment_health": "N/A"
        }

    counts = df["sentiment"].value_counts().to_dict()
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)
    neu = counts.get("neutral", 0)

    pos_pct = (pos / total) * 100
    neg_pct = (neg / total) * 100
    neu_pct = (neu / total) * 100

    # Net Sentiment Score: -100 to +100
    net_score = ((pos - neg) / total) * 100

    if net_score >= 20:
        health = "Positive Outlook"
    elif net_score >= -10:
        health = "Balanced / Moderate"
    else:
        health = "Requires Attention"

    return {
        "total": total,
        "positive_count": pos,
        "positive_pct": f"{pos_pct:.1f}%",
        "negative_count": neg,
        "negative_pct": f"{neg_pct:.1f}%",
        "neutral_count": neu,
        "neutral_pct": f"{neu_pct:.1f}%",
        "net_sentiment_score": net_score,
        "sentiment_health": health
    }
