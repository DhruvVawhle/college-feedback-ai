"""
Machine Learning Pipeline and Evaluation Engine for College Feedback AI.
Preserves existing TF-IDF feature extraction, model configurations,
and metrics computation across Logistic Regression, Naive Bayes, and Linear SVM.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODEL_NAMES = [
    "Logistic Regression",
    "Naive Bayes",
    "Linear SVM"
]

MODEL_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Naive Bayes": lambda: MultinomialNB(),
    "Linear SVM": lambda: LinearSVC(
        class_weight="balanced", random_state=42
    )
}


@st.cache_resource
def train_models(
    text_train_tuple: Tuple[str, ...],
    labels_train_tuple: Tuple[str, ...]
) -> Dict[str, Pipeline]:
    """
    Train and cache all 3 classification pipelines.
    Uses TF-IDF unigram + bigram representations up to 5,000 features.
    """
    fitted = {}
    for name, factory in MODEL_FACTORIES.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000
            )),
            ("classifier", factory())
        ])
        pipeline.fit(list(text_train_tuple), list(labels_train_tuple))
        fitted[name] = pipeline

    return fitted


def prepare_train_test_split(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Perform stratified 75/25 train-test split."""
    return train_test_split(
        df["feedback"],
        df["sentiment"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentiment"]
    )


def evaluate_all_models(
    models: Dict[str, Pipeline],
    X_test: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """Calculate and return actual performance metrics across all fitted classifiers."""
    records = []
    for name, pipeline in models.items():
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1-score (Macro)": f1
        })
    return pd.DataFrame(records)


def evaluate_single_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    labels: List[str]
) -> Dict[str, Any]:
    """Calculate detailed metrics, confusion matrix, and class report for active model."""
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "labels": labels,
        "report": report_dict,
        "y_pred": y_pred
    }


def predict_sentiment_with_probabilities(
    pipeline: Pipeline,
    text: str
) -> Dict[str, Any]:
    """Predict sentiment and extract probabilities or decision score if available."""
    clean_text = text.strip()
    if not clean_text:
        return {"sentiment": "neutral", "probabilities": None}

    prediction = pipeline.predict([clean_text])[0]
    clf = pipeline.named_steps["classifier"]
    classes = pipeline.classes_

    probabilities = {}
    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba([clean_text])[0]
        for c, p in zip(classes, probs):
            probabilities[c] = float(p)
    elif hasattr(clf, "decision_function"):
        # For LinearSVC, apply softmax to decision function for illustrative distribution
        decision = pipeline.decision_function([clean_text])
        if len(classes) == 2 and decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        exp_d = np.exp(decision[0] - np.max(decision[0]))
        probs = exp_d / exp_d.sum()
        for c, p in zip(classes, probs):
            probabilities[c] = float(p)

    return {
        "sentiment": prediction,
        "probabilities": probabilities
    }
