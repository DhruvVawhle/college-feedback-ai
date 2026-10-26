"""
OpenRouter API Client and Context Grounding for College Feedback AI.

Provides integration with OpenRouter's OpenAI-compatible Chat Completions API
to analyze college student feedback datasets while ensuring responses are strictly
grounded in actual dataset statistics and excerpts.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import requests
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/auto"

# Standard topic dictionary for keyword-based distribution analysis
TOPICS = {
    "teaching": ["teacher", "teaching", "faculty", "professor", "lecture", "instructor"],
    "laboratory": ["lab", "laboratory", "computer", "equipment", "practical"],
    "internet": ["internet", "wifi", "wi-fi", "connectivity", "network"],
    "library": ["library", "book", "books", "journal", "reading"],
    "canteen": ["canteen", "food", "meal", "snack"],
    "infrastructure": ["infrastructure", "classroom", "building", "campus", "ventilation"],
    "cleanliness": ["clean", "cleanliness", "washroom", "hygiene", "toilet"],
    "timetable": ["timetable", "schedule", "class timing", "deadline"]
}

def find_topic(text: str) -> Optional[str]:
    """Identify the first matching feedback topic from keyword list."""
    text_lower = text.lower()
    for topic, words in TOPICS.items():
        if any(word in text_lower for word in words):
            return topic
    return None

logger = logging.getLogger(__name__)

__all__ = [
    "TOPICS",
    "find_topic",
    "get_api_key",
    "get_configured_model",
    "check_api_key_configured",
    "detect_teacher_columns",
    "build_dataset_context",
    "create_grounded_system_prompt",
    "call_openrouter_api",
    "feedback_chatbot",
    "generate_chat_response"
]


def get_api_key() -> str:
    """
    Retrieve OpenRouter API key strictly from environment variables.
    Never returns hardcoded keys.
    """
    return os.getenv("OPENROUTER_API_KEY", "").strip()


def get_configured_model() -> str:
    """
    Retrieve the configured OpenRouter model identifier from the environment
    or default to 'openrouter/auto'.
    """
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def check_api_key_configured() -> bool:
    """Check if OPENROUTER_API_KEY is present and non-empty."""
    return bool(get_api_key())


def detect_teacher_columns(df) -> List[str]:
    """Detect if the dataframe explicitly contains teacher or faculty columns."""
    candidate_names = {"teacher", "teacher_name", "faculty", "professor", "instructor", "lecturer"}
    return [col for col in df.columns if col.lower() in candidate_names]


def build_dataset_context(df, ml_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Locally summarize and ground dataset statistics before making API calls.
    Extracts counts, distributions, topic frequencies, and a small set of anonymized excerpts.
    Limits data volume to protect privacy and respect token limits.
    """
    total_records = len(df)
    if total_records == 0:
        return {
            "total_records": 0,
            "sentiment_counts": {},
            "sentiment_percentages": {},
            "teacher_columns": [],
            "topic_counts": {},
            "sample_positive": [],
            "sample_negative": [],
            "sample_neutral": [],
            "ml_metrics": ml_metrics or {}
        }

    sentiments = df["sentiment"].value_counts().to_dict()
    sentiment_pcts = {
        k: f"{(v / total_records) * 100:.1f}%" for k, v in sentiments.items()
    }

    teacher_cols = detect_teacher_columns(df)
    teacher_summary = {}
    if teacher_cols:
        col = teacher_cols[0]
        teacher_summary = df[col].value_counts().head(10).to_dict()

    # Topic distribution
    topic_counts = {}
    feedback_lower = df["feedback"].str.lower()
    for topic, keywords in TOPICS.items():
        pattern = r"\b(?:" + "|".join(re_escape_words(keywords)) + r")\b"
        match_count = int(df["feedback"].str.contains(pattern, case=False, regex=True, na=False).sum())
        if match_count > 0:
            topic_counts[topic] = match_count

    # Extract limited, anonymized excerpts (max 6 per sentiment category, truncated to 120 chars)
    sample_pos = _extract_clean_excerpts(df, "positive", max_items=6)
    sample_neg = _extract_clean_excerpts(df, "negative", max_items=6)
    sample_neu = _extract_clean_excerpts(df, "neutral", max_items=4)

    return {
        "total_records": total_records,
        "sentiment_counts": sentiments,
        "sentiment_percentages": sentiment_pcts,
        "teacher_columns": teacher_cols,
        "teacher_summary": teacher_summary,
        "topic_counts": topic_counts,
        "sample_positive": sample_pos,
        "sample_negative": sample_neg,
        "sample_neutral": sample_neu,
        "ml_metrics": ml_metrics or {}
    }


def re_escape_words(words: List[str]) -> List[str]:
    """Helper to prepare keywords for regex boundary match."""
    import re
    return [re.escape(w) for w in words]


def _extract_clean_excerpts(df, sentiment_label: str, max_items: int = 5) -> List[str]:
    """Extract up to max_items distinct feedback excerpts truncated to 120 chars."""
    subset = df[df["sentiment"] == sentiment_label]
    if subset.empty:
        return []
    excerpts = (
        subset["feedback"]
        .dropna()
        .drop_duplicates()
        .head(max_items)
        .tolist()
    )
    return [e[:120].strip() + ("..." if len(e) > 120 else "") for e in excerpts]


def create_grounded_system_prompt(dataset_context: Dict[str, Any]) -> str:
    """
    Construct a strict, grounded system prompt injecting verified dataset statistics.
    Explicitly forbids inventing teacher names, departments, or statistical counts.
    """
    total = dataset_context.get("total_records", 0)
    sentiment_counts = dataset_context.get("sentiment_counts", {})
    sentiment_pcts = dataset_context.get("sentiment_percentages", {})
    teacher_cols = dataset_context.get("teacher_columns", [])
    teacher_summary = dataset_context.get("teacher_summary", {})
    topic_counts = dataset_context.get("topic_counts", {})
    sample_pos = dataset_context.get("sample_positive", [])
    sample_neg = dataset_context.get("sample_negative", [])
    sample_neu = dataset_context.get("sample_neutral", [])
    ml_metrics = dataset_context.get("ml_metrics", {})

    sentiment_text = ", ".join(
        f"{k}: {v} ({sentiment_pcts.get(k, '')})" for k, v in sentiment_counts.items()
    ) or "None"

    topic_text = ", ".join(
        f"{k} ({v} mentions)" for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    ) or "None detected"

    if teacher_cols:
        teacher_info = f"Explicit teacher columns present: {', '.join(teacher_cols)}. Known records: {json.dumps(teacher_summary)}"
    else:
        teacher_info = (
            "NO teacher/faculty name column is present in the dataset. "
            "CRITICAL: If the user asks about specific or individual teachers, "
            "you MUST explicitly state that teacher names are not available in the dataset "
            "and NEVER invent or assume any teacher names."
        )

    pos_excerpts = "\n".join(f"- {s}" for s in sample_pos) if sample_pos else "None"
    neg_excerpts = "\n".join(f"- {s}" for s in sample_neg) if sample_neg else "None"
    neu_excerpts = "\n".join(f"- {s}" for s in sample_neu) if sample_neu else "None"

    ml_info = ""
    if ml_metrics:
        ml_info = (
            f"\nActive Machine Learning Model: {ml_metrics.get('model_name', 'N/A')}\n"
            f"- Model Accuracy: {ml_metrics.get('accuracy', 'N/A')}\n"
            f"- Macro F1-score: {ml_metrics.get('f1_score', 'N/A')}\n"
        )

    return f"""You are the AI Assistant for the College Feedback Sentiment Analysis and Visualization System.
Your job is to answer user questions about student feedback, sentiment patterns, complaints, strengths, and areas for improvement.

=== VERIFIED DATASET FACTS (Use ONLY this data) ===
- Total Feedback Entries: {total}
- Sentiment Distribution: {sentiment_text}
- Identified Topic Mentions: {topic_text}
- Teacher / Faculty Information: {teacher_info}
{ml_info}
Representative Positive Feedback Excerpts:
{pos_excerpts}

Representative Negative Feedback Excerpts:
{neg_excerpts}

Representative Neutral Feedback Excerpts:
{neu_excerpts}
===================================================

RULES AND CONSTRAINTS:
1. TRUTHFULNESS & GROUNDING: Ground all findings strictly in the verified dataset facts provided above. Do NOT invent, assume, or hallucinate teacher names, course numbers, departments, counts, or percentages.
2. TEACHER INQUIRIES: If asked about specific teachers, look at the Teacher / Faculty Information. If no teacher column exists, clearly explain that the dataset contains no teacher names, so teacher-specific analysis cannot be performed. Never infer a teacher's identity from review text.
3. INSUFFICIENT DATA: If the uploaded dataset lacks information to answer a question, clearly and politely state that the dataset does not contain sufficient details.
4. RECOMMENDATIONS VS FACTS: Clearly differentiate measured dataset findings from AI-generated recommendations. Present recommendations as helpful suggestions or next steps, not measured facts.
5. FORMATTING: Use clean, professional Markdown (bullet points, bold highlights, concise sections). Keep responses direct and relevant to college administrators and students.
"""


def call_openrouter_api(
    user_prompt: str,
    dataset_context: Dict[str, Any],
    chat_history: Optional[List[Tuple[str, str]]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 30
) -> Tuple[bool, str, Optional[str]]:
    """
    Execute a chat completion request to OpenRouter.

    Returns:
        (success: bool, response_or_error_message: str, error_type: Optional[str])
    """
    key = get_api_key() if api_key is None else (api_key or "").strip()
    if not key:
        return (
            False,
            "⚠️ **OpenRouter API Key Not Found**\n\n"
            "Please configure the `OPENROUTER_API_KEY` environment variable in your `.env` file or environment.\n\n"
            "The ML classification dashboard and local features remain fully operational.",
            "MISSING_API_KEY"
        )

    active_model = model or get_configured_model()
    system_prompt = create_grounded_system_prompt(dataset_context)

    # Build message thread with limited history (last 6 turns)
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        # Keep only the last 6 messages to stay concise and avoid token limits
        recent_history = chat_history[-6:]
        for role, text in recent_history:
            if role in ["user", "assistant"] and text:
                messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": user_prompt})

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "College Feedback AI System"
    }

    payload = {
        "model": active_model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        # Check HTTP status codes with user-friendly messages
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices")
            if choices and len(choices) > 0:
                message_content = choices[0].get("message", {}).get("content", "")
                if message_content.strip():
                    return True, message_content.strip(), None
                return False, "⚠️ OpenRouter returned an empty message. Please try again.", "EMPTY_RESPONSE"
            return False, "⚠️ Malformed response received from OpenRouter API.", "MALFORMED_RESPONSE"

        elif response.status_code == 401:
            return (
                False,
                "⚠️ **Authentication Error (401)**: The provided OpenRouter API key is invalid or unauthorized. "
                "Please verify the `OPENROUTER_API_KEY` environment variable.",
                "UNAUTHORIZED"
            )

        elif response.status_code == 402:
            return (
                False,
                "⚠️ **Insufficient Credits (402)**: Your OpenRouter account has insufficient credits. "
                "Please check your account balance at openrouter.ai.",
                "INSUFFICIENT_CREDITS"
            )

        elif response.status_code == 429:
            return (
                False,
                "⚠️ **Rate Limit Exceeded (429)**: Too many requests were sent to OpenRouter in a short period. "
                "Please wait a moment before asking another question.",
                "RATE_LIMITED"
            )

        elif response.status_code == 400:
            try:
                err_data = response.json().get("error", {})
                err_msg = err_data.get("message", "Bad request.")
            except Exception:
                err_msg = "Invalid request format or model identifier."
            return (
                False,
                f"⚠️ **Request Error (400)**: {err_msg}",
                "BAD_REQUEST"
            )

        else:
            return (
                False,
                f"⚠️ **OpenRouter API Error**: Received HTTP status {response.status_code}. "
                "Please try again or select another model.",
                f"HTTP_{response.status_code}"
            )

    except requests.exceptions.Timeout:
        return (
            False,
            "⚠️ **Request Timeout**: The OpenRouter API did not respond within the time limit. "
            "Please check your internet connection and try again.",
            "TIMEOUT"
        )

    except requests.exceptions.ConnectionError:
        return (
            False,
            "⚠️ **Network Connection Error**: Unable to reach OpenRouter (`https://openrouter.ai`). "
            "Please verify your network connection.",
            "CONNECTION_ERROR"
        )

    except Exception as e:
        logger.error("Unexpected error querying OpenRouter: %s", str(e), exc_info=False)
        return (
            False,
            "⚠️ **Service Unavailable**: An unexpected error occurred while communicating with the OpenRouter API. "
            "The rest of the dashboard remains operational.",
            "UNKNOWN_ERROR"
        )


def feedback_chatbot(question: str, data, trained_model) -> str:
    """
    Offline/Rule-based chatbot fallback providing accurate statistics
    and structured responses from the local dataset.
    """
    import re
    q = question.lower().strip()
    words_in_q = set(re.findall(r"\b[a-z]+\b", q))

    if words_in_q.intersection({"hello", "hi", "hey", "greetings"}):
        return (
            "Hello! I am your College Feedback Assistant. "
            "Ask me about sentiment statistics, student complaints, "
            "feedback topics, or individual sentiment predictions."
        )

    if ("help" in words_in_q) or ("what can you do" in q):
        return (
            "You can ask me:\n"
            "- How many positive or negative feedback entries are there?\n"
            "- What are the common complaints?\n"
            "- What do students say about laboratories?\n"
            "- Analyze feedback: The teachers are helpful.\n"
            "- Show me overall sentiment statistics."
        )

    sentiments = data["sentiment"].value_counts()
    total = len(data)

    # Overall statistics
    if any(word in q for word in ["overall", "statistics", "summary", "total feedback"]):
        distribution = ", ".join(
            f"{s}: {n} ({n / total:.1%})"
            for s, n in sentiments.items()
        )
        return f"Dataset summary: {total} feedback entries. {distribution}."

    # Sentiment counts (e.g. "how many positive", "count of negative")
    count_triggers = ["how many", "count", "number of", "total number"]
    for sentiment in ["positive", "negative", "neutral"]:
        if sentiment in q and any(trig in q for trig in count_triggers):
            count = int(sentiments.get(sentiment, 0))
            return (
                f"There are {count} {sentiment} feedback entries "
                f"out of {total} total records."
            )

    # Analyze arbitrary text
    if q.startswith("analyze feedback:") or q.startswith("predict:"):
        text = question.split(":", 1)[1].strip()
        if not text:
            return "Please provide feedback after the colon."
        result = trained_model.predict([text])[0]
        return f"The model classifies this feedback as {result.upper()}."

    # Teacher-specific positive and negative feedback
    teacher_words = [
        "teacher", "teachers", "faculty",
        "professor", "professors", "lecturer", "lecturers"
    ]

    is_teacher_query = any(word in q for word in teacher_words)

    if is_teacher_query and any(
        word in q for word in ["positive", "good", "best", "appreciated", "praise"]
    ):
        positive_feedback = data[
            data["sentiment"].eq("positive")
            & data["feedback"].str.contains(
                r"\b(?:teacher|teachers|faculty|professor|professors|lecturer|lecturers)\b",
                case=False,
                regex=True,
                na=False
            )
        ]

        if positive_feedback.empty:
            return "No positive teacher-related feedback was found in the dataset."

        reviews = "\n".join(
            f"- {review}"
            for review in positive_feedback["feedback"].drop_duplicates().head(10)
        )

        return (
            "### Positive Teacher Reviews\n\n"
            "The dataset does not contain teacher names, so I cannot identify "
            "individual teachers. However, these positive reviews mention "
            "teachers or faculty:\n\n"
            f"{reviews}\n\n"
            "To identify individual teachers, add a `teacher_name` column "
            "to your CSV dataset."
        )

    if is_teacher_query and any(
        word in q for word in ["negative", "bad", "complaint", "poor"]
    ):
        negative_feedback = data[
            data["sentiment"].eq("negative")
            & data["feedback"].str.contains(
                r"\b(?:teacher|teachers|faculty|professor|professors|lecturer|lecturers)\b",
                case=False,
                regex=True,
                na=False
            )
        ]

        if negative_feedback.empty:
            return "No negative teacher-related feedback was found in the dataset."

        reviews = "\n".join(
            f"- {review}"
            for review in negative_feedback["feedback"].drop_duplicates().head(10)
        )

        return (
            "### Negative Teacher Reviews\n\n"
            "These negative reviews mention teachers or faculty:\n\n"
            f"{reviews}"
        )

    # Topic analysis
    topic = find_topic(q)

    if topic:
        words = TOPICS[topic]
        mask = data["feedback"].str.lower().apply(
            lambda text: any(word in text for word in words)
        )
        subset = data[mask]

        if subset.empty:
            return (
                f"No feedback mentioning {topic} was found in the "
                "current dataset."
            )

        counts = subset["sentiment"].value_counts()
        breakdown = ", ".join(
            f"{s}: {n}" for s, n in counts.items()
        )

        examples = subset["feedback"].head(3).tolist()
        example_text = "\n".join(f"- {x}" for x in examples)

        return (
            f"Feedback about {topic}: {len(subset)} entries. "
            f"Sentiment breakdown: {breakdown}.\n\n"
            f"Example feedback:\n{example_text}"
        )

    # Common complaints
    if any(word in q for word in ["complaint", "problem", "issue", "improve"]):
        negative = data[data["sentiment"] == "negative"]

        if negative.empty:
            return "No negative feedback entries are present in this dataset."

        found = []
        for topic_name, words in TOPICS.items():
            n = negative["feedback"].str.lower().apply(
                lambda text: any(word in text for word in words)
            ).sum()
            if n > 0:
                found.append((topic_name, int(n)))

        found.sort(key=lambda x: x[1], reverse=True)

        if not found:
            return (
                f"There are {len(negative)} negative feedback entries, "
                "but no common topic keywords were identified."
            )

        summary = ", ".join(
            f"{name} ({count} entries)" for name, count in found[:5]
        )
        return f"Common topics in negative feedback: {summary}."

    # Generic sentiment distribution
    if "sentiment" in q or "satisfaction" in q:
        return (
            "Current sentiment distribution: "
            + ", ".join(
                f"{s}: {n}" for s, n in sentiments.items()
            )
            + ". These are sentiment labels, not a verified measure "
              "of overall student satisfaction."
        )

    return (
        "Sorry, I could not identify that question. Try asking about "
        "positive, negative, or neutral feedback, common complaints, "
        "laboratories, teaching, internet, library, or overall statistics."
    )


def generate_chat_response(
    user_query: str,
    dataset,
    trained_clf,
    active_clf_name: str,
    active_eval_metrics: Optional[Dict[str, Any]] = None,
    target_llm_model: Optional[str] = None,
    chat_history: Optional[List[Tuple[str, str]]] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Orchestrate between machine learning predictions, OpenRouter AI,
    and local offline fallback based on query type and API availability.
    """
    clean_q = user_query.strip()
    if not clean_q:
        return "Please enter a valid question or prompt."

    # Direct ML model prediction requests
    if clean_q.lower().startswith("analyze feedback:") or clean_q.lower().startswith("predict:"):
        text = clean_q.split(":", 1)[1].strip()
        if not text:
            return "Please provide feedback text after the colon to predict its sentiment."
        pred = trained_clf.predict([text])[0]
        return f"**ML Classification ({active_clf_name}):**\n\nThe predicted sentiment is **{pred.upper()}**."

    # If OpenRouter is configured, call OpenRouter Chat Completions
    key = get_api_key() if api_key is None else (api_key or "").strip()
    if key:
        ctx = build_dataset_context(dataset, ml_metrics=active_eval_metrics)
        success, api_reply, err_type = call_openrouter_api(
            user_prompt=clean_q,
            dataset_context=ctx,
            chat_history=chat_history,
            model=target_llm_model,
            api_key=key
        )

        if success:
            return api_reply
        else:
            # API failure: provide clear status and seamlessly fallback to local analysis
            local_fallback = feedback_chatbot(clean_q, dataset, trained_clf)
            return (
                f"{api_reply}\n\n"
                f"---\n"
                f"**Local Analysis Fallback:**\n\n{local_fallback}"
            )
    else:
        # No API key: informative note and local analysis
        local_ans = feedback_chatbot(clean_q, dataset, trained_clf)
        return (
            "*(OpenRouter API key is not configured. Displaying local dataset analysis.)*\n\n"
            + local_ans
        )
