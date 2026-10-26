"""
Comprehensive Test Suite for College Feedback AI System & OpenRouter Integration.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import requests

import openrouter_client
from openrouter_client import (
    build_dataset_context,
    create_grounded_system_prompt,
    call_openrouter_api,
    check_api_key_configured,
    get_configured_model,
    detect_teacher_columns,
    find_topic,
    TOPICS,
    feedback_chatbot,
    generate_chat_response
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def get_sample_data():
    return pd.DataFrame({
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
    })


class TestOpenRouterIntegration(unittest.TestCase):

    def setUp(self):
        self.sample_df = get_sample_data()
        self.feedback_csv_df = pd.read_csv("feedback.csv")
        self.model_factories = {
            "Logistic Regression": lambda: LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
            "Naive Bayes": lambda: MultinomialNB(),
            "Linear SVM": lambda: LinearSVC(
                class_weight="balanced", random_state=42
            )
        }

    def test_sample_dataset_structure(self):
        """Verify sample dataset has correct columns and sentiment classes."""
        self.assertIn("feedback", self.sample_df.columns)
        self.assertIn("sentiment", self.sample_df.columns)
        classes = set(self.sample_df["sentiment"].unique())
        self.assertEqual(classes, {"positive", "negative", "neutral"})
        self.assertEqual(len(self.sample_df), 36)

    def test_csv_dataset_structure(self):
        """Verify feedback.csv has correct columns and rows."""
        self.assertIn("feedback", self.feedback_csv_df.columns)
        self.assertIn("sentiment", self.feedback_csv_df.columns)
        self.assertGreater(len(self.feedback_csv_df), 30)

    def test_ml_pipeline_and_models(self):
        """Test training and evaluation across Logistic Regression, Naive Bayes, and Linear SVM."""
        df = self.sample_df
        X_train, X_test, y_train, y_test = train_test_split(
            df["feedback"],
            df["sentiment"],
            test_size=0.25,
            random_state=42,
            stratify=df["sentiment"]
        )

        for name, factory in self.model_factories.items():
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=5000
                )),
                ("classifier", factory())
            ])
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 1.0)

            # Test individual prediction
            single_pred = pipeline.predict(["The classes and professors are amazing"])[0]
            self.assertIn(single_pred, ["positive", "negative", "neutral"])

    def test_dataset_grounding_context(self):
        """Verify build_dataset_context computes correct statistics and detects lack of teacher columns."""
        ctx = build_dataset_context(self.sample_df)
        self.assertEqual(ctx["total_records"], 36)
        self.assertEqual(ctx["sentiment_counts"]["positive"], 12)
        self.assertEqual(ctx["sentiment_counts"]["negative"], 12)
        self.assertEqual(ctx["sentiment_counts"]["neutral"], 12)
        self.assertEqual(ctx["teacher_columns"], [])

        # Verify teacher columns detection when present
        df_with_teacher = self.sample_df.copy()
        df_with_teacher["teacher_name"] = ["Prof. Smith"] * len(df_with_teacher)
        ctx_teacher = build_dataset_context(df_with_teacher)
        self.assertEqual(ctx_teacher["teacher_columns"], ["teacher_name"])
        self.assertIn("Prof. Smith", ctx_teacher["teacher_summary"])

    def test_grounded_system_prompt_teacher_constraint(self):
        """Verify prompt explicitly instructs the LLM not to invent teacher names when absent."""
        ctx = build_dataset_context(self.sample_df)
        prompt = create_grounded_system_prompt(ctx)
        self.assertIn("NO teacher/faculty name column is present in the dataset", prompt)
        self.assertIn("NEVER invent or assume any teacher names", prompt)

    def test_missing_api_key(self):
        """Verify call_openrouter_api handles missing API key gracefully."""
        success, msg, err_type = call_openrouter_api(
            user_prompt="Summarize feedback",
            dataset_context={},
            api_key=""
        )
        self.assertFalse(success)
        self.assertEqual(err_type, "MISSING_API_KEY")
        self.assertIn("OpenRouter API Key Not Found", msg)

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_success(self, mock_post):
        """Verify call_openrouter_api parses successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Students generally appreciate the teaching quality."}}]
        }
        mock_post.return_value = mock_response

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="What do students appreciate?",
            dataset_context=ctx,
            api_key="sk-test-mock-key"
        )
        self.assertTrue(success)
        self.assertIsNone(err)
        self.assertEqual(reply, "Students generally appreciate the teaching quality.")

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_401_unauthorized(self, mock_post):
        """Verify call_openrouter_api handles 401 error gracefully without crashing."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="Hello",
            dataset_context=ctx,
            api_key="sk-invalid-key"
        )
        self.assertFalse(success)
        self.assertEqual(err, "UNAUTHORIZED")
        self.assertIn("401", reply)

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_402_insufficient_credits(self, mock_post):
        """Verify call_openrouter_api handles 402 insufficient credits error gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 402
        mock_post.return_value = mock_response

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="Hello",
            dataset_context=ctx,
            api_key="sk-valid-key"
        )
        self.assertFalse(success)
        self.assertEqual(err, "INSUFFICIENT_CREDITS")
        self.assertIn("Insufficient Credits", reply)

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_429_rate_limit(self, mock_post):
        """Verify call_openrouter_api handles 429 rate limit gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_post.return_value = mock_response

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="Hello",
            dataset_context=ctx,
            api_key="sk-valid-key"
        )
        self.assertFalse(success)
        self.assertEqual(err, "RATE_LIMITED")
        self.assertIn("Rate Limit", reply)

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_timeout(self, mock_post):
        """Verify call_openrouter_api handles request timeout gracefully."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="Hello",
            dataset_context=ctx,
            api_key="sk-valid-key"
        )
        self.assertFalse(success)
        self.assertEqual(err, "TIMEOUT")
        self.assertIn("Request Timeout", reply)

    @patch("openrouter_client.requests.post")
    def test_openrouter_api_empty_response(self, mock_post):
        """Verify call_openrouter_api handles empty response content gracefully."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "   "}}]
        }
        mock_post.return_value = mock_response

        ctx = build_dataset_context(self.sample_df)
        success, reply, err = call_openrouter_api(
            user_prompt="Hello",
            dataset_context=ctx,
            api_key="sk-valid-key"
        )
        self.assertFalse(success)
        self.assertEqual(err, "EMPTY_RESPONSE")

    def test_local_feedback_chatbot_teacher_query_no_names(self):
        """Verify local chatbot states absence of teacher names when queried."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.model_factories["Logistic Regression"]())
        ])
        pipeline.fit(self.sample_df["feedback"], self.sample_df["sentiment"])

        res = feedback_chatbot("Which teacher received the most positive feedback?", self.sample_df, pipeline)
        self.assertIn("The dataset does not contain teacher names", res)

    def test_hybrid_response_orchestrator_offline(self):
        """Verify generate_chat_response falls back cleanly when API key is not configured."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.model_factories["Logistic Regression"]())
        ])
        pipeline.fit(self.sample_df["feedback"], self.sample_df["sentiment"])

        with patch("openrouter_client.get_api_key", return_value=""):
            ans = generate_chat_response(
                user_query="What is the overall student sentiment?",
                dataset=self.sample_df,
                trained_clf=pipeline,
                active_clf_name="Logistic Regression",
                active_eval_metrics={"model_name": "Logistic Regression", "accuracy": "90%"},
                target_llm_model="openrouter/auto",
                api_key=""
            )
            self.assertIn("OpenRouter API key is not configured", ans)
            self.assertIn("Dataset summary:", ans)

    def test_hybrid_response_orchestrator_direct_prediction(self):
        """Verify generate_chat_response handles 'analyze feedback:' using local ML."""
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.model_factories["Logistic Regression"]())
        ])
        pipeline.fit(self.sample_df["feedback"], self.sample_df["sentiment"])

        ans = generate_chat_response(
            user_query="analyze feedback: The teaching quality is excellent",
            dataset=self.sample_df,
            trained_clf=pipeline,
            active_clf_name="Logistic Regression",
            active_eval_metrics={"model_name": "Logistic Regression", "accuracy": "90%"},
            target_llm_model="openrouter/auto"
        )
        self.assertIn("ML Classification", ans)
        self.assertIn("POSITIVE", ans)

    @patch("openrouter_client.requests.post")
    def test_hybrid_response_orchestrator_api_fallback(self, mock_post):
        """Verify that when OpenRouter API fails (e.g. 500), generate_chat_response provides error + fallback."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", self.model_factories["Logistic Regression"]())
        ])
        pipeline.fit(self.sample_df["feedback"], self.sample_df["sentiment"])

        ans = generate_chat_response(
            user_query="What are the common complaints?",
            dataset=self.sample_df,
            trained_clf=pipeline,
            active_clf_name="Logistic Regression",
            active_eval_metrics={"model_name": "Logistic Regression", "accuracy": "90%"},
            target_llm_model="openrouter/auto",
            api_key="sk-test-key"
        )
        self.assertIn("OpenRouter API Error", ans)
        self.assertIn("Local Analysis Fallback", ans)


if __name__ == "__main__":
    unittest.main()

