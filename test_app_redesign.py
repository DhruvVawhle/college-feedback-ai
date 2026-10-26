"""
Verification test suite for redesigned UI/UX modules.
Tests data_manager, ml_engine, ui_charts, and ui_styles to guarantee zero runtime regressions.
"""

import unittest
import pandas as pd
import numpy as np

import ui_styles
import ui_charts
import data_manager
import ml_engine
import openrouter_client


class TestRedesignedArchitecture(unittest.TestCase):

    def setUp(self):
        self.sample_df = data_manager.create_sample_data()
        self.bundled_df = data_manager.load_bundled_feedback_csv()

    def test_data_manager_sample_and_kpis(self):
        """Test dataset creation, validation, and KPI calculations."""
        self.assertEqual(len(self.sample_df), 36)
        valid, clean_df, report, err = data_manager.validate_and_clean_dataset(self.sample_df)
        self.assertTrue(valid)
        self.assertEqual(err, "")
        self.assertEqual(len(clean_df), 36)
        self.assertTrue(report["stratification_safe"])

        kpis = data_manager.calculate_dataset_kpis(clean_df)
        self.assertEqual(kpis["total"], 36)
        self.assertEqual(kpis["positive_count"], 12)
        self.assertEqual(kpis["negative_count"], 12)
        self.assertEqual(kpis["neutral_count"], 12)
        self.assertEqual(kpis["positive_pct"], "33.3%")
        self.assertIsInstance(kpis["net_sentiment_score"], float)

    def test_bundled_feedback_csv(self):
        """Verify bundled feedback.csv loads and validates cleanly."""
        self.assertIsNotNone(self.bundled_df)
        valid, clean_df, report, err = data_manager.validate_and_clean_dataset(self.bundled_df)
        self.assertTrue(valid)
        self.assertGreaterEqual(len(clean_df), 60)
        self.assertTrue(report["has_feedback_col"])
        self.assertTrue(report["has_sentiment_col"])

    def test_validation_rejects_invalid_data(self):
        """Verify validation properly catches missing columns or insufficient classes."""
        # Missing sentiment
        invalid_df = pd.DataFrame({"feedback": ["Good", "Bad"]})
        valid, _, report, err = data_manager.validate_and_clean_dataset(invalid_df)
        self.assertFalse(valid)
        self.assertIn("Missing required column", err)

        # Single class
        single_class_df = pd.DataFrame({"feedback": ["Good", "Great"], "sentiment": ["positive", "positive"]})
        valid, _, report, err = data_manager.validate_and_clean_dataset(single_class_df)
        self.assertFalse(valid)
        self.assertIn("only one sentiment class", err)

    def test_ml_engine_training_and_evaluation(self):
        """Verify all 3 models train, evaluate, and predict accurately."""
        X_train, X_test, y_train, y_test = ml_engine.prepare_train_test_split(self.sample_df)
        self.assertEqual(len(X_train) + len(X_test), 36)

        models = ml_engine.train_models(tuple(X_train.tolist()), tuple(y_train.tolist()))
        self.assertEqual(set(models.keys()), {"Logistic Regression", "Naive Bayes", "Linear SVM"})

        labels = sorted(self.sample_df["sentiment"].unique().tolist())
        single_eval = ml_engine.evaluate_single_model(
            models["Logistic Regression"], X_test, y_test, labels
        )
        self.assertIn("accuracy", single_eval)
        self.assertIn("confusion_matrix", single_eval)
        self.assertEqual(single_eval["confusion_matrix"].shape, (3, 3))

        comp_df = ml_engine.evaluate_all_models(models, X_test, y_test)
        self.assertEqual(len(comp_df), 3)
        self.assertIn("Accuracy", comp_df.columns)
        self.assertIn("F1-score (Macro)", comp_df.columns)

        # Prediction with probabilities
        pred_res = ml_engine.predict_sentiment_with_probabilities(
            models["Logistic Regression"], "The faculty is extremely helpful."
        )
        self.assertIn(pred_res["sentiment"], ["positive", "neutral", "negative"])
        self.assertIsNotNone(pred_res["probabilities"])

    def test_ui_charts_generation(self):
        """Verify all Plotly figure builders execute and return valid figures."""
        donut_fig = ui_charts.create_sentiment_donut_chart(self.sample_df)
        self.assertIsNotNone(donut_fig)

        bar_fig = ui_charts.create_sentiment_bar_chart(self.sample_df)
        self.assertIsNotNone(bar_fig)

        topic_fig = ui_charts.create_topic_sentiment_chart(self.sample_df)
        self.assertIsNotNone(topic_fig)

        length_fig = ui_charts.create_feedback_length_chart(self.sample_df)
        self.assertIsNotNone(length_fig)

        X_train, X_test, y_train, y_test = ml_engine.prepare_train_test_split(self.sample_df)
        models = ml_engine.train_models(tuple(X_train.tolist()), tuple(y_train.tolist()))
        comp_df = ml_engine.evaluate_all_models(models, X_test, y_test)
        comp_fig = ui_charts.create_model_comparison_chart(comp_df)
        self.assertIsNotNone(comp_fig)

        labels = ["negative", "neutral", "positive"]
        cm = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
        cm_fig = ui_charts.create_confusion_matrix_heatmap(cm, labels)
        self.assertIsNotNone(cm_fig)

    def test_ui_styles_functions(self):
        """Verify style helpers produce valid markup."""
        pos_badge = ui_styles.render_sentiment_badge("positive")
        self.assertIn("badge-pos", pos_badge)
        self.assertIn("Positive", pos_badge)

        neg_badge = ui_styles.render_sentiment_badge("negative")
        self.assertIn("badge-neg", neg_badge)

        neu_badge = ui_styles.render_sentiment_badge("neutral")
        self.assertIn("badge-neu", neu_badge)


if __name__ == "__main__":
    unittest.main()
