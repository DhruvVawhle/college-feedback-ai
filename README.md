# 🎓 College Feedback AI

> **Executive SaaS Analytics & Sentiment Intelligence Platform for Academic Institutions**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter%20API-6366F1.svg)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

**College Feedback AI** is an end-to-end sentiment intelligence and text analytics platform designed for colleges, universities, and educational administrators. It transforms raw student, faculty, and institutional feedback into actionable insights through high-performance natural language processing (NLP), machine learning classification models, and conversational AI grounded in real institutional data.

---

## ✨ Key Features

### 📊 1. Executive Overview Dashboard
- High-level KPIs: Total feedback count, positive/negative/neutral sentiment proportions, institutional health index.
- Interactive Plotly visualizations for sentiment breakdowns, category distributions, and trend analysis.
- Instant executive summary designed with a modern dark SaaS interface.

### 🧠 2. Real-Time Sentiment Prediction
- Live inference sandbox: test custom feedback statements in real time.
- Probability distribution bars and confidence scoring across positive, neutral, and negative classes.
- Model selector allowing dynamic switching between trained classifiers.

### 🔎 3. Feedback Explorer & Filtering
- Search and query feedback entries with sub-string keyword search.
- Filter by sentiment classification or specific institutional aspects (infrastructure, academics, campus, faculty).
- Export filtered subsets for reporting and institutional audits.

### 📈 4. ML Model Evaluation & Benchmarking
- Multi-model evaluation comparing:
  - **Logistic Regression** (Balanced class weighting)
  - **Multinomial Naive Bayes**
  - **Linear Support Vector Machine (LinearSVC)**
- TF-IDF Feature Extraction with unigram and bigram representation (`ngram_range=(1, 2)`).
- Metric scorecards: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- Interactive Confusion Matrix heatmaps and full Scikit-Learn classification reports.

### 🤖 5. Grounded AI Assistant (OpenRouter Integration)
- Conversational chat assistant powered by state-of-the-art LLMs via OpenRouter.
- **Strict Anti-Hallucination Guardrails**: Prompts are dynamically grounded with real aggregated statistics and dataset excerpts.
- Synthesizes institutional recommendations, identifies recurring pain points, and drafts actionable administrative summaries.
- Graceful offline fallback mode when no API key is provided.

### 📂 6. Dataset Management & Validation
- Upload custom CSV feedback files with automated column detection.
- Data validation engine checking for schema integrity, null/empty values, and label distributions.
- Includes pre-packaged, validated sample datasets for immediate testing.

---

## 🛠️ Tech Stack

| Domain | Technology |
|---|---|
| **Frontend Framework** | [Streamlit](https://streamlit.io/) with custom executive dark SaaS styling |
| **Data Visualization** | [Plotly](https://plotly.com/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) (TF-IDF, Logistic Regression, Naive Bayes, LinearSVC) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Generative AI** | [OpenRouter API](https://openrouter.ai/) for multi-model LLM inference |
| **Environment & Config** | [python-dotenv](https://github.com/theskumar/python-dotenv) |

---

## 📁 Repository Structure

```text
college-feedback-ai/
├── app.py                  # Main Streamlit application entry point & router
├── ml_engine.py            # ML model training, TF-IDF vectorization & evaluation
├── openrouter_client.py    # OpenRouter API client, prompt grounding & chat engine
├── data_manager.py         # CSV loading, validation & synthetic data generation
├── ui_styles.py            # Custom CSS & executive dark SaaS design system
├── ui_charts.py            # Plotly & Seaborn charting utilities
├── feedback.csv            # Bundled institutional feedback dataset
├── requirements.txt        # Python package dependencies
├── test_system.py          # Comprehensive unit and integration test suite
├── test_app_redesign.py    # App redesign test script
├── .env.example            # Environment configuration template
├── .gitignore              # Git ignore rules for virtualenvs, caches & secrets
└── views/                  # Streamlit view modules
    ├── __init__.py
    ├── overview_view.py    # Executive overview view
    ├── sentiment_view.py   # Live sentiment classification view
    ├── explorer_view.py    # Feedback query & explorer view
    ├── evaluation_view.py  # ML performance benchmarking view
    ├── assistant_view.py   # Grounded AI Chat Assistant view
    └── dataset_view.py     # Dataset upload & validation view
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.9** or higher installed on your system.
- Git installed.

### 2. Clone the Repository
```bash
git clone https://github.com/DhruvVawhle/college-feedback-ai.git
cd college-feedback-ai
```

### 3. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

Open `.env` and configure your OpenRouter API key (optional, for AI Assistant features):
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/auto
```
> [!NOTE]
> If `OPENROUTER_API_KEY` is not provided, the platform functions seamlessly in **Local Mode** with classical machine learning classifiers.

---

## 🖥️ Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

The application will start locally and open in your default browser at:
```
http://localhost:8501
```

---

## 🧪 Running Tests

Execute the comprehensive test suite covering data management, ML pipelines, and API grounding:
```bash
python -m unittest test_system.py
```

---

## 📊 Dataset Format

When uploading your own dataset via the **Dataset Management** page, ensure your CSV file includes at minimum the following column format:

| feedback | sentiment |
|---|---|
| "The teachers explain difficult concepts clearly and patiently." | positive |
| "The internet connection is very slow in the laboratory." | negative |
| "The library working hours are standard." | neutral |

*(Optional columns such as `category`, `department`, or `faculty` will automatically be detected and leveraged for deeper analytics).*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
