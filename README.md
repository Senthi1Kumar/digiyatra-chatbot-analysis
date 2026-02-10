# DigiYatra Chatbot Analytics Dashboard

## Features

- **📊 Executive Overview**: High-level KPIs, volume trends, and status distribution.
- **🧠 NLP Insights**:
- Automated intent classification (Registration, Check-in, Tech Issues, etc.)
- Sentiment analysis (Polarity & Subjectivity)
- Keyword extraction using TF-IDF
- **📈 Time-Series Analysis**:
- Hourly/Daily/Weekly volume trends
- Peak usage heatmaps
- Rolling averages for trend smoothing
- **💬 Conversation Analytics**:
- Session reconstruction
- User journey depth (turns per session)
- Conversation inspector
- **⚙️ Operations**:
- Latency distribution (p50, p95, p99)
- Cost analysis and token usage
- **🔬 Advanced Insights**:
- **Frustration Index**: Sentiment flow analysis (Start vs End)
- **Anomaly Detection**: Z-Score based latency outliers
- **Correlation Heatmap**: Hidden relationships between metrics

## Installation

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Ensure Data is Present**:
    Place your `all_requests.csv` file in the root directory.

## Running the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```
