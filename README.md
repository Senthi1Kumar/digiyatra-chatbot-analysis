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

Note: this project uses PyTorch for some NLP models. Install the correct
PyTorch wheel for your environment (CPU-only or CUDA-enabled GPU). Use
`nvidia-smi` to check for an NVIDIA GPU and available CUDA version.

- CPU-only (recommended if you don't have an NVIDIA GPU):

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- CUDA-enabled GPU (example for CUDA 12.8):

```bash
# For Linux x86 use:
pip3 install torch torchvision

# For Linux Aarch64:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# For Windows
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

2.**Ensure Data is Present**:
    Place your `all_requests.csv` file in the root directory.

## Running the Dashboard

Run the Streamlit application:

```bash
streamlit run app.py
```
