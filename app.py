import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="DigiYatra Analytics",
    page_icon="✈️",
    layout="wide"
)

st.title("DigiYatra Chatbot Analytics")

st.markdown("""
### Welcome to the DigiYatra Analytics Dashboard

This dashboard provides deep insights into the chatbot's performance, user behavior, and operational metrics.

**Navigate using the sidebar to explore:**

*   **📊 Overview**: High-level KPIs, volume trends, and status distribution.
*   **🧠 NLP Insights**: Intent classification, sentiment analysis, and topic extraction.
*   **📈 Time-Series**: Detailed hourly/daily trends, seasonality, and rolling averages.
*   **💬 Conversations**: Session duration, depth analysis, and conversation reconstruction.
*   **⚙️ Operations**: Latency performance, cost analysis, and token usage.

---
""")

