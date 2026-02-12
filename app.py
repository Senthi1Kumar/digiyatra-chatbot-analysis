import streamlit as st
from src.data_loader import get_data_upload_widget

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

### 📤 Upload Your Data

Upload a CSV file to get started with the analysis:
""")

# File uploader widget
df_uploaded = get_data_upload_widget()

if df_uploaded is not None:
    st.markdown(f"""
    **Dataset Summary:**
    - Rows: {len(df_uploaded):,}
    - Columns: {len(df_uploaded.columns)}
    - Date range: {df_uploaded['Timestamp'].min() if 'Timestamp' in df_uploaded.columns else 'N/A'} to {df_uploaded['Timestamp'].max() if 'Timestamp' in df_uploaded.columns else 'N/A'}
    """)
    st.success("Your data is ready. Use the sidebar to navigate to other pages.")
else:
    st.info("👆 Please upload a CSV file above to begin the analysis.")

