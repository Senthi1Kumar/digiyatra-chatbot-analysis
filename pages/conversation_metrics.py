
"""import pandas as pd
import re
from pathlib import Path

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "all-requests (1).csv.xlsx"

# ---------------- CONFIG ----------------
FALLBACK_PATTERNS = [
    r"didn.?t understand",
    r"can you rephrase",
    r"sorry",
    r"unable to",
    r"not sure",
    r"didn.?t get"
]

fallback_regex = re.compile("|".join(FALLBACK_PATTERNS))

# ---------------- LOAD DATA ----------------
df = pd.read_excel(DATA_PATH)

# Normalize text columns
TEXT_COLS = ["Request", "Response", "Clarification", "User Feedback"]
for col in TEXT_COLS:
    df[col] = df[col].fillna("").astype(str).str.lower()

# ---------------- BASIC COUNTS ----------------
total_messages = len(df)
conversation_sizes = df.groupby("Conversation ID")["Request"].count()
total_conversations = conversation_sizes.count()

# ---------------- METRIC 1: BOUNCE RATE ----------------
bounced_conversations = (conversation_sizes == 1).sum()
bounce_rate = (bounced_conversations / total_conversations) * 100

# ---------------- METRIC 2: MISSED MESSAGES ----------------
missed_mask = (
    (df["Response"].str.strip() == "") |
    (df["Clarification"].str.strip() != "") |
    (df["Response"].str.contains(fallback_regex))
)

missed_messages = missed_mask.sum()
missed_message_rate = (missed_messages / total_messages) * 100

# ---------------- METRIC 3: FALLBACK RATE ----------------
fallback_mask = df["Response"].str.contains(fallback_regex)
fallback_messages = fallback_mask.sum()
fallback_rate = (fallback_messages / total_messages) * 100

# ---------------- FEEDBACK ANALYSIS ----------------
feedback_distribution = (
    df["User Feedback"]
    .replace("", "no_feedback")
    .value_counts(normalize=True) * 100
)

# ---------------- PRINT RESULTS ----------------
print("\n--- Conversation Metrics ---")
print(f"Total Conversations      : {total_conversations}")
print(f"Total Messages           : {total_messages}")
print(f"Bounce Rate (%)          : {bounce_rate:.2f}")
print(f"Missed Message Rate (%)  : {missed_message_rate:.2f}")
print(f"Fallback Rate (%)        : {fallback_rate:.2f}")

print("\n--- User Feedback Distribution (%) ---")
print(feedback_distribution.round(2))

# ---------------- SAVE SUMMARY ----------------
metrics_summary = pd.DataFrame({
    "Metric": [
        "Total Conversations",
        "Total Messages",
        "Bounce Rate (%)",
        "Missed Message Rate (%)",
        "Fallback Rate (%)"
    ],
    "Value": [
        total_conversations,
        total_messages,
        round(bounce_rate, 2),
        round(missed_message_rate, 2),
        round(fallback_rate, 2)
    ]
})

metrics_summary.to_csv("conversation_metrics_summary.csv", index=False)
"""

import streamlit as st
import pandas as pd
import re
import plotly.express as px
from pathlib import Path

# Set page config
st.set_page_config(page_title="Conversation Metrics", layout="wide")

# Title
st.title("📊 Conversation Metrics Dashboard")

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "all_requests.csv"

def run_metrics_report():
    try:
        # Load data
        try:
            df = pd.read_excel(DATA_PATH)
        except Exception:
            df = pd.read_csv(DATA_PATH)

        # Normalize text columns
        TEXT_COLS = ["Request", "Response", "Clarification", "User Feedback"]
        for col in TEXT_COLS:
            df[col] = df[col].fillna("").astype(str).str.lower()

        # Config
        FALLBACK_PATTERNS = [
            r"didn.?t understand", r"can you rephrase", r"sorry", 
            r"unable to", r"not sure", r"didn.?t get", 
            r"could not find", r"no information", r"don't have enough context"
        ]
        fallback_regex = re.compile("|".join(FALLBACK_PATTERNS))

        # Calculate metrics
        conversation_sizes = df.groupby("Conversation ID").size()
        total_conversations = len(conversation_sizes)
        bounced_conversations = (conversation_sizes == 1).sum()
        bounce_rate = (bounced_conversations / total_conversations) * 100

        fallback_mask = df["Response"].str.contains(fallback_regex)
        fallback_messages = fallback_mask.sum()
        fallback_rate = (fallback_messages / len(df)) * 100

        missed_mask = (df["Response"].str.strip() == "") | fallback_mask
        missed_messages = missed_mask.sum()
        missed_message_rate = (missed_messages / len(df)) * 100

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Conversations", f"{total_conversations:,}")
        col2.metric("Total Messages", f"{len(df):,}")
        col3.metric("Bounce Rate", f"{bounce_rate:.2f}%")
        col4.metric("Missed Message Rate", f"{missed_message_rate:.2f}%")
        col5.metric("Fallback Rate", f"{fallback_rate:.2f}%")

        # Visualizations (Plotly)
        col1, col2 = st.columns(2)

        with col1:
            engagement_fig = px.pie(
                names=['Bounced', 'Engaged'],
                values=[bounced_conversations, max(total_conversations - bounced_conversations, 0)],
                title='User Engagement (Bounce Rate)',
                hole=0.3,
                color_discrete_sequence=['#ff9999', '#66b3ff'],
            )
            st.plotly_chart(engagement_fig, use_container_width=True)

        with col2:
            response_fig = px.pie(
                names=['Success', 'Missed/Fallback'],
                values=[max(len(df) - missed_messages, 0), missed_messages],
                title='Response Quality',
                hole=0.3,
                color_discrete_sequence=['#99ff99', '#ffcc99'],
            )
            st.plotly_chart(response_fig, use_container_width=True)

        # Summary table
        st.subheader("📋 Metrics Summary")
        metrics_summary = pd.DataFrame({
            "Metric": ["Total Conversations", "Total Messages", "Bounce Rate (%)", "Missed Rate (%)", "Fallback Rate (%)"],
            "Value": [total_conversations, len(df), round(bounce_rate, 2), round(missed_message_rate, 2), round(fallback_rate, 2)]
        })
        st.dataframe(metrics_summary, use_container_width=True)

        # Download button
        csv = metrics_summary.to_csv(index=False)
        st.download_button(label="Download Summary CSV", data=csv, file_name="conversation_metrics_summary.csv")

    except FileNotFoundError:
        st.error(f"Data file not found at: {DATA_PATH}")
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

if __name__ == "__main__":
    run_metrics_report()
