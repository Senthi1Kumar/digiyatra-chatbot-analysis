
import streamlit as st
import pandas as pd
import re
import plotly.express as px
from pathlib import Path

from src.data_loader import get_session_data
from src.preprocessing import preprocess_data

# Set page config
st.set_page_config(page_title="Conversation Metrics", layout="wide")

# Title
st.title("📊 Conversation Metrics Dashboard")

def run_metrics_report():
    try:
        # Load and preprocess data from session state
        with st.spinner("Loading and preprocessing data..."):
            df = get_session_data()
            if df.empty:
                st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
                return
            df = preprocess_data(df)

        # Ensure key text columns are normalized strings
        TEXT_COLS = ["Request", "Response", "Clarification", "User Feedback"]
        for col in TEXT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str).str.lower()
            else:
                # Add missing column if necessary to avoid errors
                df[col] = ""

        # Config
        FALLBACK_PATTERNS = [
            r"didn.?t understand", r"can you rephrase", r"sorry", 
            r"unable to", r"not sure", r"didn.?t get", 
            r"could not find", r"no information", r"don't have enough context"
        ]
        fallback_regex = re.compile("|".join(FALLBACK_PATTERNS))

        # Calculate metrics
        if "Conversation ID" not in df.columns:
            st.error("Required column 'Conversation ID' is missing from the dataset.")
            return

        conversation_sizes = df.groupby("Conversation ID").size()
        total_conversations = len(conversation_sizes)
        bounced_conversations = (conversation_sizes == 1).sum()
        bounce_rate = (bounced_conversations / total_conversations * 100) if total_conversations > 0 else 0

        # Metrics for messages
        total_messages = len(df)
        fallback_mask = df["Response"].str.contains(fallback_regex, na=False)
        fallback_messages = fallback_mask.sum()
        fallback_rate = (fallback_messages / total_messages * 100) if total_messages > 0 else 0

        # Missed message logic: Empty response OR clarification requested OR fallback pattern triggered
        missed_mask = (df["Response"].str.strip() == "") | (df["Clarification"].str.strip() != "") | fallback_mask
        missed_messages = missed_mask.sum()
        missed_message_rate = (missed_messages / total_messages * 100) if total_messages > 0 else 0

        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Conversations", f"{total_conversations:,}")
        col2.metric("Total Messages", f"{total_messages:,}")
        col3.metric("Bounce Rate", f"{bounce_rate:.2f}%")
        col4.metric("Missed Message Rate", f"{missed_message_rate:.2f}%")
        col5.metric("Fallback Rate", f"{fallback_rate:.2f}%")

        # Visualizations (Plotly)
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            engagement_fig = px.pie(
                names=['Bounced', 'Engaged'],
                values=[bounced_conversations, max(total_conversations - bounced_conversations, 0)],
                title='User Engagement (Bounce Rate)',
                hole=0.3,
                color_discrete_sequence=['#ff9999', '#66b3ff'],
            )
            st.plotly_chart(engagement_fig, use_container_width=True)

        with v_col2:
            response_fig = px.pie(
                names=['Success', 'Missed/Fallback'],
                values=[max(total_messages - missed_messages, 0), missed_messages],
                title='Response Quality',
                hole=0.3,
                color_discrete_sequence=['#99ff99', '#ffcc99'],
            )
            st.plotly_chart(response_fig, use_container_width=True)

        # Summary table
        st.subheader("📋 Metrics Summary")
        metrics_summary = pd.DataFrame({
            "Metric": ["Total Conversations", "Total Messages", "Bounce Rate (%)", "Missed Rate (%)", "Fallback Rate (%)"],
            "Value": [total_conversations, total_messages, round(bounce_rate, 2), round(missed_message_rate, 2), round(fallback_rate, 2)]
        })
        st.dataframe(metrics_summary, use_container_width=True)

        # Download button
        csv = metrics_summary.to_csv(index=False)
        st.download_button(label="Download Summary CSV", data=csv, file_name="conversation_metrics_summary.csv")

    except Exception as e:
        st.error(f"Error processing metrics: {str(e)}")

if __name__ == "__main__":
    run_metrics_report()
