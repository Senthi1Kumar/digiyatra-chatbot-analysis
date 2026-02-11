import re

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import parse_user_feedback


st.set_page_config(page_title="AI Performance Metrics", layout="wide")

st.title("🤖 AI Performance Metrics")
st.caption(
    "AI response feedback, intent recognition accuracy, and drop-off analysis, "
    "aligned with the AI performance section of your to-do."
)


def run_ai_metrics_dashboard():
    # ---------------- 1. LOAD & PREP DATA ----------------
    with st.spinner("Loading data..."):
        df = load_data("all_requests.csv")
        df = preprocess_data(df)

    if df.empty:
        st.warning("No data available.")
        return

    # Ensure key text columns are strings
    for col in ["Request", "Response", "User Feedback"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # ---------------- 2. AI RESPONSE FEEDBACK (CSAT-LIKE) ----------------
    st.subheader("📝 AI Response Feedback")

    feedback_df = df[df["User Feedback"].notna() & (df["User Feedback"] != "")].copy()
    if not feedback_df.empty:
        feedback_strings = feedback_df["User Feedback"].astype(str)
        parsed = feedback_strings.apply(parse_user_feedback).tolist()
        parsed_df = pd.DataFrame(parsed, index=feedback_df.index)
        feedback_df = pd.concat([feedback_df, parsed_df], axis=1)

        # Normalize rating labels
        feedback_df["rating_norm"] = (
            feedback_df["rating"].astype(str).str.lower().replace({"none": "no_feedback"})
        )
        rating_counts = feedback_df["rating_norm"].value_counts().reset_index()
        rating_counts.columns = ["Rating", "Count"]

        c1, c2 = st.columns(2)
        with c1:
            total_fb = int(len(feedback_df))
            positive_fb = int(
                rating_counts[rating_counts["Rating"].isin(["good", "positive", "thumbs_up"])]["Count"].sum()
            )
            csat_percent = (positive_fb / total_fb * 100.0) if total_fb > 0 else 0.0
            st.metric("Feedback Responses", f"{total_fb:,}")
            st.metric("Approx. CSAT (from feedback)", f"{csat_percent:.1f}%")

        with c2:
            fig_fb = px.pie(
                rating_counts,
                values="Count",
                names="Rating",
                title="AI Response Feedback Distribution",
                hole=0.4,
            )
            st.plotly_chart(fig_fb, use_container_width=True)
    else:
        st.info("No user feedback found in the dataset.")

    # ---------------- 3. INTENT RECOGNITION ACCURACY ----------------
    st.subheader("🎯 Intent Recognition Accuracy & Fallbacks")

    FALLBACK_PATTERNS = [
        r"didn.?t understand",
        r"can you rephrase",
        r"sorry",
        r"unable to",
        r"not sure",
        r"didn.?t get",
        r"could not find",
        r"no information",
        r"don't have enough context",
    ]
    fallback_regex = re.compile("|".join(FALLBACK_PATTERNS), re.IGNORECASE)

    df["is_fallback"] = df["Response"].str.contains(fallback_regex, na=False)
    total_msgs = len(df)
    fallback_msgs = int(df["is_fallback"].sum())
    successful_intents = total_msgs - fallback_msgs
    intent_accuracy = (successful_intents / total_msgs * 100.0) if total_msgs > 0 else 0.0
    fallback_rate = (fallback_msgs / total_msgs * 100.0) if total_msgs > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total Messages", f"{total_msgs:,}")
    with m2:
        st.metric("Intent Recognition Accuracy", f"{intent_accuracy:.2f}%")
    with m3:
        st.metric("Fallback Rate", f"{fallback_rate:.2f}%")

    # ---------------- 4. DROP-OFF POINTS IN CONVERSATION FLOWS ----------------
    st.subheader("📉 Drop-off Points in Conversation Flows")

    if "Conversation ID" in df.columns and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        drop_off_data = (
            df.sort_values(["Conversation ID", "Timestamp"])
            .groupby("Conversation ID")
            .tail(1)
        )

        top_drop_offs = drop_off_data["Request"].value_counts().head(10).reset_index()
        top_drop_offs.columns = ["Request", "Count"]

        if not top_drop_offs.empty:
            fig_drop = px.bar(
                top_drop_offs.sort_values("Count"),
                x="Count",
                y="Request",
                orientation="h",
                title="Top Conversation Drop-off Requests",
            )
            fig_drop.update_layout(yaxis=dict(automargin=True))
            st.plotly_chart(fig_drop, use_container_width=True)

            st.caption(
                "These are the final user requests in conversations that ended here — "
                "use them to inspect confusing flows or unmet needs."
            )
        else:
            st.info("No drop-off data available (no conversations or timestamps missing).")
    else:
        st.info("Conversation ID or Timestamp column missing; cannot compute drop-off points.")


if __name__ == "__main__":
    run_ai_metrics_dashboard()