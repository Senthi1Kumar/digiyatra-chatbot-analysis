import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import (
    analyze_sentiment,
    extract_top_keywords,
    categorise_intent_basic,
    detect_language,
    parse_user_feedback,
    add_frustration_index,
    intent_summary,
    language_quality_summary,
    cluster_unseen_queries,
    compute_ood_metrics,
    compute_csat_metrics,
    compute_chat_duration_metrics,
    compute_clarification_engagement_metrics,
)

st.set_page_config(page_title="NLP Insights - DigiYatra", page_icon="🧠", layout="wide")

st.title("🧠 NLP & Sentiment Analysis")

with st.spinner("Loading data..."):
    df = load_data("all_requests.csv")
    df = preprocess_data(df)

if df.empty:
    st.stop()

# --- Sample Control for Performance ---
st.sidebar.subheader("Analysis Settings")
sample_size = st.sidebar.slider("Sample Size (for heavy NLP tasks)", 1000, 50000, 10000, step=1000)
use_sample = st.sidebar.checkbox("Use Sample", value=True)

if use_sample:
    df_analysis = df.sample(n=min(sample_size, len(df)), random_state=42)
else:
    df_analysis = df

# --- Ensure we have basic rule-based intents for downstream analytics ---
with st.spinner("Categorizing intents (rule-based)..."):
    if 'Intent' not in df_analysis.columns:
        df_analysis['Intent'] = df_analysis['Request'].apply(categorise_intent_basic)

# # --- Intent Classification ---
# st.subheader("Intent Distribution")
# with st.spinner("Categorizing intents..."):
#     df_analysis['Intent'] = df_analysis['Request'].apply(categorise_intent_basic)

# intent_counts = df_analysis['Intent'].value_counts().reset_index()
# intent_counts.columns = ['Intent', 'Count']

# fig_intent = px.bar(intent_counts, x='Intent', y='Count', color='Intent', title="Identified User Intents (Rule-based)")
# st.plotly_chart(fig_intent, width='stretch')

# --- Sentiment Analysis ---
st.subheader("Sentiment & Frustration")
with st.spinner("Analyzing sentiment & frustration..."):
    df_analysis = analyze_sentiment(df_analysis, text_col='Request')
    # Best-effort frustration index; works even if some optional cols missing
    df_analysis = add_frustration_index(
        df_analysis,
        text_col='Request',
        conv_id_col=df_analysis.columns.intersection(['Conversation ID', 'conversation_id']).tolist()[0]
        if len(df_analysis.columns.intersection(['Conversation ID', 'conversation_id'])) > 0
        else 'Conversation ID',
    )

col1, col2 = st.columns(2)
with col1:
    fig_sent = px.histogram(df_analysis, x='Sentiment_Polarity', nbins=20, title="Sentiment Polarity Distribution (-1 to 1)", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_sent, width='stretch')

with col2:
    # Frustration distribution
    if 'Frustration_Score' in df_analysis.columns:
        fig_frust = px.histogram(
            df_analysis,
            x='Frustration_Score',
            nbins=20,
            title="Frustration Index Distribution (0 = calm, 1 = very frustrated)",
            color_discrete_sequence=['#EF553B'],
        )
        st.plotly_chart(fig_frust, width='stretch')
    else:
        # Fallback to scatter if for some reason frustration is missing
        fig_scatter = px.scatter(
            df_analysis,
            x='Sentiment_Polarity',
            y='Sentiment_Subjectivity',
            color='Intent',
            title="Sentiment vs Subjectivity",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, width='stretch')

# --- Pain Points / Intent-Level Summary ---
st.subheader("Top User Pain Points by Intent")
intent_summary_df = intent_summary(df_analysis)
if not intent_summary_df.empty:
    c1, c2 = st.columns(2)
    with c1:
        fig_intent_pain = px.bar(
            intent_summary_df.head(10),
            x='Intent',
            y='Count',
            color='Intent',
            title="Top 10 Intents by Volume",
        )
        st.plotly_chart(fig_intent_pain, width='stretch')
    with c2:
        fig_intent_frust = px.bar(
            intent_summary_df.head(10),
            x='Intent',
            y='Avg_Conv_Frustration',
            color='Avg_Conv_Frustration',
            title="Average Conversation Frustration by Intent",
            color_continuous_scale='Reds',
        )
        st.plotly_chart(fig_intent_frust, width='stretch')
    st.dataframe(intent_summary_df, use_container_width=True)

# --- Language Detection ---
st.subheader("🌐 Multilingual Quality Analysis")
with st.spinner("Detecting languages (sample)..."):
    # Sample for speed
    lang_sample = df_analysis.head(5000).copy()
    lang_sample['Language'] = lang_sample['Request'].apply(detect_language)

lang_counts = lang_sample['Language'].value_counts().reset_index()
lang_counts.columns = ['Language', 'Count']

lang_quality_df = language_quality_summary(lang_sample)

l1, l2 = st.columns(2)
with l1:
    fig_lang = px.pie(lang_counts.head(10), values='Count', names='Language', title="Language Distribution (Top 10)", hole=0.4)
    st.plotly_chart(fig_lang, width='stretch')

with l2:
    if not lang_quality_df.empty:
        fig_lang_quality = px.bar(
            lang_quality_df,
            x='Language',
            y='Resolution_Rate',
            color='Avg_Conv_Frustration',
            title="Resolution Rate & Frustration by Language",
            color_continuous_scale='Reds',
        )
        st.plotly_chart(fig_lang_quality, width='stretch')
    else:
        st.info("Not enough data to compute language quality metrics.")

st.caption("Below are sample non-English requests detected in the analysis sample.")
hindi_samples = lang_sample[lang_sample['Language'].isin(['hi', 'mr', 'ta', 'te', 'bn'])]
if not hindi_samples.empty:
    st.dataframe(hindi_samples[['Request', 'Language']].head(10), width='stretch')
else:
    st.info("No non-English requests detected in sample.")

# --- User Feedback Sentiment ---
st.subheader("📝 User Feedback Sentiment")
# Use FULL dataframe for feedback as it's sparse
feedback_df = df[df['User Feedback'].notna() & (df['User Feedback'] != '')].copy()
if not feedback_df.empty:
    # Parse feedback - safe method handling potential categorical types
    # Ensure it's string first
    feedback_strings = feedback_df['User Feedback'].astype(str)
    parsed_data = feedback_strings.apply(parse_user_feedback).tolist()
    parsed_df = pd.DataFrame(parsed_data, index=feedback_df.index)
    
    feedback_df = pd.concat([feedback_df, parsed_df], axis=1)
    
    fb1, fb2 = st.columns(2)
    with fb1:
        rating_counts = feedback_df['rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_rating = px.pie(rating_counts, values='Count', names='Rating', title="User Ratings", hole=0.4, color_discrete_map={'good': 'green', 'bad': 'red'})
        st.plotly_chart(fig_rating, width='stretch')
    
    with fb2:
        # Sentiment of comments
        comments_df = feedback_df[feedback_df['comments'].notna() & (feedback_df['comments'] != '')]
        if not comments_df.empty:
            comments_df = analyze_sentiment(comments_df, text_col='comments')
            fig_fb_sent = px.histogram(comments_df, x='Sentiment_Polarity', nbins=15, color='rating', title="Comment Sentiment by Rating")
            st.plotly_chart(fig_fb_sent, width='stretch')
        else:
            st.info("No text comments in feedback.")
else:
    st.warning("No user feedback data available.")

# --- Top Keywords ---
st.subheader("Top Keywords (TF-IDF)")
with st.spinner("Extracting keywords..."):
    keywords_df = extract_top_keywords(df_analysis, text_col='Request', n=20)

if not keywords_df.empty:
    fig_words = px.bar(keywords_df, x='count', y='word', orientation='h', title="Top 20 Key Terms", color='count')
    fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_words, width='stretch')

# # --- Token & Cost Overview ---
# st.subheader("🧾 Token & Cost Overview")
# token_metrics_df = compute_token_metrics(df)
# if not token_metrics_df.empty:
#     tm = token_metrics_df.iloc[0]
#     m1, m2, m3, m4 = st.columns(4)
#     with m1:
#         st.metric("Total Conversations", int(tm.get("Total Conversations", 0)))
#     with m2:
#         st.metric("Total Tokens", f"{int(tm.get('Sum Total Tokens', 0)):,}")
#     with m3:
#         st.metric("Total Cost", f"${tm.get('Sum Cost', 0):.2f}")
#     with m4:
#         if "Prompt_Token_Share" in tm:
#             st.metric(
#                 "Prompt vs Completion",
#                 f"{tm['Prompt_Token_Share']*100:.1f}% / {tm['Completion_Token_Share']*100:.1f}%",
#                 help="Share of prompt tokens vs completion tokens.",
#             )
# else:
#     st.info("Token / cost columns not available in the dataset.")

# --- Conversation & Engagement KPIs ---
st.subheader("📊 Core Conversation KPIs")

ood_row = compute_ood_metrics(df_analysis).iloc[0] if not compute_ood_metrics(df_analysis).empty else None
csat_row = compute_csat_metrics(df).iloc[0] if not compute_csat_metrics(df).empty else None
duration_row = compute_chat_duration_metrics(df).iloc[0] if not compute_chat_duration_metrics(df).empty else None
engagement_row = compute_clarification_engagement_metrics(df).iloc[0] if not compute_clarification_engagement_metrics(df).empty else None

k1, k2, k3, k4 = st.columns(4)

with k1:
    if ood_row is not None:
        st.metric(
            "Out-of-domain Requests",
            int(ood_row["OOD_Requests"]),
            f"{ood_row['OOD_Rate']*100:.1f}% of all requests",
        )
    else:
        st.metric("Out-of-domain Requests", "N/A")

with k2:
    if csat_row is not None:
        st.metric(
            "Customer Satisfaction Score (CSAT)",
            f"{csat_row['CSAT_Percent']:.1f}%",
            help="Percentage of positive ratings from User Feedback.",
        )
    else:
        st.metric("CSAT", "N/A")

with k3:
    if duration_row is not None:
        avg_conv_min = duration_row["Avg_Conversation_Duration_Seconds"] / 60.0
        st.metric(
            "Avg Chat Duration",
            f"{avg_conv_min:.1f} min",
            help="Average duration between first and last message per conversation.",
        )
    else:
        st.metric("Avg Chat Duration", "N/A")

with k4:
    if engagement_row is not None:
        st.metric(
            "Clarification Engagement",
            f"{engagement_row['Engagement_Rate_Requests']*100:.1f}%",
            help="Share of requests where users interacted with Clarification options.",
        )
    else:
        st.metric("Clarification Engagement", "N/A")

# --- Emerging Topics (Unseen Intents) ---
st.subheader("🔍 Emerging Topics (Unseen Intents)")
with st.spinner("Discovering clusters of General/Other queries..."):
    clusters_df = cluster_unseen_queries(df_analysis, text_col='Request', intent_col='Intent')

if not clusters_df.empty:
    st.dataframe(clusters_df.head(15), use_container_width=True)
else:
    st.info("Not enough 'General/Other' queries to form meaningful clusters right now.")

# # --- Word Cloud ---
# st.subheader("Word Cloud")
# from src.nlp_analytics import generate_wordcloud_img
# with st.spinner("Generating Word Cloud..."):
#     # Generate for selected intent if any, or all
#     # We'll use the filtered analysis dataframe
#     fig_wc = generate_wordcloud_img(df_analysis['Request'])
    
#     if fig_wc:
#         st.pyplot(fig_wc)
#     else:
#         st.info("Not enough text data for Word Cloud.")

# --- Raw Data Explorer ---
st.subheader("Explore Raw Queries")
selected_intent = st.selectbox("Filter by Intent", ['All'] + list(df_analysis['Intent'].unique()))

if selected_intent != 'All':
    display_df = df_analysis[df_analysis['Intent'] == selected_intent]
else:
    display_df = df_analysis

st.dataframe(display_df[['Timestamp', 'Request', 'Response', 'Intent', 'Sentiment_Polarity']].head(50), width='stretch')
