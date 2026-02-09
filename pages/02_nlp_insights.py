import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import clean_text, analyze_sentiment, extract_top_keywords, categorise_intent_basic

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

# --- Intent Classification ---
st.subheader("Intent Distribution")
with st.spinner("Categorizing intents..."):
    df_analysis['Intent'] = df_analysis['Request'].apply(categorise_intent_basic)

intent_counts = df_analysis['Intent'].value_counts().reset_index()
intent_counts.columns = ['Intent', 'Count']

fig_intent = px.bar(intent_counts, x='Intent', y='Count', color='Intent', title="Identified User Intents (Rule-based)")
st.plotly_chart(fig_intent, width='stretch')

# --- Sentiment Analysis ---
st.subheader("Sentiment Analysis")
with st.spinner("Analyzing sentiment..."):
    df_analysis = analyze_sentiment(df_analysis, text_col='Request')

col1, col2 = st.columns(2)
with col1:
    fig_sent = px.histogram(df_analysis, x='Sentiment_Polarity', nbins=20, title="Sentiment Polarity Distribution (-1 to 1)", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_sent, width='stretch')

with col2:
    # Scatter of Sentiment vs Subjectivity
    fig_scatter = px.scatter(df_analysis, x='Sentiment_Polarity', y='Sentiment_Subjectivity', color='Intent', title="Sentiment vs Subjectivity", opacity=0.6)
    st.plotly_chart(fig_scatter, width='stretch')

# --- Top Keywords ---
st.subheader("Top Keywords (TF-IDF)")
with st.spinner("Extracting keywords..."):
    keywords_df = extract_top_keywords(df_analysis, text_col='Request', n=20)

if not keywords_df.empty:
    fig_words = px.bar(keywords_df, x='count', y='word', orientation='h', title="Top 20 Key Terms", color='count')
    fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_words, width='stretch')

# --- Word Cloud ---
st.subheader("Word Cloud")
from src.nlp_analytics import generate_wordcloud_img
with st.spinner("Generating Word Cloud..."):
    # Generate for selected intent if any, or all
    # We'll use the filtered analysis dataframe
    fig_wc = generate_wordcloud_img(df_analysis['Request'])
    
    if fig_wc:
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text data for Word Cloud.")

# --- Raw Data Explorer ---
st.subheader("Explore Raw Queries")
selected_intent = st.selectbox("Filter by Intent", ['All'] + list(df_analysis['Intent'].unique()))

if selected_intent != 'All':
    display_df = df_analysis[df_analysis['Intent'] == selected_intent]
else:
    display_df = df_analysis

st.dataframe(display_df[['Timestamp', 'Request', 'Response', 'Intent', 'Sentiment_Polarity']].head(50), width='stretch')
