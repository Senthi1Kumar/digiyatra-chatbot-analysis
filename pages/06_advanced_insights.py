import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import analyze_sentiment
from src.advanced_analytics import calculate_advanced_features, analyze_conversation_sentiment_flow, correlation_analysis

st.set_page_config(page_title="Insights - DigiYatra", page_icon="🔬", layout="wide")

st.title("Insights")
st.markdown("Statistical analysis to uncover hidden user behaviors and system anomalies.")

with st.spinner("Loading & Processing Data (this may take a moment)..."):
    df = load_data("all_requests.csv")
    df = preprocess_data(df)
    
    # Calculate Basic Sentiment if not cached
    # For advanced page, we might need to run this on a sample if full data is too big
    # But let's try sample for interactivity
    df_sample = df.sample(n=min(20000, len(df)), random_state=42).copy()
    
    # 1. Advanced Features on Message Level
    df_sample = calculate_advanced_features(df_sample)
    
    # 2. Conversation Level Sentiment Flow (Needs sentiment first)
    df_sample = analyze_sentiment(df_sample, text_col='Request')
    conv_flow = analyze_conversation_sentiment_flow(df_sample)

# --- Frustration Analysis ---
st.subheader("1. User Frustration Index (Sentiment Flow)")
st.caption("Tracking how user sentiment changes from the start to the end of a conversation.")

start_vs_end_col, frust_col = st.columns(2)

with start_vs_end_col:
    # Scatter plot: Start Sentiment vs End Sentiment
    # Points below the diagonal mean sentiment got worse
    fig_flow = px.scatter(
        conv_flow, 
        x='Start_Sentiment', 
        y='End_Sentiment', 
        color='Sentiment_Change',
        size='Turn_Count',
        color_continuous_scale='RdYlGn',
        title="Sentiment Migration: Start vs End",
        labels={'Start_Sentiment': 'Sentiment at Start', 'End_Sentiment': 'Sentiment at End'}
    )
    # Add diagonal line
    fig_flow.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="Gray", dash="dash"))
    st.plotly_chart(fig_flow, width='stretch')

with frust_col:
    # Distribution of Sentiment Change
    fig_hist = px.histogram(
        conv_flow, 
        x='Sentiment_Change', 
        nbins=30, 
        title="Distribution of Sentiment Change",
        color_discrete_sequence=['#ffaa00']
    )
    st.plotly_chart(fig_hist, width='stretch')

# Frustrated Conversations Count
frustrated_count = conv_flow['Is_Frustrated'].sum()
st.metric("Potential Frustrated Conversations (in sample)", f"{frustrated_count} ({frustrated_count/len(conv_flow)*100:.1f}%)", 
          help="Conversations > 2 turns where sentiment dropped significantly or ended negative.")

# --- Operational Anomalies ---
st.subheader("2. Operational Anomalies (Z-Score)")
st.caption("Identifying statistical outliers in system latency using Z-Score analysis (> 3-sigma).")

anomalies = df_sample[df_sample['Is_Latency_Anomaly']]
if not anomalies.empty:
    fig_anom = px.scatter(
        df_sample, 
        x='Timestamp', 
        y='Latency', 
        color='Is_Latency_Anomaly', 
        color_discrete_map={True: 'red', False: 'blue'},
        title="Latency Anomalies Over Time",
        hover_data=['Request']
    )
    st.plotly_chart(fig_anom, width='stretch')
else:
    st.info("No statistical latency anomalies found in sample.")

# --- Correlation Analysis ---
st.subheader("3. Multi-variate Correlation Map")
st.caption("Discovering hidden relationships between Cost, Latency, Length, and Sentiment.")

corr_matrix = correlation_analysis(df_sample)

if not corr_matrix.empty:
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale='RdBu_r', 
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, width='stretch')
