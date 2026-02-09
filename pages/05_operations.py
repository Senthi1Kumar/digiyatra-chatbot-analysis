import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.descriptive_analytics import get_latency_stats

st.set_page_config(page_title="Operations - DigiYatra", page_icon="⚙️", layout="wide")

st.title("⚙️ Operations & Cost Analysis")

with st.spinner("Loading data..."):
    df = load_data("all_requests.csv")
    df = preprocess_data(df)

if df.empty:
    st.stop()

# --- Latency Analysis ---
st.subheader("System Latency")
lat_stats = get_latency_stats(df)

l1, l2, l3, l4 = st.columns(4)
l1.metric("p50 (Median)", f"{lat_stats.get('p50',0):.2f}s")
l2.metric("p95 (Slow)", f"{lat_stats.get('p95',0):.2f}s")
l3.metric("p99 (Very Slow)", f"{lat_stats.get('p99',0):.2f}s")
l4.metric("Max Latency", f"{lat_stats.get('max',0):.2f}s")

# Latency Histogram
fig_lat = px.histogram(df[df['Latency'] < 10], x='Latency', nbins=50, title="Latency Distribution (filtered < 10s)", color_discrete_sequence=['#EF553B'])
st.plotly_chart(fig_lat, width='stretch')

# --- Cost Analysis ---
st.subheader("Cost & Token Usage")

c1, c2 = st.columns(2)

with c1:
    # Cost over time
    daily_cost = df.groupby('Date')['Cost'].sum().reset_index()
    fig_cost = px.line(daily_cost, x='Date', y='Cost', title="Daily Operational Cost ($)")
    st.plotly_chart(fig_cost, width='stretch')

    # Monthly Cost
    monthly_df = df.groupby(df['Timestamp'].dt.to_period('M'))['Cost'].sum().reset_index()
    monthly_df['Timestamp'] = monthly_df['Timestamp'].dt.to_timestamp()

    fig2 = px.line(
        monthly_df, 
        x='Timestamp', 
        y='Cost', 
        markers=True,
        title='Total Monthly Cost (Min Date to Max Date)',
        labels={'Timestamp': 'Month', 'Cost': 'Total Cost (USD)'}
    )
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, width='stretch')

with c2:
    # Token distribution
    if 'Prompt Tokens' in df.columns and 'Completion Tokens' in df.columns:
        token_df = pd.DataFrame({
            'Type': ['Prompt', 'Completion'],
            'Tokens': [df['Prompt Tokens'].sum(), df['Completion Tokens'].sum()]
        })
        fig_tokens = px.pie(token_df, names='Type', values='Tokens', title="Token Usage Split", hole=0.4)
        st.plotly_chart(fig_tokens, width='stretch')

# --- Errors & Failures ---
st.subheader("Error Analysis")
failures = df[df['Status'] != 'success']

if not failures.empty:
    st.warning(f"Found {len(failures)} failed requests ({len(failures)/len(df)*100:.2f}%)")
    
    # Failure reasons if available
    # Assuming 'Response' might contain error messages in failed state
    st.dataframe(failures[['Timestamp', 'Status', 'Response', 'Latency']].head(50), width='stretch')
else:
    st.success("No failed requests found in the dataset.")
