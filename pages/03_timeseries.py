import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.timeseries_analytics import resample_time_series, get_busiest_periods

st.set_page_config(page_title="Time Series - DigiYatra", page_icon="📊", layout="wide")

st.title("📊 Time-Series Analysis")

with st.spinner("Loading data..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    df = preprocess_data(df)

if df.empty:
    st.stop()

# --- Resampling Controls ---
col1, col2 = st.columns(2)
with col1:
    freq = st.selectbox("Resampling Frequency", ["H", "D", "W", "M"], index=1, format_func=lambda x: {"H":"Hourly", "D":"Daily", "W":"Weekly", "M":"Monthly"}[x])

# --- Volume Trend ---
st.subheader("Request Volume Trend")

ts_data = resample_time_series(df, freq=freq, metric='count')

if not ts_data.empty:
    fig_trend = px.line(ts_data, x='Timestamp', y='Requests', title=f"Requests over Time ({freq})")
    fig_trend.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig_trend, width='stretch')

# --- Seasonality / Patterns ---
st.subheader("Cyclical Patterns")

hourly_counts, daily_counts = get_busiest_periods(df)

c1, c2 = st.columns(2)

with c1:
    if not hourly_counts.empty:
        fig_hourly = px.bar(x=hourly_counts.index, y=hourly_counts.values, title="Total Requests by Hour of Day", labels={'x':'Hour (0-23)', 'y':'Total Requests'})
        st.plotly_chart(fig_hourly, width='stretch')

with c2:
    if not daily_counts.empty:
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_df = daily_counts.reindex(days_order).reset_index()
        daily_df.columns = ['Day', 'Count']
        
        fig_daily = px.bar(daily_df, x='Day', y='Count', title="Total Requests by Day of Week", color='Count')
        st.plotly_chart(fig_daily, width='stretch')

# --- Rolling Average ---
st.subheader("Smoothing / Rolling Average")
window = st.slider("Rolling Window Size", 1, 30, 7)

if not ts_data.empty:
    ts_data['Rolling_Avg'] = ts_data['Requests'].rolling(window=window).mean()
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=ts_data['Timestamp'], y=ts_data['Requests'], mode='lines', name='Actual', opacity=0.4))
    fig_roll.add_trace(go.Scatter(x=ts_data['Timestamp'], y=ts_data['Rolling_Avg'], mode='lines', name=f'{window}-Period Moving Avg', line=dict(width=3, color='red')))
    fig_roll.update_layout(title="Volume with Moving Average")
    st.plotly_chart(fig_roll, width='stretch')
