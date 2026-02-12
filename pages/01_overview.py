import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.descriptive_analytics import get_key_metrics, get_status_distribution, get_top_users

st.set_page_config(page_title="Overview - DigiYatra Analytics", page_icon="📊", layout="wide")

st.title("📊 Executive Overview")

# Load & Preprocess
with st.spinner("Loading data..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    df = preprocess_data(df)

if df.empty:
    st.error("No data available.")
    st.stop()

# --- Ensure Timestamp is datetime ---
if 'Timestamp' in df.columns and df['Timestamp'].dtype == 'object':
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y, %H:%M:%S', errors='coerce')

# Create a Date column for easier filtering (keep as datetime, not date object)
if 'Timestamp' in df.columns:
    df['Date'] = df['Timestamp'].dt.normalize()

# --- Date Filter ---
# Safely extract min/max dates, handling NaT values
if 'Timestamp' in df.columns and pd.notna(df['Timestamp'].min()) and pd.notna(df['Timestamp'].max()):
    min_date = df['Timestamp'].min().date()
    max_date = df['Timestamp'].max().date()
else:
    st.warning("⚠️ Timestamp column is missing or contains invalid data.")
    min_date = pd.Timestamp.now().date()
    max_date = pd.Timestamp.now().date()

col1, col2 = st.columns([2, 1])
with col2:
    date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
    if 'Date' in df.columns:
        # Convert date objects to datetime64 for proper comparison
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        mask = (df['Date'] >= start_dt) & (df['Date'] <= end_dt)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df
else:
    filtered_df = df# --- KPIs ---
metrics = get_key_metrics(filtered_df)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Total Requests", f"{metrics.get('total_requests', 0):,}")
kpi2.metric("Unique Sessions", f"{metrics.get('total_users', 0):,}", help="Based on unique Conversation IDs")
kpi3.metric("Total Cost", f"${metrics.get('total_cost', 0):,.2f}")
kpi4.metric("Avg Latency", f"{metrics.get('avg_latency', 0):.2f}s")
kpi5.metric("Success Rate", f"{metrics.get('success_rate', 0):.1f}%")

st.markdown("---")

# --- Charts ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Request Volume Over Time")
    # Resample for the chart based on range
    if (end_date - start_date).days > 30:
        freq = 'd'
    else:
        freq = 'h'
        
    vol_chart = filtered_df.set_index('Timestamp').resample(freq).size().reset_index(name='Requests')
    fig_vol = px.area(vol_chart, x='Timestamp', y='Requests', title=f"Volume ({freq})")
    st.plotly_chart(fig_vol, width='stretch')

with c2:
    st.subheader("Status Distribution")
    status_dist = get_status_distribution(filtered_df)
    if not status_dist.empty:
        fig_status = px.pie(status_dist, names='Status', values='count', hole=0.4, title="Request Status")
        st.plotly_chart(fig_status, width='stretch')
    else:
        st.info("No status data available.")

# --- Hourly Heatmap ---
st.subheader("Peak Usage Patterns (Heatmap)")
if 'Hour' in filtered_df.columns and 'DayOfWeek' in filtered_df.columns:
    heatmap_data = filtered_df.groupby(['DayOfWeek', 'Hour']).size().reset_index(name='Count')
    # Sort days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x='Hour', 
        y='DayOfWeek', 
        z='Count', 
        category_orders={'DayOfWeek': days_order},
        color_continuous_scale='Viridis',
        title="Request Intensity by Day & Hour"
    )
    st.plotly_chart(fig_heat, width='stretch')

# --- Top 10 Active Sessions ---
st.subheader("Top 10 Active Sessions (by Message Count)")
top_users = get_top_users(filtered_df, n=10)
if not top_users.empty:
    fig_top = px.bar(top_users, x='Conversation ID', y='Message Count', title="Most Active Sessions")
    st.plotly_chart(fig_top, width='stretch')
else:
    st.info("No conversation data available.")

