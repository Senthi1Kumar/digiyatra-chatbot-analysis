import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.temporal_cohort_analysis import (
    analyze_hourly_patterns,
    detect_seasonal_trends,
    cohort_analysis,
    predict_peak_loads,
    analyze_user_journey_timing,
    identify_anomalous_periods,
    generate_temporal_insights_report
)

st.set_page_config(
    page_title="Temporal Analysis - DigiYatra", 
    page_icon="⏰", 
    layout="wide"
)

st.title("⏰ Temporal & Cohort Analysis")
st.markdown("""
**Time-based intelligence for capacity planning and trend detection**
""")

# Load data
with st.spinner("Loading and analyzing temporal patterns..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    
    df = preprocess_data(df)
    
    # Cache temporal analysis
    if 'temporal_analysis_done' not in st.session_state:
        with st.spinner("📊 Running temporal analysis..."):
            temporal_report = generate_temporal_insights_report(df)
            
            st.session_state.temporal_report = temporal_report
            st.session_state.temporal_analysis_done = True
    
    temporal_report = st.session_state.temporal_report

# ============================================================================
# HOURLY & DAILY PATTERNS
# ============================================================================

st.header("⏰ Usage Patterns")

hourly_data = temporal_report.get('hourly_patterns', {})

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    busiest_hour = hourly_data.get('busiest_hour', 'N/A')
    st.metric(
        "🔥 Busiest Hour", 
        f"{busiest_hour}:00" if isinstance(busiest_hour, int) else busiest_hour,
        help="Hour of day with highest request volume"
    )

with col2:
    quietest_hour = hourly_data.get('quietest_hour', 'N/A')
    st.metric(
        "😴 Quietest Hour", 
        f"{quietest_hour}:00" if isinstance(quietest_hour, int) else quietest_hour,
        help="Hour of day with lowest request volume"
    )

with col3:
    busiest_day = hourly_data.get('busiest_day', 'N/A')
    st.metric(
        "📅 Busiest Day", 
        busiest_day,
        help="Day of week with highest request volume"
    )

with col4:
    quietest_day = hourly_data.get('quietest_day', 'N/A')
    st.metric(
        "📅 Quietest Day", 
        quietest_day,
        help="Day of week with lowest request volume"
    )

# Hourly volume chart
st.subheader("📊 Hourly Traffic Distribution")

hourly_stats = hourly_data.get('hourly_stats', {})
if hourly_stats:
    hourly_df = pd.DataFrame.from_dict(hourly_stats, orient='index').reset_index()
    hourly_df.columns = ['Hour', 'Volume', 'Avg_Latency', 'Total_Cost']
    hourly_df = hourly_df.sort_values('Hour')
    
    # Identify peak vs off-peak
    peak_hours = hourly_data.get('peak_hours', [])
    hourly_df['Period'] = hourly_df['Hour'].apply(lambda x: 'Peak' if x in peak_hours else 'Off-Peak')
    
    fig_hourly = px.bar(
        hourly_df,
        x='Hour',
        y='Volume',
        color='Period',
        color_discrete_map={'Peak': '#FF6B6B', 'Off-Peak': '#4ECDC4'},
        title="Request Volume by Hour of Day",
        labels={'Hour': 'Hour of Day (0-23)', 'Volume': 'Number of Requests'}
    )
    
    st.plotly_chart(fig_hourly, width='stretch')
    
    # Latency by hour
    fig_latency = px.line(
        hourly_df,
        x='Hour',
        y='Avg_Latency',
        title="Average Response Latency by Hour",
        markers=True,
        labels={'Hour': 'Hour of Day', 'Avg_Latency': 'Latency (seconds)'}
    )
    st.plotly_chart(fig_latency, width='stretch')

# Daily patterns
st.subheader("📅 Day-of-Week Patterns")

daily_stats = hourly_data.get('daily_stats', {})
if daily_stats:
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_df = pd.DataFrame.from_dict(daily_stats, orient='index').reset_index()
    daily_df.columns = ['Day', 'Volume', 'Avg_Latency']
    daily_df['Day'] = pd.Categorical(daily_df['Day'], categories=days_order, ordered=True)
    daily_df = daily_df.sort_values('Day')
    
    fig_daily = px.bar(
        daily_df,
        x='Day',
        y='Volume',
        color='Volume',
        color_continuous_scale='Blues',
        title="Request Volume by Day of Week"
    )
    st.plotly_chart(fig_daily, width='stretch')

# Peak hours list
with st.expander("📋 Peak Hours Details"):
    peak_hours = hourly_data.get('peak_hours', [])
    off_peak_hours = hourly_data.get('off_peak_hours', [])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Peak Hours (High Traffic):**")
        st.write(f"{', '.join([f'{h}:00' for h in sorted(peak_hours)])}")
    with col_b:
        st.markdown("**Off-Peak Hours (Low Traffic):**")
        st.write(f"{', '.join([f'{h}:00' for h in sorted(off_peak_hours[:5])])}")

st.markdown("---")

# ============================================================================
# SEASONAL TRENDS
# ============================================================================

st.header("📈 Seasonal Trends & Growth")

seasonal_data = temporal_report.get('seasonal_trends', {})

col1, col2, col3 = st.columns(3)

with col1:
    trend = seasonal_data.get('overall_trend', 'N/A')
    if trend == 'Growing':
        st.success(f"📈 **{trend}**")
    elif trend == 'Declining':
        st.error(f"📉 **{trend}**")
    else:
        st.info(f"➡️ **{trend}**")
    st.caption("Overall Trend")

with col2:
    recent_avg = seasonal_data.get('recent_avg_daily_volume', 0)
    st.metric(
        "Recent Daily Avg", 
        f"{recent_avg:.0f} requests/day",
        help="Average daily request volume over the last 7 days"
    )

with col3:
    volatility = seasonal_data.get('volatility', 0)
    st.metric(
        "Volatility (σ)", 
        f"{volatility:.0f}",
        help="Standard deviation of daily volume. Lower values indicate more stable/predictable traffic"
    )

# Daily volume trend
st.subheader("📊 Daily Volume Trend")

daily_volume = seasonal_data.get('daily_volume', [])
if daily_volume:
    daily_df = pd.DataFrame(daily_volume)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    fig_trend = go.Figure()
    
    # Actual volume
    fig_trend.add_trace(go.Scatter(
        x=daily_df['Date'],
        y=daily_df['volume'],
        mode='lines',
        name='Actual Volume',
        line=dict(color='lightblue', width=1),
        opacity=0.6
    ))
    
    # Rolling average
    fig_trend.add_trace(go.Scatter(
        x=daily_df['Date'],
        y=daily_df['rolling_avg'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='red', width=3)
    ))
    
    fig_trend.update_layout(
        title="Daily Request Volume with Trend Line",
        xaxis_title="Date",
        yaxis_title="Requests",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trend, width='stretch')

# Growth & decline periods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Growth Periods")
    growth_periods = seasonal_data.get('growth_periods', [])
    if growth_periods:
        growth_df = pd.DataFrame(growth_periods)
        growth_df['Date'] = pd.to_datetime(growth_df['Date']).dt.date
        growth_df['Growth Rate'] = (growth_df['growth_rate'] * 100).round(1)
        st.dataframe(growth_df[['Date', 'Growth Rate']].head(5), width='stretch')
    else:
        st.info("No significant growth periods detected")

with col2:
    st.subheader("📉 Decline Periods")
    decline_periods = seasonal_data.get('decline_periods', [])
    if decline_periods:
        decline_df = pd.DataFrame(decline_periods)
        decline_df['Date'] = pd.to_datetime(decline_df['Date']).dt.date
        decline_df['Decline Rate'] = (decline_df['growth_rate'] * 100).round(1)
        st.dataframe(decline_df[['Date', 'Decline Rate']].head(5), width='stretch')
    else:
        st.info("No significant decline periods detected")

st.markdown("---")

# ============================================================================
# PEAK LOAD FORECASTING
# ============================================================================

st.header("🔮 Peak Load Forecasting")

forecast_data = temporal_report.get('peak_load_forecast', {})

col1, col2 = st.columns(2)

with col1:
    hist_max = forecast_data.get('historical_max_daily', 0)
    st.metric(
        "Historical Peak", 
        f"{hist_max:,} requests/day",
        help="Highest daily request volume observed in the dataset"
    )
    
    hist_avg = forecast_data.get('historical_avg_daily', 0)
    st.metric(
        "Historical Average", 
        f"{hist_avg:.0f} requests/day",
        help="Average daily request volume across entire dataset"
    )

with col2:
    current_trend = forecast_data.get('current_trend', 'N/A')
    st.info(f"**Current Trend:** {current_trend.capitalize()}")
    
    capacity_rec = forecast_data.get('capacity_recommendation', '')
    st.warning(f"**Capacity Planning:** {capacity_rec}")

# 7-day forecast
st.subheader("📅 7-Day Volume Forecast")

forecast = forecast_data.get('forecast', [])
if forecast:
    forecast_df = pd.DataFrame(forecast)
    
    fig_forecast = px.bar(
        forecast_df,
        x='day_offset',
        y='predicted_volume',
        color='confidence',
        color_discrete_map={'high': '#00CC66', 'medium': '#FFAA00', 'low': '#FF4B4B'},
        title="Predicted Request Volume (Next 7 Days)",
        labels={'day_offset': 'Days from Now', 'predicted_volume': 'Predicted Requests'},
        text='predicted_volume'
    )
    fig_forecast.update_traces(texttemplate='%{text}', textposition='outside')
    
    st.plotly_chart(fig_forecast, width='stretch')
    
    st.caption("""
    **Confidence Levels:**
    - 🟢 High: Days 1-3 (based on recent patterns)
    - 🟡 Medium: Days 4-5 (moderate uncertainty)
    - 🔴 Low: Days 6-7 (high uncertainty)
    """)

# Peak hours in forecast
peak_hours_forecast = forecast_data.get('peak_hours', [])
if peak_hours_forecast:
    st.info(f"**Expected Peak Hours:** {', '.join([f'{h}:00' for h in peak_hours_forecast])}")

st.markdown("---")

# ============================================================================
# USER JOURNEY TIMING
# ============================================================================

st.header("⏱️ User Journey Timing Analysis")

timing_data = temporal_report.get('user_journey_timing', {})

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_duration = timing_data.get('avg_session_duration_seconds', 0)
    st.metric(
        "Avg Session Duration", 
        f"{avg_duration:.0f}s",
        help="Mean time from first message to last message in a conversation"
    )

with col2:
    median_duration = timing_data.get('median_session_duration_seconds', 0)
    st.metric(
        "Median Duration", 
        f"{median_duration:.0f}s",
        help="Middle value of session durations (less affected by outliers)"
    )

with col3:
    p95_duration = timing_data.get('p95_session_duration_seconds', 0)
    st.metric(
        "95th Percentile", 
        f"{p95_duration:.0f}s",
        help="95% of sessions complete within this time"
    )

with col4:
    avg_messages = timing_data.get('avg_messages_per_session', 0)
    st.metric(
        "Avg Messages/Session", 
        f"{avg_messages:.1f}",
        help="Average number of messages (user + bot) per conversation"
    )

# Session type distribution
st.subheader("📊 Session Complexity Distribution")

session_types = pd.DataFrame([
    {'Type': 'Quick (1-2 messages)', 'Count': timing_data.get('quick_sessions', 0), 'Color': 'green'},
    {'Type': 'Normal (3-5 messages)', 'Count': timing_data.get('normal_sessions', 0), 'Color': 'blue'},
    {'Type': 'Complex (6+ messages)', 'Count': timing_data.get('complex_sessions', 0), 'Color': 'red'}
])

fig_sessions = px.pie(
    session_types,
    values='Count',
    names='Type',
    title="Session Types by Message Count",
    hole=0.4,
    color='Type',
    color_discrete_map={
        'Quick (1-2 messages)': '#00CC66',
        'Normal (3-5 messages)': '#4ECDC4',
        'Complex (6+ messages)': '#FF6B6B'
    }
)

st.plotly_chart(fig_sessions, width='stretch')

st.markdown("""
**Interpretation:**
- **Quick Sessions:** Simple queries resolved fast (good!)
- **Normal Sessions:** Standard conversations (expected)
- **Complex Sessions:** May indicate confusion or complex issues (needs review)
""")

st.markdown("---")

# ============================================================================
# COHORT ANALYSIS
# ============================================================================

st.header("👥 Cohort Analysis")

cohort_data = temporal_report.get('cohort_analysis', {})

col1, col2 = st.columns(2)

with col1:
    total_cohorts = cohort_data.get('total_cohorts', 0)
    st.metric(
        "Total Cohorts", 
        total_cohorts,
        help="Number of weekly user cohorts (users grouped by first interaction week)"
    )

with col2:
    most_engaged = cohort_data.get('most_engaged_cohort', 'N/A')
    st.metric(
        "Most Engaged Cohort", 
        most_engaged,
        help="Week with highest average messages per user"
    )

# Cohort performance table
st.subheader("📊 Cohort Performance")

cohort_stats = cohort_data.get('cohort_stats', [])
if cohort_stats:
    cohort_df = pd.DataFrame(cohort_stats)
    
    # Convert period to readable date
    if 'Cohort' in cohort_df.columns:
        cohort_df['Cohort'] = cohort_df['Cohort'].astype(str)
    
    # Display table
    st.dataframe(
        cohort_df[['Cohort', 'unique_users', 'total_messages', 'messages_per_user', 'avg_latency']].head(10),
        width='stretch'
    )
    
    # Engagement trend by cohort
    fig_cohort = px.line(
        cohort_df.head(10),
        x='Cohort',
        y='messages_per_user',
        title="Engagement Level by Cohort (Messages per User)",
        markers=True
    )
    st.plotly_chart(fig_cohort, width='stretch')

st.markdown("---")

# ============================================================================
# ANOMALY DETECTION
# ============================================================================

st.header("🚨 Anomaly Detection")

anomalies = temporal_report.get('anomalous_periods', [])

if anomalies:
    st.warning(f"**{len(anomalies)} anomalous periods detected**")
    
    # Group by type
    anomaly_df = pd.DataFrame(anomalies)
    
    for anom_type in anomaly_df['type'].unique():
        type_anomalies = anomaly_df[anomaly_df['type'] == anom_type]
        
        st.subheader(f"📊 {anom_type.capitalize()} Anomalies")
        
        for _, anom in type_anomalies.iterrows():
            severity_colors = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }
            
            st.write(f"{severity_colors.get(anom['severity'], '⚪')} **{anom['date']}**: {anom['direction']} ({anom['value']})")
    
    # Visualize anomalies
    if 'date' in anomaly_df.columns and 'value' in anomaly_df.columns:
        fig_anom = px.scatter(
            anomaly_df,
            x='date',
            y='value',
            color='severity',
            size='value',
            color_discrete_map={'high': '#FF4B4B', 'medium': '#FFAA00', 'low': '#4ECDC4'},
            title="Anomalous Events Over Time",
            hover_data=['type', 'direction']
        )
        st.plotly_chart(fig_anom, width='stretch')
else:
    st.success("✅ No significant anomalies detected in the time period")

st.markdown("---")

# ============================================================================
# INSIGHTS & RECOMMENDATIONS
# ============================================================================

st.header("💡 Temporal Insights & Recommendations")

insights = []

# Staffing recommendations
if hourly_data:
    peak_hours = hourly_data.get('peak_hours', [])
    if peak_hours:
        insights.append({
            'category': '👥 Staffing',
            'insight': f"Peak traffic during hours: {', '.join([f'{h}:00' for h in sorted(peak_hours)])}",
            'recommendation': "Ensure adequate support staff availability during these hours"
        })

# Growth trend insights
if seasonal_data.get('overall_trend') == 'Growing':
    insights.append({
        'category': '📈 Capacity Planning',
        'insight': "Usage is growing consistently",
        'recommendation': "Plan for infrastructure scaling within next quarter"
    })

# Anomaly alerts
if len(anomalies) > 5:
    insights.append({
        'category': '🚨 Stability',
        'insight': f"Detected {len(anomalies)} anomalous events",
        'recommendation': "Investigate root causes - possible system issues or marketing campaigns"
    })

# Complex sessions
complex_pct = timing_data.get('complex_sessions', 0) / max(
    timing_data.get('quick_sessions', 1) + 
    timing_data.get('normal_sessions', 1) + 
    timing_data.get('complex_sessions', 1), 1
) * 100

if complex_pct > 20:
    insights.append({
        'category': '💬 User Experience',
        'insight': f"{complex_pct:.1f}% of sessions are complex (6+ messages)",
        'recommendation': "Review complex conversations to identify UX friction points"
    })

# Display insights
if insights:
    for insight in insights:
        with st.expander(f"{insight['category']}", expanded=True):
            st.info(f"**Insight:** {insight['insight']}")
            st.success(f"**Recommendation:** {insight['recommendation']}")
else:
    st.success("✅ All temporal patterns are within normal ranges")

# Footer
st.markdown("---")
st.caption("⏰ Temporal & Cohort Analysis | Predictive Intelligence | DigiYatra 2026")
