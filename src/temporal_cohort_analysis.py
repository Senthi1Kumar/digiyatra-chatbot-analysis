"""
Temporal & Cohort Analysis
===========================

Advanced time-based analytics:
- User behavior patterns over time
- Cohort analysis (users grouped by first interaction date)
- Seasonal/cyclical pattern detection
- Retention and churn analysis
- Peak load prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
from collections import defaultdict


def analyze_hourly_patterns(df: pd.DataFrame) -> Dict:
    """
    Identify hourly usage patterns and anomalies.
    """
    
    if 'Timestamp' not in df.columns:
        return {}
    
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Timestamp']).dt.day_name()
    
    hourly_stats = df.groupby('Hour').agg({
        'Request': 'count',
        'Latency': 'mean',
        'Cost': 'sum'
    }).rename(columns={'Request': 'volume'})
    
    # Identify peak and off-peak hours
    peak_threshold = hourly_stats['volume'].quantile(0.75)
    peak_hours = hourly_stats[hourly_stats['volume'] >= peak_threshold].index.tolist()
    off_peak_hours = hourly_stats[hourly_stats['volume'] < peak_threshold].index.tolist()
    
    # Day-of-week patterns
    daily_stats = df.groupby('DayOfWeek').agg({
        'Request': 'count',
        'Latency': 'mean'
    }).rename(columns={'Request': 'volume'})
    
    return {
        'hourly_stats': hourly_stats.to_dict('index'),
        'peak_hours': peak_hours,
        'off_peak_hours': off_peak_hours,
        'daily_stats': daily_stats.to_dict('index'),
        'busiest_hour': int(hourly_stats['volume'].idxmax()),
        'quietest_hour': int(hourly_stats['volume'].idxmin()),
        'busiest_day': daily_stats['volume'].idxmax(),
        'quietest_day': daily_stats['volume'].idxmin()
    }


def detect_seasonal_trends(df: pd.DataFrame, window_days: int = 7) -> Dict:
    """
    Detect weekly/monthly trends and seasonal patterns.
    """
    
    if 'Timestamp' not in df.columns:
        return {}
    
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    
    # Daily volume
    daily_volume = df.groupby('Date').size().reset_index(name='volume')
    daily_volume['Date'] = pd.to_datetime(daily_volume['Date'])
    
    # Calculate rolling average
    daily_volume['rolling_avg'] = daily_volume['volume'].rolling(window=window_days, min_periods=1).mean()
    daily_volume['trend'] = daily_volume['volume'] - daily_volume['rolling_avg']
    
    # Identify growth/decline periods
    daily_volume['growth_rate'] = daily_volume['volume'].pct_change()
    
    # Find periods of significant change
    growth_threshold = daily_volume['growth_rate'].quantile(0.75)
    decline_threshold = daily_volume['growth_rate'].quantile(0.25)
    
    growth_periods = daily_volume[daily_volume['growth_rate'] > growth_threshold][['Date', 'growth_rate']].to_dict('records')
    decline_periods = daily_volume[daily_volume['growth_rate'] < decline_threshold][['Date', 'growth_rate']].to_dict('records')
    
    # Overall trend
    if len(daily_volume) >= 7:
        recent_avg = daily_volume.tail(7)['volume'].mean()
        earlier_avg = daily_volume.head(7)['volume'].mean()
        
        if recent_avg > earlier_avg * 1.1:
            overall_trend = 'Growing'
        elif recent_avg < earlier_avg * 0.9:
            overall_trend = 'Declining'
        else:
            overall_trend = 'Stable'
    else:
        overall_trend = 'Insufficient data'
    
    return {
        'daily_volume': daily_volume.to_dict('records'),
        'overall_trend': overall_trend,
        'recent_avg_daily_volume': daily_volume.tail(7)['volume'].mean(),
        'growth_periods': growth_periods[:5],  # Top 5
        'decline_periods': decline_periods[:5],
        'volatility': daily_volume['volume'].std()
    }


def cohort_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze user cohorts based on first interaction date.
    
    Cohorts = Users grouped by month/week of first interaction
    Tracks: Retention, behavior changes, satisfaction over time
    """
    
    if 'Conversation ID' not in df.columns or 'Timestamp' not in df.columns:
        return {}
    
    df['Date'] = pd.to_datetime(df['Timestamp'])
    
    # Get first interaction date for each conversation (proxy for user)
    first_interactions = df.groupby('Conversation ID')['Date'].min().reset_index()
    first_interactions.columns = ['Conversation ID', 'First_Interaction']
    first_interactions['Cohort'] = first_interactions['First_Interaction'].dt.to_period('W')  # Weekly cohorts
    
    # Merge back
    df_cohort = df.merge(first_interactions[['Conversation ID', 'Cohort']], on='Conversation ID', how='left')
    
    # Analyze cohort behavior
    cohort_stats = df_cohort.groupby('Cohort').agg({
        'Conversation ID': 'nunique',
        'Request': 'count',
        'Latency': 'mean',
        'Cost': 'sum'
    })
    cohort_stats.columns = ['unique_users', 'total_messages', 'avg_latency', 'total_cost']
    
    # Calculate engagement rate (messages per user)
    cohort_stats['messages_per_user'] = cohort_stats['total_messages'] / cohort_stats['unique_users']
    
    # Calculate retention (users returning in subsequent periods)
    # This is simplified - in production, would track specific return dates
    
    return {
        'cohort_stats': cohort_stats.reset_index().to_dict('records'),
        'total_cohorts': len(cohort_stats),
        'most_engaged_cohort': cohort_stats['messages_per_user'].idxmax().to_timestamp().strftime('%Y-%m-%d'),
        'least_engaged_cohort': cohort_stats['messages_per_user'].idxmin().to_timestamp().strftime('%Y-%m-%d')
    }


def predict_peak_loads(df: pd.DataFrame, forecast_days: int = 7) -> Dict:
    """
    Simple peak load forecasting based on historical patterns.
    
    Uses moving average and historical peaks.
    """
    
    if 'Timestamp' not in df.columns:
        return {}
    
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    
    # Daily volume
    daily_volume = df.groupby('Date').size().reset_index(name='volume')
    
    # Calculate moving average
    window = min(7, len(daily_volume) // 2)
    if window > 0:
        moving_avg = daily_volume['volume'].rolling(window=window, min_periods=1).mean().iloc[-1]
    else:
        moving_avg = daily_volume['volume'].mean()
    
    # Identify typical peak hours
    hourly_volume = df.groupby('Hour').size()
    top_3_hours = hourly_volume.nlargest(3).index.tolist()
    
    # Historical max
    historical_max = daily_volume['volume'].max()
    historical_avg = daily_volume['volume'].mean()
    
    # Simple forecast
    forecast = []
    for day in range(1, forecast_days + 1):
        # Use moving average with slight increase for growth
        predicted_volume = moving_avg * 1.02  # 2% growth assumption
        
        forecast.append({
            'day_offset': day,
            'predicted_volume': round(predicted_volume),
            'confidence': 'medium' if day <= 3 else 'low'
        })
    
    return {
        'forecast': forecast,
        'peak_hours': top_3_hours,
        'historical_max_daily': int(historical_max),
        'historical_avg_daily': round(historical_avg),
        'current_trend': 'growing' if moving_avg > historical_avg else 'stable',
        'capacity_recommendation': f"Prepare for {round(historical_max * 1.2)} messages/day during peak periods"
    }


def analyze_user_journey_timing(df: pd.DataFrame) -> Dict:
    """
    Analyze timing patterns in user conversations:
    - Time between messages
    - Session duration distribution
    - Time to resolution
    """
    
    if 'Conversation ID' not in df.columns or 'Timestamp' not in df.columns:
        return {}
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    timing_stats = {
        'session_durations': [],
        'time_to_resolution': [],
        'messages_per_session': []
    }
    
    for conv_id, group in df.groupby('Conversation ID'):
        group = group.sort_values('Timestamp')
        
        # Session duration
        if len(group) > 1:
            duration = (group['Timestamp'].max() - group['Timestamp'].min()).total_seconds()
            timing_stats['session_durations'].append(duration)
            
            # Time between messages
            time_diffs = group['Timestamp'].diff().dt.total_seconds().dropna()
            
            # If long gaps (> 5 min), user might have left and returned
            if (time_diffs > 300).any():
                timing_stats['time_to_resolution'].append(duration)
        
        timing_stats['messages_per_session'].append(len(group))
    
    # Calculate statistics
    results = {}
    
    if timing_stats['session_durations']:
        durations = np.array(timing_stats['session_durations'])
        results['avg_session_duration_seconds'] = float(np.mean(durations))
        results['median_session_duration_seconds'] = float(np.median(durations))
        results['p95_session_duration_seconds'] = float(np.percentile(durations, 95))
    
    if timing_stats['messages_per_session']:
        messages = np.array(timing_stats['messages_per_session'])
        results['avg_messages_per_session'] = float(np.mean(messages))
        results['median_messages_per_session'] = float(np.median(messages))
        
        # Classify session types
        results['quick_sessions'] = int((messages <= 2).sum())  # 1-2 messages
        results['normal_sessions'] = int(((messages > 2) & (messages <= 5)).sum())
        results['complex_sessions'] = int((messages > 5).sum())
    
    return results


def identify_anomalous_periods(df: pd.DataFrame, std_threshold: float = 2.0) -> List[Dict]:
    """
    Detect anomalous time periods based on volume, latency, or error rate.
    """
    
    if 'Timestamp' not in df.columns:
        return []
    
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    
    # Daily aggregation
    daily_stats = df.groupby('Date').agg({
        'Request': 'count',
        'Latency': 'mean',
        'Status': lambda x: (x != 'success').sum() if 'Status' in df.columns else 0
    })
    daily_stats.columns = ['volume', 'avg_latency', 'errors']
    
    # Calculate z-scores
    for col in ['volume', 'avg_latency', 'errors']:
        mean = daily_stats[col].mean()
        std = daily_stats[col].std()
        if std > 0:
            daily_stats[f'{col}_zscore'] = (daily_stats[col] - mean) / std
    
    # Identify anomalies
    anomalies = []
    
    for date, row in daily_stats.iterrows():
        anomaly = {}
        
        if abs(row.get('volume_zscore', 0)) > std_threshold:
            anomaly['date'] = str(date)
            anomaly['type'] = 'volume'
            anomaly['value'] = int(row['volume'])
            anomaly['severity'] = 'high' if abs(row['volume_zscore']) > 3 else 'medium'
            anomaly['direction'] = 'spike' if row['volume_zscore'] > 0 else 'drop'
            anomalies.append(anomaly.copy())
        
        if abs(row.get('avg_latency_zscore', 0)) > std_threshold:
            anomaly['date'] = str(date)
            anomaly['type'] = 'latency'
            anomaly['value'] = round(row['avg_latency'], 2)
            anomaly['severity'] = 'high' if abs(row['avg_latency_zscore']) > 3 else 'medium'
            anomaly['direction'] = 'high' if row['avg_latency_zscore'] > 0 else 'low'
            anomalies.append(anomaly.copy())
    
    return anomalies


def generate_temporal_insights_report(df: pd.DataFrame) -> Dict:
    """
    Comprehensive temporal analysis report.
    """
    
    return {
        'hourly_patterns': analyze_hourly_patterns(df),
        'seasonal_trends': detect_seasonal_trends(df),
        'cohort_analysis': cohort_analysis(df),
        'peak_load_forecast': predict_peak_loads(df),
        'user_journey_timing': analyze_user_journey_timing(df),
        'anomalous_periods': identify_anomalous_periods(df)
    }


if __name__ == "__main__":
    print("Temporal & Cohort Analysis Module Loaded!")
    print("Available functions:")
    print("- analyze_hourly_patterns(df)")
    print("- detect_seasonal_trends(df)")
    print("- cohort_analysis(df)")
    print("- predict_peak_loads(df)")
    print("- analyze_user_journey_timing(df)")
    print("- identify_anomalous_periods(df)")
    print("- generate_temporal_insights_report(df) - Run all analyses")
