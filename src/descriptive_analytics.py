import pandas as pd
import numpy as np

def get_key_metrics(df: pd.DataFrame):
    """
    Calculate high-level KPIs.
    """
    if df.empty:
        return {}
        
    metrics = {
        "total_requests": len(df),
        "total_conversations": df['Conversation ID'].nunique() if 'Conversation ID' in df.columns else 0,
        "total_users": df['User ID'].nunique() if 'User ID' in df.columns else 0,
        "total_cost": df['Cost'].sum() if 'Cost' in df.columns else 0,
        "avg_latency": df['Latency'].mean() if 'Latency' in df.columns else 0,
        "success_rate": (df['Status'] == 'success').mean() * 100 if 'Status' in df.columns else 0
    }
    return metrics

def get_hourly_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate request volume by hour of day.
    """
    if 'Hour' not in df.columns:
        return pd.DataFrame()
        
    return df.groupby('Hour').size().reset_index(name='Requests')

def get_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get breakdown of request statuses.
    """
    if 'Status' not in df.columns:
        return pd.DataFrame()
        
    return df['Status'].value_counts().reset_index()

def get_latency_stats(df: pd.DataFrame) -> dict:
    """
    Calculate latency percentiles.
    """
    if 'Latency' not in df.columns:
        return {}
        
    return {
        "p50": df['Latency'].quantile(0.50),
        "p90": df['Latency'].quantile(0.90),
        "p95": df['Latency'].quantile(0.95),
        "p99": df['Latency'].quantile(0.99),
        "max": df['Latency'].max()
    }
