import pandas as pd

def resample_time_series(df: pd.DataFrame, freq='H', metric='count'):
    """
    Resample time series data.
    
    Args:
        freq: 'H' for hourly, 'D' for daily, 'W' for weekly.
        metric: 'count' for volume, 'mean' for averages of numeric cols.
    """
    if 'Timestamp' not in df.columns:
        return pd.DataFrame()
        
    df_indexed = df.set_index('Timestamp')
    
    if metric == 'count':
        return df_indexed.resample(freq).size().reset_index(name='Requests')
    elif metric == 'sum':
        return df_indexed.resample(freq).sum(numeric_only=True).reset_index()
    elif metric == 'mean':
        return df_indexed.resample(freq).mean(numeric_only=True).reset_index()
    
    return pd.DataFrame()

def get_busiest_periods(df: pd.DataFrame):
    """
    Identify peak usage hours and days.
    """
    if 'Hour' not in df.columns or 'DayOfWeek' not in df.columns:
        return {}, {}
        
    hourly_counts = df.groupby('Hour').size().sort_values(ascending=False)
    daily_counts = df.groupby('DayOfWeek').size().sort_values(ascending=False)
    
    return hourly_counts, daily_counts
