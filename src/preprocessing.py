import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing steps to the dataframe.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with new features.
    """
    if df.empty:
        return df
    
    # Check if we have Timestamp column
    if 'Timestamp' not in df.columns:
        return df

    # 0. Parse Timestamp if it's still a string
    if df['Timestamp'].dtype == 'object':
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y, %I:%M:%S %p', errors='coerce')

    # 1. Temporal Features
    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.day_name()
    df['Month'] = df['Timestamp'].dt.month_name()
    
    # 2. Boolean cleanup
    if 'Cache Hit' in df.columns:
        # Convert string 'true'/'false' to boolean, handling NaN
        df['Cache Hit'] = df['Cache Hit'].astype(str).str.lower() == 'true'

    # 3. Numeric cleanup
    numeric_cols = ['Prompt Tokens', 'Completion Tokens', 'Total Tokens', 'Cost', 'Latency']
    for col in numeric_cols:
        if col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Text length features
    if 'Request' in df.columns:
        df['Request_Length'] = df['Request'].astype(str).str.len()
    
    return df
