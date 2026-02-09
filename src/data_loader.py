import pandas as pd
import streamlit as st
import os

@st.cache_data(ttl=3600)
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with caching.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return pd.DataFrame()

    try:
        # Read CSV - using python engine for more robust parsing if needed, but c is faster usually
        # Optimizing types helps memory usage
        dtype_map = {
            'Status': 'category',
            'Model': 'category',
            'Cache Hit': 'str', # Will convert to bool later
            'User Feedback': 'category'
        }
        
        # Using iterator to handle large file if needed, but 1.18M rows fits in memory
        df = pd.read_csv(file_path, dtype=dtype_map)
        
        # Initial Timestamp cleaning - critical based on user feedback
        # The user noted specific format issues, standardizing to dayfirst=True
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True, errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('Timestamp')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()
