import pandas as pd
import streamlit as st
import os
import polars as pl
from io import BytesIO

@st.cache_data(ttl=3600)
def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load data from CSV file (path or uploaded file) with caching.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, user must upload via UI.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    if file_path is None:
        return pd.DataFrame()
    
    try:
        # Handle both file paths and file-like objects (from st.file_uploader)
        if isinstance(file_path, str):
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
                return pd.DataFrame()
            data_source = file_path
        else:
            # Assume it's a file-like object from st.file_uploader
            data_source = file_path

        # Use Polars for fast CSV loading and type handling, then convert to pandas
        df_pl = (
            pl.read_csv(
                data_source,
                try_parse_dates=False,
                dtypes={
                    "Status": pl.Categorical,
                    "Model": pl.Categorical,
                    "Cache Hit": pl.Utf8,
                    "User Feedback": pl.Utf8,
                },
            )
            .with_columns(
                pl.col("Timestamp")
                .str.strptime(pl.Datetime, format="%d/%m/%Y, %H:%M:%S", strict=False)
                .alias("Timestamp")
            )
            .sort("Timestamp")
        )

        # Convert to pandas for downstream compatibility with existing code
        df = df_pl.to_pandas()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def get_data_upload_widget():
    """
    Render a Streamlit file uploader widget for CSV files.
    Stores the uploaded file in st.session_state['uploaded_data'] for use across pages.
    
    Returns:
        pd.DataFrame or None: Loaded dataframe if file is uploaded, otherwise None.
    """
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="data_uploader")
    
    if uploaded_file is not None:
        try:
            # Pass the file object directly to load_data
            df = load_data(uploaded_file)
            if not df.empty:
                # Store in session state for use in other pages
                st.session_state['uploaded_data'] = df
                st.session_state['uploaded_filename'] = uploaded_file.name
                st.success(f"✅ Loaded {len(df):,} rows from {uploaded_file.name}")
                return df
        except Exception as e:
            st.error(f"Failed to load file: {e}")
    
    return None


def get_session_data() -> pd.DataFrame:
    """
    Retrieve uploaded data from session state.
    Returns the cached dataframe if available, otherwise an empty dataframe.
    
    Returns:
        pd.DataFrame: Loaded dataframe from session state, or empty dataframe.
    """
    if 'uploaded_data' in st.session_state:
        return st.session_state['uploaded_data']
    return pd.DataFrame()
