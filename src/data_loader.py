import pandas as pd
import streamlit as st
import os
import polars as pl

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
        # Use Polars for fast CSV loading and type handling, then convert to pandas
        df_pl = (
            pl.read_csv(
                file_path,
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
