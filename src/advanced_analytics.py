import pandas as pd
import numpy as np
from src.nlp_analytics import analyze_sentiment

def calculate_advanced_features(df: pd.DataFrame, conv_df: pd.DataFrame = None):
    """
    Generate advanced features for deeper insights.
    """
    # 1. Response Verbosity (Instruction Adherence Proxy)
    # Ratio > 1 means model talks more than user. 
    # Extremely high ratio might indicate unnecessary verbosity.
    if 'Completion Tokens' in df.columns and 'Prompt Tokens' in df.columns:
        df['Verbosity_Ratio'] = df['Completion Tokens'] / (df['Prompt Tokens'] + 1) # Avoid div by zero

    # 2. Latency Z-Score (Anomaly Detection)
    if 'Latency' in df.columns:
        latency_mean = df['Latency'].mean()
        latency_std = df['Latency'].std()
        df['Latency_Z_Score'] = (df['Latency'] - latency_mean) / latency_std
        df['Is_Latency_Anomaly'] = df['Latency_Z_Score'] > 3  # 3 Sigma rule

    return df

def analyze_conversation_sentiment_flow(df: pd.DataFrame):
    """
    Analyze how sentiment changes within a conversation.
    Returns conversation-level stats with 'Sentiment_Change'.
    """
    # Ensure sentiment exists
    if 'Sentiment_Polarity' not in df.columns:
        df = analyze_sentiment(df, text_col='Request')
        
    if 'Conversation ID' not in df.columns:
        return pd.DataFrame()

    # Sort by time
    df_sorted = df.sort_values(['Conversation ID', 'Timestamp'])

    # Get first and last sentiment per conversation
    # We group by conversation and take the first and last valid sentiment
    
    conv_sentiment = df_sorted.groupby('Conversation ID')['Sentiment_Polarity'].agg(['first', 'last', 'mean', 'min', 'count'])
    conv_sentiment.columns = ['Start_Sentiment', 'End_Sentiment', 'Avg_Sentiment', 'Min_Sentiment', 'Turn_Count']
    
    # Feature 3: Sentiment Flow / Delta
    conv_sentiment['Sentiment_Change'] = conv_sentiment['End_Sentiment'] - conv_sentiment['Start_Sentiment']
    
    # Feature 4: Frustration Index
    # Logic: Conversation is long (>3 turns) AND ends with negative sentiment OR has large negative drop
    conv_sentiment['Is_Frustrated'] = (
        (conv_sentiment['Turn_Count'] > 2) & 
        ((conv_sentiment['End_Sentiment'] < -0.1) | (conv_sentiment['Sentiment_Change'] < -0.3))
    )
    
    return conv_sentiment.reset_index()

def correlation_analysis(df: pd.DataFrame):
    """
    Get correlation between numerical features to find hidden relationships.
    e.g. Does Latency correlate with Sentiment?
    """
    cols = ['Latency', 'Cost', 'Total Tokens', 'Sentiment_Polarity', 'Request_Length', 'Verbosity_Ratio', 'Latency_Z_Score']
    available_cols = [c for c in cols if c in df.columns]
    
    if len(available_cols) > 1:
        return df[available_cols].corr()
    return pd.DataFrame()
