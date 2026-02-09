import pandas as pd

def reconstruct_conversations(df: pd.DataFrame):
    """
    Group messages into conversations and calculate session metrics.
    """
    if 'Conversation ID' not in df.columns:
        return pd.DataFrame()
        
    # Group by Conversation ID
    conv_stats = df.groupby('Conversation ID').agg({
        'Timestamp': ['min', 'max', 'count'],
        'User ID': 'first',
        'Model': 'first',
        'Cost': 'sum',
        'Latency': 'mean'
    }).reset_index()
    
    # Flatten columns
    conv_stats.columns = ['Conversation ID', 'Start_Time', 'End_Time', 'Message_Count', 'User ID', 'Model', 'Total_Cost', 'Avg_Latency']
    
    # Calculate duration
    conv_stats['Duration_Seconds'] = (conv_stats['End_Time'] - conv_stats['Start_Time']).dt.total_seconds()
    
    return conv_stats

def get_conversation_depth(conv_df: pd.DataFrame):
    """
    Analyze the distribution of conversation lengths (turns).
    """
    if conv_df.empty:
        return pd.DataFrame()
        
    return conv_df['Message_Count'].value_counts().sort_index().reset_index(name='Frequency')
