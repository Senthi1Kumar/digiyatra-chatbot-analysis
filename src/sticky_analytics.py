import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import plotly.graph_objects as go

def extract_topics_nmf(df: pd.DataFrame, text_col='Request', n_topics=5):
    """
    Apply Non-Negative Matrix Factorization (NMF) to discover hidden topics
    beyond the rule-based categories.
    """
    if df.empty or text_col not in df.columns:
        return df, []

    # 1. TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(df[text_col].fillna('').astype(str))
    
    # 2. NMF Decomposition
    nmf_model = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    nmf_features = nmf_model.fit_transform(tfidf)
    
    # 3. Assign Topic to each document
    df['Topic_ID'] = nmf_features.argmax(axis=1)
    
    # 4. Extract Top Words per Topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topics_summary = []
    
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[:-6:-1] # Top 5 words
        top_words = [feature_names[i] for i in top_indices]
        topics_summary.append(f"Topic {topic_idx+1}: {', '.join(top_words)}")
        
    # Map ID to Name
    topic_map = {i: summary for i, summary in enumerate(topics_summary)}
    df['Topic_Name'] = df['Topic_ID'].map(topic_map)
    
    return df, topics_summary

def generate_sankey_data(df: pd.DataFrame, min_flow_count=5):
    """
    Map the user journey: Intent A -> Intent B -> Intent C.
    Requires 'Conversation ID' and 'Timestamp'.
    """
    if 'Intent' not in df.columns or 'Conversation ID' not in df.columns:
        return None

    # Sort by time
    df = df.sort_values(['Conversation ID', 'Timestamp'])
    
    # Shift to get next intent
    df['Next_Intent'] = df.groupby('Conversation ID')['Intent'].shift(-1)
    
    # Filter out end of conversations (Next_Intent is NaN)
    flow_df = df.dropna(subset=['Next_Intent'])
    
    # Count transitions
    flow_counts = flow_df.groupby(['Intent', 'Next_Intent']).size().reset_index(name='Count')
    
    # Filter noise
    flow_counts = flow_counts[flow_counts['Count'] >= min_flow_count]
    
    if flow_counts.empty:
        return None

    # Create Sankey Nodes and Links
    all_intents = list(set(flow_counts['Intent'].unique()) | set(flow_counts['Next_Intent'].unique()))
    intent_map = {intent: i for i, intent in enumerate(all_intents)}
    
    sankey_data = {
        'node_labels': all_intents,
        'source': flow_counts['Intent'].map(intent_map).tolist(),
        'target': flow_counts['Next_Intent'].map(intent_map).tolist(),
        'value': flow_counts['Count'].tolist()
    }
    
    return sankey_data

def analyze_friction(df: pd.DataFrame):
    """
    Identify "Friction Points":
    1. Clarification Loops (Bot asks for clarification)
    2. Negative Feedback
    3. High Latency
    """
    friction_stats = {}
    
    # 1. Clarification Rate (Assuming 'Clarification' col exists or we detect it)
    if 'Clarification' in df.columns:
        friction_stats['Clarification_Rate'] = df['Clarification'].notna().mean() * 100
        # Identify top intents leading to clarification
        friction_stats['Top_Confusing_Intents'] = df[df['Clarification'].notna()]['Intent'].value_counts().head(5)
    
    # 2. Negative Feedback Hotspots
    if 'User Feedback' in df.columns:
        neg_feedback = df[df['User Feedback'].astype(str).str.lower().isin(['thumbs down', 'negative', 'bad'])]
        friction_stats['Neg_Feedback_Rate'] = len(neg_feedback) / len(df) * 100
        friction_stats['Top_Disliked_Intents'] = neg_feedback['Intent'].value_counts().head(5)
        
    return friction_stats