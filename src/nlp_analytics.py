import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Simple stop words list to avoid heavy downloads if simple is enough
STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment(df: pd.DataFrame, text_col='Request') -> pd.DataFrame:
    """
    Add sentiment polarity and subjectivity columns.
    Uses TextBlob for efficient, rule-based sentiment.
    """
    if text_col not in df.columns:
        return df
        
    # Operate on a copy to avoid SettingWithCopy warnings
    # Sampling for large datasets if needed, but TextBlob is reasonably fast
    # For 1M rows, this might take a few minutes. We might want to cache or sample in the dashboard.
    
    def get_sentiment(text):
        blob = TextBlob(str(text))
        return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

    # Apply directly
    df[['Sentiment_Polarity', 'Sentiment_Subjectivity']] = df[text_col].apply(get_sentiment)
    return df

def extract_top_keywords(df: pd.DataFrame, text_col='Request', n=20):
    """
    Extract top N keywords using TF-IDF.
    """
    if text_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Use a sample for speed if dataset is huge calculate on subset
    sample_df = df.head(50000) if len(df) > 50000 else df
    
    vectorizer = CountVectorizer(stop_words='english', max_features=1000, max_df=0.95, min_df=2)
    try:
        X = vectorizer.fit_transform(sample_df[text_col].fillna('').apply(clean_text))
        counts = X.sum(axis=0).A1
        words = vectorizer.get_feature_names_out()
        
        freq_df = pd.DataFrame({'word': words, 'count': counts})
        return freq_df.sort_values('count', ascending=False).head(n)
    except ValueError:
        return pd.DataFrame(columns=['word', 'count'])

def categorise_intent_basic(text):
    """
    Rule-based intent classification for immediate insights.
    """
    text = str(text).lower()
    if any(x in text for x in ['otp', 'code', 'sms', 'receive']):
        return 'OTP/Login Issue'
    elif any(x in text for x in ['registration', 'register', 'sign up']):
        return 'Registration'
    elif any(x in text for x in ['boarding', 'pass', 'upload']):
        return 'Boarding Pass'
    elif any(x in text for x in ['face', 'facial', 'photo', 'selfie']):
        return 'Face Verification'
    elif any(x in text for x in ['airport', 'delhi', 'bengaluru', 'gate']):
        return 'Airport/Gate'
    elif any(x in text for x in ['crash', 'slow', 'working', 'bug', 'error']):
        return 'Technical Issue'
    elif any(x in text for x in ['minor', 'dependent', 'child', 'kid']):
        return 'Dependent/Minor'
    else:
        return 'General/Other'

def generate_wordcloud_img(text_series: pd.Series):
    """
    Generate a WordCloud image from a pandas series of text.
    """
    text = ' '.join(text_series.dropna().astype(str).apply(clean_text))
    
    if not text.strip():
        return None
        
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
