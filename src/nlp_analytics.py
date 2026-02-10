import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from langdetect import detect, LangDetectException
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

def detect_language(text: str) -> str:
    """
    Detect language of text using langdetect.
    Returns language code (e.g., 'en', 'hi') or 'unknown'.
    """
    try:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        return 'unknown'
    except Exception:
        return 'unknown'

def parse_user_feedback(feedback_str: str) -> dict:
    """
    Parse JSON-like user feedback string.
    Returns dict with 'rating' and 'comments'.
    """
    import json
    try:
        if pd.isna(feedback_str) or feedback_str == '':
            return {'rating': None, 'comments': None}
        data = json.loads(feedback_str.replace("'", '"'))
        return {'rating': data.get('rating'), 'comments': data.get('comments', '')}
    except Exception:
        return {'rating': None, 'comments': None}

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


def compute_frustration_score(
    polarity: float,
    subjectivity: float,
    num_fallbacks: int = 0,
    num_repeats: int = 0,
    has_negative_words: bool = False,
) -> float:
    """
    Heuristic frustration score in range [0, 1].

    - More negative polarity -> higher frustration
    - Higher subjectivity (more opinionated) slightly increases score
    - Fallbacks / repeated messages amplify frustration
    - Explicit negative cues (e.g. 'not working', 'worst') boost it
    """
    # Map polarity [-1,1] to [0,1] where 1 = most negative
    polarity_component = (1 - polarity) / 2.0
    subjectivity_component = subjectivity

    fallback_component = min(num_fallbacks / 3.0, 1.0)
    repeat_component = min(num_repeats / 3.0, 1.0)
    negative_flag_component = 1.0 if has_negative_words else 0.0

    # Weighted combination
    score = (
        0.45 * polarity_component
        + 0.15 * subjectivity_component
        + 0.15 * fallback_component
        + 0.15 * repeat_component
        + 0.10 * negative_flag_component
    )
    # Clip for safety
    return float(max(0.0, min(1.0, score)))


NEGATIVE_PATTERNS = [
    "not working",
    "worst",
    "useless",
    "waste",
    "bad",
    "stupid",
    "idiot",
    "pathetic",
    "frustrated",
    "angry",
    "slow",
    "error",
]


def add_frustration_index(
    df: pd.DataFrame,
    text_col: str = "Request",
    sentiment_polarity_col: str = "Sentiment_Polarity",
    sentiment_subjectivity_col: str = "Sentiment_Subjectivity",
    fallback_col: str = "Is Fallback",
    conv_id_col: str = "Conversation ID",
) -> pd.DataFrame:
    """
    Add per-message and per-conversation frustration metrics.

    Requirements (best effort – works even if some cols missing):
    - Sentiment columns already computed via analyze_sentiment.
    - Optional: boolean fallback column and conversation id column.
    """
    if df.empty or text_col not in df.columns:
        return df

    tmp = df.copy()

    if sentiment_polarity_col not in tmp.columns or sentiment_subjectivity_col not in tmp.columns:
        tmp = analyze_sentiment(tmp, text_col=text_col)

    # Fallback / repeats
    if fallback_col in tmp.columns:
        tmp[fallback_col] = tmp[fallback_col].fillna(False).astype(bool)
    else:
        tmp[fallback_col] = False

    # Simple repeated-text count within a conversation if ID exists
    if conv_id_col in tmp.columns:
        tmp["__text_lower"] = tmp[text_col].astype(str).str.lower()
        tmp["Repeat_Count"] = (
            tmp.groupby(conv_id_col)["__text_lower"]
            .transform(lambda s: s.map(s.value_counts()))
        )
    else:
        tmp["Repeat_Count"] = 1

    def has_negative(text: str) -> bool:
        t = str(text).lower()
        return any(p in t for p in NEGATIVE_PATTERNS)

    tmp["Has_Negative_Words"] = tmp[text_col].apply(has_negative)

    tmp["Frustration_Score"] = tmp.apply(
        lambda row: compute_frustration_score(
            polarity=row.get(sentiment_polarity_col, 0.0),
            subjectivity=row.get(sentiment_subjectivity_col, 0.0),
            num_fallbacks=int(row.get(fallback_col, 0)),
            num_repeats=int(max(row.get("Repeat_Count", 1) - 1, 0)),
            has_negative_words=bool(row.get("Has_Negative_Words", False)),
        ),
        axis=1,
    )

    # Aggregate to conversation-level frustration if conv id exists
    if conv_id_col in tmp.columns:
        agg = (
            tmp.groupby(conv_id_col)["Frustration_Score"]
            .agg(["mean", "max"])
            .rename(
                columns={
                    "mean": "Conv_Frustration_Mean",
                    "max": "Conv_Frustration_Max",
                }
            )
        )
        tmp = tmp.merge(agg, left_on=conv_id_col, right_index=True, how="left")

    return tmp

def extract_top_keywords(df: pd.DataFrame, text_col='Request', n=20):
    """
    Extract top N keywords using TF-IDF.
    """
    if text_col not in df.columns or df.empty:
        return pd.DataFrame()
    
    # Use a sample for speed if dataset is huge calculate on subset
    sample_df = df.head(50000) if len(df) > 50000 else df
    
    vectorizer = CountVectorizer(stop_words="english", max_features=1000, max_df=0.95, min_df=2)
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


def intent_summary(
    df: pd.DataFrame,
    intent_col: str = "Intent",
    frustration_col: str = "Conv_Frustration_Mean",
    resolution_col: str = "Status",
) -> pd.DataFrame:
    """
    Aggregate high-level stats per intent.

    Expects:
    - intent_col: rule-based or model-based intent label
    - optional frustration_col: conversation frustration metric
    - optional resolution_col: e.g. Status == 'success'
    """
    if df.empty or intent_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()

    if resolution_col in tmp.columns:
        tmp["Resolved"] = tmp[resolution_col].astype(str).str.lower().isin(
            ["success", "ok", "solved", "resolved"]
        )
    else:
        tmp["Resolved"] = False

    if frustration_col not in tmp.columns:
        tmp[frustration_col] = np.nan

    agg = (
        tmp.groupby(intent_col)
        .agg(
            Count=(intent_col, "size"),
            Avg_Conv_Frustration=(frustration_col, "mean"),
            Resolution_Rate=("Resolved", "mean"),
        )
        .reset_index()
        .rename(columns={intent_col: "Intent"})
        .sort_values("Count", ascending=False)
    )
    return agg

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


def language_quality_summary(
    df: pd.DataFrame,
    text_col: str = "Request",
    lang_col: str = "Language",
    frustration_col: str = "Conv_Frustration_Mean",
    resolution_col: str = "Status",
) -> pd.DataFrame:
    """
    Summarise core metrics by detected language.

    Requires a language column (use detect_language on Request).
    """
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()

    if lang_col not in tmp.columns:
        tmp[lang_col] = tmp[text_col].apply(detect_language)

    if resolution_col in tmp.columns:
        tmp["Resolved"] = tmp[resolution_col].astype(str).str.lower().isin(
            ["success", "ok", "solved", "resolved"]
        )
    else:
        tmp["Resolved"] = False

    if frustration_col not in tmp.columns:
        tmp[frustration_col] = np.nan

    numeric_cols = [c for c in ["Prompt Tokens", "Completion Tokens", "Total Tokens", "Latency"] if c in tmp.columns]

    group = tmp.groupby(lang_col)

    # Conversation counts per language
    size_series = group.size().rename("Count")

    # Mean metrics per language
    agg_dict = {
        "Resolved": "mean",
        frustration_col: "mean",
    }
    for c in numeric_cols:
        agg_dict[c] = "mean"

    metrics = group.agg(agg_dict)

    agg = (
        pd.concat([size_series, metrics], axis=1)
        .reset_index()
        .rename(
            columns={
                lang_col: "Language",
                "Resolved": "Resolution_Rate",
                frustration_col: "Avg_Conv_Frustration",
            }
        )
        .sort_values("Language")
    )

    return agg


def compute_token_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple token / cost aggregates for dashboard use.

    Expects preprocessed numeric columns:
    - Prompt Tokens, Completion Tokens, Total Tokens, Cost
    """
    if df.empty:
        return pd.DataFrame()

    cols = [c for c in ["Prompt Tokens", "Completion Tokens", "Total Tokens", "Cost"] if c in df.columns]
    if not cols:
        return pd.DataFrame()

    summary = {
        "Total Conversations": len(df),
    }
    for c in cols:
        summary[f"Sum {c}"] = float(df[c].sum())
        summary[f"Avg {c}"] = float(df[c].mean())

    if "Prompt Tokens" in df.columns and "Completion Tokens" in df.columns:
        total_prompt = df["Prompt Tokens"].sum()
        total_completion = df["Completion Tokens"].sum()
        total_tokens = max(total_prompt + total_completion, 1)
        summary["Prompt_Token_Share"] = float(total_prompt / total_tokens)
        summary["Completion_Token_Share"] = float(total_completion / total_tokens)

    return pd.DataFrame([summary])


def compute_ood_metrics(
    df: pd.DataFrame,
    intent_col: str = "Intent",
    ood_labels: tuple = ("Out-of-domain", "General/Other"),
) -> pd.DataFrame:
    """
    Compute simple out-of-domain (OOD) request stats.

    By default treats 'Out-of-domain' and 'General/Other' intents as OOD.
    Returns a single-row dataframe with:
    - Total_Requests
    - OOD_Requests
    - OOD_Rate (0-1)
    """
    if df.empty or intent_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    total = len(tmp)
    ood_mask = tmp[intent_col].isin(ood_labels)
    ood_count = int(ood_mask.sum())
    rate = float(ood_count / total) if total > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "Total_Requests": int(total),
                "OOD_Requests": ood_count,
                "OOD_Rate": rate,
            }
        ]
    )


def compute_csat_metrics(
    df: pd.DataFrame,
    user_feedback_col: str = "User Feedback",
) -> pd.DataFrame:
    """
    Compute CSAT metrics from the User Feedback column.

    Expects feedback strings that parse via parse_user_feedback into:
    - rating: e.g. 'good', 'bad', numeric string, etc.

    Returns a single-row dataframe with:
    - CSAT_Responses
    - CSAT_Positive
    - CSAT_Percent (0-100)
    """
    if df.empty or user_feedback_col not in df.columns:
        return pd.DataFrame()

    feedback_df = df[df[user_feedback_col].notna() & (df[user_feedback_col] != "")].copy()
    if feedback_df.empty:
        return pd.DataFrame()

    feedback_strings = feedback_df[user_feedback_col].astype(str)
    parsed = feedback_strings.apply(parse_user_feedback).tolist()
    parsed_df = pd.DataFrame(parsed, index=feedback_df.index)

    if "rating" not in parsed_df.columns:
        return pd.DataFrame()

    ratings = parsed_df["rating"].dropna().astype(str)
    if ratings.empty:
        return pd.DataFrame()

    # Define what counts as "satisfied"
    def is_positive(r: str) -> bool:
        r_low = r.lower()
        if r_low in {"good", "positive", "satisfied", "thumbs_up"}:
            return True
        if r_low in {"bad", "negative", "unsatisfied", "thumbs_down"}:
            return False
        # Try numeric interpretation (e.g. 1-5 scale)
        try:
            val = float(r)
            return val >= 4.0
        except ValueError:
            return False

    total_responses = len(ratings)
    positive = int(ratings.apply(is_positive).sum())
    percent = float(positive / total_responses * 100.0) if total_responses > 0 else 0.0

    return pd.DataFrame(
        [
            {
                "CSAT_Responses": int(total_responses),
                "CSAT_Positive": positive,
                "CSAT_Percent": percent,
            }
        ]
    )


def compute_chat_duration_metrics(
    df: pd.DataFrame,
    conv_id_col: str = "Conversation ID",
    user_id_col: str = "User ID",
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """
    Compute chat duration metrics using conversation IDs and timestamps.

    Returns a single-row dataframe with:
    - Avg_Conversation_Duration_Seconds
    - Median_Conversation_Duration_Seconds
    - Avg_Duration_Per_User_Seconds
    """
    required_cols = {conv_id_col, timestamp_col}
    if df.empty or not required_cols.issubset(df.columns):
        return pd.DataFrame()

    tmp = df[[c for c in [conv_id_col, user_id_col, timestamp_col] if c in df.columns]].copy()
    if not np.issubdtype(tmp[timestamp_col].dtype, np.datetime64):
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], errors="coerce")

    tmp = tmp.dropna(subset=[timestamp_col])
    if tmp.empty:
        return pd.DataFrame()

    grp = tmp.groupby(conv_id_col)
    conv_df = grp.agg(
        Start=(timestamp_col, "min"),
        End=(timestamp_col, "max"),
        User_First=(user_id_col, "first") if user_id_col in tmp.columns else (timestamp_col, "size"),
    )
    conv_df["Duration_Seconds"] = (conv_df["End"] - conv_df["Start"]).dt.total_seconds().clip(lower=0)

    if conv_df.empty:
        return pd.DataFrame()

    avg_conv = float(conv_df["Duration_Seconds"].mean())
    med_conv = float(conv_df["Duration_Seconds"].median())

    if user_id_col in tmp.columns:
        user_mean = float(
            conv_df.groupby("User_First")["Duration_Seconds"].mean().mean()
        )
    else:
        user_mean = avg_conv

    return pd.DataFrame(
        [
            {
                "Avg_Conversation_Duration_Seconds": avg_conv,
                "Median_Conversation_Duration_Seconds": med_conv,
                "Avg_Duration_Per_User_Seconds": user_mean,
            }
        ]
    )


def compute_clarification_engagement_metrics(
    df: pd.DataFrame,
    clarification_col: str = "Clarification",
    conv_id_col: str = "Conversation ID",
) -> pd.DataFrame:
    """
    Compute engagement metrics based on the Clarification column.

    Returns a single-row dataframe with:
    - Engagement_Rate_Requests
    - Engagement_Rate_Conversations
    """
    if df.empty or clarification_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()

    def has_clarification(val) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip()
        return s not in {"", "[]", "nan", "None"}

    tmp["Clarification_Engaged"] = tmp[clarification_col].apply(has_clarification)

    total_requests = len(tmp)
    engaged_requests = int(tmp["Clarification_Engaged"].sum())
    req_rate = float(engaged_requests / total_requests) if total_requests > 0 else 0.0

    if conv_id_col in tmp.columns:
        conv_engaged = tmp.groupby(conv_id_col)["Clarification_Engaged"].any()
        engaged_convs = int(conv_engaged.sum())
        total_convs = len(conv_engaged)
        conv_rate = float(engaged_convs / total_convs) if total_convs > 0 else 0.0
    else:
        conv_rate = req_rate

    return pd.DataFrame(
        [
            {
                "Engagement_Rate_Requests": req_rate,
                "Engagement_Rate_Conversations": conv_rate,
            }
        ]
    )


def cluster_unseen_queries(
    df: pd.DataFrame,
    text_col: str = "Request",
    intent_col: str = "Intent",
    min_samples: int = 30,
    max_clusters: int = 15,
) -> pd.DataFrame:
    """
    Cluster queries that are currently labelled as General/Other (or missing intent)
    to discover emerging topics / unseen intents.

    Uses a simple bag-of-words + MiniBatchKMeans for speed.
    Returns a dataframe with cluster label, size, and representative example.
    """
    if df.empty or text_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    if intent_col in tmp.columns:
        mask_unseen = tmp[intent_col].fillna("Unknown").isin(["General/Other", "Unknown"])
        tmp = tmp[mask_unseen]

    tmp[text_col] = tmp[text_col].fillna("").astype(str)
    tmp = tmp[tmp[text_col].str.strip() != ""]

    if len(tmp) < min_samples:
        return pd.DataFrame()

    vectorizer = CountVectorizer(stop_words="english", max_features=2000, max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(tmp[text_col].apply(clean_text))

    n_clusters = min(max_clusters, max(2, len(tmp) // min_samples))

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(X)

    tmp = tmp.copy()
    tmp["Cluster"] = labels

    # Representative message per cluster (closest to centroid)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    rows = []
    for cluster_id in range(n_clusters):
        cluster_df = tmp[tmp["Cluster"] == cluster_id]
        if cluster_df.empty:
            continue
        size = len(cluster_df)
        if size < min_samples:
            continue

        rep_idx = closest[cluster_id]
        rep_text = tmp.iloc[rep_idx][text_col]
        rows.append(
            {
                "Cluster": int(cluster_id),
                "Size": int(size),
                "Representative_Query": rep_text,
            }
        )

    return pd.DataFrame(rows).sort_values("Size", ascending=False)
