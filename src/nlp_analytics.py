import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import re
# from langdetect import detect, LangDetectException
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

_indiclid_model = None
_lingua_detector = None


def _get_lingua_detector():
    """
    Lazy-load Lingua language detector.
    Includes all supported languages in high-accuracy mode by default.
    """
    global _lingua_detector
    if _lingua_detector is not None:
        return _lingua_detector

    import logging
    try:
        from lingua import LanguageDetectorBuilder
        # Using all languages for maximum coverage
        _lingua_detector = LanguageDetectorBuilder.from_all_languages().build()
        return _lingua_detector
    except Exception as e:
        logging.error(f"Lingua initialization failed: {e}")
        return None


def _get_indiclid_model():
    """
    Lazy-load IndicLID model from local repo or pip package.
    
    Tries to import from local cloned repo first (IndicLID/Inference),
    then falls back to pip-installed ai4bharat.IndicLID.
    """
    global _indiclid_model
    if _indiclid_model is not None:
        return _indiclid_model

    import logging
    try:
        # Try local cloned repo first
        base_dir = Path(__file__).resolve().parent.parent
        indiclid_inference_path = base_dir / "IndicLID" / "Inference"
        if indiclid_inference_path.exists():
            sys.path.insert(0, str(indiclid_inference_path))
            os.chdir(str(indiclid_inference_path))
        from ai4bharat.IndicLID import IndicLID  # type: ignore
        _indiclid_model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
        return _indiclid_model
    except Exception as e:
        logging.warning(f"IndicLID local import failed: {e}")
        # Fallback: try pip-installed version
        try:
            from ai4bharat.IndicLID import IndicLID  # type: ignore
            _indiclid_model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
            return _indiclid_model
        except Exception as e2:
            logging.error(f"IndicLID pip import failed: {e2}")
            _indiclid_model = None
            return None



# def _langdetect_fallback(text: str) -> str:
#     try:
#         if not isinstance(text, str) or len(text.strip()) < 3:
#             return "unknown"
#         return detect(text)
#     except LangDetectException:
#         return "unknown"
#     except Exception:
#         return "unknown"


# Minimum character length for IndicLID to produce reliable results.
# Shorter texts (greetings like "hi", "ok") get random Indic labels.
_INDICLID_MIN_LENGTH = 5

# Confidence thresholds removed as requested to use IndicLID by default without fallback.

# --- Lingua ISO Mapping ---
# Maps Lingua ISO-639-3 or ISO-639-1 names/codes to human-readable labels if needed.
# Lingua's Language object has .name (e.g., 'ENGLISH', 'HINDI')
# and ISO codes (e.g., .iso_code_639_1.name -> 'EN', .iso_code_639_3.name -> 'HIN')

def detect_language(text: str, engine: str = "IndicLID") -> str:
    """
    Detect language of a single text using the specified engine.
    
    Engines:
    - "IndicLID": Best for Indian languages (Standard + Romanised)
    - "Lingua": Best for general multi-language support (75 languages)
    """
    if not isinstance(text, str) or len(text.strip()) < 1:
        return "unknown"

    import logging
    cleaned = text.strip()

    if engine == "Lingua":
        detector = _get_lingua_detector()
        if detector is not None:
            try:
                lang = detector.detect_language_of(cleaned)
                if lang:
                    return f"{lang.name} ({lang.iso_code_639_3.name})"
                return "unknown"
            except Exception as e:
                logging.error(f"Lingua detection failed: {e}")
        return "unknown"

    # Default: IndicLID
    model = _get_indiclid_model()
    if model is not None:
        try:
            outputs = model.batch_predict([cleaned], batch_size=1)
            if outputs:
                _, label, *_ = outputs[0]
                return label or "unknown"
        except Exception as e:
            logging.error(f"IndicLID batch_predict failed: {e}")

    return "unknown"


def detect_language_series(text_series: pd.Series, engine: str = "IndicLID") -> pd.Series:
    """
    Vectorised language detection over a pandas Series.
    """
    if text_series.empty:
        return text_series.copy()

    import logging
    texts = text_series.fillna("").astype(str)
    labels = pd.Series("unknown", index=texts.index)

    if engine == "Lingua":
        detector = _get_lingua_detector()
        if detector is not None:
            try:
                # Use list comprehension for compatibility if parallel method is missing
                results = [detector.detect_language_of(t) for t in texts.tolist()]
                labels = pd.Series([
                    f"{r.name} ({r.iso_code_639_3.name})" if r else "unknown" 
                    for r in results
                ], index=texts.index)
            except Exception as e:
                logging.error(f"Lingua series detection failed: {e}")
        return labels

    # Default: IndicLID
    model = _get_indiclid_model()
    if model is not None:
        try:
            outputs = model.batch_predict(texts.tolist(), batch_size=64)
            if outputs and len(outputs) == len(texts):
                labels = pd.Series([o[1] if o[1] else "unknown" for o in outputs], index=texts.index)
        except Exception as e:
            logging.error(f"IndicLID batch_predict failed for series: {e}")

    return labels


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
# STOP_WORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

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

_indic_sent_tokenizer = None
_indic_sent_model = None
_indic_sent_device = None


def _get_indic_sentiment_model():
    """
    Lazy-load an IndicBERT-based sentiment classifier if configured.

    Set INDIC_SENTIMENT_MODEL_ID to a Hugging Face model id that is
    fine-tuned for sentiment classification (e.g., an IndicBERT variant).

    If not configured or loading fails, returns (None, None, None).
    """
    global _indic_sent_tokenizer, _indic_sent_model, _indic_sent_device
    if _indic_sent_model is not None and _indic_sent_tokenizer is not None:
        return _indic_sent_tokenizer, _indic_sent_model, _indic_sent_device

    model_id = os.getenv("INDIC_SENTIMENT_MODEL_ID")
    if not model_id:
        return None, None, None

    try:
        _indic_sent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _indic_sent_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _indic_sent_model = AutoModelForSequenceClassification.from_pretrained(model_id).to(
            _indic_sent_device
        )
        _indic_sent_model.eval()
        return _indic_sent_tokenizer, _indic_sent_model, _indic_sent_device
    except Exception:
        _indic_sent_tokenizer = None
        _indic_sent_model = None
        _indic_sent_device = None
        return None, None, None


def _indic_sentiment_batch(texts):
    """
    Run Indic sentiment model in batches, returning labels and scores.
    """
    tok, model, device = _get_indic_sentiment_model()
    if tok is None or model is None:
        return [], []

    labels_out = []
    scores_out = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = [str(t) if not pd.isna(t) else "" for t in texts[i : i + batch_size]]
        if not batch:
            continue
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        batch_scores, batch_ids = probs.max(dim=-1)
        for score, idx in zip(batch_scores, batch_ids):
            labels_out.append(model.config.id2label[int(idx)])
            scores_out.append(float(score.cpu().item()))

    return labels_out, scores_out


def analyze_sentiment(df: pd.DataFrame, text_col: str = "Request") -> pd.DataFrame:
    """
    Add sentiment columns without internal sampling.

    Always computes TextBlob-based polarity/subjectivity:
    - Sentiment_Polarity
    - Sentiment_Subjectivity

    If an IndicBERT-based sentiment model is configured via INDIC_SENTIMENT_MODEL_ID,
    also adds:
    - Indic_Sentiment_Label
    - Indic_Sentiment_Score
    """
    if df.empty or text_col not in df.columns:
        return df

    # TextBlob sentiment baseline
    def get_sentiment(text):
        blob = TextBlob(str(text))
        return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])

    df[["Sentiment_Polarity", "Sentiment_Subjectivity"]] = df[text_col].apply(get_sentiment)

    # IndicBERT sentiment model is disabled for now
    # tok, model, device = _get_indic_sentiment_model()
    # if tok is not None and model is not None:
    #     texts = df[text_col].astype(str).tolist()
    #     labels, scores = _indic_sentiment_batch(texts)
    #     if labels and len(labels) == len(df):
    #         df["Indic_Sentiment_Label"] = labels
    #         df["Indic_Sentiment_Score"] = scores

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

# def generate_wordcloud_img(text_series: pd.Series):
#     """
#     Generate a WordCloud image from a pandas series of text.
#     """
#     text = ' '.join(text_series.dropna().astype(str).apply(clean_text))
    
#     if not text.strip():
#         return None
        
#     wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOP_WORDS).generate(text)
    
#     # Create matplotlib figure
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis('off')
#     return fig


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
    
    Returns metrics including:
    - Count: number of messages per language
    - Resolution_Rate: percentage (0-100) of successful/resolved messages
    - Avg_Conv_Frustration: average frustration score (0-1) per language
    - Other numeric column means (tokens, latency)
    """
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()

    if lang_col not in tmp.columns:
        # Use vectorised IndicLID-based detection when available
        tmp[lang_col] = detect_language_series(tmp[text_col])

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
                "Resolved": "Resolution_Rate_Decimal",
                frustration_col: "Avg_Conv_Frustration",
            }
        )
    )
    
    # Convert resolution rate from 0-1 to 0-100 percentage for clarity
    agg["Resolution_Rate"] = (agg["Resolution_Rate_Decimal"] * 100).round(1)
    agg = agg.drop(columns=["Resolution_Rate_Decimal"])
    agg = agg.sort_values("Language")

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
        tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], format='%m/%d/%Y, %I:%M:%S %p', errors="coerce")

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


_tanaos_tokenizer = None
_tanaos_model = None
_tanaos_device = None


def _get_tanaos_model():
    """
    Lazy-load the tanaos/tanaos-intent-classifier-v1 model.

    Uses GPU if available, otherwise CPU.
    """
    global _tanaos_tokenizer, _tanaos_model, _tanaos_device

    if _tanaos_model is not None and _tanaos_tokenizer is not None:
        return _tanaos_tokenizer, _tanaos_model, _tanaos_device

    try:
        _tanaos_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tanaos_tokenizer = AutoTokenizer.from_pretrained("tanaos/tanaos-intent-classifier-v1")
        _tanaos_model = AutoModelForSequenceClassification.from_pretrained(
            "tanaos/tanaos-intent-classifier-v1"
        ).to(_tanaos_device)
        _tanaos_model.eval()
        return _tanaos_tokenizer, _tanaos_model, _tanaos_device
    except Exception:
        # If transformers/torch is not available or model can't be loaded,
        # leave globals as None so callers can handle gracefully.
        _tanaos_tokenizer = None
        _tanaos_model = None
        _tanaos_device = None
        return None, None, None


def tanaos_intent_batch(texts, batch_size: int = 64):
    """
    Run the Tanaos intent classifier in batches.

    Returns two lists: labels and scores (floats).
    If the model is unavailable, returns ([], []).
    """
    tok, model, device = _get_tanaos_model()
    if tok is None or model is None:
        return [], []

    labels_out = []
    scores_out = []

    for i in range(0, len(texts), batch_size):
        batch = [str(t) if not pd.isna(t) else "" for t in texts[i : i + batch_size]]
        if not batch:
            continue
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        batch_scores, batch_ids = probs.max(dim=-1)
        for score, idx in zip(batch_scores, batch_ids):
            labels_out.append(model.config.id2label[int(idx)])
            scores_out.append(float(score.cpu().item()))

    return labels_out, scores_out


def annotate_tanaos_intent(
    df: pd.DataFrame,
    text_col: str = "Request",
    label_col: str = "Tanaos_Intent",
    score_col: str = "Tanaos_Intent_Score",
) -> pd.DataFrame:
    """
    Attach Tanaos conversation-style intent labels to a dataframe.

    - Uses GPU if available (via torch.cuda.is_available()).
    - Treat low-confidence predictions (score < 0.5) as implicit fallback/unknown in analytics.
    """
    if df.empty or text_col not in df.columns:
        return df

    tok, model, device = _get_tanaos_model()
    if tok is None or model is None:
        # Model not available; just return df unchanged
        return df

    tmp = df.copy()
    texts = tmp[text_col].astype(str).tolist()
    labels, scores = tanaos_intent_batch(texts)
    if not labels or len(labels) != len(tmp):
        return tmp

    tmp[label_col] = labels
    tmp[score_col] = scores
    return tmp


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


# --- Semantic Intent Classification ---

_semantic_model = None
_intent_prototypes = None
_intent_embeddings = None

def _get_semantic_model():
    """
    Lazy-load sentence-transformers model.
    """
    global _semantic_model
    if _semantic_model is not None:
        return _semantic_model
    
    try:
        from sentence_transformers import SentenceTransformer
        # 'all-MiniLM-L6-v2' is fast and effective
        _semantic_model = SentenceTransformer('all-MiniLM-L6-v2') 
        return _semantic_model
    except ImportError:
        print("sentence-transformers not installed. Falling back to None.")
        return None
    except Exception as e:
        print(f"Error loading semantic model: {e}")
        return None

def _get_intent_embeddings():
    """
    Define and embed intent prototypes.
    """
    global _intent_prototypes, _intent_embeddings
    
    if _intent_embeddings is not None:
        return _intent_prototypes, _intent_embeddings

    model = _get_semantic_model()
    if model is None:
        return {}, None

    # Define prototypes - mapping intent label to a descriptive sentence
    # We can have multiple prototypes per intent for better coverage
    prototypes_map = {
        "OTP/Login Issue": [
            "I cannot receive the OTP code for login.",
            "My SMS code is not arriving.",
            "I am unable to sign in to the app.",
            "Problem logging into my account."
        ],
        "Registration Issue": [
            "I cannot register on the app.",
            "Sign up process is failing.",
            "Error during registration step.",
            "I am unable to create a new account."
        ],
        "Boarding Pass Issue": [
            "I cannot upload my boarding pass.",
            "My flight details are not fetching.",
            "Scanning the barcode failed.",
            "Boarding pass upload error."
        ],
        "Face Verification": [
            "My face verification failed.",
            "Selfie upload is not working.",
            "Facial recognition rejected my photo.",
            "I cannot verify my identity with the camera."
        ],
        "Airport/Gate": [
            "Where is the DigiYatra gate?",
            "Which entry gate should I use at the airport?",
            "Is security check faster with DigiYatra?",
            "Airport terminal and gate information."
        ],
        "Technical Issue": [
            "The app keeps crashing.",
            "I am facing a bug or error in the app.",
            "The application is very slow and lagging.",
            "It is not working on my iPhone or Android."
        ],
        "Dependent/Minor": [
            "How do I add my child or kid?",
            "Can I travel with my minor dependent?",
            "Adding a family member to the app.",
            "Travelling with an infant."
        ],
        "Feedback/Complaint": [
            "I am very frustrated with this service.",
            "This is a terrible experience.",
            "Great app, works well.",
            "I want to complain about the delay."
        ],
        "General Query": [
            "What is DigiYatra?",
            "How does this work?",
            "Is it mandatory?",
            "General information about the service."
        ]
    }
    
    # Flatten for embedding
    flat_prototypes = []
    flat_labels = []
    
    for label, sentences in prototypes_map.items():
        for s in sentences:
            flat_prototypes.append(s)
            flat_labels.append(label)
            
    _intent_prototypes = flat_labels
    _intent_embeddings = model.encode(flat_prototypes, convert_to_tensor=True)
    
    return _intent_prototypes, _intent_embeddings

def categorise_intent_semantic(text):
    """
    Classify intent using semantic similarity.
    """
    if not isinstance(text, str) or not text.strip():
        return "Unknown"
        
    model = _get_semantic_model()
    if model is None:
        return categorise_intent_basic(text) # Fallback
        
    try:
        from sentence_transformers import util
        # Ensure embeddings are ready
        prototypes, embeddings = _get_intent_embeddings()
        
        # Embed input
        query_embedding = model.encode(text, convert_to_tensor=True)
        
        # Compute cosine similarity
        hits = util.semantic_search(query_embedding, embeddings, top_k=1)
        
        if hits and hits[0]:
            best_hit = hits[0][0] # {corpus_id, score}
            best_idx = best_hit['corpus_id']
            score = best_hit['score']
            
            # Threshold?
            if score > 0.35: # Reasonable threshold for MiniLM
                return prototypes[best_idx]
            else:
                return "General/Other"
        
        return "General/Other"
        
    except Exception as e:
        print(f"Semantic classification error: {e}")
        return categorise_intent_basic(text)

def categorise_intent_semantic_batch(texts):
    """
    Batch classify intents.
    """
    model = _get_semantic_model()
    if model is None:
        return [categorise_intent_basic(t) for t in texts]
        
    try:
        from sentence_transformers import util
        prototypes, embeddings = _get_intent_embeddings()
        
        # Embed all texts
        # Clean nulls
        clean_texts = [str(t) if not pd.isna(t) else "" for t in texts]
        query_embeddings = model.encode(clean_texts, convert_to_tensor=True)
        
        # Search
        hits = util.semantic_search(query_embeddings, embeddings, top_k=1)
        
        results = []
        for hit in hits:
            if hit:
                best_hit = hit[0]
                if best_hit['score'] > 0.35:
                    results.append(prototypes[best_hit['corpus_id']])
                else:
                    results.append("General/Other")
            else:
                results.append("General/Other")
                
        return results
        
    except Exception as e:
        print(f"Batch semantic classification error: {e}")
        return [categorise_intent_basic(t) for t in texts]
