import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import (
    analyze_sentiment,
    extract_top_keywords,
    categorise_intent_basic,
    detect_language_series,
    parse_user_feedback,
    add_frustration_index,
    intent_summary,
    language_quality_summary,
    cluster_unseen_queries,
    compute_ood_metrics,
    compute_csat_metrics,
    compute_chat_duration_metrics,
    compute_clarification_engagement_metrics,
    annotate_tanaos_intent,
)

st.set_page_config(page_title="NLP Insights - DigiYatra", page_icon="🧠", layout="wide")

st.title("🧠 NLP & Sentiment Analysis")

with st.spinner("Loading data..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    df = preprocess_data(df)

if df.empty:
    st.stop()

# --- Analysis Settings ---
st.sidebar.subheader("Analysis Settings")
use_tanaos = st.sidebar.checkbox("Use Tanaos intent model (GPU if available)", value=True)

df_analysis = df

# --- Ensure we have basic rule-based intents for downstream analytics ---
with st.spinner("Categorizing intents (rule-based)..."):
    if 'Intent' not in df_analysis.columns:
        df_analysis['Intent'] = df_analysis['Request'].apply(categorise_intent_basic)

# # --- Intent Classification ---
# st.subheader("Intent Distribution")
# with st.spinner("Categorizing intents..."):
#     df_analysis['Intent'] = df_analysis['Request'].apply(categorise_intent_basic)

# intent_counts = df_analysis['Intent'].value_counts().reset_index()
# intent_counts.columns = ['Intent', 'Count']

# fig_intent = px.bar(intent_counts, x='Intent', y='Count', color='Intent', title="Identified User Intents (Rule-based)")
# st.plotly_chart(fig_intent, width='stretch')

# --- Sentiment Analysis ---
st.subheader("Sentiment & Frustration")
with st.spinner("Analyzing sentiment & frustration..."):
    df_analysis = analyze_sentiment(df_analysis, text_col='Request')
    # Best-effort frustration index; works even if some optional cols missing
    df_analysis = add_frustration_index(
        df_analysis,
        text_col='Request',
        conv_id_col=df_analysis.columns.intersection(['Conversation ID', 'conversation_id']).tolist()[0]
        if len(df_analysis.columns.intersection(['Conversation ID', 'conversation_id'])) > 0
        else 'Conversation ID',
    )

# --- Optional Tanaos intent annotation on the analysis sample ---
if use_tanaos:
    with st.spinner("Running Tanaos intent classifier on analysis sample..."):
        df_analysis = annotate_tanaos_intent(df_analysis, text_col='Request')

col1, col2 = st.columns(2)
with col1:
    fig_sent = px.histogram(df_analysis, x='Sentiment_Polarity', nbins=20, title="Sentiment Polarity Distribution (-1 to 1)", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig_sent, width='stretch')

with col2:
    # Frustration distribution
    if 'Frustration_Score' in df_analysis.columns:
        fig_frust = px.histogram(
            df_analysis,
            x='Frustration_Score',
            nbins=20,
            title="Frustration Index Distribution (0 = calm, 1 = very frustrated)",
            color_discrete_sequence=['#EF553B'],
        )
        st.plotly_chart(fig_frust, width='stretch')
    else:
        # Fallback to scatter if for some reason frustration is missing
        fig_scatter = px.scatter(
            df_analysis,
            x='Sentiment_Polarity',
            y='Sentiment_Subjectivity',
            color='Intent',
            title="Sentiment vs Subjectivity",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, width='stretch')

# --- Intent-Level Summary ---
st.subheader("Top User Intents")
intent_summary_df = intent_summary(df_analysis)
if not intent_summary_df.empty:
    c1, c2 = st.columns(2)
    with c1:
        fig_intent_pain = px.bar(
            intent_summary_df.head(10),
            x='Intent',
            y='Count',
            color='Intent',
            title="Top 10 Intents by Volume",
        )
        st.plotly_chart(fig_intent_pain, width='stretch')
    with c2:
        fig_intent_frust = px.bar(
            intent_summary_df.head(10),
            x='Intent',
            y='Avg_Conv_Frustration',
            color='Avg_Conv_Frustration',
            title="Average Conversation Frustration by Intent",
            color_continuous_scale='Reds',
        )
        st.plotly_chart(fig_intent_frust, width='stretch')
    st.dataframe(intent_summary_df, width='stretch')

# --- Language Detection ---
st.subheader("🌐 Multilingual Quality Analysis")

# Sidebar toggle for LID engine
lid_engine = st.sidebar.radio(
    "Language Detection Engine",
    options=["IndicLID", "Lingua"],
    index=0,
    help="IndicLID is best for Romanised Indian languages. Lingua is 75-language general purpose detector."
)

# IndicLID code -> human-readable name mapping
INDICLID_LANG_NAMES = {
    'asm_Latn': 'Assamese - Roman', 'asm_Beng': 'Assamese - Bengali',
    'ben_Latn': 'Bengali - Roman', 'ben_Beng': 'Bengali - Bengali',
    'brx_Latn': 'Bodo - Roman', 'brx_Deva': 'Bodo - Devanagari',
    'doi_Deva': 'Dogri - Devanagari',
    'guj_Latn': 'Gujarati - Roman', 'guj_Gujr': 'Gujarati - Gujarati',
    'hin_Latn': 'Hindi - Roman', 'hin_Deva': 'Hindi - Devanagari',
    'kan_Latn': 'Kannada - Roman', 'kan_Knda': 'Kannada - Kannada',
    'kas_Latn': 'Kashmiri - Roman', 'kas_Arab': 'Kashmiri - Arabic', 'kas_Deva': 'Kashmiri - Devanagari',
    'kok_Latn': 'Konkani - Roman', 'kok_Deva': 'Konkani - Devanagari',
    'mai_Latn': 'Maithili - Roman', 'mai_Deva': 'Maithili - Devanagari',
    'mal_Latn': 'Malayalam - Roman', 'mal_Mlym': 'Malayalam - Malayalam',
    'mni_Latn': 'Manipuri - Roman', 'mni_Beng': 'Manipuri - Bengali', 'mni_Meti': 'Manipuri - Meetei',
    'mar_Latn': 'Marathi - Roman', 'mar_Deva': 'Marathi - Devanagari',
    'nep_Latn': 'Nepali - Roman', 'nep_Deva': 'Nepali - Devanagari',
    'ori_Latn': 'Odia - Roman', 'ori_Orya': 'Odia - Oriya',
    'pan_Latn': 'Punjabi - Roman', 'pan_Guru': 'Punjabi - Gurmukhi',
    'san_Latn': 'Sanskrit - Roman', 'san_Deva': 'Sanskrit - Devanagari',
    'sat_Olch': 'Santali - Ol Chiki',
    'snd_Latn': 'Sindhi - Roman', 'snd_Arab': 'Sindhi - Arabic',
    'tam_Latn': 'Tamil - Roman', 'tam_Tamil': 'Tamil - Tamil',
    'tel_Latn': 'Telugu - Roman', 'tel_Telu': 'Telugu - Telugu',
    'urd_Latn': 'Urdu - Roman', 'urd_Arab': 'Urdu - Arabic',
    'eng_Latn': 'English', 'other': 'Other',
}

def _readable_lang(code: str) -> str:
    # If it's already a formatted string from Lingua (e.g. "ENGLISH (ENG)"), just return it
    if "(" in code and ")" in code:
        return code
    name = INDICLID_LANG_NAMES.get(code, code)
    return f"{name} ({code})"

with st.spinner(f"Detecting languages using {lid_engine}..."):
    df_analysis["Language"] = detect_language_series(df_analysis["Request"], engine=lid_engine)

lang_counts = df_analysis['Language'].value_counts().reset_index()
lang_counts.columns = ['Language', 'Count']
lang_counts['Label'] = lang_counts['Language'].apply(_readable_lang)

lang_quality_df = language_quality_summary(df_analysis)
if not lang_quality_df.empty:
    lang_quality_df['Label'] = lang_quality_df['Language'].apply(_readable_lang)

total_langs = len(lang_counts)
show_k = st.sidebar.slider(
    "Languages to show in distribution",
    min_value=1, max_value=max(total_langs, 1), value=total_langs, step=1,
)

l1, l2 = st.columns([3, 2])
with l1:
    pie_data = lang_counts.head(show_k)
    title_suffix = f"(Top {show_k})" if show_k < total_langs else "(All)"
    total_count = int(pie_data['Count'].sum())
    # Only show percentage text on slices >= 2% so small ones stay clean
    pct_threshold = 0.02
    text_info = []
    for _, row in pie_data.iterrows():
        if row['Count'] / total_count >= pct_threshold:
            text_info.append(f"{row['Count'] / total_count * 100:.1f}%")
        else:
            text_info.append("")
    fig_lang = px.pie(
        pie_data, values='Count', names='Label',
        title=f"Language Distribution {title_suffix}", hole=0.4,
    )
    fig_lang.update_traces(
        text=text_info,
        textposition='outside',
        textinfo='text',
        hoverinfo='label+percent+value',
    )
    fig_lang.update_layout(
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_lang, width='stretch')

with l2:
    # Table with all languages, counts, and percentages
    total_all = int(lang_counts['Count'].sum())
    table_df = lang_counts.head(show_k)[['Label', 'Count']].copy()
    table_df['Percentage'] = (table_df['Count'] / total_all * 100).round(3).astype(str) + '%'
    table_df = table_df.rename(columns={'Label': 'Language'})
    st.dataframe(table_df, width='stretch', hide_index=True, height=min(400, 35 * len(table_df) + 40))

# Frustration chart full-width below
if not lang_quality_df.empty:
    fig_frust = px.bar(
        lang_quality_df,
        x='Label',
        y='Avg_Conv_Frustration',
        color='Count',
        title="Average Frustration by Language",
        color_continuous_scale='Blues',
        labels={'Label': 'Language', 'Avg_Conv_Frustration': 'Avg Frustration (0-1)'},
    )
    fig_frust.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_frust, width='stretch')
else:
    st.info("Not enough data to compute language quality metrics.")

st.caption("Below are sample non-English requests detected in the data.")
non_english = df_analysis[~df_analysis['Language'].isin(['eng_Latn', 'other'])]
if not non_english.empty:
    display_ne = non_english[['Request', 'Language']].copy()
    display_ne['Language'] = display_ne['Language'].apply(_readable_lang)
    st.dataframe(display_ne.head(20), width='stretch')
else:
    st.info("No non-English requests detected.")

# --- User Feedback Sentiment ---
st.subheader("📝 User Feedback Sentiment")
# Use FULL dataframe for feedback as it's sparse
feedback_df = df[df['User Feedback'].notna() & (df['User Feedback'] != '')].copy()
if not feedback_df.empty:
    # Parse feedback - safe method handling potential categorical types
    # Ensure it's string first
    feedback_strings = feedback_df['User Feedback'].astype(str)
    parsed_data = feedback_strings.apply(parse_user_feedback).tolist()
    parsed_df = pd.DataFrame(parsed_data, index=feedback_df.index)
    
    feedback_df = pd.concat([feedback_df, parsed_df], axis=1)
    
    fb1, fb2 = st.columns(2)
    with fb1:
        rating_counts = feedback_df['rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig_rating = px.pie(rating_counts, values='Count', names='Rating', title="User Ratings", hole=0.4, color_discrete_map={'good': 'green', 'bad': 'red'})
        st.plotly_chart(fig_rating, width='stretch')
    
    with fb2:
        # Sentiment of comments
        comments_df = feedback_df[feedback_df['comments'].notna() & (feedback_df['comments'] != '')]
        if not comments_df.empty:
            comments_df = analyze_sentiment(comments_df, text_col='comments')
            fig_fb_sent = px.histogram(comments_df, x='Sentiment_Polarity', nbins=15, color='rating', title="Comment Sentiment by Rating")
            st.plotly_chart(fig_fb_sent, width='stretch')
        else:
            st.info("No text comments in feedback.")
else:
    st.warning("No user feedback data available.")

# --- Top Keywords ---
st.subheader("Top Keywords (TF-IDF)")
with st.spinner("Extracting keywords..."):
    keywords_df = extract_top_keywords(df_analysis, text_col='Request', n=20)

if not keywords_df.empty:
    fig_words = px.bar(keywords_df, x='count', y='word', orientation='h', title="Top 20 Key Terms", color='count')
    fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_words, width='stretch')

# --- Tanaos Conversation-style Intents Overview ---
if use_tanaos and 'Tanaos_Intent' in df_analysis.columns:
    st.subheader("Conversation-style Intents (Tanaos model)")
    tanaos_counts = df_analysis['Tanaos_Intent'].value_counts().reset_index()
    tanaos_counts.columns = ['Intent', 'Count']
    fig_tanaos = px.bar(
        tanaos_counts,
        x='Intent',
        y='Count',
        color='Intent',
        title="Tanaos Conversation-style Intent Distribution (sample)",
    )
    st.plotly_chart(fig_tanaos, width='stretch')

# --- 3D Intent & Sentiment Map ---
st.subheader("3D Intent & Sentiment Map")
feature_cols = []
for col in ['Sentiment_Polarity', 'Frustration_Score']:
    if col in df_analysis.columns:
        feature_cols.append(col)
if 'Tanaos_Intent_Score' in df_analysis.columns:
    feature_cols.append('Tanaos_Intent_Score')

if len(feature_cols) >= 2 and len(df_analysis) >= 10:
    feats = df_analysis[feature_cols].fillna(0.0).to_numpy()
    n_components = min(3, feats.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(feats)

    df_plot = df_analysis.copy()
    df_plot['x'] = coords[:, 0]
    df_plot['y'] = coords[:, 1]
    if n_components > 2:
        df_plot['z'] = coords[:, 2]
    else:
        df_plot['z'] = 0.0

    color_col = 'Tanaos_Intent' if 'Tanaos_Intent' in df_plot.columns else 'Intent'

    fig_3d = px.scatter_3d(
        df_plot,
        x='x',
        y='y',
        z='z',
        color=color_col,
        hover_data=['Request', 'Intent', 'Conversation ID', 'Message ID'],
        title="3D Cluster of Requests by Intent & Sentiment (PCA)",
        opacity=0.8,
        height=720,
    )
    # slightly larger markers and improved layout for dashboard readability
    fig_3d.update_traces(marker=dict(size=6))
    fig_3d.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig_3d, width='stretch', height=720)

    # --- Small details panel (left) showing selected point details ---
    # Streamlit does not provide native hover callbacks for Plotly charts, so
    # provide two complementary interactions:
    # 1) Hover tooltip (built-in) shows Request, Intent, Conversation ID, Message ID
    # 2) Select a point index below to inspect full text and IDs in a small panel

    # details_col1, details_col2 = st.columns([1, 3])
    # with details_col1:
    #     st.subheader("Point Details")
    #     st.caption("Pick a point index to inspect (or hover over points for a tooltip)")
    #     # Provide a selectbox of available indices for inspection
    #     idx_options = df_plot.index.astype(str).tolist()
    #     if idx_options:
    #         selected_idx = st.selectbox("Select point index", options=idx_options, index=0)
    #         sel_row = df_plot.loc[int(selected_idx)]
    #         # Show concise info in a small card-like layout
    #         st.markdown(f"**Conversation ID:** `{sel_row.get('Conversation ID', '')}`")
    #         st.markdown(f"**Message ID:** `{sel_row.get('Message ID', '')}`")
    #         st.markdown("**Request**:")
    #         st.write(sel_row.get('Request', '')[:400])
    #     else:
    #         st.info("No points available to inspect.")

    # with details_col2:
    #     # Keep the right column empty so the details panel visually appears bottom-left
    #     st.write("")
else:
    st.info("Not enough features or rows to build a 3D intent & sentiment map.")

# # --- Token & Cost Overview ---
# st.subheader("🧾 Token & Cost Overview")
# token_metrics_df = compute_token_metrics(df)
# if not token_metrics_df.empty:
#     tm = token_metrics_df.iloc[0]
#     m1, m2, m3, m4 = st.columns(4)
#     with m1:
#         st.metric("Total Conversations", int(tm.get("Total Conversations", 0)))
#     with m2:
#         st.metric("Total Tokens", f"{int(tm.get('Sum Total Tokens', 0)):,}")
#     with m3:
#         st.metric("Total Cost", f"${tm.get('Sum Cost', 0):.2f}")
#     with m4:
#         if "Prompt_Token_Share" in tm:
#             st.metric(
#                 "Prompt vs Completion",
#                 f"{tm['Prompt_Token_Share']*100:.1f}% / {tm['Completion_Token_Share']*100:.1f}%",
#                 help="Share of prompt tokens vs completion tokens.",
#             )
# else:
#     st.info("Token / cost columns not available in the dataset.")

# --- Conversation & Engagement KPIs ---
st.subheader("📊 Core Conversation KPIs")

ood_row = compute_ood_metrics(df_analysis).iloc[0] if not compute_ood_metrics(df_analysis).empty else None
csat_row = compute_csat_metrics(df).iloc[0] if not compute_csat_metrics(df).empty else None
duration_row = compute_chat_duration_metrics(df).iloc[0] if not compute_chat_duration_metrics(df).empty else None
engagement_row = compute_clarification_engagement_metrics(df).iloc[0] if not compute_clarification_engagement_metrics(df).empty else None

k1, k2, k3, k4 = st.columns(4)

with k1:
    if ood_row is not None:
        st.metric(
            "Out-of-domain Requests",
            int(ood_row["OOD_Requests"]),
            f"{ood_row['OOD_Rate']*100:.1f}% of all requests",
        )
    else:
        st.metric("Out-of-domain Requests", "N/A")

with k2:
    if csat_row is not None:
        st.metric(
            "Customer Satisfaction Score (CSAT)",
            f"{csat_row['CSAT_Percent']:.1f}%",
            help="Percentage of positive ratings from User Feedback.",
        )
    else:
        st.metric("CSAT", "N/A")

with k3:
    if duration_row is not None:
        avg_conv_min = duration_row["Avg_Conversation_Duration_Seconds"] / 60.0
        st.metric(
            "Avg Chat Duration",
            f"{avg_conv_min:.1f} min",
            help="Average duration between first and last message per conversation.",
        )
    else:
        st.metric("Avg Chat Duration", "N/A")

with k4:
    if engagement_row is not None:
        st.metric(
            "Clarification Engagement",
            f"{engagement_row['Engagement_Rate_Requests']*100:.1f}%",
            help="Share of requests where users interacted with Clarification options.",
        )
    else:
        st.metric("Clarification Engagement", "N/A")

# --- Emerging Topics (Unseen Intents) ---
st.subheader("🔍 Emerging Topics (Unseen Intents)")
with st.spinner("Discovering clusters of General/Other queries..."):
    clusters_df = cluster_unseen_queries(df_analysis, text_col='Request', intent_col='Intent')

if not clusters_df.empty:
    st.dataframe(clusters_df.head(15), width='stretch')
else:
    st.info("Not enough 'General/Other' queries to form meaningful clusters right now.")

# # --- Word Cloud ---
# st.subheader("Word Cloud")
# from src.nlp_analytics import generate_wordcloud_img
# with st.spinner("Generating Word Cloud..."):
#     # Generate for selected intent if any, or all
#     # We'll use the filtered analysis dataframe
#     fig_wc = generate_wordcloud_img(df_analysis['Request'])
    
#     if fig_wc:
#         st.pyplot(fig_wc)
#     else:
#         st.info("Not enough text data for Word Cloud.")

# --- Raw Data Explorer ---
st.subheader("Explore Raw Queries")
selected_intent = st.selectbox("Filter by Intent", ['All'] + list(df_analysis['Intent'].unique()))

if selected_intent != 'All':
    display_df = df_analysis[df_analysis['Intent'] == selected_intent]
else:
    display_df = df_analysis

st.dataframe(display_df[['Timestamp', 'Request', 'Response', 'Intent', 'Sentiment_Polarity']].head(50), width='stretch')
