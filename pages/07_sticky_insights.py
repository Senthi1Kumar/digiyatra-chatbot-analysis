import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.nlp_analytics import categorise_intent_basic
from src.sticky_analytics import extract_topics_nmf, generate_sankey_data, analyze_friction

st.set_page_config(page_title="Sticky Insights - DigiYatra", page_icon="🧲", layout="wide")

st.title("🧲 Strategic 'Sticky' Insights")
st.markdown("Deep-dive analytics designed to identify **User Journeys**, **Hidden Topics**, and **Friction Points**.")

# Load Data
with st.spinner("Crunching advanced data..."):
    df = load_data("all_requests.csv")
    df = preprocess_data(df)
    
    # Ensure Intent exists (using your basic rule-based one first)
    if 'Intent' not in df.columns:
        df['Intent'] = df['Request'].apply(categorise_intent_basic)

# --- Section 1: Dynamic Topic Modeling (NMF) ---
st.header("1. Beyond Rules: Discovering Hidden Topics")
st.caption("Using NMF (Non-Negative Matrix Factorization) to find what users are *really* asking, beyond our hardcoded rules.")

n_topics = st.slider("Number of Topics to Discover", 3, 10, 5)

if st.button("Run Topic Modeling"):
    with st.spinner("Extracting topics..."):
        # Use a sample for speed if dataset > 10k
        df_sample = df.sample(min(10000, len(df)), random_state=42).copy()
        
        df_labeled, topics_list = extract_topics_nmf(df_sample, n_topics=n_topics)
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Discovered Topics")
            for t in topics_list:
                st.write(f"• **{t}**")
                
        with c2:
            st.subheader("Topic Distribution")
            topic_counts = df_labeled['Topic_Name'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            fig_topic = px.bar(topic_counts, x='Count', y='Topic', orientation='h', title="Volume by Discovered Topic")
            st.plotly_chart(fig_topic, width='stretch')

# --- Section 2: User Journey (Sankey) ---
st.markdown("---")
st.header("2. User Journey Flow (Sankey Diagram)")
st.caption("Visualizing how users move from one topic to another. Where do they go after 'Registration'?")

sankey_data = generate_sankey_data(df, min_flow_count=10)

if sankey_data:
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_data['node_labels'],
            color="blue"
        ),
        link=dict(
            source=sankey_data['source'],
            target=sankey_data['target'],
            value=sankey_data['value']
        )
    )])
    fig_sankey.update_layout(title_text="Conversation Intent Flow", font_size=10, height=600)
    st.plotly_chart(fig_sankey, width='stretch')
else:
    st.info("Not enough conversation flow data to generate Sankey.")

# --- Section 3: The Friction Matrix ---
st.markdown("---")
st.header("3. The Friction Matrix")
st.caption("Where is the chatbot failing? Identifying the 'Why' behind user drop-offs.")

friction = analyze_friction(df)

f1, f2 = st.columns(2)

with f1:
    st.subheader("Confusing Topics (Clarification Triggers)")
    if 'Top_Confusing_Intents' in friction and not friction['Top_Confusing_Intents'].empty:
        st.write("Users get confused most often when discussing:")
        st.dataframe(friction['Top_Confusing_Intents'])
    else:
        st.info("No clarification data available.")

with f2:
    st.subheader("Disliked Topics (Negative Feedback)")
    if 'Top_Disliked_Intents' in friction and not friction['Top_Disliked_Intents'].empty:
        st.write("Users give negative feedback most often on:")
        st.dataframe(friction['Top_Disliked_Intents'])
    else:
        st.info("No negative feedback data available.")