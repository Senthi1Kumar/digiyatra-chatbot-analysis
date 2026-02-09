import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.conversation_analytics import reconstruct_conversations, get_conversation_depth

st.set_page_config(page_title="Conversations - DigiYatra", page_icon="💬", layout="wide")

st.title("💬 Conversation Analytics")

with st.spinner("Loading data & Reconstructing Sessions..."):
    df = load_data("all_requests.csv")
    df = preprocess_data(df)
    
    # This can be expensive, so we cache it in the session state if needed, or rely on fast vectorized pandas
    if 'conv_df' not in st.session_state:
        st.session_state.conv_df = reconstruct_conversations(df)
    
    conv_df = st.session_state.conv_df

if conv_df.empty:
    st.warning("No conversation data available (requires 'Conversation ID')")
    st.stop()

# --- Session Metrics ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Sessions", f"{len(conv_df):,}")
m2.metric("Avg Messages / Session", f"{conv_df['Message_Count'].mean():.1f}")
m3.metric("Avg Duration (sec)", f"{conv_df['Duration_Seconds'].mean():.1f}s")
m4.metric("Avg Cost / Session", f"${conv_df['Total_Cost'].mean():.4f}")

# --- Distribution Charts ---
st.subheader("Session Depth Analysis")

c1, c2 = st.columns(2)

with c1:
    # Message Count Distribution
    depth_dist = get_conversation_depth(conv_df)
    # Clip tails for better viz
    depth_dist_clipped = depth_dist[depth_dist['Message_Count'] <= 20] if not depth_dist.empty else depth_dist
    
    if not depth_dist_clipped.empty:
        fig_depth = px.bar(depth_dist_clipped, x='Message_Count', y='Frequency', title="Messages per Session (Distribution)", labels={'Message_Count': 'Messages Count'})
        st.plotly_chart(fig_depth, width='stretch')

with c2:
    # Duration Distribution
    # Filter out 0 duration (single message) for log plot
    durations = conv_df[conv_df['Duration_Seconds'] > 0]['Duration_Seconds']
    fig_dur = px.histogram(durations, nbins=50, title="Session Duration (Seconds)", log_y=True)
    st.plotly_chart(fig_dur, width='stretch')

# --- User Journey Inspection ---
st.subheader("Inspect Specific Conversation")
selected_conv_id = st.selectbox("Select Conversation ID (Sample)", conv_df['Conversation ID'].head(100))

if selected_conv_id:
    msgs = df[df['Conversation ID'] == selected_conv_id].sort_values('Timestamp')
    
    for _, row in msgs.iterrows():
        with st.chat_message("user"):
            st.write(f"**User:** {row['Request']}")
        with st.chat_message("assistant"):
            st.write(f"**Bot:** {row['Response']}")
            st.caption(f"Latency: {row['Latency']}s | Tokens: {row['Total Tokens']}")
