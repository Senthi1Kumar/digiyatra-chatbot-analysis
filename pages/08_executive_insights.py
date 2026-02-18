import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.advanced_nlp_insights import (
    run_advanced_nlp_analysis,
    fix_language_detection
)

st.set_page_config(
    page_title="Executive Insights - DigiYatra", 
    page_icon="💼", 
    layout="wide"
)

st.title("💼 NLP Insights Dashboard")
st.markdown("""
**Deep business intelligence powered by advanced NLP analytics**

Actionable insights for:
- 🎯 Product Teams: User experience pain points and feature gaps
- ⚙️ Engineering: Performance bottlenecks and optimization opportunities  
- 💰 Business/Ops: Cost efficiency and resource allocation
- 🤖 ML Teams: Model performance and training priorities
""")

# Load data
with st.spinner("Loading and analyzing data..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    
    df = preprocess_data(df)
    
    # Cache the analysis in session state
    if 'advanced_analysis_done' not in st.session_state:
        with st.spinner("🧠 Running advanced NLP analysis..."):
            # Always use full dataset
            df_analysis = df.copy()
            
            # Fix language detection
            df_analysis = fix_language_detection(df_analysis)
            
            # Run comprehensive analysis
            df_analyzed, report = run_advanced_nlp_analysis(df_analysis)
            
            st.session_state.df_analyzed = df_analyzed
            st.session_state.analysis_report = report
            st.session_state.advanced_analysis_done = True
    
    df_analyzed = st.session_state.df_analyzed
    report = st.session_state.analysis_report

# ============================================================================
# EXECUTIVE SUMMARY METRICS
# ============================================================================

st.header("📊 Summary Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ires = report['business_metrics'].get('intent_resolution_efficiency_score', 0)
    st.metric(
        "Intent Resolution Efficiency",
        f"{ires:.1f}%",
        delta=f"{ires - 75:.1f}% vs target",
        delta_color="normal" if ires >= 75 else "inverse",
        help="Measures how well intents are recognized AND resolved quickly. Combines intent confidence (50%) and fast response rate (50%). Target: 75%+"
    )

with col2:
    ufi = report['business_metrics'].get('user_frustration_index', 0)
    st.metric(
        "User Frustration Index",
        f"{ufi:.1f}%",
        delta=f"{10 - ufi:.1f}% vs target",
        delta_color="inverse" if ufi > 10 else "normal",
        help="Percentage of users showing frustrated emotion based on sentiment analysis. Target: <10%"
    )

with col3:
    sssr = report['business_metrics'].get('self_service_success_rate', 0)
    st.metric(
        "Self-Service Success",
        f"{sssr:.1f}%",
        delta=f"{sssr - 80:.1f}% vs target",
        delta_color="normal" if sssr >= 80 else "inverse",
        help="Percentage of queries resolved without fallbacks or escalations. Target: >80%"
    )

with col4:
    friction = report['business_metrics'].get('friction_impact_score', 0)
    st.metric(
        "Friction Impact Score",
        f"{friction:.1f}%",
        delta=f"{5 - friction:.1f}% vs target",
        delta_color="inverse" if friction > 5 else "normal",
        help="Percentage of conversations with loops (repeated intents) or drop-offs. Lower is better. Target: <5%"
    )

with col5:
    cost = report['business_metrics'].get('avg_cost_per_conversation', 0)
    st.metric(
        "Avg Cost/Conversation",
        f"${cost:.4f}",
        help="Average LLM cost per conversation (sum of all message costs in conversation)"
    )

st.markdown("---")

# ============================================================================
# ACTIONABLE INSIGHTS (TOP PRIORITY)
# ============================================================================

st.header("🎯 Prioritized Action Items")

insights = report.get('actionable_insights', [])

if insights:
    # Create tabs for different stakeholders
    stakeholders = list(set(i['stakeholder'] for i in insights))
    tabs = st.tabs(stakeholders)
    
    for tab, stakeholder in zip(tabs, stakeholders):
        with tab:
            stakeholder_insights = [i for i in insights if i['stakeholder'] == stakeholder]
            
            for idx, insight in enumerate(stakeholder_insights, 1):
                priority_colors = {
                    'HIGH': '🔴',
                    'MEDIUM': '🟡', 
                    'LOW': '🟢'
                }
                
                with st.expander(
                    f"{priority_colors[insight['priority']]} **{insight['priority']}** - {insight['category']}", 
                    expanded=(insight['priority'] == 'HIGH')
                ):
                    st.markdown(f"**📊 Insight:** {insight['insight']}")
                    st.markdown(f"**💡 Recommended Action:** {insight['action']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.info(f"**Expected Impact:** {insight['expected_impact']}")
                    with col_b:
                        st.info(f"**Effort Required:** {insight['effort']}")
else:
    st.success("✅ No critical issues detected. System performing within acceptable ranges.")

st.markdown("---")

# ============================================================================
# INTENT PERFORMANCE ANALYSIS
# ============================================================================

st.header("🎯 Intent Classification Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Intent Distribution with Confidence")
    
    if 'Intent' in df_analyzed.columns and 'Intent_Confidence' in df_analyzed.columns:
        intent_stats = df_analyzed.groupby('Intent').agg({
            'Intent_Confidence': ['mean', 'count'],
            'Request': 'count'
        })
        intent_stats.columns = ['Avg_Confidence', 'Count', 'Volume']
        intent_stats = intent_stats.reset_index()
        
        # Create bubble chart: Volume vs Confidence
        fig_intent = px.scatter(
            intent_stats,
            x='Avg_Confidence',
            y='Volume',
            size='Volume',
            color='Intent',
            hover_data=['Intent', 'Volume', 'Avg_Confidence'],
            title="Intent Volume vs Classification Confidence",
            labels={
                'Avg_Confidence': 'Average Confidence Score',
                'Volume': 'Number of Requests'
            }
        )
        
        # Add threshold lines
        fig_intent.add_vline(x=0.7, line_dash="dash", line_color="red", 
                            annotation_text="Confidence Threshold")
        
        st.plotly_chart(fig_intent, width='stretch')
        
        # Gap analysis
        st.subheader("🔍 Intent Gaps (High Volume, Low Confidence)")
        gaps = report['business_metrics'].get('top_intent_gaps', {})
        if gaps:
            gap_df = pd.DataFrame([
                {
                    'Intent': k,
                    'Volume': v['volume'],
                    'Avg Confidence': f"{v['Intent_Confidence']:.2f}",
                    'Gap Score': f"{v['gap_score']:.2f}"
                }
                for k, v in gaps.items()
            ])
            st.dataframe(gap_df, width='stretch')

with col2:
    st.subheader("Sentiment Analysis Results")
    
    if 'emotion' in df_analyzed.columns:
        emotion_dist = df_analyzed['emotion'].value_counts().reset_index()
        emotion_dist.columns = ['Emotion', 'Count']
        
        # Color mapping
        emotion_colors = {
            'frustrated': '#FF4B4B',
            'satisfied': '#00CC66',
            'neutral': '#FFAA00',
            'confused': '#4B8BFF'
        }
        
        fig_emotion = px.pie(
            emotion_dist,
            values='Count',
            names='Emotion',
            title="User Emotion Distribution",
            hole=0.4,
            color='Emotion',
            color_discrete_map=emotion_colors
        )
        
        st.plotly_chart(fig_emotion, width='stretch')
        
        # Sentiment trend over time
        if 'Timestamp' in df_analyzed.columns:
            st.subheader("😊 Sentiment Trend Over Time")
            df_analyzed['Date'] = pd.to_datetime(df_analyzed['Timestamp']).dt.date
            
            sentiment_trend = df_analyzed.groupby('Date')['polarity'].mean().reset_index()
            sentiment_trend.columns = ['Date', 'Avg_Sentiment']
            
            fig_trend = px.line(
                sentiment_trend,
                x='Date',
                y='Avg_Sentiment',
                title="Daily Sentiment Trend",
                markers=True
            )
            fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_trend.update_yaxes(range=[-1, 1])
            
            st.plotly_chart(fig_trend, width='stretch')

st.markdown("---")

# ============================================================================
# FRICTION ANALYSIS
# ============================================================================

st.header("⚠️ User Journey Friction Points")

friction_data = report.get('friction_analysis', {})

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚪 Top Drop-off Points")
    drop_offs = friction_data.get('drop_off_points', {})
    
    if drop_offs:
        drop_df = pd.DataFrame([
            {'Intent': k, 'Drop-offs': v}
            for k, v in sorted(drop_offs.items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        fig_drops = px.bar(
            drop_df,
            x='Drop-offs',
            y='Intent',
            orientation='h',
            title="Intents Where Users Abandon Conversations",
            color='Drop-offs',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_drops, width='stretch')
        
        st.caption("💡 **Action:** Review these intents for unclear responses or missing information")
    else:
        st.info("No significant drop-off patterns detected")

with col2:
    st.subheader("🔄 Conversation Loops")
    loops = friction_data.get('conversation_loops', {})
    
    if loops:
        loop_df = pd.DataFrame([
            {'Intent': k, 'Loops': v}
            for k, v in sorted(loops.items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        
        fig_loops = px.bar(
            loop_df,
            x='Loops',
            y='Intent',
            orientation='h',
            title="Intents With Repeated Questions (User Not Satisfied)",
            color='Loops',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_loops, width='stretch')
        
        st.caption("💡 **Action:** Improve response clarity or add follow-up suggestions")
    else:
        st.info("No significant conversation loops detected")

# Unresolved Patterns
st.subheader("❌ Fallback/Unresolved Patterns")
unresolved = friction_data.get('unresolved_patterns', [])

if unresolved:
    unresolved_df = pd.DataFrame([
        {
            'Pattern': p['pattern'],
            'Count': p['count'],
            'Percentage': f"{(p['count'] / len(df_analyzed)) * 100:.2f}%",
            'Top Intent': ', '.join([f"{k} ({v})" for k, v in list(p.get('top_intents', {}).items())[:3]])
        }
        for p in unresolved
    ])
    
    st.dataframe(unresolved_df, width='stretch')
    st.caption("💡 **Action:** These are chatbot responses indicating failure - expand knowledge base for these intents")
else:
    st.success("✅ Low fallback rate - chatbot is handling queries well!")

st.markdown("---")

# ============================================================================
# COST ANALYSIS
# ============================================================================

st.header("💰 Cost Efficiency Analysis")

col1, col2 = st.columns(2)

with col1:
    cost_by_intent = report['business_metrics'].get('cost_by_intent', {})
    
    if cost_by_intent:
        cost_df = pd.DataFrame([
            {'Intent': k, 'Avg Cost': v}
            for k, v in sorted(cost_by_intent.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig_cost = px.bar(
            cost_df,
            x='Avg Cost',
            y='Intent',
            orientation='h',
            title="Average Cost per Intent",
            color='Avg Cost',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cost, width='stretch')
        
        st.caption("💡 **Optimization:** Focus on high-cost intents for prompt optimization")

with col2:
    if 'Intent' in df_analyzed.columns and 'Cost' in df_analyzed.columns and 'Latency' in df_analyzed.columns:
        st.subheader("⚡ Cost vs Performance Trade-off")
        
        perf_df = df_analyzed.groupby('Intent').agg({
            'Cost': 'mean',
            'Latency': 'mean',
            'Request': 'count'
        }).reset_index()
        perf_df.columns = ['Intent', 'Avg_Cost', 'Avg_Latency', 'Volume']
        
        fig_perf = px.scatter(
            perf_df,
            x='Avg_Latency',
            y='Avg_Cost',
            size='Volume',
            color='Intent',
            title="Cost vs Latency by Intent",
            labels={
                'Avg_Latency': 'Average Latency (seconds)',
                'Avg_Cost': 'Average Cost ($)'
            }
        )
        
        st.plotly_chart(fig_perf, width='stretch')

st.markdown("---")

# ============================================================================
# LANGUAGE ANALYSIS
# ============================================================================

st.header("🌍 Multilingual Support Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'Language_Corrected' in df_analyzed.columns:
        lang_dist = df_analyzed['Language_Corrected'].value_counts().head(10).reset_index()
        lang_dist.columns = ['Language', 'Count']
        
        fig_lang = px.bar(
            lang_dist,
            x='Count',
            y='Language',
            orientation='h',
            title="Top 10 Languages Detected",
            color='Count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_lang, width='stretch')

with col2:
    if 'Language_Corrected' in df_analyzed.columns and 'polarity' in df_analyzed.columns:
        st.subheader("😊 Sentiment by Language")
        
        lang_sentiment = df_analyzed.groupby('Language_Corrected')['polarity'].mean().sort_values().head(10).reset_index()
        lang_sentiment.columns = ['Language', 'Avg_Sentiment']
        
        fig_lang_sent = px.bar(
            lang_sentiment,
            x='Avg_Sentiment',
            y='Language',
            orientation='h',
            title="Average Sentiment by Language",
            color='Avg_Sentiment',
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        
        st.plotly_chart(fig_lang_sent, width='stretch')
        
        st.caption("💡 **Note:** Negative sentiment in specific languages may indicate translation/localization issues")

st.markdown("---")

# ============================================================================
# DOWNLOAD REPORT
# ============================================================================

st.header("📥 Export Analysis")

col1, col2 = st.columns(2)

with col1:
    # Export enhanced dataframe
    csv = df_analyzed.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Enhanced Dataset (with NLP features)",
        data=csv,
        file_name="digiyatra_nlp_analysis.csv",
        mime="text/csv"
    )

with col2:
    # Export executive report
    import json
    report_json = json.dumps(report, indent=2, default=str)
    st.download_button(
        label="⬇️ Download Executive Report (JSON)",
        data=report_json,
        file_name="digiyatra_executive_report.json",
        mime="application/json"
    )

# Footer
st.markdown("---")
st.caption("💼 Executive Insights Dashboard | Powered by Advanced NLP Analytics | DigiYatra 2026")
