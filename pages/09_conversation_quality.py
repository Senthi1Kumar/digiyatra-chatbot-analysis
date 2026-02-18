import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.data_loader import get_session_data
from src.preprocessing import preprocess_data
from src.conversation_quality_metrics import (
    calculate_fcr,
    calculate_csat_from_feedback,
    calculate_deflection_rate,
    analyze_response_quality,
    calculate_conversation_effectiveness_score,
    generate_quality_improvement_plan
)

st.set_page_config(
    page_title="Conversation Quality - DigiYatra", 
    page_icon="⭐", 
    layout="wide"
)

st.title("⭐ Conversation Quality & Effectiveness")
st.markdown("""
**Industry-standard quality metrics for chatbot performance**
""")

# Load data
with st.spinner("Loading and analyzing conversation quality..."):
    df = get_session_data()
    if df.empty:
        st.error("❌ No data uploaded. Please upload a CSV file on the home page first.")
        st.stop()
    
    df = preprocess_data(df)
    
    # Cache quality analysis
    if 'quality_analysis_done' not in st.session_state:
        with st.spinner("📊 Calculating quality metrics..."):
            # Calculate all quality metrics
            fcr_data = calculate_fcr(df)
            csat_data = calculate_csat_from_feedback(df)
            deflection_data = calculate_deflection_rate(df)
            quality_data = analyze_response_quality(df)
            effectiveness_data = calculate_conversation_effectiveness_score(df)
            improvement_plan = generate_quality_improvement_plan(effectiveness_data)
            
            st.session_state.fcr_data = fcr_data
            st.session_state.csat_data = csat_data
            st.session_state.deflection_data = deflection_data
            st.session_state.quality_data = quality_data
            st.session_state.effectiveness_data = effectiveness_data
            st.session_state.improvement_plan = improvement_plan
            st.session_state.quality_analysis_done = True
    
    fcr_data = st.session_state.fcr_data
    csat_data = st.session_state.csat_data
    deflection_data = st.session_state.deflection_data
    quality_data = st.session_state.quality_data
    effectiveness_data = st.session_state.effectiveness_data
    improvement_plan = st.session_state.improvement_plan

# ============================================================================
# OVERALL EFFECTIVENESS SCORE (Hero Metric)
# ============================================================================

st.header("🎯 Overall Conversation Effectiveness")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    score = effectiveness_data['overall_effectiveness_score']
    grade = effectiveness_data['grade']
    
    # Color based on grade
    if score >= 85:
        color = "normal"
        emoji = "🌟"
    elif score >= 75:
        color = "normal"
        emoji = "✅"
    elif score >= 65:
        color = "off"
        emoji = "⚠️"
    else:
        color = "inverse"
        emoji = "❌"
    
    st.metric(
        "Effectiveness Score",
        f"{score:.1f}%",
        delta=f"{score - 75:.1f}% vs target",
        delta_color=color,
        help="Weighted composite score: 30% FCR + 35% CSAT + 20% Deflection + 15% Response Quality"
    )
    st.markdown(f"### {emoji} Grade: **{grade}**")

with col2:
    with st.expander("ℹ️ Score Calculation", expanded=False):
        st.write("**Formula:**")
        st.write("- FCR × 30%")
        st.write("- CSAT × 35%")
        st.write("- Deflection × 20%")
        st.write("- Response Quality × 15%")

with col3:
    with st.expander("ℹ️ Grade Scale", expanded=False):
        st.write("**Benchmarks:**")
        st.write("- A: 85-100%")
        st.write("- B: 75-84%")
        st.write("- C: 65-74%")
        st.write("- D: 50-64%")
        st.write("- F: <50%")

# Component breakdown
st.subheader("📊 Component Breakdown")

components = effectiveness_data['components']
weights = effectiveness_data['weights']

component_df = pd.DataFrame([
    {
        'Metric': 'First Contact Resolution',
        'Score': components['fcr'],
        'Weight': weights['fcr'] * 100,
        'Contribution': components['fcr'] * weights['fcr']
    },
    {
        'Metric': 'Customer Satisfaction',
        'Score': components['csat'],
        'Weight': weights['csat'] * 100,
        'Contribution': components['csat'] * weights['csat']
    },
    {
        'Metric': 'Deflection Rate',
        'Score': components['deflection'],
        'Weight': weights['deflection'] * 100,
        'Contribution': components['deflection'] * weights['deflection']
    },
    {
        'Metric': 'Response Quality',
        'Score': components['response_quality'],
        'Weight': weights['quality'] * 100,
        'Contribution': components['response_quality'] * weights['quality']
    }
])

fig_components = go.Figure()

fig_components.add_trace(go.Bar(
    name='Score',
    x=component_df['Metric'],
    y=component_df['Score'],
    marker_color='lightblue',
    text=component_df['Score'].apply(lambda x: f"{x:.1f}%"),
    textposition='auto'
))

fig_components.add_trace(go.Scatter(
    name='Target (75%)',
    x=component_df['Metric'],
    y=[75, 75, 75, 75],
    mode='lines',
    line=dict(color='red', dash='dash'),
    showlegend=True
))

fig_components.update_layout(
    title="Quality Metrics vs Target",
    yaxis_title="Score (%)",
    barmode='group',
    height=400
)

st.plotly_chart(fig_components, width='stretch')

st.markdown("---")

# ============================================================================
# FIRST CONTACT RESOLUTION (FCR)
# ============================================================================

st.header("🎯 First Contact Resolution (FCR)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "FCR Rate",
        f"{fcr_data.get('fcr_rate', 0):.1f}%",
        delta=f"{fcr_data.get('fcr_rate', 0) - 75:.1f}% vs target",
        help="Percentage of conversations resolved in a single session without repeated intents, clarifications, negative feedback, or >5 turns"
    )

with col2:
    st.metric(
        "Resolved on First Contact",
        f"{fcr_data.get('first_contact_resolved', 0):,}",
        help="Number of conversations successfully resolved in first interaction"
    )

with col3:
    st.metric(
        "Total Conversations",
        f"{fcr_data.get('total_conversations', 0):,}",
        help="Total number of unique conversations analyzed"
    )

with col4:
    if fcr_data.get('fcr_rate', 0) >= 75:
        st.success("✅ Meeting Target")
    else:
        st.error("❌ Below Target")

with st.expander("ℹ️ What is FCR?", expanded=False):
    st.markdown("""
    **First Contact Resolution** measures conversations resolved in a single session **without:**
    - Repeated intents (user asking same thing multiple times)
    - Clarification loops
    - Negative feedback
    - Long conversation chains (>5 turns)
    
    **Target:** 75% (industry standard for chatbots)
    """)

# FCR by Intent
st.subheader("📊 FCR Performance by Intent")

if fcr_data.get('by_intent'):
    intent_fcr = []
    for intent, stats in fcr_data['by_intent'].items():
        if stats['total'] >= 5:  # Only show intents with enough data
            intent_fcr.append({
                'Intent': intent,
                'FCR Rate': stats['fcr_rate'],
                'Resolved': stats['resolved'],
                'Total': stats['total']
            })
    
    if intent_fcr:
        intent_fcr_df = pd.DataFrame(intent_fcr).sort_values('FCR Rate')
        
        fig_fcr = px.bar(
            intent_fcr_df,
            x='FCR Rate',
            y='Intent',
            orientation='h',
            color='FCR Rate',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            title="First Contact Resolution by Intent",
            text='FCR Rate'
        )
        fig_fcr.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_fcr.add_vline(x=75, line_dash="dash", line_color="red", annotation_text="Target: 75%")
        
        st.plotly_chart(fig_fcr, width='stretch')
        
        # Show data table
        with st.expander("📋 View Detailed FCR Data"):
            st.dataframe(intent_fcr_df, width='stretch')

st.markdown("---")

# ============================================================================
# CUSTOMER SATISFACTION (CSAT)
# ============================================================================

st.header("😊 Customer Satisfaction (CSAT)")

col1, col2, col3, col4 = st.columns(4)

csat_score = csat_data.get('csat_score')

with col1:
    if csat_score is not None:
        st.metric(
            "CSAT Score",
            f"{csat_score:.1f}%",
            delta=f"{csat_score - 80:.1f}% vs target",
            help="Percentage of positive feedback responses. Calculated as: (Positive Responses / Total Responses) × 100"
        )
    else:
        st.metric("CSAT Score", "N/A", help="No feedback data available in dataset")

with col2:
    st.metric(
        "Feedback Responses",
        f"{csat_data.get('total_feedback_responses', 0):,}",
        help="Total number of users who provided feedback (thumbs up/down, ratings, or comments)"
    )

with col3:
    rating = csat_data.get('industry_rating', 'N/A')
    if rating == 'excellent':
        st.success(f"⭐ **{rating.upper()}**")
    elif rating == 'good':
        st.info(f"✅ **{rating.upper()}**")
    elif rating == 'fair':
        st.warning(f"⚠️ **{rating.upper()}**")
    else:
        st.error(f"❌ **{rating.upper()}**")
    st.caption("Industry Rating")

with col4:
    breakdown = csat_data.get('feedback_breakdown', {})
    if breakdown:
        positive = breakdown.get('positive', 0)
        total = sum(breakdown.values())
        if total > 0:
            st.metric("Positive Rate", f"{(positive/total)*100:.1f}%", help="Percentage of feedback that was positive")

# Feedback breakdown
st.subheader("📊 Feedback Distribution")

breakdown = csat_data.get('feedback_breakdown', {})
if breakdown:
    feedback_df = pd.DataFrame([
        {'Type': k.capitalize(), 'Count': v}
        for k, v in breakdown.items()
    ])
    
    colors = {'Positive': '#00CC66', 'Neutral': '#FFAA00', 'Negative': '#FF4B4B'}
    
    fig_feedback = px.pie(
        feedback_df,
        values='Count',
        names='Type',
        title="User Feedback Breakdown",
        hole=0.4,
        color='Type',
        color_discrete_map=colors
    )
    
    st.plotly_chart(fig_feedback, width='stretch')
else:
    st.info("No feedback data available in the dataset")

# CSAT Benchmarks
st.subheader("📏 Industry Benchmarks")
benchmark_df = pd.DataFrame([
    {'Rating': 'Excellent', 'Range': '80-100%', 'Your Score': '✅' if csat_score and csat_score >= 80 else ''},
    {'Rating': 'Good', 'Range': '70-80%', 'Your Score': '✅' if csat_score and 70 <= csat_score < 80 else ''},
    {'Rating': 'Fair', 'Range': '60-70%', 'Your Score': '✅' if csat_score and 60 <= csat_score < 70 else ''},
    {'Rating': 'Poor', 'Range': '0-60%', 'Your Score': '✅' if csat_score and csat_score < 60 else ''}
])
st.table(benchmark_df)

st.markdown("---")

# ============================================================================
# DEFLECTION RATE
# ============================================================================

st.header("🚀 Deflection Rate")

col1, col2, col3, col4 = st.columns(4)

with col1:
    deflection_rate = deflection_data.get('deflection_rate', 0)
    st.metric(
        "Deflection Rate",
        f"{deflection_rate:.1f}%",
        delta=f"{deflection_rate - 75:.1f}% vs target",
        help="Percentage of inquiries resolved through self-service without human agent intervention"
    )

with col2:
    st.metric(
        "Self-Service Resolved",
        f"{deflection_data.get('self_service_resolved', 0):,}",
        help="Number of conversations resolved by chatbot without escalation"
    )

with col3:
    st.metric(
        "Escalated to Human",
        f"{deflection_data.get('escalated', 0):,}",
        help="Number of conversations that required human agent assistance"
    )

with col4:
    total = deflection_data.get('total_inquiries', 1)
    escalated = deflection_data.get('escalated', 0)
    st.metric(
        "Escalation Rate",
        f"{(escalated/total)*100:.1f}%",
        help="Percentage of conversations that needed human intervention"
    )

with st.expander("ℹ️ Business Impact", expanded=False):
    st.markdown("""
    **Deflection Rate** = Percentage of inquiries resolved through self-service without human agent intervention
    
    **Cost Savings:**
    - Each 10% improvement in deflection = ₹5-10L/month savings
    - Reduced wait times for complex queries requiring human agents
    - Better agent productivity (handle complex cases only)
    
    **Target:** 75-85% (industry standard)
    """)

# Deflection visualization
deflection_viz_data = pd.DataFrame([
    {'Category': 'Self-Service Resolved', 'Count': deflection_data.get('self_service_resolved', 0), 'Color': 'green'},
    {'Category': 'Escalated to Human', 'Count': deflection_data.get('escalated', 0), 'Color': 'red'}
])

fig_deflection = px.pie(
    deflection_viz_data,
    values='Count',
    names='Category',
    title="Self-Service vs Escalation",
    hole=0.4,
    color='Category',
    color_discrete_map={'Self-Service Resolved': '#00CC66', 'Escalated to Human': '#FF4B4B'}
)

st.plotly_chart(fig_deflection, width='stretch')

st.markdown("---")

# ============================================================================
# RESPONSE QUALITY
# ============================================================================

st.header("📝 Response Quality Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Avg Response Length",
        f"{quality_data.get('avg_response_length', 0):.0f} chars",
        help="Average character count of chatbot responses"
    )

with col2:
    optimal_rate = quality_data.get('optimal_rate', 0)
    st.metric(
        "Optimal Length Rate",
        f"{optimal_rate:.1f}%",
        help="Percentage of responses between 100-500 characters (concise yet complete)"
    )

with col3:
    st.metric(
        "Empty Responses",
        f"{quality_data.get('empty_responses', 0):,}",
        delta=f"{quality_data.get('empty_response_rate', 0):.1f}%",
        help="Number of responses with 0 characters (system errors or no content)"
    )

with col4:
    st.metric(
        "Too Verbose",
        f"{quality_data.get('too_verbose_responses', 0):,}",
        delta=f"{quality_data.get('verbose_rate', 0):.1f}%",
        help="Number of responses >1000 characters (may overwhelm users)"
    )

with st.expander("ℹ️ Quality Guidelines", expanded=False):
    st.markdown("""
    **Response Length Benchmarks:**
    - **Optimal:** 100-500 characters (concise yet complete)
    - **Too Short:** <100 characters (may lack necessary detail)
    - **Too Verbose:** >1000 characters (users may not read entire response)
    
    **Target:** 70%+ responses in optimal range
    """)

# Response length distribution
quality_dist = pd.DataFrame([
    {'Category': 'Optimal (100-500)', 'Count': quality_data.get('optimal_responses', 0), 'Status': 'Good'},
    {'Category': 'Too Short (<100)', 'Count': quality_data.get('empty_responses', 0), 'Status': 'Warning'},
    {'Category': 'Too Verbose (>1000)', 'Count': quality_data.get('too_verbose_responses', 0), 'Status': 'Warning'}
])

fig_quality = px.bar(
    quality_dist,
    x='Category',
    y='Count',
    color='Status',
    color_discrete_map={'Good': '#00CC66', 'Warning': '#FFAA00'},
    title="Response Length Distribution"
)

st.plotly_chart(fig_quality, width='stretch')

st.markdown("---")

# ============================================================================
# IMPROVEMENT PLAN
# ============================================================================

st.header("🎯 Quality Improvement Roadmap")

if improvement_plan:
    st.markdown("**Auto-generated recommendations based on quality metrics:**")
    
    for idx, rec in enumerate(improvement_plan, 1):
        priority_colors = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        
        with st.expander(
            f"{priority_colors[rec['priority']]} **{rec['priority']}** - {rec['metric']} (Current: {rec['current_value']}, Target: {rec['target']})",
            expanded=(rec['priority'] == 'HIGH')
        ):
            st.markdown(f"**Current Performance:** {rec['current_value']}")
            st.markdown(f"**Target:** {rec['target']}")
            
            st.markdown("**Action Items:**")
            for action in rec['actions']:
                st.markdown(f"- {action}")
            
            st.success(f"**Expected Improvement:** {rec['expected_improvement']}")
else:
    st.success("✅ All quality metrics are within acceptable ranges!")

st.markdown("---")

# ============================================================================
# EXPORT
# ============================================================================

st.header("📥 Export Quality Report")

col1, col2 = st.columns(2)

with col1:
    # Create summary report
    summary_report = {
        'Overall Effectiveness Score': effectiveness_data['overall_effectiveness_score'],
        'Grade': effectiveness_data['grade'],
        'FCR Rate': fcr_data.get('fcr_rate', 0),
        'CSAT Score': csat_data.get('csat_score', 0),
        'Deflection Rate': deflection_data.get('deflection_rate', 0),
        'Response Quality Score': quality_data.get('optimal_rate', 0)
    }
    
    import json
    report_json = json.dumps({
        'summary': summary_report,
        'fcr_details': fcr_data,
        'csat_details': csat_data,
        'deflection_details': deflection_data,
        'quality_details': quality_data,
        'improvement_plan': improvement_plan
    }, indent=2, default=str)
    
    st.download_button(
        label="⬇️ Download Quality Report (JSON)",
        data=report_json,
        file_name="conversation_quality_report.json",
        mime="application/json"
    )

with col2:
    # Create CSV summary
    summary_df = pd.DataFrame([summary_report])
    csv = summary_df.to_csv(index=False)
    
    st.download_button(
        label="⬇️ Download Summary (CSV)",
        data=csv,
        file_name="quality_summary.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("⭐ Conversation Quality Dashboard | Industry-Standard Metrics | DigiYatra 2026")
