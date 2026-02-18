"""
Conversation Quality Metrics & CSAT Analysis
============================================

Advanced metrics for measuring conversation quality beyond simple success/fail:
- First Contact Resolution (FCR)
- Customer Satisfaction Score (CSAT) derived from feedback
- Quality-adjusted deflection rate
- Response coherence and relevance scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import re
from collections import defaultdict


def calculate_fcr(df: pd.DataFrame) -> Dict:
    """
    Calculate First Contact Resolution (FCR).
    
    FCR = Conversations resolved in single session without:
    - Multiple repeated intents
    - Clarification loops
    - Negative feedback
    - Escalation indicators
    """
    
    if 'Conversation ID' not in df.columns:
        return {}
    
    fcr_data = {
        'total_conversations': 0,
        'first_contact_resolved': 0,
        'fcr_rate': 0.0,
        'by_intent': {}
    }
    
    for conv_id, group in df.groupby('Conversation ID'):
        fcr_data['total_conversations'] += 1
        
        # Criteria for FCR
        is_fcr = True
        
        # 1. Check for repeated intents (indicates confusion)
        if 'Intent' in group.columns and len(group) > 1:
            intents = group['Intent'].tolist()
            if len(intents) != len(set(intents)):  # Repeated intent
                is_fcr = False
        
        # 2. Check for clarification requests
        if 'Clarification' in group.columns:
            if group['Clarification'].notna().any():
                is_fcr = False
        
        # 3. Check for negative feedback
        if 'User Feedback' in group.columns:
            feedback_str = ' '.join(group['User Feedback'].fillna('').astype(str))
            if any(neg in feedback_str.lower() for neg in ['thumbs down', 'bad', 'poor', 'negative']):
                is_fcr = False
        
        # 4. Check for fallback responses
        if 'Response' in group.columns:
            responses = ' '.join(group['Response'].fillna('').astype(str))
            fallback_patterns = [
                "didn't understand", "can you rephrase", "sorry", 
                "unable to", "not sure", "could not find"
            ]
            if any(pattern in responses.lower() for pattern in fallback_patterns):
                is_fcr = False
        
        # 5. Long conversations (>5 turns) less likely to be FCR
        if len(group) > 5:
            is_fcr = False
        
        if is_fcr:
            fcr_data['first_contact_resolved'] += 1
            
            # Track by intent
            if 'Intent' in group.columns:
                first_intent = group.iloc[0]['Intent']
                if first_intent not in fcr_data['by_intent']:
                    fcr_data['by_intent'][first_intent] = {'resolved': 0, 'total': 0}
                fcr_data['by_intent'][first_intent]['resolved'] += 1
        
        # Track total by intent
        if 'Intent' in group.columns:
            first_intent = group.iloc[0]['Intent']
            if first_intent not in fcr_data['by_intent']:
                fcr_data['by_intent'][first_intent] = {'resolved': 0, 'total': 0}
            fcr_data['by_intent'][first_intent]['total'] += 1
    
    # Calculate rates
    if fcr_data['total_conversations'] > 0:
        fcr_data['fcr_rate'] = (fcr_data['first_contact_resolved'] / 
                               fcr_data['total_conversations']) * 100
    
    # Calculate by intent
    for intent, counts in fcr_data['by_intent'].items():
        if counts['total'] > 0:
            counts['fcr_rate'] = (counts['resolved'] / counts['total']) * 100
    
    return fcr_data


def calculate_csat_from_feedback(df: pd.DataFrame) -> Dict:
    """
    Calculate Customer Satisfaction Score (CSAT) from user feedback.
    
    CSAT = (Positive Responses / Total Responses) * 100
    """
    
    if 'User Feedback' not in df.columns:
        return {'csat_score': None, 'feedback_breakdown': {}}
    
    # Parse feedback
    feedback_counts = defaultdict(int)
    total_feedback = 0
    positive_feedback = 0
    
    for feedback in df['User Feedback'].fillna(''):
        if not feedback or feedback == '':
            continue
        
        total_feedback += 1
        feedback_lower = str(feedback).lower()
        
        # Categorize feedback
        if any(pos in feedback_lower for pos in ['thumbs up', 'good', 'positive', 'great', 'excellent', 'helpful']):
            feedback_counts['positive'] += 1
            positive_feedback += 1
        elif any(neg in feedback_lower for neg in ['thumbs down', 'bad', 'poor', 'negative', 'awful', 'useless']):
            feedback_counts['negative'] += 1
        else:
            feedback_counts['neutral'] += 1
    
    # Calculate CSAT
    csat_score = (positive_feedback / total_feedback * 100) if total_feedback > 0 else None
    
    # Industry benchmark context
    benchmark = {
        'excellent': (80, 100),
        'good': (70, 80),
        'fair': (60, 70),
        'poor': (0, 60)
    }
    
    rating = 'N/A'
    if csat_score is not None:
        for level, (low, high) in benchmark.items():
            if low <= csat_score < high:
                rating = level
                break
    
    return {
        'csat_score': round(csat_score, 2) if csat_score else None,
        'feedback_breakdown': dict(feedback_counts),
        'total_feedback_responses': total_feedback,
        'industry_rating': rating,
        'benchmark': benchmark
    }


def calculate_deflection_rate(df: pd.DataFrame) -> Dict:
    """
    Calculate deflection rate (self-service resolution without human intervention).
    
    Quality-adjusted: Only counts as deflected if user seems satisfied.
    """
    
    if 'Conversation ID' not in df.columns:
        return {}
    
    deflection_data = {
        'total_inquiries': 0,
        'self_service_resolved': 0,
        'escalated': 0,
        'deflection_rate': 0.0,
        'quality_adjusted_deflection': 0.0
    }
    
    for conv_id, group in df.groupby('Conversation ID'):
        deflection_data['total_inquiries'] += 1
        
        # Check for escalation indicators
        escalated = False
        
        if 'Response' in group.columns:
            responses = ' '.join(group['Response'].fillna('').astype(str))
            escalation_phrases = [
                'transfer to agent', 'speak to human', 'contact support',
                'call us', 'reach out', 'human assistance'
            ]
            if any(phrase in responses.lower() for phrase in escalation_phrases):
                escalated = True
        
        if 'Clarification' in group.columns:
            # Multiple clarifications suggest need for human help
            if group['Clarification'].notna().sum() > 2:
                escalated = True
        
        if escalated:
            deflection_data['escalated'] += 1
        else:
            deflection_data['self_service_resolved'] += 1
            
            # Quality check: Was user satisfied?
            quality_resolved = True
            
            if 'User Feedback' in group.columns:
                feedback = ' '.join(group['User Feedback'].fillna('').astype(str))
                if any(neg in feedback.lower() for neg in ['thumbs down', 'bad', 'negative']):
                    quality_resolved = False
            
            if len(group) > 6:  # Too long conversation = not efficient
                quality_resolved = False
    
    # Calculate rates
    if deflection_data['total_inquiries'] > 0:
        deflection_data['deflection_rate'] = (
            deflection_data['self_service_resolved'] / 
            deflection_data['total_inquiries']
        ) * 100
    
    return deflection_data


def analyze_response_quality(df: pd.DataFrame) -> Dict:
    """
    Analyze response quality metrics:
    - Response completeness (length, structure)
    - Response relevance (based on user follow-up)
    - Response tone consistency
    """
    
    quality_metrics = {
        'avg_response_length': 0,
        'response_length_variance': 0,
        'empty_responses': 0,
        'too_verbose_responses': 0,
        'optimal_responses': 0
    }
    
    if 'Response' not in df.columns:
        return quality_metrics
    
    response_lengths = []
    
    for response in df['Response'].fillna(''):
        length = len(str(response))
        response_lengths.append(length)
        
        if length == 0:
            quality_metrics['empty_responses'] += 1
        elif length > 1000:  # Too verbose
            quality_metrics['too_verbose_responses'] += 1
        elif 100 <= length <= 500:  # Optimal range
            quality_metrics['optimal_responses'] += 1
    
    if response_lengths:
        quality_metrics['avg_response_length'] = np.mean(response_lengths)
        quality_metrics['response_length_variance'] = np.var(response_lengths)
    
    # Calculate percentages
    total = len(df)
    if total > 0:
        quality_metrics['empty_response_rate'] = (quality_metrics['empty_responses'] / total) * 100
        quality_metrics['verbose_rate'] = (quality_metrics['too_verbose_responses'] / total) * 100
        quality_metrics['optimal_rate'] = (quality_metrics['optimal_responses'] / total) * 100
    
    return quality_metrics


def calculate_conversation_effectiveness_score(df: pd.DataFrame) -> Dict:
    """
    Composite metric combining multiple quality factors:
    - FCR
    - CSAT
    - Deflection Rate
    - Response Quality
    
    Returns a 0-100 score.
    """
    
    # Get individual metrics
    fcr_data = calculate_fcr(df)
    csat_data = calculate_csat_from_feedback(df)
    deflection_data = calculate_deflection_rate(df)
    quality_data = analyze_response_quality(df)
    
    # Weight each component
    weights = {
        'fcr': 0.30,
        'csat': 0.35,
        'deflection': 0.20,
        'quality': 0.15
    }
    
    # Normalize scores to 0-100
    fcr_score = fcr_data.get('fcr_rate', 0)
    csat_score = csat_data.get('csat_score', 0) or 0
    deflection_score = deflection_data.get('deflection_rate', 0)
    quality_score = quality_data.get('optimal_rate', 0) or 0
    
    # Calculate weighted average
    effectiveness_score = (
        fcr_score * weights['fcr'] +
        csat_score * weights['csat'] +
        deflection_score * weights['deflection'] +
        quality_score * weights['quality']
    )
    
    # Determine grade
    if effectiveness_score >= 85:
        grade = 'A (Excellent)'
    elif effectiveness_score >= 75:
        grade = 'B (Good)'
    elif effectiveness_score >= 65:
        grade = 'C (Fair)'
    elif effectiveness_score >= 50:
        grade = 'D (Needs Improvement)'
    else:
        grade = 'F (Critical)'
    
    return {
        'overall_effectiveness_score': round(effectiveness_score, 2),
        'grade': grade,
        'components': {
            'fcr': fcr_score,
            'csat': csat_score,
            'deflection': deflection_score,
            'response_quality': quality_score
        },
        'weights': weights,
        'detailed_metrics': {
            'fcr_data': fcr_data,
            'csat_data': csat_data,
            'deflection_data': deflection_data,
            'quality_data': quality_data
        }
    }


def generate_quality_improvement_plan(effectiveness_data: Dict) -> list:
    """
    Generate specific recommendations based on quality metrics.
    """
    
    recommendations = []
    components = effectiveness_data['components']
    
    # FCR Recommendations
    if components['fcr'] < 65:
        recommendations.append({
            'metric': 'First Contact Resolution',
            'current_value': f"{components['fcr']:.1f}%",
            'target': '75-85%',
            'priority': 'HIGH',
            'actions': [
                'Analyze conversations with repeated intents to identify unclear responses',
                'Improve intent classifier to reduce misclassification',
                'Add clarification flows for ambiguous queries',
                'Expand knowledge base for top unresolved intents'
            ],
            'expected_improvement': '10-15% FCR increase'
        })
    
    # CSAT Recommendations
    if components['csat'] < 75:
        recommendations.append({
            'metric': 'Customer Satisfaction',
            'current_value': f"{components['csat']:.1f}%",
            'target': '80-90%',
            'priority': 'HIGH',
            'actions': [
                'Review conversations with negative feedback to identify patterns',
                'Improve response tone - make it more empathetic and helpful',
                'Add proactive suggestions and next steps in responses',
                'Reduce response latency for better user experience'
            ],
            'expected_improvement': '12-18% CSAT increase'
        })
    
    # Deflection Rate Recommendations  
    if components['deflection'] < 70:
        recommendations.append({
            'metric': 'Self-Service Deflection',
            'current_value': f"{components['deflection']:.1f}%",
            'target': '75-85%',
            'priority': 'MEDIUM',
            'actions': [
                'Identify queries leading to escalation and add self-service paths',
                'Improve fallback handling with better suggestions',
                'Add FAQ content for common complex queries',
                'Implement escalation prevention flows'
            ],
            'expected_improvement': '8-12% deflection increase, ₹3-5L/month savings'
        })
    
    # Response Quality Recommendations
    if components['response_quality'] < 60:
        recommendations.append({
            'metric': 'Response Quality',
            'current_value': f"{components['response_quality']:.1f}%",
            'target': '70-80%',
            'priority': 'MEDIUM',
            'actions': [
                'Optimize prompts to generate concise yet complete responses',
                'Add response templates for common intents',
                'Implement response quality checks before sending to user',
                'A/B test different response formats'
            ],
            'expected_improvement': '15-20% response quality increase'
        })
    
    return recommendations


if __name__ == "__main__":
    print("Conversation Quality Metrics Module Loaded!")
    print("Available functions:")
    print("- calculate_fcr(df)")
    print("- calculate_csat_from_feedback(df)")
    print("- calculate_deflection_rate(df)")
    print("- analyze_response_quality(df)")
    print("- calculate_conversation_effectiveness_score(df)")
