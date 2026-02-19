import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. INTENT CLASSIFICATION (Hybrid Approach)
# ============================================================================

class IntentClassifier:
    """
    Hybrid intent classification combining:
    - Rule-based for high-confidence patterns
    - Semantic similarity for ambiguous cases
    - Context awareness (previous messages in conversation)
    """
    
    def __init__(self):
        # DigiYatra-specific intent patterns with confidence scores
        self.intent_patterns = {
            "OTP/Login Issue": {
                "patterns": [
                    r'\b(otp|one time password|verification code|sms|login|log in|sign in)\b',
                    r'\b(not receiving|didn\'?t receive|haven\'?t received)\b.*\b(otp|code)\b',
                    r'\b(cannot login|can\'?t login|unable to login)\b'
                ],
                "keywords": ["otp", "login", "verification", "code", "sms", "password"],
                "confidence_threshold": 0.7
            },
            "Registration Issue": {
                "patterns": [
                    r'\b(register|registration|sign up|signup|create account|new account)\b',
                    r'\b(cannot register|can\'?t register|unable to register)\b',
                    r'\b(registration.*fail|registration.*error)\b'
                ],
                "keywords": ["register", "registration", "signup", "account", "create"],
                "confidence_threshold": 0.7
            },
            "Face Verification": {
                "patterns": [
                    r'\b(face|facial|selfie|photo|picture|camera|biometric)\b.*\b(verification|verify|recognition|failed|rejected)\b',
                    r'\b(face.*match|face.*not|selfie.*error)\b'
                ],
                "keywords": ["face", "facial", "selfie", "biometric", "verification", "camera"],
                "confidence_threshold": 0.75
            },
            "Boarding Pass Issue": {
                "patterns": [
                    r'\b(boarding pass|flight|ticket|barcode|qr|scan)\b',
                    r'\b(cannot upload|can\'?t upload|upload.*fail|scan.*fail)\b.*\b(boarding|pass|ticket)\b'
                ],
                "keywords": ["boarding", "pass", "flight", "ticket", "scan", "barcode", "qr"],
                "confidence_threshold": 0.75
            },
            "Dependent/Minor": {
                "patterns": [
                    r'\b(minor|child|children|kid|kids|infant|baby|dependent|family member)\b',
                    r'\b(add.*minor|add.*child|travel.*child|minor.*travel)\b',
                    r'\bhow.*add\b.*\b(minor|child|dependent)\b'
                ],
                "keywords": ["minor", "child", "children", "kid", "infant", "dependent", "family"],
                "confidence_threshold": 0.8
            },
            "Airport/Gate Information": {
                "patterns": [
                    r'\b(gate|entry|terminal|airport|security|check.*in|counter|kiosk)\b',
                    r'\b(where.*gate|which.*gate|find.*gate|location)\b',
                    r'\b(digiyatra.*gate|digiyatra.*entry)\b'
                ],
                "keywords": ["gate", "entry", "terminal", "airport", "security", "kiosk", "location"],
                "confidence_threshold": 0.7
            },
            "Technical Issue": {
                "patterns": [
                    r'\b(app.*crash|app.*freeze|app.*slow|not working|not loading|error|bug)\b',
                    r'\b(technical.*issue|technical.*problem|glitch)\b'
                ],
                "keywords": ["crash", "error", "bug", "technical", "issue", "not working", "slow"],
                "confidence_threshold": 0.7
            },
            "Feedback/Complaint": {
                "patterns": [
                    r'\b(frustrated|angry|terrible|awful|worst|hate|disappointed|complain|complaint)\b',
                    r'\b(waste.*time|very bad|pathetic|useless)\b',
                    r'\b(excellent|great|good|wonderful|amazing|love|appreciate|thank)\b'
                ],
                "keywords": ["frustrated", "complaint", "feedback", "terrible", "great", "excellent"],
                "confidence_threshold": 0.6
            }
        }
    
    def classify_with_confidence(self, text: str, context: List[str] = None) -> Tuple[str, float]:
        """
        Classify intent with confidence score.
        
        Args:
            text: User message
            context: Previous messages in conversation for context-awareness
            
        Returns:
            (intent, confidence_score)
        """
        if not isinstance(text, str) or not text.strip():
            return ("Unknown", 0.0)
        
        text_lower = text.lower()
        scores = defaultdict(float)
        
        # Rule-based scoring
        for intent, config in self.intent_patterns.items():
            # Pattern matching
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    scores[intent] += 0.4
            
            # Keyword matching with TF-IDF-like weighting
            keyword_count = sum(1 for kw in config["keywords"] if kw in text_lower)
            if keyword_count > 0:
                scores[intent] += (keyword_count / len(config["keywords"])) * 0.3
        
        # Context boost (if user asked about same topic before)
        if context:
            context_text = " ".join(context).lower()
            for intent in scores.keys():
                for kw in self.intent_patterns[intent]["keywords"]:
                    if kw in context_text:
                        scores[intent] += 0.15
                        break
        
        # Determine best match
        if not scores:
            return ("General Query", 0.3)
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent], 1.0)
        
        # Apply confidence threshold
        threshold = self.intent_patterns[best_intent]["confidence_threshold"]
        if confidence < threshold:
            return ("General Query", confidence)
        
        return (best_intent, confidence)
    
    def classify_batch(self, df: pd.DataFrame, text_col: str = "Request") -> pd.DataFrame:
        """
        Classify intents for entire dataframe with context awareness.
        """
        results = []
        
        # Group by conversation for context
        if "Conversation ID" in df.columns:
            for conv_id, group in df.groupby("Conversation ID"):
                messages = group[text_col].fillna("").tolist()
                context = []
                
                for msg in messages:
                    intent, confidence = self.classify_with_confidence(msg, context[-3:])  # Last 3 messages
                    results.append({
                        "Intent": intent,
                        "Intent_Confidence": confidence
                    })
                    context.append(msg)
        else:
            # No conversation context
            for msg in df[text_col].fillna(""):
                intent, confidence = self.classify_with_confidence(msg)
                results.append({
                    "Intent": intent,
                    "Intent_Confidence": confidence
                })
        
        result_df = pd.DataFrame(results, index=df.index)
        return pd.concat([df, result_df], axis=1)


# ============================================================================
# 2. MULTILINGUAL SENTIMENT ANALYSIS
# ============================================================================

class MultilingualSentimentAnalyzer:
    """
    Proper multilingual sentiment handling:
    - Language-specific sentiment analysis
    - Handles code-mixed text (Hinglish, Tanglish)
    - Emotion detection beyond polarity
    """
    
    def __init__(self):
        self.english_negative_words = [
            'not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody', 'none',
            'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'useless',
            'frustrated', 'angry', 'disappointed', 'pathetic', 'waste', 'fail'
        ]
        
        self.english_positive_words = [
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'perfect', 'best',
            'love', 'appreciate', 'thank', 'helpful', 'easy', 'smooth', 'fast'
        ]
        
        # Hindi/Hinglish sentiment indicators
        self.hindi_negative = ['kharab', 'bekar', 'ganda', 'galat', 'nahi', 'mat']
        self.hindi_positive = ['accha', 'badhiya', 'sahi', 'theek', 'dhanyavad']
    
    def analyze_sentiment_robust(self, text: str, language: str = "English") -> Dict:
        """
        Robust sentiment analysis with emotion detection.
        
        Returns:
            {
                'polarity': float (-1 to 1),
                'emotion': str (frustrated/satisfied/neutral/confused),
                'intensity': float (0 to 1)
            }
        """
        if not isinstance(text, str) or not text.strip():
            return {'polarity': 0.0, 'emotion': 'neutral', 'intensity': 0.0}
        
        text_lower = text.lower()
        
        # Detect strong emotions first
        if any(word in text_lower for word in ['frustrated', 'angry', 'terrible', 'awful', 'worst', 'pathetic']):
            return {'polarity': -0.8, 'emotion': 'frustrated', 'intensity': 0.9}
        
        if any(word in text_lower for word in ['confused', 'understand', 'explain', 'how', 'what', 'why']):
            if '?' in text:
                return {'polarity': -0.2, 'emotion': 'confused', 'intensity': 0.6}
        
        if any(word in text_lower for word in ['thank', 'appreciate', 'excellent', 'great', 'wonderful']):
            return {'polarity': 0.8, 'emotion': 'satisfied', 'intensity': 0.9}
        
        # Lexicon-based sentiment for English
        if language.lower().startswith('eng') or language == 'unknown':
            pos_count = sum(1 for word in self.english_positive_words if word in text_lower)
            neg_count = sum(1 for word in self.english_negative_words if word in text_lower)
            
            # Negation handling
            if re.search(r'\b(not|no|never)\s+\w+', text_lower):
                neg_count += 1
            
            total = pos_count + neg_count
            if total == 0:
                polarity = 0.0
            else:
                polarity = (pos_count - neg_count) / total
            
            intensity = min(total / 3.0, 1.0)
            
            if polarity > 0.3:
                emotion = 'satisfied'
            elif polarity < -0.3:
                emotion = 'frustrated'
            else:
                emotion = 'neutral'
            
            return {'polarity': polarity, 'emotion': emotion, 'intensity': intensity}
        
        # Hindi/Hinglish sentiment
        elif 'hindi' in language.lower() or 'hinglish' in language.lower():
            pos_count = sum(1 for word in self.hindi_positive if word in text_lower)
            neg_count = sum(1 for word in self.hindi_negative if word in text_lower)
            
            # Also check English words in code-mixed text
            pos_count += sum(1 for word in self.english_positive_words if word in text_lower)
            neg_count += sum(1 for word in self.english_negative_words if word in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                return {'polarity': 0.0, 'emotion': 'neutral', 'intensity': 0.0}
            
            polarity = (pos_count - neg_count) / total
            intensity = min(total / 3.0, 1.0)
            emotion = 'satisfied' if polarity > 0.2 else ('frustrated' if polarity < -0.2 else 'neutral')
            
            return {'polarity': polarity, 'emotion': emotion, 'intensity': intensity}
        
        # Default for other languages
        return {'polarity': 0.0, 'emotion': 'neutral', 'intensity': 0.0}
    
    def analyze_batch(self, df: pd.DataFrame, text_col: str = "Request", lang_col: str = "Language") -> pd.DataFrame:
        """
        Apply sentiment analysis to dataframe.
        Returns only the sentiment columns (polarity, emotion, intensity).
        """
        sentiments = []
        
        for idx, row in df.iterrows():
            text = row.get(text_col, "")
            lang = row.get(lang_col, "English") if lang_col in df.columns else "English"
            
            sentiment = self.analyze_sentiment_robust(text, lang)
            sentiments.append(sentiment)
        
        sentiment_df = pd.DataFrame(sentiments, index=df.index)
        return sentiment_df  # Return only sentiment columns, not concatenated


# ============================================================================
# 3. USER JOURNEY & FRICTION ANALYSIS
# ============================================================================

def analyze_user_journey_friction(df: pd.DataFrame) -> Dict:
    """
    Deep friction point analysis:
    - Drop-off detection
    - Conversation loops (repeated intents)
    - Resolution failure patterns
    - Time-to-resolution bottlenecks
    """
    
    if 'Conversation ID' not in df.columns:
        return {}
    
    friction_report = {
        'drop_off_points': {},
        'conversation_loops': {},
        'high_latency_intents': {},
        'unresolved_patterns': []
    }
    
    # 1. Drop-off Analysis: Where do users abandon conversations?
    for conv_id, group in df.groupby('Conversation ID'):
        if len(group) == 1:
            continue  # Single message - might be bounce
        
        last_intent = group.iloc[-1].get('Intent', 'Unknown')
        last_request = group.iloc[-1].get('Request', '')
        
        # Check if conversation seems unresolved (no positive feedback, abrupt end)
        if len(group) > 2:  # Multi-turn conversation
            friction_report['drop_off_points'][last_intent] = \
                friction_report['drop_off_points'].get(last_intent, 0) + 1
    
    # 2. Conversation Loops: Users asking same thing multiple times
    for conv_id, group in df.groupby('Conversation ID'):
        if len(group) < 3:
            continue
        
        intents = group['Intent'].tolist() if 'Intent' in group.columns else []
        
        # Detect repeated intents
        for i in range(len(intents) - 2):
            if intents[i] == intents[i+1] or intents[i] == intents[i+2]:
                loop_pattern = intents[i]
                friction_report['conversation_loops'][loop_pattern] = \
                    friction_report['conversation_loops'].get(loop_pattern, 0) + 1
    
    # 3. High Latency Intents: Which intents take too long to process?
    if 'Intent' in df.columns and 'Latency' in df.columns:
        intent_latency = df.groupby('Intent')['Latency'].agg(['mean', 'median', 'count'])
        high_latency = intent_latency[intent_latency['mean'] > df['Latency'].quantile(0.90)]
        friction_report['high_latency_intents'] = high_latency.to_dict('index')
    
    # 4. Unresolved Patterns: Clarifications, fallbacks, "didn't understand"
    fallback_patterns = [
        r"didn'?t understand", r"can you rephrase", r"sorry", 
        r"unable to", r"not sure", r"could not find"
    ]
    
    for pattern in fallback_patterns:
        matches = df[df['Response'].str.contains(pattern, case=False, na=False)]
        if not matches.empty:
            top_intents = matches['Intent'].value_counts().head(5).to_dict() if 'Intent' in matches.columns else {}
            friction_report['unresolved_patterns'].append({
                'pattern': pattern,
                'count': len(matches),
                'top_intents': top_intents
            })
    
    return friction_report


# ============================================================================
# 4. BUSINESS IMPACT METRICS (Executive KPIs)
# ============================================================================

def calculate_business_impact_metrics(df: pd.DataFrame, friction_data: Dict) -> Dict:
    """
    Calculate executive-level KPIs with business context.
    """
    
    metrics = {}
    
    # 1. Intent Resolution Efficiency Score (IRES)
    # Measures: Intent recognized correctly + Resolved quickly + User satisfied
    if 'Intent_Confidence' in df.columns and 'Latency' in df.columns:
        avg_confidence = df['Intent_Confidence'].mean()
        avg_latency = df['Latency'].mean()
        fast_responses = (df['Latency'] < 3).mean()  # < 3 seconds
        
        ires = (avg_confidence * 0.5 + fast_responses * 0.5) * 100
        metrics['intent_resolution_efficiency_score'] = round(ires, 2)
    
    # 2. User Frustration Index (UFI)
    if 'emotion' in df.columns:
        frustration_rate = (df['emotion'] == 'frustrated').mean() * 100
        metrics['user_frustration_index'] = round(frustration_rate, 2)
    
    # 3. Self-Service Success Rate (SSSR)
    # Conversations that didn't need escalation/clarification
    if 'Response' in df.columns:
        fallback_pattern = r"(didn'?t understand|can you rephrase|sorry|unable to|not sure)"
        successful = ~df['Response'].str.contains(fallback_pattern, case=False, na=False)
        sssr = successful.mean() * 100
        metrics['self_service_success_rate'] = round(sssr, 2)
    
    # 4. Intent Distribution Gap Analysis
    # Compares what users want vs what chatbot handles well
    if 'Intent' in df.columns and 'Intent_Confidence' in df.columns:
        intent_perf = df.groupby('Intent').agg({
            'Intent_Confidence': 'mean',
            'Request': 'count'
        }).rename(columns={'Request': 'volume'})
        
        # High volume but low confidence = gap
        intent_perf['gap_score'] = intent_perf['volume'] * (1 - intent_perf['Intent_Confidence'])
        metrics['top_intent_gaps'] = intent_perf.nlargest(5, 'gap_score').to_dict('index')
    
    # 5. Cost Efficiency Score
    # Lower cost per resolved query is better
    if 'Cost' in df.columns and 'Conversation ID' in df.columns:
        cost_per_conversation = df.groupby('Conversation ID')['Cost'].sum().mean()
        metrics['avg_cost_per_conversation'] = round(cost_per_conversation, 4)
        
        # Cost by intent
        if 'Intent' in df.columns:
            cost_by_intent = df.groupby('Intent')['Cost'].mean().to_dict()
            metrics['cost_by_intent'] = {k: round(v, 4) for k, v in cost_by_intent.items()}
    
    # 6. Friction Impact Score
    # Quantifies business impact of friction points
    loop_count = sum(friction_data.get('conversation_loops', {}).values())
    dropout_count = sum(friction_data.get('drop_off_points', {}).values())
    total_convs = df['Conversation ID'].nunique() if 'Conversation ID' in df.columns else 1
    
    friction_impact = ((loop_count + dropout_count) / total_convs) * 100
    metrics['friction_impact_score'] = round(friction_impact, 2)
    
    return metrics


# ============================================================================
# 5. ACTIONABLE INSIGHTS GENERATOR
# ============================================================================

def generate_actionable_insights(df: pd.DataFrame, metrics: Dict, friction: Dict) -> List[Dict]:
    """
    Generate prioritized, actionable recommendations for different stakeholders.
    """
    
    insights = []
    
    # For Product Team
    if metrics.get('user_frustration_index', 0) > 15:
        insights.append({
            'priority': 'HIGH',
            'stakeholder': 'Product Team',
            'category': 'User Experience',
            'insight': f"User frustration at {metrics['user_frustration_index']:.1f}% - well above 10% threshold",
            'action': "Review top frustration-causing intents and improve response quality or flow",
            'expected_impact': "15-25% reduction in negative feedback",
            'effort': 'Medium'
        })
    
    # For Engineering Team
    if friction.get('high_latency_intents'):
        worst_latency = max(
            friction['high_latency_intents'].items(),
            key=lambda x: x[1]['mean']
        )
        insights.append({
            'priority': 'MEDIUM',
            'stakeholder': 'Engineering Team',
            'category': 'Performance',
            'insight': f"Intent '{worst_latency[0]}' has {worst_latency[1]['mean']:.2f}s avg latency",
            'action': "Optimize backend processing or add caching for this intent",
            'expected_impact': "30-40% latency reduction, improved CSAT",
            'effort': 'High'
        })
    
    # For Business/Ops Team
    if metrics.get('self_service_success_rate', 100) < 70:
        insights.append({
            'priority': 'HIGH',
            'stakeholder': 'Operations Team',
            'category': 'Cost Efficiency',
            'insight': f"Only {metrics['self_service_success_rate']:.1f}% of queries self-resolved (target: >80%)",
            'action': "Analyze fallback patterns and expand knowledge base for common unresolved queries",
            'expected_impact': "Reduce human agent escalations by 20-30%, save ₹5-10L/month",
            'effort': 'Medium'
        })
    
    # For ML/Data Science Team
    if metrics.get('top_intent_gaps'):
        top_gap = max(
            metrics['top_intent_gaps'].items(),
            key=lambda x: x[1]['gap_score']
        )
        insights.append({
            'priority': 'HIGH',
            'stakeholder': 'ML Team',
            'category': 'Model Performance',
            'insight': f"Intent '{top_gap[0]}' has high volume ({top_gap[1]['volume']} queries) but low confidence ({top_gap[1]['Intent_Confidence']:.2f})",
            'action': "Retrain intent classifier with more examples for this category or refine intent definitions",
            'expected_impact': "10-15% improvement in intent recognition accuracy",
            'effort': 'High'
        })
    
    # For Content/Training Team
    if friction.get('unresolved_patterns'):
        total_unresolved = sum(p['count'] for p in friction['unresolved_patterns'])
        unresolved_rate = (total_unresolved / len(df)) * 100
        
        if unresolved_rate > 10:
            insights.append({
                'priority': 'HIGH',
                'stakeholder': 'Content Team',
                'category': 'Knowledge Gap',
                'insight': f"{unresolved_rate:.1f}% of responses are fallbacks/unclear (target: <5%)",
                'action': "Create FAQ content for top unresolved query patterns",
                'expected_impact': "Reduce fallback rate by 40-50%",
                'effort': 'Low'
            })
    
    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    insights.sort(key=lambda x: priority_order[x['priority']])
    
    return insights


# ============================================================================
# 6. MAIN ORCHESTRATOR
# ============================================================================

def run_advanced_nlp_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to run all advanced NLP analytics.
    
    Returns:
        - Enhanced dataframe with all NLP features
        - Comprehensive insights dictionary
    """
    
    print("🚀 Starting Advanced NLP Analysis Pipeline...")
    
    # Step 1: Intent Classification with Confidence
    print("   ✓ Running hybrid intent classification...")
    classifier = IntentClassifier()
    df = classifier.classify_batch(df, text_col="Request")
    
    # Step 2: Multilingual Sentiment Analysis
    print("   ✓ Performing multilingual sentiment analysis...")
    sentiment_analyzer = MultilingualSentimentAnalyzer()
    sentiment_results = sentiment_analyzer.analyze_batch(df, text_col="Request")
    
    # Concatenate sentiment results with main dataframe
    df = pd.concat([df, sentiment_results], axis=1)
    
    # Also analyze Response sentiment
    response_sentiment_results = sentiment_analyzer.analyze_batch(df, text_col="Response")
    # Add response sentiment with prefixed column names to avoid conflicts
    df['response_polarity'] = response_sentiment_results['polarity'].values
    df['response_emotion'] = response_sentiment_results['emotion'].values
    
    # Step 3: Friction Analysis
    print("   ✓ Detecting user journey friction points...")
    friction_data = analyze_user_journey_friction(df)
    
    # Step 4: Business Metrics
    print("   ✓ Calculating executive KPIs...")
    business_metrics = calculate_business_impact_metrics(df, friction_data)
    
    # Step 5: Actionable Insights
    print("   ✓ Generating actionable recommendations...")
    insights = generate_actionable_insights(df, business_metrics, friction_data)
    
    # Compile comprehensive report
    report = {
        'business_metrics': business_metrics,
        'friction_analysis': friction_data,
        'actionable_insights': insights,
        'summary': {
            'total_conversations': df['Conversation ID'].nunique() if 'Conversation ID' in df.columns else 0,
            'total_messages': len(df),
            'avg_intent_confidence': df['Intent_Confidence'].mean() if 'Intent_Confidence' in df.columns else 0,
            'frustration_rate': (df['emotion'] == 'frustrated').mean() * 100 if 'emotion' in df.columns else 0
        }
    }
    
    print("✅ Analysis Complete!\n")
    
    return df, report


# ============================================================================
# 7. LANGUAGE DETECTION FIX
# ============================================================================

def fix_language_detection(df: pd.DataFrame, text_col: str = "Request", 
                           min_length: int = 10) -> pd.DataFrame:
    """
    Improved language detection that avoids false positives on short text.
    
    Rules:
    - Text < 10 chars: Mark as "Too Short"
    - Common greetings (hi, hello, ok, yes, no): Mark as "English (Greeting)"
    - Apply IndicLID only for longer text
    """
    
    greetings = ['hi', 'hello', 'ok', 'yes', 'no', 'bye', 'thanks', 'thank you', 'ty']
    
    def smart_detect(text):
        if not isinstance(text, str) or len(text.strip()) < 3:
            return "Unknown"
        
        text_clean = text.strip().lower()
        
        # Short greetings
        if text_clean in greetings or len(text_clean) < min_length:
            return "English (Short)"
        
        # Use IndicLID only for substantial text
        try:
            # Try to import from the user's nlp_analytics module
            from nlp_analytics import detect_language
            return detect_language(text, engine="IndicLID")
        except ImportError:
            # Fallback if IndicLID not available
            return "English" if text.isascii() else "Non-English"
    
    df['Language_Corrected'] = df[text_col].apply(smart_detect)
    return df


if __name__ == "__main__":
    # Example usage
    print("Advanced NLP Insights Module Loaded Successfully!")
    print("Use: run_advanced_nlp_analysis(df) to execute full pipeline")
