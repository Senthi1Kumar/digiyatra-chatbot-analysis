import pandas as pd
import re
import json
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PATH SETUP (FROM YOUR CODE) ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
# Ensure this filename matches your actual file on disk
DATA_PATH = BASE_DIR / "all-requests (1).csv.xlsx" 

def run_advanced_metrics():
    # ---------------- 1. LOAD DATA ----------------
    try:
        # Checking if it's an Excel file or CSV (handles the .csv.xlsx naming quirk)
        if DATA_PATH.suffix == '.xlsx':
            df = pd.read_excel(DATA_PATH)
        else:
            df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Error loading file at {DATA_PATH}: {e}")
        return

    # Normalize columns to lowercase for consistent matching
    TEXT_COLS = ["Request", "Response", "User Feedback"]
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str)

    # ---------------- 2. AI RESPONSE FEEDBACK ----------------
    def extract_rating(val):
        if not val or val.lower() == "nan":
            return "No Feedback"
        try:
            # Clean and parse JSON from the 'User Feedback' column
            clean_json = val.replace("'", '"')
            data = json.loads(clean_json)
            return data.get('rating', 'Unknown').capitalize()
        except:
            # Fallback for simple string feedback
            v = val.lower()
            if 'good' in v: return 'Good'
            if 'bad' in v: return 'Bad'
            return "Other"

    df['Rating'] = df['User Feedback'].apply(extract_rating)
    feedback_dist = df['Rating'].value_counts()

    # ---------------- 3. INTENT RECOGNITION ACCURACY ----------------
    # Logic: Accuracy is calculated as the % of messages that did NOT trigger a fallback
    FALLBACK_PATTERNS = [
        r"didn.?t understand", r"can you rephrase", r"sorry", 
        r"unable to", r"not sure", r"didn.?t get", 
        r"could not find", r"no information", r"don't have enough context"
    ]
    fallback_regex = re.compile("|".join(FALLBACK_PATTERNS), re.IGNORECASE)
    
    # Identify failures
    df['is_fallback'] = df['Response'].str.contains(fallback_regex)
    total_msgs = len(df)
    successful_intents = total_msgs - df['is_fallback'].sum()
    intent_accuracy = (successful_intents / total_msgs) * 100

    # ---------------- 4. DROP-OFF POINTS ----------------
    # Sort by time to find the last message of every conversation
    # Note: Ensure your Timestamp column is in a sortable format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Get the last row of every unique Conversation ID
    drop_off_data = df.sort_values(['Conversation ID', 'Timestamp']).groupby('Conversation ID').tail(1)
    
    # Top 5 requests that caused the user to stop talking
    top_drop_offs = drop_off_data['Request'].value_counts().head(5)

    # ---------------- 5. PRINT NEAT SUMMARY ----------------
    print("\n" + "═"*60)
    print(f"{'ADVANCED AI PERFORMANCE INSIGHTS':^60}")
    print("═"*60)
    
    print(f"\n[1] INTENT RECOGNITION ACCURACY")
    print(f"Score: {intent_accuracy:.2f}%")
    print(f"Definition: % of queries where the AI successfully mapped a valid response.")

    print(f"\n[2] USER FEEDBACK DISTRIBUTION")
    for label, count in feedback_dist.items():
        if label != "No Feedback":
            print(f"- {label}: {count} responses")
    
    print(f"\n[3] TOP CONVERSATION DROP-OFF POINTS")
    print("Requests that most frequently ended the session:")
    for i, (req, count) in enumerate(top_drop_offs.items(), 1):
        print(f" {i}. \"{req[:60]}...\" ({count} drop-offs)")
    
    print("═"*60)

    # ---------------- 6. VISUALIZATION ----------------
    plt.figure(figsize=(10, 6))
    top_drop_offs.plot(kind='barh', color='salmon')
    plt.title('Top 5 Conversation Drop-off Points', fontweight='bold')
    plt.xlabel('Number of Users who Abandoned')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('ai_dropoff_analysis.png')
    print("\n[System] Drop-off analysis chart saved as 'ai_dropoff_analysis.png'")

if __name__ == "__main__":
    run_advanced_metrics()