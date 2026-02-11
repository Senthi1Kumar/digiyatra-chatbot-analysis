import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PATH SETUP (FROM YOUR CODE) ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "all-requests (1).csv.xlsx"

def run_metrics_report():
    # ---------------- LOAD DATA ----------------
    # Note: Using try-except to handle different formats (CSV/Excel)
    try:
        # Since the file extension might be .csv.xlsx, we try reading as Excel first per your code
        df = pd.read_excel(DATA_PATH)
    except Exception:
        # Fallback to CSV if the internal format is actually CSV
        df = pd.read_csv(DATA_PATH)

    # Normalize text columns
    TEXT_COLS = ["Request", "Response", "Clarification", "User Feedback"]
    for col in TEXT_COLS:
        df[col] = df[col].fillna("").astype(str).str.lower()

    # ---------------- CONFIG (REFINED REGEX) ----------------
    FALLBACK_PATTERNS = [
        r"didn.?t understand", r"can you rephrase", r"sorry", 
        r"unable to", r"not sure", r"didn.?t get", 
        r"could not find", r"no information", r"don't have enough context"
    ]
    fallback_regex = re.compile("|".join(FALLBACK_PATTERNS))

    # ---------------- METRIC 1: BOUNCE RATE ----------------
    conversation_sizes = df.groupby("Conversation ID").size()
    total_conversations = len(conversation_sizes)
    bounced_conversations = (conversation_sizes == 1).sum()
    bounce_rate = (bounced_conversations / total_conversations) * 100

    # ---------------- METRIC 2: FALLBACK RATE ----------------
    fallback_mask = df["Response"].str.contains(fallback_regex)
    fallback_messages = fallback_mask.sum()
    fallback_rate = (fallback_messages / len(df)) * 100

    # ---------------- METRIC 3: MISSED MESSAGES ----------------
    # We define missed as either a fallback pattern OR an empty response
    missed_mask = (df["Response"].str.strip() == "") | fallback_mask
    missed_messages = missed_mask.sum()
    missed_message_rate = (missed_messages / len(df)) * 100

    # ---------------- PRINT RESULTS ----------------
    print("\n" + "="*50)
    print(f"{'CONVERSATION METRICS REPORT':^50}")
    print("="*50)
    print(f"{'Total Conversations':<30} | {total_conversations:,}")
    print(f"{'Total Messages':<30} | {len(df):,}")
    print(f"{'Bounce Rate (%)':<30} | {bounce_rate:>14.2f}%")
    print(f"{'Missed Message Rate (%)':<30} | {missed_message_rate:>14.2f}%")
    print(f"{'Fallback Rate (%)':<30} | {fallback_rate:>14.2f}%")
    print("-" * 50)
    print(f"{'NLP Success Rate (%)':<30} | {100 - missed_message_rate:>14.2f}%")
    print("="*50)

    # ---------------- VISUALIZATION ----------------
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Engagement Pie Chart
    ax1.pie([bounced_conversations, total_conversations - bounced_conversations], 
            labels=['Bounced', 'Engaged'], autopct='%1.1f%%', 
            startangle=140, colors=['#ff9999', '#66b3ff'], explode=(0.07, 0), shadow=True)
    ax1.set_title('User Engagement (Bounce Rate)', fontsize=14, fontweight='bold')

    # Response Quality Pie Chart
    ax2.pie([len(df) - missed_messages, missed_messages], 
            labels=['Success', 'Missed/Fallback'], autopct='%1.1f%%', 
            startangle=140, colors=['#99ff99', '#ffcc99'], explode=(0, 0.15), shadow=True)
    ax2.set_title('NLP Response Quality', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('nlp_metrics_dashboard.png', dpi=300)
    print(f"\n[System] Dashboard saved to: nlp_metrics_dashboard.png")

    # Save summary data
    metrics_summary = pd.DataFrame({
        "Metric": ["Total Conversations", "Total Messages", "Bounce Rate (%)", "Missed Rate (%)", "Fallback Rate (%)"],
        "Value": [total_conversations, len(df), bounce_rate, missed_message_rate, fallback_rate]
    })
    metrics_summary.to_csv("conversation_metrics_summary.csv", index=False)

if __name__ == "__main__":
    run_metrics_report()


