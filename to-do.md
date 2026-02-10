# NLP Analytics

## 1. What are the customer's issues w/ the DigiYatra platform

- Analyze the user's requests
  - How:

## 2. Detect multiple languages and do analyze it as well

<!-- - Use open multilingual models from sarvam, AI4Bharat, HF -->
  - Identify the frustrated user requests and their intents and look the time and date and feedback


## Outcome metrics

- Show no. of Out-of-domain responses and requests

- customer satisfication score (CSAT)

- Chat duration: w.r.t conversation id, filter the message ids then get timestamps and calculate avg. chat duration/per user

- `Clarification` col. if any user clicks the options so we can calculate engagement rate

### Conversation metrics

- analyze feedback and responses w.r.t requests

- bounce rate (This is the percentage of users who start a chat but leave quickly without any real interaction. A high bounce rate often points to a problem with the bot’s opening, unclear options, or an inability to engage the user right away.) - calculation w.r.t conv. id

- Missed Messages (These are messages the chatbot didn’t understand or couldn’t respond to. They often pop up due to regional slang or idiomatic expressions it hasn’t learned yet.) -- based on the requests and responses

- Fallback Rate (This is the percentage of user messages your chatbot flat-out doesn’t understand or can’t respond to appropriately. A high fallback rate, often calculated as ((Total Messages - Fallback Messages) / Total Messages) * 100, signals that its Natural Language Processing (NLP) needs work or its training data needs an update. NLP is the technology that allows computers to understand human language.)


### AI Perf. metrics

- AI Response Feedback (This captures direct user reactions (like thumbs up/down, or phrases like “that’s not what I meant”) to the AI’s answers. This data is gold for judging the accuracy and user satisfaction with generative models, and it helps fine-tune responses.)
- Intent Recognition Accuracy


- Drop-off Points in Conversation Flows:
What it tells you: This pinpoints specific stages or messages in a conversation where users are most likely to abandon the chat. It flags points of friction, confusion, unclear instructions, or where the bot simply fails to meet expectations.
  - Action: Dig into these points. Can you simplify the flow? Clarify the language? Provide better options?


Ref:
- https://quickchat.ai/post/chatbot-analytics
