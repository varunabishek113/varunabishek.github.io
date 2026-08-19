Title: Managing Conversation History Without Blowing Your Token Budget
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LLM, LangChain, Python, Memory, Token Management, Chatbot
Slug: managing-conversation-history-tokens
Status: draft

LLMs have no memory between calls. Every time you hit the API, the model starts fresh. To build a chatbot that remembers what was said three messages ago, you have to manually pass that history back in on every request. The problem: each message adds tokens, and token budgets are finite. A naive implementation that keeps the full conversation will eventually hit the context limit — or cost you more than it should.

## The naive approach (and why it breaks)

```python
history = []

def chat(user_message):
    history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=history
    )
    reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply
```

This works until it doesn't. After enough turns, `history` overflows the context window and the API throws an error. Even before that, you're paying for tokens from 40 messages ago that have nothing to do with the current question.

LangChain's memory modules solve this by managing what gets kept.

## ConversationBufferMemory — full history, simple

The most basic option: keep every message, no trimming.

```python
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory()

chain = ConversationChain(llm=llm, memory=memory, verbose=False)

chain.predict(input="My name is Varun.")
chain.predict(input="What's my name?")   # → "Your name is Varun."
```

Fine for short conversations. Breaks on long ones for exactly the same reason the naive approach does.

## ConversationBufferWindowMemory — sliding window

Keep only the last `k` exchanges. Everything older gets dropped:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # last 5 human+AI turns
chain = ConversationChain(llm=llm, memory=memory)
```

Simple and predictable. The tradeoff is that context beyond `k` turns ago is gone entirely. A user who mentioned their project name 10 messages back and references it now will confuse the model.

Good for: customer support bots, short-session apps, anywhere long-term context doesn't matter.

## ConversationSummaryMemory — compress old turns

Summarize older messages instead of dropping them. The LLM itself writes the summary as history grows:

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
chain = ConversationChain(llm=llm, memory=memory)
```

Recent turns are kept verbatim; older ones get collapsed into a running summary. The model always has the gist of what happened earlier, just not the exact words.

The catch: summarization costs extra tokens and adds latency on each turn. For long conversations, this is usually worth it. For short ones, it's overhead you don't need.

## ConversationSummaryBufferMemory — the practical middle ground

Keeps recent messages verbatim and summarizes anything older than `max_token_limit` tokens:

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000    # keep last ~1000 tokens verbatim; summarize the rest
)
chain = ConversationChain(llm=llm, memory=memory)
```

This is the most practical default for production chatbots. Recent context stays sharp; older context survives in compressed form. The token budget stays bounded without losing the thread of a long session.

## ConversationTokenBufferMemory — token-precise trimming

Trims by exact token count rather than by message count:

```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000   # never exceed 2000 tokens in memory
)
```

More precise than window-based trimming since different messages have very different lengths. A user who sends one-word replies needs a bigger window than one who writes paragraphs. Token-based trimming handles both correctly.

## Persisting memory across sessions

All the memory classes above are in-memory. Close the process and the history is gone. For a real chatbot, you need to save and reload it.

**Manual serialization:**

```python
import json

# Save
history = memory.chat_memory.messages
serialized = [{"role": m.type, "content": m.content} for m in history]
json.dump(serialized, open("session_123.json", "w"))

# Load
from langchain.schema import HumanMessage, AIMessage
data = json.load(open("session_123.json"))
messages = [
    HumanMessage(content=m["content"]) if m["role"] == "human"
    else AIMessage(content=m["content"])
    for m in data
]
memory.chat_memory.messages = messages
```

**With a database (Redis, MongoDB, Postgres):**

LangChain has `RedisChatMessageHistory`, `MongoDBChatMessageHistory`, and others that write directly to an external store. They slot into the same memory interface:

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferMemory

message_history = RedisChatMessageHistory(
    session_id="user_456",
    url="redis://localhost:6379"
)

memory = ConversationBufferMemory(
    chat_memory=message_history,
    return_messages=True
)
```

Each session ID gets its own history. Users pick up where they left off, even after server restarts.

## Token counting before you send

Sometimes you want to check token usage programmatically before committing to an API call:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Count tokens in a message list
from langchain_community.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.predict(input="How many tokens did that use?")
    print(f"Total tokens: {cb.total_tokens}")
    print(f"Prompt tokens: {cb.prompt_tokens}")
    print(f"Completion tokens: {cb.completion_tokens}")
    print(f"Cost: ${cb.total_cost:.4f}")
```

`get_openai_callback` wraps any LangChain call and reports usage. Useful for auditing costs in development and for setting hard limits in production.

## The summary

| Memory type | What it keeps | Best for |
|---|---|---|
| `BufferMemory` | Everything | Short conversations |
| `BufferWindowMemory` | Last k turns | Short-session apps |
| `SummaryMemory` | Running summary | Long sessions, cost-sensitive |
| `SummaryBufferMemory` | Recent verbatim + summary | Most production chatbots |
| `TokenBufferMemory` | Bounded by token count | Variable-length message flows |

Start with `ConversationSummaryBufferMemory` at around 1000–1500 tokens and tune from there.