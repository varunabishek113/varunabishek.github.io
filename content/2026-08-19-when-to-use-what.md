Title: LangChain vs. LlamaIndex vs. Raw API Calls — When to Use What
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LLM, LangChain, LlamaIndex, API, Python, Architecture
Slug: langchain-vs-llamaindex-vs-raw-api
Status: draft

Three ways to build with LLMs. Each makes different tradeoffs, and picking the wrong one early costs you time when the project grows. This isn't a ranking — it's a map of when each tool actually fits.

## Raw API calls

Direct calls to OpenAI, Anthropic, Gemini, or any provider. No framework, just HTTP.

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in one paragraph."}
    ]
)
print(response.choices[0].message.content)
```

**When this fits:**
- You're doing one thing — a single prompt, a classification call, a rewrite task.
- You need full control over request structure and don't want a framework deciding how to format things.
- You're building a lightweight microservice where a framework would add weight for no reason.
- You're learning. Frameworks abstract the parts you should understand first.

**Where it breaks down:**
- You start needing RAG. Now you're writing a vector store wrapper, a chunking function, a retrieval loop, and token management yourself.
- You want conversation memory. Now you're manually tracking message history per session.
- You want to swap providers. Now you're rewriting every call.

Raw API calls don't scale to complex pipelines without you rebuilding the things frameworks already built.

## LangChain

LangChain is a general-purpose framework for chaining LLM calls with tools, memory, retrievers, and data loaders. It covers almost every use case — sometimes too broadly.

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-3.5-turbo")
vectorstore = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
result = chain.invoke({"query": "What does the contract say about late fees?"})
```

**When this fits:**
- You're building a multi-step pipeline: load → split → embed → retrieve → generate.
- You need agents — LLMs that pick tools, take actions, and loop based on output.
- You want memory management handled for you (buffer, window, summary).
- You're connecting to external tools: web search, APIs, databases, code interpreters.
- You want to swap LLMs or vector stores without rewriting the whole pipeline.

**Where it breaks down:**
- Retrieval quality is your primary concern. LangChain supports many retrieval strategies, but tuning them requires working around the abstractions rather than with them.
- The abstraction layers make debugging harder. When something returns wrong results, the error is often buried three layers deep.
- LangChain has gone through significant API churn (v0.1 → LCEL → v0.3). Code from tutorials written six months ago may not run without changes.

LangChain is best when you need breadth — many connected components — and are willing to trade some debuggability for speed of assembly.

## LlamaIndex

LlamaIndex (formerly GPT Index) is purpose-built for indexing and retrieval. Its primitives — nodes, indexes, query engines, retrievers — are designed around document ingestion and high-quality search. LangChain can do RAG; LlamaIndex was built specifically for it.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load and index
documents = SimpleDirectoryReader("docs/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What are the payment terms?")
print(response)
```

Four lines. No manual chunking, no retriever config, no chain setup.

**When this fits:**
- Document Q&A is the core feature, not a component inside a larger pipeline.
- You need fine-grained retrieval: hierarchical indexes, sub-question decomposition, hybrid search, reranking.
- You want to index structured data (SQL, JSON, dataframes) alongside unstructured docs.
- You care about retrieval evaluation — LlamaIndex has built-in evals for this.

**Where it breaks down:**
- You need agents with complex tool use. LlamaIndex has agents, but LangChain's are more mature and have more integrations.
- Your pipeline has many non-retrieval components (custom memory, complex branching logic, tool chains). LangChain's LCEL is better suited for composing those.
- You need a wider ecosystem of third-party integrations. LangChain has more.

LlamaIndex is better when retrieval quality is what you're optimizing for and the rest of the pipeline is simple.

## Side by side

| | Raw API | LangChain | LlamaIndex |
|---|---|---|---|
| Learning curve | Low | High | Medium |
| RAG support | DIY | Good | Excellent |
| Agent support | DIY | Excellent | Good |
| Memory management | DIY | Built-in | Basic |
| Retrieval tuning | DIY | Moderate | Deep |
| Debugging | Easy | Hard | Medium |
| Ecosystem / integrations | Provider-specific | Large | Growing |
| API stability | Stable | Churn | Moderate |

## The real decision flow

**Start with raw API calls if:** You're learning, prototyping a single prompt interaction, or building something where a framework would be overhead with no benefit.

**Reach for LangChain if:** Your pipeline connects multiple components — loaders, retrievers, tools, memory, agents — and you want the ecosystem to handle the wiring.

**Reach for LlamaIndex if:** Your core problem is document retrieval and Q&A, and you want the best retrieval quality with the least configuration overhead.

**Use both if:** It's more common than you'd think. LangChain for orchestration, LlamaIndex as the retrieval backend. They interoperate:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from langchain_community.retrievers import LlamaIndexRetriever

documents = SimpleDirectoryReader("docs/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Use LlamaIndex's query engine as a LangChain retriever
retriever = LlamaIndexRetriever(index=index.as_query_engine())
```

The frameworks aren't mutually exclusive. Pick the best tool for each layer of your stack.