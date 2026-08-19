Title: Intro to Vector Stores (Chroma, FAISS, Pinecone) with LangChain
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LangChain, Vector Store, Chroma, FAISS, Pinecone, Embeddings
Slug: intro-to-vector-stores-langchain

A vector store is a database that finds things by meaning, not exact match. You convert text into a numerical vector (an embedding), store it, and later search for semantically similar vectors. That's the retrieval half of RAG. Which store you pick depends on scale, whether you want persistence, and whether you're staying local or going cloud.

## How similarity search works

When you embed text, you map it into a high-dimensional space where similar meanings land close to each other. "How do I reset my password?" and "Forgot password steps" produce vectors that are geometrically near each other, even though they share no words.

At query time:

1. Embed the user's question into the same space.
2. Find the `k` stored vectors closest to it (cosine similarity or Euclidean distance).
3. Return the corresponding text chunks.

That's what every vector store does. The differences are in speed, persistence, and infrastructure.

## Chroma — local, zero-config

Chroma runs in-process. No server, no Docker, just install and go. The right choice for development, prototypes, and small production apps where you control the machine.

```bash
pip install chromadb
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"   # omit for in-memory only
)

# Load an existing store
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Search
results = vectorstore.similarity_search("What is the refund policy?", k=4)
```

`persist_directory` writes the index to disk. Without it, everything lives in memory and resets when the process ends. Chroma handles up to a few hundred thousand documents comfortably on a standard laptop.

**Metadata filtering in Chroma:**

```python
results = vectorstore.similarity_search(
    query="security protocols",
    k=3,
    filter={"source": "handbook.pdf"}
)
```

Filter keys must match what you set in `.metadata` at index time.

## FAISS — fast, in-memory, CPU-native

FAISS (Facebook AI Similarity Search) is a C++ library optimized for large-scale nearest-neighbor search. It's faster than Chroma for big indexes and doesn't need a server, but it's in-memory — persistence requires manually serializing and deserializing the index.

```bash
pip install faiss-cpu   # or faiss-gpu if you have CUDA
```

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save to disk
vectorstore.save_local("faiss_index")

# Load from disk
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Search
results = vectorstore.similarity_search("explain the leave policy", k=5)
```

FAISS doesn't have built-in metadata filtering the way Chroma does. You filter post-retrieval by inspecting `doc.metadata` yourself. For production use cases needing filter logic, Chroma or Pinecone are easier.

FAISS shines when you have millions of vectors and want fast retrieval on your own hardware — local LLM setups, air-gapped environments, research workloads.

## Pinecone — managed, cloud-native

Pinecone is a hosted vector database. You push vectors to their servers; they handle indexing, scaling, and replication. No infra to manage, but it's a paid service and your data leaves your machine.

```bash
pip install pinecone-client
```

```python
from langchain_pinecone import PineconeVectorStore
import pinecone

pinecone.init(
    api_key="your-api-key",
    environment="us-east-1-aws"
)

# Create index (one-time setup, not per run)
pinecone.create_index(
    name="my-index",
    dimension=1536,       # OpenAI ada-002 output dimension
    metric="cosine"
)

# Upsert documents
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# Connect to existing index
vectorstore = PineconeVectorStore(
    index_name="my-index",
    embedding=embeddings
)

# Search with metadata filter
results = vectorstore.similarity_search(
    "quarterly revenue targets",
    k=4,
    filter={"department": "finance"}
)
```

Pinecone's main advantage is that it scales horizontally without any work on your side. Billions of vectors, concurrent queries, namespace isolation — all handled. The tradeoff is cost and the fact that your data lives in their infrastructure.

## Comparing the three

| | Chroma | FAISS | Pinecone |
|---|---|---|---|
| Hosting | Local | Local | Cloud |
| Persistence | Built-in | Manual (save/load) | Always on |
| Metadata filtering | Yes | Post-retrieval only | Yes |
| Scale | ~100k docs | Millions | Billions |
| Cost | Free | Free | Paid |
| Best for | Dev / small prod | Large local indexes | Production at scale |

## Which retriever mode to use

All three expose `.as_retriever()`, which is what you pass into LangChain chains:

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",          # or "similarity" (default)
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

**`similarity`** — Returns the top-k most similar chunks. Simple, fast, but can return redundant results if your docs have repeated content.

**`mmr` (Maximal Marginal Relevance)** — Balances similarity with diversity. Fetches a larger candidate set (`fetch_k`), then picks `k` results that are relevant but not too similar to each other. Better for documents with repeated sections or when you want varied context.

## Swapping stores without rewriting your chain

LangChain's interfaces are consistent across stores. Switch from Chroma to Pinecone by changing one line:

```python
# Before
retriever = chroma_store.as_retriever(search_kwargs={"k": 4})

# After
retriever = pinecone_store.as_retriever(search_kwargs={"k": 4})
```

The chain, prompts, and LLM stay the same. This makes it easy to start local with Chroma and migrate to Pinecone when you need to scale.