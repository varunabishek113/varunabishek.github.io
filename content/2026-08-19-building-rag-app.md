Title: Building Your First RAG (Retrieval-Augmented Generation) App
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LLM, RAG, LangChain, Python, Vector Store
Slug: building-your-first-rag-app
Status: draft

LLMs are trained on static data. Ask one about your internal docs, last week's meeting notes, or a product spec — it has no idea. RAG solves this by fetching relevant content at query time and injecting it into the prompt. The model still does the reasoning; you just give it the right context first.

## How RAG works

The pipeline has two phases:

**Indexing (offline)** — Load your documents, split them into chunks, convert each chunk into a vector embedding, and store those vectors in a vector database.

**Retrieval + generation (online)** — When a user asks a question, convert the question into an embedding, find the closest-matching chunks in the vector DB, shove those chunks into the prompt alongside the question, and let the LLM answer.

That's it. The LLM never "learns" your docs — it reads them fresh on every query.

## Dependencies

```bash
pip install langchain langchain-community langchain-openai \
    chromadb tiktoken
```

## Step 1 — Load your documents

LangChain ships with loaders for almost every format. For a plain text file:

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("docs/handbook.txt")
docs = loader.load()
```

For a folder of files:

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("docs/", glob="**/*.txt")
docs = loader.load()
```

Each loaded item is a `Document` with `.page_content` (the text) and `.metadata` (source path, page number, etc.).

## Step 2 — Split into chunks

Embeddings work best on short, focused passages. Long documents get split up:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

`chunk_overlap` keeps a window of context at chunk boundaries. Tune `chunk_size` based on what you're embedding — 500 characters is a reasonable starting point for dense technical text.

## Step 3 — Embed and store

Convert chunks to vectors and store them in Chroma (a local vector DB):

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

`persist_directory` saves the index to disk so you don't re-embed on every run. Drop it if you want an in-memory store that resets each session.

## Step 4 — Build the retrieval chain

LangChain's `RetrievalQA` chain wraps the retriever and the LLM together:

```python
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

`k=4` means retrieve the 4 most similar chunks per query. `chain_type="stuff"` concatenates them into the prompt. For larger doc sets, `map_reduce` or `refine` handle overflow better.

## Step 5 — Query it

```python
result = qa_chain.invoke({"query": "What is the refund policy?"})

print(result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print(doc.metadata["source"])
```

The `source_documents` field shows exactly which chunks the model drew from — useful for debugging and for building citation UIs.

## Full pipeline

```python
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

os.environ["OPENAI_API_KEY"] = "your-key-here"

# Load
loader = DirectoryLoader("docs/", glob="**/*.txt")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed + store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain.invoke({"query": "Summarize the onboarding process"})
print(result["result"])
```

## Common failure modes

**Bad retrieval** — The right answer exists in your docs but the LLM gives a wrong or vague response. Usually means your chunks are too large or your embeddings aren't granular enough. Shrink `chunk_size` and re-index.

**Hallucination despite good retrieval** — The LLM drifts from the retrieved context. Add a system prompt that says: "Answer only using the provided context. If the context doesn't contain the answer, say so."

**Slow indexing** — Embedding thousands of chunks hits API rate limits fast. Batch your calls, add retry logic, or switch to a local embedding model like `sentence-transformers`.

## Going further

Once the basic pipeline works, two things make the biggest difference in real apps:

**Metadata filtering** — Tag chunks with source, date, or category at index time, then filter by those fields at retrieval time. Keeps queries scoped and cuts noise.

**Hybrid search** — Combine vector similarity with keyword search (BM25). Pure vector search misses exact-match queries; hybrid search handles both.