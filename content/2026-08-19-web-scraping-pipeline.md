Title: Web Scraping + Summarization Pipeline with LangChain
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LLM, LangChain, Python, Web Scraping, NLP
Slug: web-scraping-summarization-langchain
Status: draft

You don't always need a fancy dataset. Sometimes the data you need is already live on the web — you just have to go get it and make sense of it. That's exactly what this pipeline does: scrape a URL, chunk the content, and summarize it using an LLM, all wired together with LangChain.

## What you're building

The pipeline has three stages:

1. Fetch raw HTML from a URL
2. Parse and clean it into readable text
3. Pass that text through an LLM with a summarization prompt

LangChain handles the glue — loaders, text splitters, chains — so you spend less time on plumbing and more time on what actually matters.

## Dependencies

```bash
pip install langchain langchain-community langchain-openai \
    beautifulsoup4 requests tiktoken
```

You'll also need an OpenAI API key (or swap in any LLM provider LangChain supports).

## Step 1 — Scrape the page

LangChain's `WebBaseLoader` does this with one line:

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
```

Under the hood it fetches the page with `requests`, strips tags using BeautifulSoup, and returns a list of `Document` objects. Each document has `.page_content` (the text) and `.metadata` (source URL, title, etc.).

If you're scraping a JavaScript-heavy site, `WebBaseLoader` won't work — it only processes static HTML. Use `SeleniumURLLoader` or `PlaywrightURLLoader` instead:

```python
from langchain_community.document_loaders import SeleniumURLLoader

loader = SeleniumURLLoader(urls=["https://js-heavy-site.com"])
docs = loader.load()
```

## Step 2 — Split the text

LLMs have context limits. A long article will overflow the context window if you send it all at once. `RecursiveCharacterTextSplitter` breaks the content into chunks that fit:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
```

`chunk_overlap` keeps a little context between adjacent chunks so the LLM doesn't lose thread between them. Tune `chunk_size` based on your model's token limit — 2000 characters is a safe starting point for most GPT-3.5/4 setups.

## Step 3 — Summarize with a chain

LangChain has built-in summarization chains. `load_summarize_chain` wires together the LLM and a prompt strategy:

```python
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(chunks)

print(summary)
```

**`stuff`** — Concatenates all chunks and sends them in one prompt. Works for short documents. Breaks on anything long.

**`map_reduce`** — Summarizes each chunk separately (map), then summarizes the summaries (reduce). Handles long content well, costs more tokens.

**`refine`** — Starts with the first chunk's summary, then iteratively refines it with each next chunk. Produces more coherent output than `map_reduce` but is slower.

For most web articles, `map_reduce` is the right default.

## Full pipeline

```python
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

os.environ["OPENAI_API_KEY"] = "your-key-here"

def summarize_url(url: str) -> str:
    # Load
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Summarize
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(chunks)

if __name__ == "__main__":
    url = "https://en.wikipedia.org/wiki/Large_language_model"
    print(summarize_url(url))
```

## Custom prompt

The default prompt is generic. You can replace it with something that fits your use case:

```python
from langchain.prompts import PromptTemplate

map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following content in 3-5 bullet points.
Focus on key facts, arguments, and conclusions.

Content:
{text}

Summary:
"""
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You have summaries from sections of an article.
Write a single, coherent paragraph summarizing all of them.

Summaries:
{text}

Final Summary:
"""
)

chain = load_summarize_chain(
    llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=combine_prompt
)
```

This gives you control over tone, format, and what the model pays attention to.

## Scaling it up

If you're summarizing many URLs, a few things to watch:

- **Rate limits** — Add `time.sleep()` between requests, or use async loaders (`AsyncHtmlLoader`).
- **Token costs** — `map_reduce` on a 10,000-word article can rack up tokens fast. Cache results if you're hitting the same URLs repeatedly.
- **Paywalled / bot-protected sites** — `WebBaseLoader` will get blocked. You need session cookies, headers spoofing, or a scraping API like Browserless or ScrapingBee.
- **Structured extraction** — If you want specific fields (author, date, key claims) rather than a freeform summary, swap the summarization chain for an extraction chain or use `PydanticOutputParser` to force structured JSON output.

## Swapping the LLM

LangChain's abstraction makes it easy to drop in a different model. For a local setup:

```python
from langchain_community.llms import Ollama

llm = Ollama(model="llama3")
```

The rest of the chain stays the same. This matters if you're building something where data can't leave your machine — which is exactly the kind of setup that makes sense for projects handling sensitive internal docs.