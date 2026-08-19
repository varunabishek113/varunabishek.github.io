Title: Document Loaders — Pulling in PDFs, Websites, CSVs, and More
Date: 2026-08-19
Category: GenAI
Tags: GenAI, LangChain, Python, Document Loaders, PDF, CSV, Web Scraping
Slug: document-loaders-langchain
Status: draft

Before any LangChain pipeline does anything useful, it needs data. Document loaders are how that data gets in — they fetch content from a source, parse it, and return a list of `Document` objects with `.page_content` and `.metadata`. The rest of the pipeline (splitting, embedding, retrieval) doesn't care where the content came from. That's the point.

## The Document object

Every loader returns the same structure:

```python
from langchain.schema import Document

doc = Document(
    page_content="This is the text content",
    metadata={"source": "handbook.pdf", "page": 3}
)
```

`.page_content` is what gets embedded and fed to the LLM. `.metadata` is free-form — source file, URL, page number, timestamps — whatever you want to carry through for filtering or citation.

## PDFs

The most common use case. Two loaders worth knowing:

```python
# PyPDF — lightweight, good for most PDFs
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("report.pdf")
docs = loader.load()  # One Document per page
```

```python
# PDFPlumber — better for tables and layout-heavy docs
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("report.pdf")
docs = loader.load()
```

`PyPDFLoader` is faster. `PDFPlumberLoader` preserves layout better and handles tables more reliably. For scanned PDFs (image-based, no text layer), neither works — you need OCR. `UnstructuredPDFLoader` with `mode="elements"` can handle these via Tesseract.

## Websites

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.python.org/3/library/os.html")
docs = loader.load()
```

This fetches static HTML and strips the tags. Works fine for documentation sites, Wikipedia, blogs. For JavaScript-rendered pages (React/Next.js SPAs), the content won't be there — use `SeleniumURLLoader` or `PlaywrightURLLoader` instead, which spin up a real browser.

Loading multiple URLs at once:

```python
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
]
loader = WebBaseLoader(urls)
docs = loader.load()
```

## CSVs

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="employees.csv",
    source_column="employee_id"   # used as the metadata source field
)
docs = loader.load()
```

Each row becomes its own `Document`. The `source_column` parameter sets which column gets used as the source identifier in metadata — useful when you want to trace an answer back to a specific record.

For large CSVs with many columns, consider specifying which columns matter:

```python
loader = CSVLoader(
    file_path="data.csv",
    csv_args={"fieldnames": ["name", "description", "date"]}
)
```

## JSON and JSONL

```python
from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="data.json",
    jq_schema=".messages[].content",  # jq syntax to extract the right field
    text_content=True
)
docs = loader.load()
```

`JSONLoader` uses `jq` syntax to navigate nested JSON. `jq_schema` tells it exactly which field to pull as the page content. For JSONL (one JSON object per line), set `json_lines=True`.

## Directories (bulk loading)

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="docs/",
    glob="**/*.pdf",          # glob pattern for file types
    loader_cls=PyPDFLoader,   # which loader to use per file
    show_progress=True
)
docs = loader.load()
```

`DirectoryLoader` walks a folder and applies the specified loader to each matching file. Mix it with different `glob` patterns and `loader_cls` values if you have a folder with mixed file types.

## Notion, Google Drive, and databases

LangChain has loaders for these too, though they need extra auth setup:

```python
# Notion
from langchain_community.document_loaders import NotionDBLoader

loader = NotionDBLoader(
    integration_token="your-token",
    database_id="your-db-id",
    request_timeout_sec=30
)

# Google Drive
from langchain_community.document_loaders import GoogleDriveLoader

loader = GoogleDriveLoader(
    folder_id="your-folder-id",
    recursive=True
)
```

SQL databases have their own path — `SQLDatabaseLoader` runs a query and turns rows into documents. Useful for pipelines that need to answer questions against live relational data.

## Custom loaders

If nothing built-in fits, subclass `BaseLoader`:

```python
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

class MyAPILoader(BaseLoader):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def load(self) -> list[Document]:
        import requests
        response = requests.get(self.endpoint).json()
        return [
            Document(
                page_content=item["text"],
                metadata={"id": item["id"], "source": self.endpoint}
            )
            for item in response["items"]
        ]
```

The interface is just one method: `load()` returning a list of `Document` objects. That's all the rest of the pipeline needs.

## What to watch for

**Encoding issues** — PDFs and older text files sometimes come with encoding problems. `PyPDFLoader` occasionally returns garbled text from scanned or password-protected files. Check `doc.page_content` before embedding — garbage in means garbage retrieved.

**Large files** — Loading a 500-page PDF returns 500 documents at once. If you're loading many large files, stream them in batches rather than calling `.load()` on everything at once.

**Metadata hygiene** — The metadata your loaders attach is what you'll use later for filtering. Set it deliberately. A chunk with `metadata={"source": "file.pdf"}` is much harder to filter than one with `metadata={"source": "file.pdf", "section": "terms", "date": "2026-01"}`.