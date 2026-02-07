# RAG Workspace (Streamlit)

Interactive Retrieval-Augmented Generation app for indexing local files, URLs, and pasted text, then chatting with source-aware answers.

## What Improved

- Refactored architecture in `main.py` into clear layers: state, ingestion, indexing, retrieval, generation, and UI rendering.
- Faster response path: one generation call per user query (removed extra retrieval-query LLM round trip).
- Better ingestion UX:
  - Tabbed intake for files, URLs, and pasted text.
  - URL parsing and validation.
  - Duplicate source detection using content/URL fingerprints.
  - Batch processing progress and clear success/skip/error summaries.
- Source management:
  - Indexed source table with metadata.
  - Remove individual sources and automatically rebuild index.
  - Clear chat and clear knowledge base controls.
- Better reliability:
  - Stronger metadata normalization.
  - Cached embeddings and cached web snippet retrieval.
  - Session timeout handling with system events.

## Features

- File ingestion: `pdf`, `txt`, `md`, `markdown`, `csv`, `xlsx`, `xls`
- URL ingestion with content extraction
- Pasted-text ingestion for ad hoc notes
- FAISS vector indexing with OpenAI embeddings
- Retrieval + optional web fallback for temporal questions
- Source-aware chat responses
- Response latency shown per answer
- URL safety controls to block internal/private network targets
- Ingestion size and rate safety limits
- Optional application password gate for shared deployments

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run main.py
```

3. In the sidebar, enter your OpenAI API key.

## Security Setup

- Optional app-level password protection:

```bash
set RAG_APP_ACCESS_TOKEN=your-strong-secret
```

- Optional server-side OpenAI key fallback:

```bash
set OPENAI_API_KEY=sk-...
```

If `RAG_APP_ACCESS_TOKEN` is set, users must sign in before accessing the app.

## Security and Tests

Install developer tools:

```bash
pip install -r requirements-dev.txt
```

Run checks:

```bash
pytest
bandit -q -r main.py
pip-audit -r requirements.txt
```

## Notes

- Web fallback is configurable in sidebar settings.
- For best performance, keep your knowledge base focused and avoid indexing duplicate content.
