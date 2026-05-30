# Paper Assistant

Research-paper RAG: upload PDFs, chunk and embed them, then ask questions with streamed answers and citations.

## Prerequisites

- Python 3.11+
- Node.js 18+
- [Qdrant Cloud](https://cloud.qdrant.io/) cluster (or self-hosted Qdrant)
- API keys: [Hugging Face](https://huggingface.co/settings/tokens) (`HF_TOKEN`), [Groq](https://console.groq.com/) (`GROQ_API_KEY`)

## Setup

```bash
# From the repository root
cp .env.example .env
# Edit .env — QDRANT_URL / QDRANT_API_KEY from your Qdrant Cloud cluster Connect tab

pip install -r requirements.txt
# or: pip install -e backend

cd frontend && npm install && cd ..
```

## Run

**Backend** (must run from repo root — `backend/run.sh` sets `PYTHONPATH`):

```bash
./backend/run.sh
# API: http://127.0.0.1:8000
# Health: http://127.0.0.1:8000/health
```

**Frontend**:

```bash
./frontend/run_frontend.sh
# UI: http://127.0.0.1:5173
```

Optional: `frontend/.env` with `VITE_BACKEND_URL=http://127.0.0.1:8000` (default).

## Configuration

- `backend/config.yaml` — Qdrant collections, embedding/rerank models
- `.env` — secrets and overrides (`QDRANT_URL`, `HF_TOKEN`, `GROQ_API_KEY`, …)

Local data (gitignored): `data/uploads/`, `data/extracted/`, `data/cache/`, `data/documents.json`, `data/ingestion_status.json`.

## Tests

```bash
PYTHONPATH=. pytest backend/tests -q
```

## Architecture

- **Backend**: FastAPI, PyMuPDF parsing, HF embeddings + reranker, Qdrant dense + sparse hybrid retrieval, Groq LLM
- **Frontend**: React + Vite, SSE streaming from `/answer/stream`
