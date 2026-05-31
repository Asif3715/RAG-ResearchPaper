<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</div>

# 📚 Paper Assistant

> **A production-grade RAG workspace for research PDFs** — Upload papers, index them in a hybrid vector store, and ask grounded questions with streamed answers, LaTeX-ready markdown, and collapsible source citations.

Built as a full-stack system with a retrieval pipeline designed around real paper workflows: multi-document libraries, per-document scoping, background ingestion, and citation-backed generation.

---

## 🌟 Why This Project Exists

Reading a 20-page ML paper and hunting for “what loss they used” or “how attention is defined” is slow. Paper Assistant treats each PDF as a **searchable knowledge base**:
- Chunks are parsed, embedded, and retrieved on demand.
- Answers are synthesized by an LLM **only from retrieved passages**.
- Explicit source attribution is provided for every claim.

This is not just a chat wrapper — it is an **end-to-end ingestion → retrieval → generation** pipeline with observable status, durable metadata, and a UI tuned for technical content (equations, tables, structured answers).

---

## ✨ Highlights

| Area | What you get |
|------|----------------|
| 🔍 **Retrieval** | Dense (BGE-M3) + sparse (hashed BoW) hybrid search with **RRF fusion**, optional **cross-encoder reranking** (BGE reranker v2-m3) |
| 🗄️ **Storage** | **Qdrant Cloud** (or self-hosted) with separate dense/sparse collections and `doc_id` payload filtering |
| 📄 **Parsing** | **PyMuPDF + pymupdf4llm** — page-aware markdown, recursive chunking, content-addressed cache (`data/extracted/`) |
| 🧠 **Generation** | **Groq** (Llama 3.3 70B) with prompts tuned for structured markdown, LaTeX, and inline citations |
| 💻 **UX** | SSE streaming, KaTeX math, GFM tables, collapsible sources, slide-out library, document scoping, and **in-app PDF viewing** |
| ⚙️ **Ops** | Health checks, background ingest, persisted ingestion status, embedding disk cache, pytest suite |

---

## 🛠️ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Vite-B73BFE?style=for-the-badge&logo=vite&logoColor=FFD62E" alt="Vite" />
  <img src="https://img.shields.io/badge/Qdrant-E53935?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Groq-000000?style=for-the-badge&logo=groq&logoColor=white" alt="Groq" />
</p>

| Layer | Technologies |
|-------|----------------|
| **API** | Python 3.11+, FastAPI, Uvicorn, Pydantic v2 |
| **Retrieval** | `qdrant-client`, Hugging Face Inference (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) |
| **LLM** | Groq API (`llama-3.3-70b-versatile`) |
| **PDF** | PyMuPDF, pymupdf4llm |
| **Frontend** | React 19, TypeScript, Vite 7 |
| **UI** | react-markdown, remark-gfm, remark-math, rehype-katex, Lucide |
| **Tests** | pytest |

---

## 🏗️ Architecture

```mermaid
flowchart TB
  subgraph Client["React + Vite"]
    UI[Chat UI]
    SSE[SSE /answer/stream]
  end

  subgraph API["FastAPI"]
    UP[POST /upload]
    ING[Ingestion worker]
    RET[Retrieval pipeline]
    GEN[Groq generator]
  end

  subgraph Parse["PDF pipeline"]
    PDF[PyMuPDF4LLM]
    CHK[Chunk + normalize]
  end

  subgraph ML["Models"]
    EMB[BGE-M3 embeddings]
    RR[BGE reranker]
    LLM[Llama 3.3 70B]
  end

  subgraph Store["Qdrant"]
    DENSE[(dense collection)]
    SPARSE[(sparse collection)]
  end

  UI --> UP
  UP --> PDF --> CHK --> ING
  ING --> EMB --> DENSE
  ING --> SPARSE
  UI --> SSE --> RET
  RET --> DENSE
  RET --> SPARSE
  RET --> RR
  SSE --> GEN
  GEN --> LLM
  RET --> GEN
```

### 🔄 Request Flow (Question → Answer)

1. **Upload** — PDF bytes are hashed, parsed to page-level markdown, chunked (~1.1k chars, overlap), and queued for background ingest. The original file is stored safely for in-app viewing.
2. **Index** — Batches are embedded (HF Inference API), sparse vectors built, and upserted to Qdrant; progress is written to `data/ingestion_status.json` and the document registry.
3. **Query** — User question is embedded; **hybrid search** pulls top-*k* chunks (optional **doc_id** filter when a paper is scoped).
4. **Rerank** — Cross-encoder rescores candidates (with graceful fallback if the API fails).
5. **Generate** — Top passages are sent to Groq; tokens stream to the client over **Server-Sent Events**; sources are attached as structured metadata for the UI.

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **[Qdrant Cloud](https://cloud.qdrant.io/)** cluster (or local Qdrant)
- API keys:
  - [Hugging Face](https://huggingface.co/settings/tokens) → `HF_TOKEN` (embeddings + reranker)
  - [Groq](https://console.groq.com/) → `GROQ_API_KEY`
  - Qdrant → `QDRANT_URL`, `QDRANT_API_KEY`

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` (from Qdrant Cloud → **Connect**):

```env
QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-key>
HF_TOKEN=<your-hf-token>
GROQ_API_KEY=<your-groq-key>
```

> **Note:** Optional override models in `backend/config.yaml` (defaults: BGE-M3, BGE reranker v2-m3).

### 2. Install Dependencies

```bash
# Backend (from repo root)
pip install -r requirements.txt
# or: pip install -e backend

# Frontend
cd frontend && npm install && cd ..
```

### 3. Run the App

**Terminal A — API** (must run from repository root):

```bash
./backend/run.sh
# → http://127.0.0.1:8000
```

**Terminal B — UI**:

```bash
./frontend/run_frontend.sh
# → http://127.0.0.1:5173
```

> *Optional frontend override: `frontend/.env` with `VITE_BACKEND_URL=http://127.0.0.1:8000`.*

### 4. How to Use

1. Open the UI → **Sources** → **Upload PDF**.
2. Wait for ingestion (status shown per document).
3. Click a paper to **scope** chat, or ask across the full library.
4. Open **search settings** (composer) to tune hybrid / dense-only, Top-K, reranker.
5. Click **View PDF** in the document menu to read the raw paper instantly.

---

## 🗂️ Repository Layout

```
paperassistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # HTTP API + SSE streaming
│   │   ├── dependencies.py      # Shared singleton clients
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── core/config.py       # Data directories
│   │   └── services/
│   │       ├── pdf/parser.py    # Parse, chunk, cache by doc hash
│   │       ├── indexing/        # Ingest, status, registry, cleanup
│   │       ├── embeddings/      # HF embed + rerank + disk cache
│   │       ├── retrieval/       # RRF, sparse vectors, pipeline
│   │       ├── storage/         # Qdrant client + config
│   │       └── llm/             # Groq prompts + streaming
│   ├── config.yaml              # Models, collection names
│   ├── run.sh                   # Uvicorn from repo root
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/          # Chat, citations, sources drawer, markdown
│   │   └── api.ts
│   └── run_frontend.sh
├── data/                        # Gitignored runtime data
│   ├── uploads/                 # Raw PDFs (viewable from UI)
│   ├── extracted/               # Parsed JSON per doc_id
│   ├── cache/                   # Embedding cache
│   ├── documents.json           # Document registry
│   └── ingestion_status.json
├── .env.example
└── requirements.txt
```

---

## 🔌 API Overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Qdrant, embedding model, Groq connectivity |
| `POST` | `/upload` | Upload PDF(s); background ingest |
| `GET` | `/documents/{doc_id}/pdf` | Serves the original uploaded PDF |
| `GET` | `/ingestion`, `/ingestion/{doc_id}` | Ingestion timeline / status |
| `GET` | `/documents` | Merged library (Qdrant + registry + status) |
| `DELETE` | `/documents/{doc_id}` | Delete one document (vectors + artifacts) |
| `DELETE` | `/documents?doc_ids=…` | Delete selected documents |
| `POST` | `/query` | Synchronous Q&A (rate-limited) |
| `GET` | `/answer/stream` | **SSE** streamed answer + source metadata |

---

## 🧠 Design Decisions (For Interviews)

- **Hybrid Retrieval:** Academic PDFs contain rare terms (architecture names, theorem labels). Combining dense semantic search with a sparse lexical leg and **Reciprocal Rank Fusion** improves recall when either signal alone misses a passage.
- **Separate dense/sparse Qdrant collections:** Lets you tune sparse vector naming per deployment (e.g., legacy `bm25` vs `sparse` on cloud) and query each leg independently before fusion.
- **Content-addressed `doc_id`:** SHA-256 of file bytes ensures stable identity across re-uploads; parsed output cached under `data/extracted/{doc_id}.json`.
- **Background Ingestion:** Upload returns immediately; embedding batches run in FastAPI `BackgroundTasks` with durable status for the UI poll loop.
- **Citation-First Prompting:** The LLM is instructed to use markdown structure, LaTeX for math, and inline `[Source Title, Page N]` citations — aligned with how researchers expect to verify claims.
- **Frontend as a Research Surface:** Not a generic chat box: document scoping, ingestion progress, advanced retrieval controls, KaTeX + GFM rendering, and user-controlled source disclosure.

---

## 🧪 Testing

Run the test suite to verify components:

```bash
PYTHONPATH=. pytest backend/tests -q
```

Covers API contracts, config/env merging, RRF fusion, sparse vectors, PDF chunking, document merge logic, ingestion status persistence, and delete semantics.

---

## 🚧 Limitations & Next Steps

- Requires live **Qdrant**, **Hugging Face Inference**, and **Groq** (no offline mode).
- Sparse leg uses custom hashed term vectors, not full BM25 index semantics.
- Figures/tables in PDFs are text-oriented (no multimodal image embeddings yet).
- Single-user local deployment (no auth/multi-tenancy).

**Plausible Next Steps:** 
- OAuth & Multi-tenancy
- Async job queue (Redis/Celery)
- Evaluation harness (nDCG / citation accuracy)
- Cloud Object Storage (S3) integration for persistent file uploads
- Vision chunks for embedded figures

---

## 👨‍💻 Author

**Asif Khan**  
🔗 [LinkedIn](https://linkedin.com/in/asif-khan-0ba826240) · ✉️ tanoli3715@gmail.com
