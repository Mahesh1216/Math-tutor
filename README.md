# Math-Tutor — Agentic RAG with Human-in-the-Loop

A full-stack **Math Tutor** that:
- retrieves first from a **vector knowledge base** (KB),
- falls back to **web/MCP search** when needed,
- answers with **clear step-by-step** explanations, and
- **learns** from user feedback to improve future answers.

> Stack: **FastAPI** (Backend) + **React** (Frontend) + **Qdrant** (Vector DB). MCP tools (search/web) .

---

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [1) Clone](#1-clone)
  - [2) Start Vector DB (Qdrant)](#2-start-vector-db-qdrant)
  - [3) Backend Setup (FastAPI)](#3-backend-setup-fastapi)
  - [4) Frontend Setup (React)](#4-frontend-setup-react)
- [Environment Variables](#environment-variables)
- [Knowledge Base Ingestion](#knowledge-base-ingestion)
- [API Reference](#api-reference)
- [Optional: MCP (Model Context Protocol) Tools](#optional-mcp-model-context-protocol-tools)
- [Troubleshooting](#troubleshooting)
- [Suggested Dev Scripts](#suggested-dev-scripts)
- [Optional: Docker Compose](#optional-docker-compose)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features
- **Routing Agent**: KB → Generate; else Web/MCP → Generate (with citations).
- **Guardrails**: math-only filtering, length/PII checks, and output format enforcement (steps → final answer).
- **Feedback Loop**: collect ratings/comments; tune thresholds and exemplars over time.
- **Simple API**: `/ask`, `/feedback`, `/ingest`, `/health`.
- **Frontend**: ask questions, view step-by-step solutions, give feedback; basic admin actions (ingest/eval).

---

## Repository Structure
```
Math-tutor/
├─ Backend/                 # FastAPI app, routing, vector client, guardrails, feedback
│  ├─ app/
│  ├─ data/                 # local SQLite, KB jsonl, etc.
│  ├─ requirements.txt
│  └─ (optional) mcp.config.json
├─ Frontend/                # React app (question/answer UI, feedback, basic admin)
│  ├─ package.json
│  └─ (src, public, etc.)
└─ README.md
```

---

## Prerequisites
- **Python** 3.10+ (3.11 recommended)
- **Node.js** 18+ and **npm**
- **Docker** (for Qdrant)
- API keys (as needed): `OPENAI_API_KEY`, search provider keys if using MCP

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/Mahesh1216/Math-tutor.git
cd Math-tutor
```

### 2) Start Vector DB (Qdrant)
```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

### 3) Backend Setup (FastAPI)
```bash
cd Backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `Backend/` (see [Environment Variables](#environment-variables)), then run:
```bash
uvicorn app.main:app --reload --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

### 4) Frontend Setup (React)
Open a second terminal:
```bash
cd Frontend
npm run dev
```
---

## Environment Variables

Create `Backend/.env`:
```
GOOGLE_API_KEY=
TAVILY_API_KEY=
```
---

## Knowledge Base Ingestion

1) Create a starter KB file at `Backend/data/knowledge_base.jsonl`:
```jsonl
{"question":"Find the derivative of x^3","solution":"d/dx x^3 = 3x^2","tags":["calculus"],"source":"seed"}
{"question":"Compute 12 + 35","solution":"47","tags":["arithmetic"],"source":"seed"}
{"question":"Evaluate ∫(0→1) x^2 dx","solution":"1/3","tags":["calculus","integral"],"source":"seed"}
```

2) Ingest into the vector DB:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"data/knowledge_base.jsonl","tags":["calculus","algebra"]}'
```

---

## API Reference

### `GET /health`
Simple liveness probe.
```json
{ "status": "ok" }
```

### `POST /ask`
Ask a math question and get a step-by-step answer.
```json
{
  "question": "Evaluate ∫_0^1 x^2 dx",
  "need_steps": true
}
```
**Response (example)**
```json
{
  "source": "kb",
  "steps": ["We integrate x^2...", "Apply limits 0→1...", "Result is 1/3."],
  "final_answer": "1/3",
  "citations": [],
  "meta": { "similarity": 0.83, "router_path": "kb-hit" }
}
```

### `POST /feedback`
Submit human feedback for continuous improvement.
```json
{
  "question": "Evaluate ∫_0^1 x^2 dx",
  "answer_id": "uuid-or-id-from-response",
  "rating": 4,
  "comment": "Good, but add substitution detail."
}
```

### `POST /ingest`
Index KB items into the vector DB.
```json
{
  "path": "data/knowledge_base.jsonl",
  "tags": ["algebra","calculus"]
}
```

---

## Optional: MCP (Model Context Protocol) Tools

Create `Backend/mcp.config.json` if you plan to use MCP search/browse:
```json
{
  "servers": {
    "search": { "command": "mcp-server-tavily", "env": { "TAVILY_API_KEY": "YOUR_TAVILY_KEY" } },
    "web":    { "command": "mcp-server-web" }
  }
}
```
Ensure `MCP_CONFIG=./mcp.config.json` is set in `Backend/.env`.

---

## Troubleshooting
- **CORS errors** (frontend → backend): add FastAPI CORS middleware and allow your dev origin (`http://localhost:3000` or `5173`).
- **Qdrant connection refused**: confirm Docker is running; container exposes `6333`; `QDRANT_URL` matches.
- **No/poor answers**: verify KB ingestion; adjust `RETRIEVAL_THRESHOLD`; ensure MCP keys are set for web fallback.
- **Rate limits/timeouts**: check LLM plan limits; add retry/backoff on the client side; tune `MCP_TIMEOUT_SECS`.

---

## Suggested Dev Scripts

**Backend**
```bash
ruff check . && ruff format .
pytest -q
```

**Frontend**
```bash
npm run lint
npm run build
```

---

## Optional: Docker Compose

Create `docker-compose.yml` at repo root to bring up **Qdrant** and the **Backend** together (adjust image/paths to your project):

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes:
      - qdrant_storage:/qdrant/storage

  backend:
    build: ./Backend
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_COLLECTION=math_kb
      - DB_URL=sqlite:///./data/app.sqlite3
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBEDDING_MODEL=text-embedding-3-small
      - RETRIEVAL_TOP_K=5
      - RETRIEVAL_THRESHOLD=0.78
      - ALLOW_WEB_SEARCH=true
    ports: ["8000:8000"]
    depends_on: [qdrant]
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

volumes:
  qdrant_storage:
```

> You can similarly add a **frontend** service if you prefer containerized UI.

---

## Roadmap
- Step-level feedback (thumbs per step)
- Export session to PDF/Markdown
- Full evaluation script (accuracy, router hit-rate, abstention)
- Docker images and GH Actions CI
- OCR input for handwritten problems

---

## License
Educational use. Verify licenses of any external datasets you ingest.
