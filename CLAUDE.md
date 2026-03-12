# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

---

## Current Status

**Branch**: `feature/langgraph-interview-agent` (active migration)

**Recently completed — LangGraph Migration (Steps 1-6 of 7)**:
- Consolidated ~2000 lines of duplicated interview orchestration into a single LangGraph StateGraph
- Deleted `interview_agent.py` (596 lines) and `session_store.py` (236 lines)
- Session state now persisted via LangGraph checkpoints (SQLite) instead of in-memory store
- `session.py` simplified from 679 to ~280 lines
- `websocket.py` simplified from 776 to ~270 lines

**Remaining**:
- Test coverage (<5%, target 70%+)
- Authentication (JWT)
- See `DEVELOPMENT_ROADMAP.md` for full plan

---

## Project Overview

**Resume Griller** is a full-stack AI-powered interview simulator. It analyzes resumes via RAG, conducts mock interviews with intelligent follow-up questions, and evaluates answers across 7 dimensions using 18 gap types.

## Quick Start

### Backend

```bash
uv sync
source .venv/bin/activate
cp .env.example .env  # Add API keys

PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend && npm install && npm run dev
```

Frontend: `http://localhost:3000`

### Docker Compose

```bash
cp .env.example .env && docker-compose up -d
```

### Tests

```bash
PYTHONPATH=. uv run pytest tests/
PYTHONPATH=. uv run pytest tests/ --cov=backend --cov=rag

# Graph-specific interactive test (uses mocks, no API keys needed)
PYTHONPATH=. uv run python scripts/test_graph.py
```

---

## Architecture

### System Overview

```
Frontend (Next.js 16 + React 19)
    ↕ HTTP / WebSocket
Backend (FastAPI)
    ├── Graph Module (LangGraph)     ← interview orchestration
    │     ├── state.py               ← InterviewState TypedDict
    │     ├── nodes.py               ← 9 node functions
    │     ├── edges.py               ← 3 routing functions
    │     ├── builder.py             ← StateGraph construction
    │     ├── services.py            ← dependency injection
    │     └── checkpointer.py        ← SQLite persistence
    ├── Grilling Engine              ← 18 gap types, 7D scoring
    ├── LLM Service                  ← Groq/Gemini/OpenAI/Claude/Custom/Hybrid
    ├── Voice Services               ← Deepgram STT, ElevenLabs TTS
    └── RAG Pipeline                 ← ChromaDB + sentence-transformers
```

### LangGraph Interview Flow

The interview is a StateGraph where each API/WS call triggers a full graph run from `route_action`. State is restored from checkpoint (SQLite) between calls.

```
START -> route_action
  "start"  -> generate_questions -> ask_question -> END
  "answer" -> evaluate_answer -> route_after_evaluate
                                   "grill"   -> generate_follow_up -> END
                                   "advance" -> advance_question -> route_after_advance
                                                                      "more" -> ask_question -> END
                                                                      "done" -> complete_interview -> END
                                   "done"    -> complete_interview -> END
  "skip"   -> handle_skip -> advance_question -> (same routing)
  "end"    -> handle_end -> complete_interview -> END
```

**Key design decisions**:
- Fresh invocation pattern (not LangGraph interrupts): each request runs the full graph
- `session_id` = LangGraph `thread_id` for checkpoint keying
- Services injected via config: `config["configurable"]["services"]`
- Graph nodes are thin wrappers delegating to GrillingEngine, LLM, RAG

### How routes use the graph

```python
from backend.app.graph import get_compiled_graph, create_initial_state, GraphServices

services = GraphServices.create(model_type, retriever, prepared_context)
graph = await get_compiled_graph()
result = await graph.ainvoke(
    {"action": "answer", "current_answer": "..."},
    config={"configurable": {"thread_id": session_id, "services": services}}
)
# result["response_type"]    -> "question" | "follow_up" | "complete" | "error"
# result["response_content"] -> the text to send back
# result["response_data"]    -> metadata (scores, question numbers, etc.)
```

---

## Project Structure

```
resume-griller/
├── backend/app/
│   ├── graph/                        # LangGraph interview orchestration (NEW)
│   │   ├── state.py                  # InterviewState TypedDict (166 lines)
│   │   ├── nodes.py                  # 9 graph nodes (607 lines)
│   │   ├── edges.py                  # 3 routing functions (103 lines)
│   │   ├── builder.py                # StateGraph construction (144 lines)
│   │   ├── services.py               # GraphServices DI (106 lines)
│   │   ├── checkpointer.py           # AsyncSqliteSaver setup (77 lines)
│   │   └── __init__.py               # Exports
│   │
│   ├── api/routes/
│   │   ├── session.py                # HTTP endpoints using graph (~280 lines)
│   │   ├── websocket.py              # WS endpoint using graph (~270 lines)
│   │   ├── resume.py                 # Resume upload & processing
│   │   └── voice.py                  # STT/TTS endpoints
│   │
│   ├── core/
│   │   ├── grilling_engine.py        # Gap detection & 7D scoring (973 lines)
│   │   └── logging_config.py         # Structured logging
│   │
│   ├── services/
│   │   ├── llm_service.py            # Multi-provider LLM abstraction (828 lines)
│   │   ├── stt_service.py            # Deepgram STT
│   │   └── tts_service.py            # ElevenLabs TTS
│   │
│   ├── middleware/rate_limit.py       # SlowAPI rate limiting
│   ├── api/deps.py                   # FastAPI dependency injection
│   ├── config.py                     # Settings from .env
│   └── main.py                       # FastAPI app entry point
│
├── frontend/src/
│   ├── app/                          # Next.js App Router pages
│   ├── components/                   # React components (interview, upload, ui)
│   ├── stores/interviewStore.ts      # Zustand state management
│   └── lib/                          # API client, WebSocket client
│
├── rag/                              # RAG Pipeline
│   ├── resume_parser.py              # PDF/TXT parser (464 lines)
│   ├── chunker.py                    # Semantic chunking (224 lines)
│   ├── embedder.py                   # ChromaDB operations (282 lines)
│   ├── retriever.py                  # Context retrieval (268 lines)
│   └── generator.py                  # LoRA model inference (185 lines)
│
├── scripts/test_graph.py             # Interactive graph test (mock services)
├── data/                             # Uploads, ChromaDB, checkpoints
├── tests/                            # pytest tests
├── pyproject.toml                    # Python deps (uv)
└── docker-compose.yml                # Multi-service orchestration
```

---

## Key Components

### Graph Module (`backend/app/graph/`)

The central orchestration layer. Replaces the old `InterviewAgent` class.

**state.py** — `InterviewState` TypedDict with fields for:
- Identity (session_id, resume_id, mode, model_type)
- Config (num_questions, max_follow_ups, focus_areas)
- Flow state (status, questions, current_question_index, follow_up_count)
- Current interaction (current_answer, current_evaluation)
- Conversation history (append-only via `Annotated[list, operator.add]`)
- Output (response_type, response_content, response_data)
- Input signal (action: start/answer/skip/end)

**nodes.py** — 9 async node functions:
- `generate_questions` — RAG + LLM question generation
- `ask_question` — format current question, retrieve RAG context
- `evaluate_answer` — GrillingEngine evaluation + consistency check
- `generate_follow_up` — targeted follow-up from detected gaps
- `advance_question` — increment index, reset follow-up count
- `complete_interview` — generate summary
- `handle_skip`, `handle_end`, `handle_error`

**edges.py** — 3 pure routing functions:
- `route_action` — entry point, reads `state["action"]`
- `route_after_evaluate` — grill / advance / done decision
- `route_after_advance` — more questions or done

**services.py** — `GraphServices` dataclass for DI:
- `GraphServices.create(model_type, retriever)` — factory
- Bundles: retriever, llm, grilling_engine, hybrid_service

**checkpointer.py** — `get_compiled_graph()` singleton:
- Uses `AsyncSqliteSaver` for persistent state
- Checkpoint DB at `data/interview_checkpoints.db`

### Grilling Engine (`backend/app/core/grilling_engine.py`)

Evaluates answers and generates follow-ups. NOT replaced by LangGraph — graph nodes delegate to it.

- **18 gap types**: no_specific_example, no_metrics, unclear_role, no_outcome, no_tech_depth, etc.
- **7D scoring**: relevancy, clarity, informativeness, specificity, quantification, depth, completeness
- **Forced first follow-up**: always grills at least once per question
- **Resume consistency checking**: validates claims against resume

### LLM Service (`backend/app/services/llm_service.py`)

Unified interface for 6 providers:
1. **Groq** (primary) — `llama-3.3-70b-versatile`
2. **Gemini** — `gemini-2.5-flash`
3. **OpenAI** — `gpt-4o`
4. **Claude** — `claude-sonnet-4-20250514`
5. **Custom Model** — vLLM on GCP with LoRA adapter
6. **Hybrid** — Groq preprocessing + Custom Model execution

### RAG Pipeline (`rag/`)

Resume parsing → semantic chunking → ChromaDB embeddings → retrieval.
Used by `generate_questions` and `ask_question` graph nodes.

---

## Configuration

### Essential `.env` Variables

```bash
# LLM (required)
LLM_MODE=api
LLM_PROVIDER=groq
GROQ_API_KEY=your_key

# Optional providers
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...

# Voice (optional)
VOICE_ENABLED=false
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...

# RAG
CHROMA_PERSIST_DIR=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Uploads
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE=10485760
```

See `.env.example` for all options.

---

## Development

### Commit Message Format

```
type(scope): short description
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
Scopes: `backend`, `frontend`, `rag`, `ml`, `graph`

### Code Quality

```bash
# Python
ruff check backend/ rag/
black backend/ rag/

# Frontend
cd frontend && npm run lint
```

### Adding Dependencies

```bash
uv add package-name              # production
uv add --dev package-name         # dev
cd frontend && npm install pkg    # frontend
```

### Key Dependencies

**Backend**: fastapi, uvicorn, langgraph, langgraph-checkpoint-sqlite, groq, anthropic, openai, google-generativeai, chromadb, sentence-transformers, structlog, slowapi, pydantic

**Frontend**: next@16, react@19, zustand, tailwindcss@4, lucide-react

---

## API Reference (Quick)

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/resumes/upload` | Upload resume (multipart) |
| POST | `/api/v1/sessions` | Create session & get first question |
| GET | `/api/v1/sessions/{id}` | Get session state from checkpoint |
| POST | `/api/v1/sessions/{id}/answer` | Submit answer |
| POST | `/api/v1/sessions/{id}/skip` | Skip question |
| POST | `/api/v1/sessions/{id}/end` | End interview early |
| GET | `/api/v1/sessions/{id}/summary` | Get interview summary |
| DELETE | `/api/v1/sessions/{id}` | Delete session checkpoint |

### WebSocket

**Endpoint**: `ws://localhost:8000/ws/interview/{session_id}`

Client sends: `{"type": "start|answer|skip|end", "content": "..."}`
Server sends: `{"type": "question|follow_up|complete|error", "content": "...", "data": {...}}`

Full API docs at `http://localhost:8000/docs`.
