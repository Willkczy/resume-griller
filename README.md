# Resume Griller

Resume Griller is a full-stack AI mock interview application. It parses a resume,
generates resume-specific questions, evaluates answers across seven dimensions,
and asks targeted follow-up questions when an answer lacks evidence or detail.

## Current status

The active architecture uses LangGraph with SQLite checkpoints. Resume uploads are
parsed into structured Markdown and the complete parsed resume is stored in graph
state for the duration of an interview.

- Internal interview phases: `pending`, `generating`, `asking`, `evaluating`,
  `completed`, `cancelled`
- Public lifecycle: `pending`, `in_progress`, `completed`, `cancelled`
- New uploads: PDF and UTF-8 TXT
- Legacy resumes: read from ChromaDB only when no parsed Markdown exists
- Session persistence: SQLite through LangGraph `AsyncSqliteSaver`

The project is a development prototype. Authentication and production deployment
are not included yet.

## Architecture

```text
Next.js frontend
  ├─ REST: upload, session management, summaries, voice
  └─ WebSocket: live interview messages
        ↓
FastAPI backend
  ├─ LangGraph interview graph
  │    ├─ question generation
  │    ├─ answer evaluation
  │    ├─ follow-up generation
  │    └─ SQLite checkpoints
  ├─ Grilling Engine: 18 gap types and 7D scoring
  ├─ LLM providers: Groq, Gemini, OpenAI, Anthropic, Custom, Hybrid
  ├─ Voice: Deepgram STT and ElevenLabs TTS
  └─ Resume processing
       ├─ primary: data/parsed_resumes/{resume_id}.md
       └─ legacy fallback: ChromaDB
```

Important code:

- `backend/app/graph/` — interview orchestration and checkpointed state
- `backend/app/core/grilling_engine.py` — evaluation and follow-up logic
- `backend/app/services/llm_service.py` — provider abstraction
- `rag/retriever.py` — Markdown-first resume access and legacy fallback
- `frontend/src/components/interview/` — chat and video interview clients

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node.js 20+.

```bash
uv sync --extra dev
cp .env.example .env
```

At minimum, configure the provider selected by `LLM_PROVIDER`. Resume parsing uses
Groq, so `GROQ_API_KEY` is required for new uploads.

```bash
cd frontend
npm ci
cp .env.example .env.local
```

The frontend defaults to `http://localhost:8000` and `ws://localhost:8000` when
the corresponding public environment variables are absent.

## Run locally

Backend:

```bash
PYTHONPATH=. uv run uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`. Backend docs are available at
`http://localhost:8000/docs`.

## Tests and quality checks

```bash
PYTHONPATH=. uv run --extra dev pytest tests/
PYTHONPATH=. uv run --extra dev pytest tests/ \
  --cov=backend.app.graph \
  --cov=backend.app.api.routes.session \
  --cov=backend.app.api.routes.websocket \
  --cov=backend.app.api.routes.resume \
  --cov=rag.retriever \
  --cov-report=term-missing

uv run --extra dev ruff check backend rag/retriever.py tests --config pyproject.toml
uv run --extra dev black --check backend rag/retriever.py tests

cd frontend
npx tsc --noEmit
npm run lint
npm run build
```

The interactive graph smoke test uses mocks and does not require API keys:

```bash
PYTHONPATH=. uv run python scripts/test_graph.py
```

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/resume/upload` | Parse and save a PDF/TXT resume |
| GET | `/api/v1/resume/{resume_id}` | Resume summary |
| DELETE | `/api/v1/resume/{resume_id}` | Delete resume data |
| POST | `/api/v1/sessions` | Create session and generate first question |
| GET | `/api/v1/sessions/{session_id}` | Current checkpointed state |
| POST | `/api/v1/sessions/{session_id}/answer` | Submit an answer |
| POST | `/api/v1/sessions/{session_id}/skip` | Skip current question |
| POST | `/api/v1/sessions/{session_id}/end` | End early |
| GET | `/api/v1/sessions/{session_id}/summary` | Interview summary |
| DELETE | `/api/v1/sessions/{session_id}` | Delete checkpoint thread |

WebSocket endpoint: `/ws/interview/{session_id}`. Client message types are
`start`, `answer`, `answer_audio`, `skip`, `end`, and `ping`.

## Data and compatibility

Runtime data is ignored by Git:

- `data/uploads/`
- `data/parsed_resumes/*.md`
- `data/chromadb/`
- `data/interview_checkpoints.db`

Do not delete ChromaDB while legacy resume IDs are still needed. New resumes do
not create embeddings; the legacy embedder is initialized lazily only for an old
resume with no Markdown representation.

## Next milestones

1. Finish the stabilization baseline and maintain core-flow coverage at 70%+.
2. Add authentication and ownership checks.
3. Evaluate PostgreSQL checkpoints for multi-instance deployment.
4. Add deployment and CI/CD configuration after the runtime contract is stable.
