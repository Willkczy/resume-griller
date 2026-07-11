# AI Onboarding Guide

This document is the short source of truth for agents and engineers resuming work
on Resume Griller.

## What exists now

Resume Griller is a FastAPI + Next.js mock interview application. Interview
orchestration lives in a LangGraph `StateGraph`; state is persisted through
SQLite checkpoints using the session ID as LangGraph's thread ID.

The old `InterviewAgent` and in-memory `session_store` were deleted. Do not
reintroduce orchestration in HTTP or WebSocket handlers. Both transports should
remain thin adapters over the same graph.

## Request flow

1. A PDF/TXT resume is uploaded.
2. Text is extracted and Groq converts it to structured Markdown.
3. `POST /api/v1/sessions` loads the complete Markdown into `InterviewState`.
4. Each answer, skip, or end request invokes the graph with the same `thread_id`.
5. Graph nodes call `GrillingEngine` and the selected LLM service.
6. REST and WebSocket adapters map internal phases to the public lifecycle.

Internal phases such as `asking` and `evaluating` must not leak to clients. The
public status values are `pending`, `in_progress`, `completed`, and `cancelled`.

## Code map

| Area | Location | Responsibility |
|---|---|---|
| Graph | `backend/app/graph/` | State, nodes, routing, checkpoint setup |
| Grilling | `backend/app/core/grilling_engine.py` | Gap detection and 7D scoring |
| Transports | `backend/app/api/routes/` | REST/WebSocket validation and mapping |
| LLMs | `backend/app/services/llm_service.py` | API, custom, and hybrid providers |
| Resume | `rag/retriever.py` | Markdown primary store; lazy Chroma fallback |
| Frontend | `frontend/src/components/interview/` | Chat/video session clients |

Legacy `rag/resume_parser.py`, `chunker.py`, and `embedder.py` are compatibility
code. New interview behavior should use the complete parsed resume in graph state.

## Development rules

- Keep graph nodes transport-independent and inject services through LangGraph
  config.
- Keep public schemas and TypeScript types synchronized.
- Mock all LLM, STT, TTS, and Chroma calls in automated tests.
- New uploads support only PDF and UTF-8 TXT.
- Preserve existing user runtime data; never migrate or delete it implicitly.
- Use `pyproject.toml` and the committed `uv.lock` as dependency sources.
- Do not commit `.env`, uploaded resumes, parsed resumes, or checkpoint databases.

## Local commands

```bash
uv sync --extra dev
PYTHONPATH=. uv run pytest tests/
PYTHONPATH=. uv run python scripts/test_graph.py
PYTHONPATH=. uv run uvicorn backend.app.main:app --reload --port 8000
```

```bash
cd frontend
npm ci
npm run dev
npm run lint
npm run build
```

## Current priorities

1. Keep upload → session → WebSocket → summary working end to end.
2. Maintain at least 70% coverage for graph, session, WebSocket, resume routes,
   and the active retriever.
3. Keep Python lint, TypeScript, ESLint, and production build green.
4. Add authentication and production persistence only after this baseline remains
   stable.

## Known boundaries

- No authentication or user ownership model.
- SQLite checkpoints target a single backend instance.
- Rate limiting is process-local and does not protect every WebSocket operation.
- Voice depends on external providers and is optional.
- Custom/Hybrid mode requires a separately hosted OpenAI-compatible model.
