# AI Onboarding Guide

This document is the short source of truth for agents and engineers resuming work
on Resume Griller.

Use `README.md` for the current runtime contract and `DEVELOPMENT_ROADMAP.md` for
future priorities. Do not copy model names, line counts, coverage numbers, or
milestone status into additional documents unless they are intentionally dated.

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
3. `POST /api/v1/sessions` loads the complete Markdown into `InterviewState`; its
   intended first-question generation currently has the startup defect noted below.
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

The configured API providers are Groq, Gemini, OpenAI, and Anthropic. `custom` is
an interview execution mode backed by an OpenAI-compatible endpoint; Hybrid uses
Groq preprocessing plus that endpoint. Although `LLM_MODE=local` exists in settings,
automatic local LoRA loading is not connected to the active service factory.

## Development rules

- Keep graph nodes transport-independent and inject services through LangGraph
  config.
- Keep public schemas and TypeScript types synchronized.
- Mock all LLM, STT, TTS, and Chroma calls in automated tests.
- New uploads support only PDF and UTF-8 TXT.
- Preserve existing user runtime data; never migrate or delete it implicitly.
- Never commit real resumes, answers, audio, parsed text, checkpoints, or traces.
- Use `pyproject.toml` and the committed `uv.lock` as dependency sources.
- Do not commit `.env`, uploaded resumes, parsed resumes, or checkpoint databases.
- Treat `DEVELOPMENT_ROADMAP.md` milestone gates as dependencies; do not skip ahead
  to adaptive behavior, fine-tuning, or public deployment.

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

## Current priorities and handoff

Read [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md) for the revised M0–M9 sequence
and [docs/REFACTOR_HANDOFF.md](docs/REFACTOR_HANDOFF.md) for current progress and the
next eligible slice. The dated test/assessment evidence lives there, not in this guide.

The first work is M0 data/provenance inventory and a tested voice support decision.
Then repair the real session-start/follow-up flow, introduce validated model contracts,
build a human-reviewed benchmark, persist feedback, and test recovery. Basic CI belongs
early, in M2. JD/adaptive functionality is optional; controlled deployment is M9.

For each slice, state the owner exercise, agent scope and acceptance checks. Let the
owner implement the concept being learned unless explicitly delegated. Update the
handoff with engineering evidence and owner learning status separately.

## Known boundaries

- No authentication or user ownership model.
- SQLite checkpoints target a single backend instance.
- Rate limiting is process-local and does not protect every WebSocket operation.
- Evaluation history is not append-only; changing questions can clear the current
  evaluation and the final report is still count-based.
- Session creation overwrites its start action with null initial state and can fail
  response validation before generating questions.
- Follow-up answers are evaluated against the original question; reconnect can return
  that original question too.
- Empty/unbounded model evaluation output is not safely rejected.
- Full resume text is stored in state but evaluation prompts truncate context.
- Voice is batch-based. `.env.example` disables it safely, but the Python setting
  defaults to enabled and the root dependency set lacks the ElevenLabs SDK.
- Custom/Hybrid mode requires a separately hosted OpenAI-compatible model.
- No CI workflow, production container configuration, or frontend test suite exists.
- No real-user processing or public beta is allowed before roadmap M9 passes.
