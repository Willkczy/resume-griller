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
3. `POST /api/v1/sessions` loads the complete Markdown into `InterviewState` and
   generates the first question through the graph.
4. REST and WS call `graph.commands.execute_command`, which validates the latest
   checkpoint before invoking the graph with the same `thread_id`. A start on an
   asking session projects its active turn without writing or regenerating it.
5. Graph nodes call `GrillingEngine` and the selected LLM service.
6. The graph records the actual active prompt and stable turn IDs. Evaluation targets
   that prompt; one pure policy selects follow-up/advance. REST and WS project the
   same active turn and map internal phases to the public lifecycle.

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

The sequence establishes a safe local baseline and correct interview flow before
validated model contracts, a human-reviewed benchmark, durable feedback and recovery.
Basic CI belongs in M2. JD/adaptive functionality is optional; controlled deployment
is M9. Current completion and scoped owner exceptions are recorded in the handoff.

For each slice, state the owner exercise, agent scope and acceptance checks. Let the
owner implement the concept being learned unless explicitly delegated. Update the
handoff with engineering evidence and owner learning status separately.

## Known boundaries

- No authentication or user ownership model.
- SQLite checkpoints target a single backend instance.
- Rate limiting is process-local and does not protect every WebSocket operation.
- Evaluation history is not append-only; changing questions can clear the current
  evaluation and the final report is still count-based.
- Sequential command validation does not provide concurrency or idempotency guarantees.
- Older checkpoints recover active prompts without migration; a missing follow-up
  prompt causes a 409 rejection rather than guessing the main question.
- Model questions/evaluations are runtime validated. Invalid output, provider failure,
  insufficient evidence and unsupported compact evaluation have no score and do not
  advance the interview. Public outcome unions and dedicated UI restore submitted
  drafts and failures from the checkpoint on manual reconnect. Allowed actions come
  from the shared command guard; reconnect does not resubmit. Provider metadata and
  deterministic CI are still M2 work.
- Full resume text is stored in state but evaluation prompts truncate context.
- Voice is batch-based and disabled by default in settings and `.env.example`.
  The root dependency set lacks ElevenLabs; live voice remains experimental.
  See `docs/decisions/0001-text-only-baseline.md`.
- Custom/Hybrid mode requires a separately hosted OpenAI-compatible model.
- No CI workflow or production container configuration exists. Frontend outcome
  fixtures exercise presentation/reconciliation; broader browser/recovery coverage
  remains unfinished.
- No real-user processing or public beta is allowed before roadmap M9 passes.
