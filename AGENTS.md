# AGENTS.md

Guidance for Codex and other coding agents working in Resume Griller.

## Canonical documentation

- `README.md`: implemented setup, architecture, routes, and runtime boundaries.
- `DEVELOPMENT_ROADMAP.md`: future work, milestone dependencies, acceptance gates,
  and non-goals.
- `AI_ONBOARDING.md`: short engineering context and invariants.
- `docs/CONTRIBUTING.md`: development and contribution commands.
- `docs/REFACTOR_HANDOFF.md`: current evidence, owner exercises, and next slice.
- `docs/PROJECT_REASSESSMENT.md`: dated assessment and recruiting rationale.

Inspect the current Git branch and worktree before acting. Do not hard-code a branch,
coverage number, model version, or file line count into new guidance. The dated
baseline and start decision live in the roadmap.

## Architecture summary

Resume Griller is a FastAPI + Next.js AI mock interview prototype.

- LangGraph owns interview orchestration and checkpointed state.
- REST and WebSocket handlers are thin adapters over the same graph.
- `session_id` is the LangGraph `thread_id`.
- New PDF/TXT uploads are parsed by Groq into Markdown.
- The complete parsed resume is stored in graph state during an interview.
- Active interviews do not perform per-question vector retrieval.
- ChromaDB and the old parser/chunker/embedder remain only for legacy resume IDs.
- API LLM providers are Groq, Gemini, OpenAI, and Anthropic.
- Custom is an OpenAI-compatible execution mode; Hybrid adds Groq preprocessing.
- `LLM_MODE=local` exists in settings but automatic local LoRA loading is not wired
  into the active factory.
- SQLite checkpoints and local files make the current runtime single-instance.

The old `InterviewAgent` and in-memory session store were deleted. Do not recreate
transport-specific orchestration or a second session state system.

## Current development gate and learning workflow

Follow the revised M0–M9 plan in `DEVELOPMENT_ROADMAP.md` and the current evidence in
`docs/REFACTOR_HANDOFF.md`. The prior M0–M8 numbering is superseded. Start with M0;
do not infer completion from documentation alignment or passing legacy tests.

- Core: M0 baseline -> M1 interview correctness -> M2 contracts/CI -> M3 benchmark
  -> M4 ledger/report -> M5 recovery -> M6 product/dependency simplification.
- M7 JD targeting and M8 adaptive selection are optional, selected by the owner.
- M9 controlled deployment depends on M6, not on optional M7/M8. Public beta and
  real-user data processing remain blocked until all M9 release/security checks pass.
- Preserve the learning objective: state the owner exercise and agent scope before
  each small slice. Default to owner implementation of the core concept and agent
  explanation/review/scaffolding. Explicit delegation authorizes completing the slice
  without repeated permission requests; record the pending learning exercise separately.
- Update the handoff after each slice with actual checks, evidence, owner learning
  status and exact next step. Never claim human understanding from agent test success.
- If provenance or another human fact is unknown, ask only for that missing fact and
  continue independent work. Never infer consent/synthetic provenance or rewrite history.

Immediate risks and reproduction targets are in the roadmap: startup action overwrite,
wrong follow-up scoring target, conflicting policy, invalid scores, missing ledger and
unresolved fixture/voice contracts. No functional fixes are complete merely because
these are documented.

## Engineering rules

- Preserve user changes in a dirty worktree and avoid unrelated edits.
- Keep graph nodes transport-independent; inject services via LangGraph config.
- Keep public Pydantic schemas and TypeScript types synchronized.
- Mock LLM, STT, TTS, and Chroma calls in automated unit/integration tests.
- For AI behavior changes, add or update an evaluation case and report the delta.
- Version prompt, rubric, schema, model, and dataset where relevant.
- Never commit `.env`, real resumes, answers, audio, parsed resumes, checkpoints, or
  unredacted traces.
- Preserve runtime data; migrations and deletion require explicit scope and tests.
- Prefer simple deterministic workflows until measured evidence justifies complexity.
- Do not add multi-agent swarms, single-document vector RAG, RL, fine-tuning,
  Kubernetes, or microservices outside the roadmap gates.

## Local commands

```bash
uv sync --extra dev
cp .env.example .env
PYTHONPATH=. uv run uvicorn backend.app.main:app --reload --port 8000
```

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

```bash
PYTHONPATH=. uv run --extra dev pytest tests/
PYTHONPATH=. uv run --extra dev pytest tests/ --cov=backend --cov=rag \
  --cov-report=term-missing
uv run --extra dev ruff check backend rag/retriever.py tests --config pyproject.toml
uv run --extra dev black --check backend rag/retriever.py tests
cd frontend && npx tsc --noEmit && npm run lint && npm run build
```

The graph smoke test uses mocks and needs no provider key:

```bash
PYTHONPATH=. uv run python scripts/test_graph.py
```

## API summary

- Health: `GET /health`
- Resume: `/api/v1/resume/...` (singular `resume`)
- Sessions: `/api/v1/sessions/...`
- Voice: `/api/v1/voice/...`
- WebSocket: `/ws/interview/{session_id}`

Use the generated FastAPI docs at `http://localhost:8000/docs` for the complete
route schema.
