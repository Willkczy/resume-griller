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

The project is a development prototype. Authentication, user ownership, production
deployment, and validated model-quality claims are not included yet. The current
implementation is suitable for local development, not for processing real user
resumes or operating a public beta.

The assessment found a session-start action merge defect and invalid-evaluation
handling that the existing tests miss. The intended flow below is not a claim of
end-to-end correctness. See [the dated assessment](docs/PROJECT_REASSESSMENT.md),
[the refactor roadmap](DEVELOPMENT_ROADMAP.md), and
[the current handoff](docs/REFACTOR_HANDOFF.md) for evidence and next work.

## Documentation map

| Document | Authority |
|---|---|
| `README.md` | Current setup, architecture, routes, and runtime boundaries |
| `DEVELOPMENT_ROADMAP.md` | Canonical future plan, milestones, gates, and non-goals |
| `AI_ONBOARDING.md` | Short engineering context and invariants |
| `docs/CONTRIBUTING.md` | Development and contribution workflow |
| `docs/REFACTOR_HANDOFF.md` | Current milestone evidence, owner learning status, next slice |
| `docs/PROJECT_REASSESSMENT.md` | Dated code assessment and recruiting rationale |
| `frontend/README.md` | Frontend-specific setup and architecture |
| `ml/README.md` | Experimental ML assets and fine-tuning prerequisites |

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
  ├─ LLM: 4 API providers plus Custom/Hybrid execution modes
  ├─ Voice: batch Deepgram STT and ElevenLabs TTS (optional, not yet verified)
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

`LLM_MODE=local` is present in settings but is not connected to the active service
factory. Use the API or Custom/Hybrid interview paths described by the application;
do not rely on automatic local LoRA loading.

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

Dated verification results and limitations are recorded in
[the roadmap](DEVELOPMENT_ROADMAP.md#verified-baseline-and-immediate-risks) and
[the handoff](docs/REFACTOR_HANDOFF.md#verification-evidence). Passing checks are not
proof that startup, scoring, recovery, or live providers work correctly.

## API overview

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/resume/upload` | Parse and save a PDF/TXT resume |
| GET | `/api/v1/resume/{resume_id}` | Resume summary |
| DELETE | `/api/v1/resume/{resume_id}` | Delete resume data |
| POST | `/api/v1/resume/{resume_id}/generate-questions` | Legacy standalone question generation |
| GET | `/api/v1/resume/{resume_id}/chunks` | Legacy ChromaDB chunk inspection |
| POST | `/api/v1/sessions` | Create session and generate first question |
| GET | `/api/v1/sessions/{session_id}` | Current checkpointed state |
| POST | `/api/v1/sessions/{session_id}/answer` | Submit an answer |
| POST | `/api/v1/sessions/{session_id}/skip` | Skip current question |
| POST | `/api/v1/sessions/{session_id}/end` | End early |
| GET | `/api/v1/sessions/{session_id}/summary` | Interview summary |
| DELETE | `/api/v1/sessions/{session_id}` | Delete checkpoint thread |

WebSocket endpoint: `/ws/interview/{session_id}`. Client message types are
`start`, `answer`, `answer_audio`, `skip`, `end`, and `ping`.

Voice routes use the `/api/v1/voice` prefix. The current audio path processes a
completed recording rather than streaming audio. Voice is not part of the verified
baseline because the root dependency set does not currently include the ElevenLabs
SDK. The safe `.env.example` disables voice, while the Python setting still defaults
to enabled when no environment value is provided; M0 resolves that code/dependency
contract.

## Data and compatibility

Runtime data is ignored by Git:

- `data/uploads/`
- `data/parsed_resumes/*.md`
- `data/chromadb/`
- `data/interview_checkpoints.db`

Only synthetic, explicitly documented fixtures may be committed. Currently tracked
PDF/TXT resume samples and the LoRA notebook still require an explicit provenance and
PII determination under roadmap milestone M0. Two PDFs are byte-identical. Do not use
any of these artifacts as real-user examples until that gate is resolved.

Do not delete ChromaDB while legacy resume IDs are still needed. New resumes do
not create embeddings; the legacy embedder is initialized lazily only for an old
resume with no Markdown representation.

## Learning-oriented refactor

Follow [DEVELOPMENT_ROADMAP.md](DEVELOPMENT_ROADMAP.md), which supersedes the previous
M0–M8 plan with an evidence-gated M0–M9 plan:

1. M0: system ownership, data provenance and voice support baseline.
2. M1–M2: correct text interview, validated model contracts and deterministic CI.
3. M3–M4: human-reviewed benchmark, durable feedback and coaching report.
4. M5–M6: retry/restart recovery and product/dependency simplification.
5. M7–M8: optional JD targeting and adaptive-selection experiment.
6. M9: controlled deployment; does not depend on optional M7/M8.

No functional refactor is complete merely because the plan is documented. Public beta
and real-user data processing remain blocked until M9 passes. A local synthetic-data
demo can be a useful portfolio result at M4 without adding optional technologies.

For work on another machine or with another coding agent, begin with
[docs/REFACTOR_HANDOFF.md](docs/REFACTOR_HANDOFF.md).
