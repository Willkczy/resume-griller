# Contributing Guide

This document defines the development workflow. See the root `README.md` for the
implemented architecture and `DEVELOPMENT_ROADMAP.md` for milestone priority and
acceptance gates.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 20+
- Git

## Setup

From the repository root:

```bash
uv sync --locked --extra dev
cp .env.example .env
```

Configure the selected interview provider. New resume parsing always requires
`GROQ_API_KEY`, even when another provider conducts the interview.

```bash
cd frontend
npm ci
cp .env.example .env.local
```

The root `pyproject.toml` and `uv.lock` are the only Python dependency sources.

## Run locally

Backend, from the repository root:

```bash
PYTHONPATH=. uv run uvicorn backend.app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

The frontend is available at `http://localhost:3000`, and generated backend API
documentation is at `http://localhost:8000/docs`.

The supported baseline is text-only: settings and `.env.example` disable voice.
Disabled voice paths require no credentials or ElevenLabs SDK. Live voice is still
experimental; see [the decision](decisions/0001-text-only-baseline.md).

## Branch and commit conventions

Inspect the current branch and worktree before starting. Preserve changes you do not
own. Human contributors may use `feature/`, `fix/`, `refactor/`, or `docs/` prefixes;
Codex-created branches use the configured `codex/` prefix.

Commit messages use:

```text
type(scope): short description
```

Common types are `feat`, `fix`, `docs`, `refactor`, `test`, and `chore`. Common scopes
are `backend`, `frontend`, `graph`, `rag`, and `ml`.

For the completed M0–M1 sequence, see the [commit reading guide](M0_M1_COMMIT_GUIDE.md)
for code paths, verification evidence and read-only history inspection commands.

Keep commits focused. A change to AI behavior should include its contract tests,
evaluation case or rationale, evaluation delta, and documentation in the same pull
request when practical.

## Quality checks

Backend tests:

```bash
PYTHONPATH=. uv run --frozen --extra dev pytest tests/
PYTHONPATH=. uv run --frozen --extra dev pytest tests/ --cov=backend --cov=rag \
  --cov-report=term-missing
```

Python quality:

```bash
uv run --frozen --extra dev ruff check backend rag/retriever.py tests scripts/test_graph.py --config pyproject.toml
uv run --frozen --extra dev black --check backend rag/retriever.py tests scripts/test_graph.py
```

Apply Python formatting when needed:

```bash
uv run --extra dev black backend rag/retriever.py tests
uv run --extra dev ruff check --fix backend rag/retriever.py tests \
  --config pyproject.toml
```

Frontend quality:

```bash
cd frontend
npm test
npx tsc --noEmit --incremental false
npm run lint
NEXT_TELEMETRY_DISABLED=1 npm run build
```

`npm test` compiles and tests the real outcome presentation/draft-reconciliation
helper with synthetic fixtures using Node's test runner. It needs no provider key or
additional test dependency. It does not exercise camera/microphone devices or replace
browser and recovery testing.

The graph smoke script uses mock services and needs no API key:

```bash
PYTHONPATH=. uv run --frozen --extra dev python scripts/test_graph.py
```

## Continuous integration

[The CI workflow](../.github/workflows/ci.yml) runs `Backend checks` and `Frontend
checks` on pushes, pull requests and manual dispatch. Both jobs must pass for a
reviewable change. The workflow grants read-only repository access, pins actions by
commit, disables persisted checkout credentials and cancels superseded runs. There
are no deployment steps, provider secrets or runtime-data artifacts. Require these
job checks in repository branch protection separately if merge enforcement is desired.

The workflow is the source for runner/tool versions and exact commands. Python uses
`uv sync --locked --extra dev` so a stale lockfile fails rather than being rewritten;
subsequent steps use `--frozen`. Frontend uses `npm ci`. Caches contain dependency
packages, and the full install is still required on a cache hit. The root legacy ML
runtime dependencies are still installed; their simplification remains M6.

`tests/conftest.py` isolates the backend before collection: Pydantic settings ignore
local dotenv files, provider credentials are removed, runtime/cache directories are
temporary, voice/custom health probes and tracing are disabled, and model downloads
are offline. Tests mock provider and Chroma calls. An autouse guard rejects DNS and
IPv4/IPv6 connections during tests; attempted I/O also fails teardown when an adapter
catches the original exception. Unix sockets used by asyncio and in-process FastAPI
TestClient requests remain available. The isolation checks run child pytest processes
to prove these failures cannot silently become a green provider-failure test.

Run the quality commands above to reproduce the job steps locally. Dependency setup
requires registry access; the deterministic tests require no live providers, model
downloads or personal environment files. To validate workflow syntax when Actionlint
is installed, run `actionlint .github/workflows/ci.yml`. Local success is evidence for
the commands tested; inspect both job results on the proposed PR revision for hosted
CI evidence. A workflow file does not by itself enforce a merge rule or demonstrate
model quality, browser/device support or M5 recovery guarantees.

## Where changes belong

| Area | Location |
|---|---|
| Graph state and orchestration | `backend/app/graph/` |
| Answer scoring and follow-ups | `backend/app/core/grilling_engine.py` |
| REST and WebSocket transports | `backend/app/api/routes/` |
| Provider implementations | `backend/app/services/llm_service.py` |
| Resume parsing and compatibility lookup | `rag/retriever.py` |
| Frontend pages and components | `frontend/src/` |
| Backend tests | `tests/` |
| Future AI benchmark | roadmap-defined evaluation package/datasets |

Do not reintroduce the deleted `InterviewAgent` or in-memory session store. Do not
place production code under `scripts/`, and do not extend legacy Chroma components
for new interviews unless a measured multi-document retrieval requirement is added.

## Dependencies

Add a Python runtime dependency with:

```bash
uv add package-name
```

Add a Python development dependency with:

```bash
uv add --dev package-name
```

Add a frontend dependency with:

```bash
cd frontend
npm install package-name
```

Commit the corresponding lockfile. Do not hand-edit generated lockfiles.

## Definition of done

A normal change is complete when:

- relevant success and failure tests pass;
- Python and frontend checks for the touched area pass;
- public schemas and TypeScript types remain aligned;
- sensitive data is not logged or committed;
- documentation reflects any changed contract;
- migration and rollback behavior are documented when state or storage changes.

For AI behavior changes, also follow the roadmap's evaluation-specific definition of
done and milestone gate.

## Learning-oriented slices and agent handoffs

Read [the roadmap](../DEVELOPMENT_ROADMAP.md) and
[the current handoff](REFACTOR_HANDOFF.md) before selecting work. Use the revised
M0–M9 dependencies; optional M7/M8 are not prerequisites for M9 deployment.

Each slice should state its invariant, owner exercise, agent scope and acceptance checks.
Use the real graph in API/graph integration tests while mocking external services and
using temporary storage. Do not replace both sides of the seam under test with fakes.
Before the benchmark exists, record deterministic regression evidence and explicitly
state that model-quality improvement has not been measured. Later AI changes include
versioned benchmark evidence, failure analysis and the measured delta.

Update REFACTOR_HANDOFF after each slice with actual commands/results, changed files,
owner explanation status, unresolved risks and exact next step. Owner delegation permits
agent implementation of that slice; it does not prove the owner completed the exercise.
Do not add a new technology or start an optional track solely to match a job posting.

On a new machine, fetch/switch to the pushed branch, preserve local edits, and verify
status before installing dependencies. Copy environment examples only when the local
files do not exist. Credentials and runtime data are not transferred through Git.


### Model execution diagnostics

`tests/test_model_execution.py` uses synthetic text and actual SDK response schemas
with mocked transports. It covers nullable usage, provider identity, deadlines,
concurrent-call isolation, finish-reason rejection, safe logs and checkpoint/transport
parity. No provider key or network call is required. Inspect `model_executions` in a
synthetic session response to correlate an application `call_id` with safe logs.
Do not enable SDK request-body logging or log model inputs/outputs when debugging.
See [decision 0003](decisions/0003-model-output-contracts.md) for counter semantics,
metadata lifetime and the pending owner exercise.
