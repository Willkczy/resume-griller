# Resume Griller Frontend

The frontend is a Next.js 16 / React 19 application for resume upload, interview
configuration, text or recorded-audio interviews, and the final session summary.

The root `README.md` is the source of truth for the complete system, and
`DEVELOPMENT_ROADMAP.md` defines planned product work.

## Setup

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

Environment variables:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

Those localhost values are also the application defaults.

## Code map

| Area | Location |
|---|---|
| App Router pages | `src/app/` |
| Interview clients | `src/components/interview/` |
| Resume upload/configuration | `src/components/upload/` |
| REST client | `src/lib/api.ts` |
| WebSocket client | `src/lib/websocket.ts` |
| Shared API/WS types | `src/types/index.ts` |
| Interview state | Currently local state in interview components; `src/stores/interviewStore.ts` is unused by those controllers |

The REST client uses `/api/v1/resume` and `/api/v1/sessions`. The WebSocket client
connects to `/ws/interview/{session_id}` and sends `start`, `answer`, `answer_audio`,
`skip`, `end`, or `ping` messages.

Audio recording is currently batch-based: the browser records a complete clip and
sends it for transcription after recording stops. It is not streaming STT.

## Quality checks

```bash
cd frontend
npx tsc --noEmit
npm run lint
npm run build
```

There is currently no frontend test framework or `npm test` script. Adding component
and end-to-end coverage is planned production-readiness work.

## Current result-page limitation

The result page mainly presents session counts and follow-up statistics. It does not
yet receive a persistent per-turn 7D evaluation history. Roadmap M4 introduces the
evaluation ledger and evidence-based coaching report; avoid adding more inferred
insights from follow-up counts before that data contract exists.

## Security boundary

The current API has no authentication or ownership checks. Do not deploy the frontend
for real-user resumes or expose it as a public beta before roadmap M9 passes.

## Refactor sequence

Use the root roadmap and `docs/REFACTOR_HANDOFF.md`; old milestone numbering is
obsolete. M2 synchronizes runtime contracts/types and adds CI. M4 adds evidence-backed
reports. M5 restores actual turns/history and safe draft/retry behavior. M6 shares the
interview controller, defaults to text, supports parse/transcript review, checks keyboard
access, and hides unsupported voice/custom behavior. JD/adaptive UI is optional M7/M8.

The current video read-aloud control is a timer placeholder. Camera preview is local;
no video analysis is implemented. Follow-up counts do not establish answer quality.
